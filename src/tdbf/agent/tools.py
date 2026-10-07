"""Gold-free tools the BRT agent calls, backed by a ContainerSession.

Every tool here operates on the **buggy checkout only** and reuses the pure
parsing/path/preamble helpers from `harness/runner.py` (so outcome semantics
match the non-agentic baseline exactly — only the execution substrate changes
from `docker run --rm` to `docker exec` on a live session).

GOLD-FREE INVARIANT (the validity control): no tool applies the gold patch or
runs against the fixed tree. `run_test` runs on buggy code only and returns the
raw fail/error signal; whether the test *passes on the fixed tree* (the L4
question) is scored offline, never exposed to the agent. `assert_gold_free()`
guards the registry against a fixed-tree tool slipping in.
"""

from __future__ import annotations

import json

from ..harness.runner import (
    _REPO_PREAMBLES,
    DJANGO_BRT_APP,
    ContainerResult,
    _parse_pytest_outcome,
    brt_module,
    brt_run_spec,
    clean_brt_output,
    native_brt,
    parse_brt_result,
    repo_test_command,
    repo_test_label,
    set_brt_module,
    valid_brt_module,
)
from .session import ContainerSession, ExecResult
from .types import Tool, ToolResult

_MAX_OUTPUT = 2500  # chars of tool output fed back to the model


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped, {len(text) - limit} more chars]"


def run_brt_in_session(session: ContainerSession, code: str, *,
                       test_name: str = "test_brt",
                       timeout: int = 180) -> tuple[ContainerResult, ExecResult]:
    """Write a BRT into the live (buggy or agent-edited) tree and run it the
    repo's way. The ONE place a BRT is executed in a session: run_test, run_brt,
    the gold-free gate and the fix loop's accept check all come through here, so
    they cannot disagree about placement, command or outcome. Gold-free: it only
    ever sees the tree the session holds."""
    spec = brt_run_spec(session.repo, code, test_name=test_name)
    for rel, content in spec.files.items():
        session.write_file(f"/testbed/{rel}", content)
    if spec.needs_pytest:
        session.ensure_pytest()
    res = session.exec(spec.command, timeout=timeout)
    outcome, reason = parse_brt_result(session.repo, res.returncode, res.output)
    return ContainerResult(outcome, reason, res.output[-3000:]), res


class _BrtModuleState:
    """The Django test package the BRT agent last ran its test in, shared by
    run_test and submit_test so a submission without `module` lands where the
    agent actually ran it."""

    def __init__(self) -> None:
        self.module: str | None = None


def _resolve_module(session: ContainerSession, requested: object,
                    code: str, state: _BrtModuleState | None) -> tuple[str | None, str | None]:
    """(module, error). Order: the `module` argument, the code's own header, the
    module of the last run_test, then the fallback app."""
    name = str(requested or "").strip().strip("/").replace("/", ".")
    if name.startswith("tests."):
        name = name[len("tests."):]
    if name:
        if not valid_brt_module(name):
            return None, (f"'module' must be a test package name under tests/, like "
                          f"'admin_views' (got {requested!r}).")
        if name != DJANGO_BRT_APP:
            # Must be a package of the ORIGINAL checkout (tracked in git), not one
            # the agent created with run_python: the gate runs in this session, but
            # the fix agent and the label run in fresh containers where a package
            # the agent made would not exist (seen in the 10-07 pilot).
            res = session.exec('git cat-file -e "HEAD:tests/$MODPATH/__init__.py"',
                               env={"MODPATH": name.replace(".", "/")})
            if res.returncode != 0:
                return None, (f"tests/{name.replace('.', '/')}/ is not a test package of this "
                              "repository. Use an existing one (glob pattern "
                              "'tests/[a-z_]*/__init__.py' lists them) or omit 'module' to use the "
                              f"empty app '{DJANGO_BRT_APP}'. Packages you create yourself are not "
                              "available when the test is used later.")
        return name, None
    return (brt_module(code) or (state.module if state else None) or DJANGO_BRT_APP), None


_DJANGO_MODULE_PROP = {
    "type": "string",
    "description": ("Existing test package under tests/ to run in, e.g. 'admin_views' "
                    "(its models and fixtures are available). Omit for an empty app of "
                    f"your own ('{DJANGO_BRT_APP}')."),
}


class _SessionTool(Tool):
    """Base for tools sharing one container session."""

    def __init__(self, session: ContainerSession) -> None:
        self.session = session


def _as_int(value: object) -> int | None:
    """Best-effort int from a possibly-string tool argument; None if absent/invalid."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ReadFileTool(_SessionTool):
    name = "read_file"
    description = (
        "Read a source file from the (buggy) repository checkout. Path is relative to the repo root. "
        "Output is prefixed with line numbers ('123\\t<code>') to help you locate code. "
        "Optionally pass start_line/end_line (1-based, inclusive) to read just that range — use this "
        "to jump straight to the code you found with grep instead of re-reading the whole file. "
        "NOTE: the 'N\\t' line-number prefix is a display aid only; never include it in edit_file's old_str."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to repo root, e.g. 'astropy/table/table.py'."},
            "start_line": {"type": "integer", "description": "Optional 1-based first line to read."},
            "end_line": {"type": "integer", "description": "Optional 1-based last line to read (inclusive)."},
        },
        "required": ["path"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        path = str(arguments.get("path", "")).strip()
        if not path:
            return ToolResult("read_file requires a non-empty 'path'.", is_error=True)
        # Read whole file (no byte cap) so line ranges land on real line numbers;
        # output to the model is still bounded by _clip below.
        res = self.session.read_file(path)
        if res.returncode != 0:
            return ToolResult(f"Could not read {path!r}: {_clip(res.output, 400)}", is_error=True)
        lines = res.output.splitlines()
        n = len(lines)
        start = _as_int(arguments.get("start_line"))
        end = _as_int(arguments.get("end_line"))
        # A line argument that is present but not an integer is a malformed call,
        # not an absent one. Dropping it silently returned the file from line 1:
        # a live free model leaked tool-call markup into start_line
        # ('219\n<parameter=end_line>\n250') and got the same wrong page three
        # turns running with no hint of why.
        for key, parsed in (("start_line", start), ("end_line", end)):
            raw = arguments.get(key)
            if parsed is None and raw not in (None, ""):
                return ToolResult(
                    f"read_file {key} must be a single integer, got {raw!r}. "
                    "Pass start_line and end_line as separate integer arguments.",
                    is_error=True,
                )
        if start is not None or end is not None:
            first = max(1, start or 1)
            last = min(n, end if end is not None else n)
            if first > n:
                return ToolResult(
                    f"start_line {first} is past end of {path!r} ({n} lines).", is_error=True
                )
            selected = lines[first - 1 : last]
        else:
            first = 1
            selected = lines
        width = len(str(first + len(selected) - 1)) if selected else 1
        numbered = "\n".join(f"{first + i:>{width}}\t{ln}" for i, ln in enumerate(selected))
        return ToolResult(_clip(numbered))


class GrepTool(_SessionTool):
    name = "grep"
    description = "Search the repository for a regular expression (ripgrep-style) across Python files. Returns matching path:line:text."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Pattern to search for (passed to grep -E)."},
            "path": {"type": "string", "description": "Optional subdir to limit the search (relative to repo root). Defaults to whole repo."},
        },
        "required": ["pattern"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return ToolResult("grep requires a non-empty 'pattern'.", is_error=True)
        path = str(arguments.get("path", "") or ".").strip()
        # Pattern/path passed via env so they cannot be shell-injected.
        res = self.session.exec(
            'grep -rnE --include="*.py" -- "$PAT" "$GPATH" | head -n 120',
            env={"PAT": pattern, "GPATH": path},
        )
        out = res.output.strip()
        if not out:
            return ToolResult(f"No matches for pattern {pattern!r} under {path!r}.")
        return ToolResult(_clip(out))


class GlobTool(_SessionTool):
    name = "glob"
    description = "List tracked repository files whose path matches a substring or regex. Useful to locate modules before reading them."
    input_schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string", "description": "Substring/regex to filter file paths, e.g. 'table' or 'io/ascii'."}},
        "required": ["pattern"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return ToolResult("glob requires a non-empty 'pattern'.", is_error=True)
        res = self.session.exec(
            'git ls-files | grep -E -- "$PAT" | head -n 200',
            env={"PAT": pattern},
        )
        out = res.output.strip()
        if not out:
            return ToolResult(f"No tracked files match {pattern!r}.")
        return ToolResult(_clip(out))


class RunPythonTool(_SessionTool):
    name = "run_python"
    description = (
        "Run a Python snippet inside the buggy repository environment and return its stdout "
        "(or the traceback if it raises). Use this to observe the actual buggy behaviour before "
        "writing assertions."
    )
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute."}},
        "required": ["code"],
    }

    # Django: the snippet runs inside a TestCase under runtests.py, so it sees the
    # same apps, models and test database as the BRT will (a bare `python` probe
    # had INSTALLED_APPS empty and crashed on any model import).
    DJANGO_DESCRIPTION = (
        "Run a Python snippet inside the buggy Django checkout, under Django's own test runner: "
        "the standard contrib apps are installed, a test database exists, and with `module` that "
        "test package's models are available too. Returns the snippet's printed output (or the "
        "traceback if it raises). Use this to observe the actual buggy behaviour before writing "
        "assertions."
    )
    _DJANGO_PROBE = (
        "from django.test import TestCase\n\n\n"
        "class TdbfProbe(TestCase):\n"
        "    def test_probe(self):\n"
        "        with open('/tmp/tdbf_probe.py') as f:\n"
        "            src = f.read()\n"
        "        exec(compile(src, '<run_python>', 'exec'), {'__name__': '__main__'})\n"
    )

    def __init__(self, session: ContainerSession) -> None:
        super().__init__(session)
        if native_brt(session.repo):
            self.description = self.DJANGO_DESCRIPTION
            self.input_schema = {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute."},
                    "module": _DJANGO_MODULE_PROP,
                },
                "required": ["code"],
            }

    def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        if not str(code).strip():
            return ToolResult("run_python requires non-empty 'code'.", is_error=True)
        if native_brt(self.session.repo):
            return self._execute_native(str(code), arguments.get("module"))
        preamble = _REPO_PREAMBLES.get(self.session.repo, "")
        self.session.write_file("/tmp/tdbf_probe.py", preamble + code)
        res = self.session.exec("python /tmp/tdbf_probe.py", timeout=90)
        body = res.output.strip()
        if not body:
            body = "(no output)" if res.returncode == 0 else "(no output; non-zero exit)"
        tag = "" if res.returncode == 0 else f"[exit {res.returncode}] "
        return ToolResult(_clip(tag + body))

    def _execute_native(self, code: str, requested: object) -> ToolResult:
        module, err = _resolve_module(self.session, requested, "", None)
        if err:
            return ToolResult(err, is_error=True)
        self.session.write_file("/tmp/tdbf_probe.py", code)
        result, res = run_brt_in_session(
            self.session, set_brt_module(self._DJANGO_PROBE, module),
            test_name="test_tdbf_probe", timeout=90)
        body = clean_brt_output(self.session.repo, res.output).strip() or "(no output)"
        tag = "" if result.outcome == "pass" else "[the snippet raised — see the traceback] "
        return ToolResult(_clip(tag + body))


class RunTestTool(_SessionTool):
    name = "run_test"
    description = (
        "Write the given pytest test into the repository and run it against the BUGGY code. "
        "Returns the outcome (fail/pass/error) and pytest output. A good Bug Reproduction Test "
        "should FAIL here for the reason described in the issue (an assertion failure, not an "
        "import/setup error). The test is NOT run against any fixed code — that is scored separately."
    )
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Full pytest test module source (framework setup is injected for you)."}},
        "required": ["code"],
    }

    DJANGO_DESCRIPTION = (
        "Write the given test module to tests/<module>/test_brt.py and run it with Django's own test "
        "runner (./tests/runtests.py <module>.test_brt) against the BUGGY code. Returns outcome "
        "(fail/pass/error) and, for a failure, failure_reason: 'assertion' (an assertion fired), "
        "'exception' (the code under test raised) or 'runtime' (the test itself is broken: an "
        "import/name/syntax error or a test-setup mistake). A good Bug Reproduction Test FAILS here "
        "with 'assertion' or 'exception', for the reason described in the issue. The test is NOT "
        "run against any fixed code — that is scored separately."
    )
    DJANGO_SCHEMA = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Full test module source: django.test TestCase / SimpleTestCase classes."},
            "module": _DJANGO_MODULE_PROP,
        },
        "required": ["code"],
    }

    def __init__(self, session: ContainerSession, state: _BrtModuleState | None = None) -> None:
        super().__init__(session)
        self.state = state
        if native_brt(session.repo):
            self.description = self.DJANGO_DESCRIPTION
            self.input_schema = self.DJANGO_SCHEMA

    def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        if not str(code).strip():
            return ToolResult("run_test requires non-empty 'code'.", is_error=True)
        if native_brt(self.session.repo):
            module, err = _resolve_module(self.session, arguments.get("module"), str(code), self.state)
            if err:
                return ToolResult(err, is_error=True)
            if self.state is not None:
                self.state.module = module
            code = set_brt_module(str(code), module)
        result, res = run_brt_in_session(self.session, str(code))
        header = f"outcome={result.outcome}" + (
            f" failure_reason={result.failure_reason}" if result.failure_reason else "")
        return ToolResult(f"{header}\n\n{_clip(clean_brt_output(self.session.repo, res.output))}")


class SubmitTestTool(Tool):
    """Terminal tool: the agent calls this to declare its final BRT and stop.

    Captures the test code as the run's artifact (offline scoring takes it from
    here). It does not itself run anything against the fixed tree.
    """

    name = "submit_test"
    description = (
        "Submit your final Bug Reproduction Test and finish. Call this only once you have used "
        "run_test to confirm the test fails cleanly on the buggy code for the right reason. "
        "Provide the complete test module source."
    )
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The final, complete pytest test module."}},
        "required": ["code"],
    }
    terminal = True

    DJANGO_DESCRIPTION = (
        "Submit your final Bug Reproduction Test and finish. Call this only once you have used "
        "run_test to confirm the test fails on the buggy code for the right reason "
        "(failure_reason 'assertion' or 'exception'). Provide the complete test module source and "
        "the same `module` you ran it in."
    )
    DJANGO_SCHEMA = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The final, complete test module."},
            "module": _DJANGO_MODULE_PROP,
        },
        "required": ["code"],
    }

    def __init__(self, session: ContainerSession | None = None,
                 state: _BrtModuleState | None = None) -> None:
        self.session = session
        self.state = state
        if session is not None and native_brt(session.repo):
            self.description = self.DJANGO_DESCRIPTION
            self.input_schema = self.DJANGO_SCHEMA

    def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        if not str(code).strip():
            return ToolResult("submit_test requires non-empty 'code'.", is_error=True)
        if self.session is not None and native_brt(self.session.repo):
            module, err = _resolve_module(self.session, arguments.get("module"), str(code), self.state)
            if err:
                return ToolResult(err, is_error=True)
            return ToolResult(set_brt_module(str(code), module))
        return ToolResult(str(code))


# ============================ Phase B: fix tools ============================ #
# These tools MUTATE the buggy checkout. They remain gold-free: none applies the
# gold patch, and none runs the held-out FAIL_TO_PASS / test_patch. The agent's
# only success signal is the validated BRT (its own reproduction, injected at
# construction so the agent cannot weaken it) plus existing repo tests it chooses
# to run. Whether the produced fix actually *resolves* the issue (held-out
# test_patch) is scored offline, never exposed to the agent.


class EditFileTool(_SessionTool):
    name = "edit_file"
    description = (
        "Edit a source file in place by replacing an exact string. `old_str` must appear in the "
        "file; every occurrence is replaced with `new_str`. Read the file first to get an exact "
        "match (including indentation). Do NOT include read_file's 'N\\t' line-number prefix in "
        "old_str — match only the real file text. Use this to apply your fix to the buggy code."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to repo root."},
            "old_str": {"type": "string", "description": "Exact existing text to replace."},
            "new_str": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_str", "new_str"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        path = str(arguments.get("path", "")).strip()
        old = arguments.get("old_str", "")
        new = arguments.get("new_str", "")
        if not path:
            return ToolResult("edit_file requires a non-empty 'path'.", is_error=True)
        # Only repo files can be edited: the fix is captured as `git diff` of the
        # checkout. An absolute path used to be READ from the real file but WRITTEN
        # to a new copy at /testbed/<path> -- "Edited" was reported and nothing
        # changed. A path that climbs out with ".." would vanish from the diff too.
        if path.startswith("/") or ".." in path.split("/"):
            return ToolResult(
                f"edit_file only edits files inside the repository; give a path relative "
                f"to the repo root (got {path!r}).",
                is_error=True,
            )
        if not isinstance(old, str) or old == "":
            return ToolResult("edit_file requires a non-empty 'old_str'.", is_error=True)
        res = self.session.read_file(path)
        if res.returncode != 0:
            return ToolResult(f"Could not read {path!r}: {_clip(res.output, 400)}", is_error=True)
        content = res.output
        count = content.count(old)
        if count == 0:
            return ToolResult(
                f"'old_str' not found in {path!r}. Read the file and copy the exact text "
                "(including whitespace) you want to replace.",
                is_error=True,
            )
        updated = content.replace(old, new)
        self.session.write_file(f"/testbed/{path}", updated)
        return ToolResult(f"Edited {path}: replaced {count} occurrence(s).")


class RunBrtTool(_SessionTool):
    """Run the validated BRT against the CURRENT (edited) tree.

    The BRT code is fixed at construction — the agent cannot supply or alter it,
    so it cannot make the test pass by weakening the test. This is the agent's
    progress signal: the fix is done when the BRT goes from fail -> pass.
    """

    name = "run_brt"
    description = (
        "Run the bug-reproduction test against the current code. It FAILS now (the bug is unfixed); "
        "your goal is to edit the source until it PASSES. Returns outcome (pass/fail/error) and output."
    )
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, session: ContainerSession, brt_code: str) -> None:
        super().__init__(session)
        # Placement and preamble come from brt_run_spec (Django: the module named
        # in the BRT's header, no preamble — so no double django.setup()).
        self._brt_code = brt_code

    def execute(self, arguments: dict) -> ToolResult:
        result, res = run_brt_in_session(self.session, self._brt_code)
        return ToolResult(f"outcome={result.outcome}\n\n"
                          f"{_clip(clean_brt_output(self.session.repo, res.output))}")


class RunTestsTool(_SessionTool):
    name = "run_tests"
    description = (
        "Run a SPECIFIC existing test file from the repo (e.g. 'astropy/modeling/tests/"
        "test_separable.py') to check your fix did not break related behaviour. Point it at the test "
        "module nearest your change — do NOT pass whole directories (slow, and they contain failures "
        "unrelated to your fix). A reported failure may be pre-existing/environmental, not a regression "
        "you caused. Returns pytest output."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Existing test file path relative to repo root (a file, not a directory)."}
        },
        "required": ["path"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        path = str(arguments.get("path", "")).strip()
        if not path:
            return ToolResult("run_tests requires a non-empty 'path'.", is_error=True)
        # Some repos cannot run their own suite under bare pytest (Django needs its
        # runner + a settings module, addressed by dotted label). repo_test_command
        # picks the invocation the SWE-bench grader itself uses for this repo, so a
        # regression the grader would catch is one the agent can see too.
        command, uses_label = repo_test_command(self.session.repo)
        if uses_label:
            label = repo_test_label(self.session.repo, path)
            if not label:
                return ToolResult(
                    f"Could not resolve {path!r} to a test module for "
                    f"{self.session.repo}. Pass a path to a single existing test "
                    "file, e.g. 'tests/check_framework/test_templates.py'.",
                    is_error=True,
                )
            env = {"TLABEL": label}
        else:
            self.session.ensure_pytest()
            env = {"TPATH": path}
        # 180s cap (matches run_brt): bounds wall-clock if the model still points
        # at a broad/slow path despite the guidance to use a single test file.
        res = self.session.exec(command, timeout=180, env=env)
        outcome = _parse_pytest_outcome(res.returncode, res.output)
        note = ""
        if res.returncode == 124:  # ExecResult timeout sentinel
            note = ("\n\n[run_tests hit the 180s timeout — the path is too broad. "
                    "Re-run on a single specific test file instead of a directory.]")
        return ToolResult(f"outcome={outcome}\n\n{_clip(res.output)}{note}")


class RevertTool(_SessionTool):
    name = "revert"
    description = (
        "Discard ALL your source edits and restore the original buggy checkout. Use this to start "
        "over if your changes made things worse. The bug-reproduction test is preserved."
    )
    input_schema = {"type": "object", "properties": {}}

    # Reactive arm: the same text minus the sentence about a BRT it does not have.
    REACTIVE_DESCRIPTION = (
        "Discard ALL your source edits and restore the original buggy checkout. Use this to start "
        "over if your changes made things worse."
    )

    def __init__(self, session: ContainerSession, *, reactive: bool = False) -> None:
        super().__init__(session)
        if reactive:
            self.description = self.REACTIVE_DESCRIPTION

    def execute(self, arguments: dict) -> ToolResult:
        self.session.reset_tree()
        return ToolResult("Reverted all source edits to the original buggy state.")


class SubmitFixTool(Tool):
    """Terminal tool: the agent declares its fix complete and stops.

    Carries no artifact itself — the fix patch is captured offline as `git diff`
    of the (tracked) source tree. The untracked BRT file is naturally excluded.
    """

    name = "submit_fix"
    description = (
        "Submit your fix and finish. Call this only once run_brt reports outcome=pass AND any "
        "existing tests you ran still pass. Your source edits are captured automatically."
    )
    # Reactive arm (RQ1 baseline): identical except it cannot name run_brt, a tool
    # that arm does not have.
    REACTIVE_DESCRIPTION = (
        "Submit your fix and finish. Call this only once your change addresses the issue AND any "
        "existing tests you ran still pass. Your source edits are captured automatically."
    )
    input_schema = {"type": "object", "properties": {}}
    terminal = True

    def __init__(self, *, reactive: bool = False) -> None:
        if reactive:
            self.description = self.REACTIVE_DESCRIPTION

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult("FIX_SUBMITTED")


# ====================== Phase C: review-role tools ========================= #
# The Critic (Phase A) and Reviewer (Phase B) are advisory roles. They inspect
# the buggy tree read-only and return a structured verdict — they NEVER write a
# test, edit source, or (of course) touch the gold/fixed tree. submit_review is
# their only terminal tool; its artifact is a JSON {verdict, feedback} the
# orchestrator parses to decide accept vs. revise.


class SubmitReviewTool(Tool):
    """Terminal tool for a critic/reviewer role: record a structured verdict."""

    name = "submit_review"
    description = (
        "Submit your review verdict and finish. `verdict` must be 'accept' (the artifact is good "
        "enough to proceed) or 'revise' (it needs specific, fixable changes). When verdict='revise', "
        "`feedback` MUST give concrete, actionable instructions the author can act on. Call once."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["accept", "revise"],
                "description": "'accept' to proceed as-is, or 'revise' to send it back with feedback.",
            },
            "feedback": {
                "type": "string",
                "description": "Concrete reasoning; required actionable guidance when verdict='revise'.",
            },
        },
        "required": ["verdict"],
    }
    terminal = True

    def execute(self, arguments: dict) -> ToolResult:
        verdict = str(arguments.get("verdict", "")).strip().lower()
        if verdict not in ("accept", "revise"):
            return ToolResult("submit_review requires verdict to be 'accept' or 'revise'.", is_error=True)
        feedback = str(arguments.get("feedback", "")).strip()
        if verdict == "revise" and not feedback:
            return ToolResult(
                "verdict='revise' requires non-empty 'feedback' describing exactly what to change.",
                is_error=True,
            )
        return ToolResult(json.dumps({"verdict": verdict, "feedback": feedback}))


class SubmitSelectionTool(Tool):
    """Terminal tool for the Selector role: pick the best candidate BRT by index."""

    name = "submit_selection"
    description = (
        "Select the single best candidate Bug Reproduction Test and finish. `index` is the "
        "0-based number of the candidate you are choosing (as labelled in the task). `reason` "
        "should briefly justify why that test most specifically pins the bug. Call once."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "0-based index of the chosen candidate."},
            "reason": {"type": "string", "description": "Why this candidate is the most specific reproduction."},
        },
        "required": ["index"],
    }
    terminal = True

    def execute(self, arguments: dict) -> ToolResult:
        idx = _as_int(arguments.get("index"))
        if idx is None:
            return ToolResult("submit_selection requires an integer 'index'.", is_error=True)
        reason = str(arguments.get("reason", "")).strip()
        return ToolResult(json.dumps({"index": idx, "reason": reason}))


class SubmitBreakTool(Tool):
    """Terminal tool for the post-fix Breaker: record whether it could defeat the fix."""

    name = "submit_break"
    description = (
        "Submit your verdict on the candidate fix and finish. `verdict` must be 'broken' (you found "
        "a concrete case where the fixed code still violates what the issue says is correct, so the "
        "fix is incomplete despite passing the BRT) or 'robust' (after genuine effort you could not "
        "defeat it). When verdict='broken', `feedback` MUST describe the exact failing case the fix "
        "author should handle. Call once."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["robust", "broken"],
                "description": "'robust' if the fix withstood your attack, 'broken' if you defeated it.",
            },
            "feedback": {
                "type": "string",
                "description": "Concrete failing case / weakness; required when verdict='broken'.",
            },
        },
        "required": ["verdict"],
    }
    terminal = True

    def execute(self, arguments: dict) -> ToolResult:
        verdict = str(arguments.get("verdict", "")).strip().lower()
        if verdict not in ("robust", "broken"):
            return ToolResult("submit_break requires verdict to be 'robust' or 'broken'.", is_error=True)
        feedback = str(arguments.get("feedback", "")).strip()
        if verdict == "broken" and not feedback:
            return ToolResult(
                "verdict='broken' requires non-empty 'feedback' describing the exact case that fails.",
                is_error=True,
            )
        return ToolResult(json.dumps({"verdict": verdict, "feedback": feedback}))


# Names allowed in a BRT agent's registry. Any tool outside this set — in
# particular anything that could touch the gold/fixed tree — is rejected.
_GOLD_FREE_TOOL_NAMES = frozenset(
    {"read_file", "grep", "glob", "run_python", "run_test", "submit_test"}
)

# Names allowed in a review-role registry (Phase C critic/reviewer). Read-only
# inspection of the buggy tree + the terminal verdict. No test-writing, no source
# edits, no gold/fixed-tree tool. run_python (critic: verify expected values) and
# run_tests (reviewer: targeted regression) are the only executors.
_ROLE_TOOL_NAMES = frozenset(
    {"read_file", "grep", "glob", "run_python", "run_tests", "submit_review"}
)

# Names allowed in a Selector registry (best-of-N BRT selection). Read-only
# inspection + run_python (verify the correct value) + the terminal choice. No
# test-writing, no source edits, no gold/fixed-tree tool.
_SELECTOR_TOOL_NAMES = frozenset(
    {"read_file", "grep", "glob", "run_python", "submit_selection"}
)

# Names allowed in a post-fix Breaker registry (Phase C verification gate). The
# breaker runs in the fix agent's session where the candidate fix is already
# applied, so read_file/run_python observe the *agent's own* fixed tree (never
# gold). Read-only inspection + run_python (probe the fix) + the terminal verdict.
# No source edit, no test-write, no run_brt/run_tests, no gold/held-out tool.
_BREAKER_TOOL_NAMES = frozenset(
    {"read_file", "grep", "glob", "run_python", "submit_break"}
)

# Names allowed in a fix agent's registry (Phase B). These mutate the buggy tree
# but stay gold-free: no gold patch, no held-out FAIL_TO_PASS / test_patch.
_FIX_TOOL_NAMES = frozenset(
    {"read_file", "grep", "glob", "edit_file", "run_brt", "run_tests", "revert", "submit_fix"}
)

# The reactive baseline (RQ1 Arm A) is the fix agent WITHOUT run_brt — nothing
# else. Deriving it from _FIX_TOOL_NAMES keeps the arms from drifting apart.
_REACTIVE_FIX_TOOL_NAMES = _FIX_TOOL_NAMES - {"run_brt"}


def _assert_allowed(tools: list[Tool], allowed: frozenset[str]) -> None:
    for t in tools:
        if t.name not in allowed:
            raise ValueError(
                f"Tool {t.name!r} is not in the gold-free allowlist {sorted(allowed)}. "
                "No agent tool may run against the gold/fixed tree or the held-out tests."
            )


def assert_gold_free(tools: list[Tool]) -> None:
    """Guard the BRT-agent tool surface against a fixed-tree / gold tool slipping in.

    This is the programmatic side of the validity invariant: the agent must
    never get a tool that observes the gold-fixed outcome.
    """
    _assert_allowed(tools, _GOLD_FREE_TOOL_NAMES)


def assert_fix_gold_free(tools: list[Tool]) -> None:
    """Guard the fix-agent tool surface (Phase B). The fix tools mutate the buggy
    tree but must never apply the gold patch nor run the held-out test_patch /
    FAIL_TO_PASS — resolution is scored offline only."""
    _assert_allowed(tools, _FIX_TOOL_NAMES)


def assert_reactive_fix_gold_free(tools: list[Tool]) -> None:
    """Guard the reactive-baseline surface: the fix tools minus run_brt. Same
    gold-free rule; and no BRT tool can slip in, or Arm A stops being reactive."""
    _assert_allowed(tools, _REACTIVE_FIX_TOOL_NAMES)


def assert_role_gold_free(tools: list[Tool]) -> None:
    """Guard a review-role tool surface (Phase C critic/reviewer). Roles inspect
    read-only and return a verdict; the same gold-free invariant applies — no
    tool may observe the gold-fixed outcome or the held-out tests."""
    _assert_allowed(tools, _ROLE_TOOL_NAMES)


def assert_selector_gold_free(tools: list[Tool]) -> None:
    """Guard the Selector tool surface (best-of-N BRT selection). The selector
    only inspects and picks — same gold-free invariant, no fixed-tree tool."""
    _assert_allowed(tools, _SELECTOR_TOOL_NAMES)


def assert_breaker_gold_free(tools: list[Tool]) -> None:
    """Guard the post-fix Breaker tool surface. The breaker probes the agent's own
    applied fix read-only and returns a verdict; the same gold-free invariant
    applies — no source edit, no test-write, and no tool that observes the gold
    patch or the held-out FAIL_TO_PASS / test_patch."""
    _assert_allowed(tools, _BREAKER_TOOL_NAMES)


def brt_agent_tools(session: ContainerSession) -> list[Tool]:
    """The standard gold-free toolset for a single BRT agent (Phase A)."""
    state = _BrtModuleState()  # Django: the module run_test last used, for submit_test
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        RunPythonTool(session),
        RunTestTool(session, state),
        SubmitTestTool(session, state),
    ]
    assert_gold_free(tools)
    return tools


def fix_agent_tools(session: ContainerSession, brt_code: str) -> list[Tool]:
    """The gold-free toolset for a fix agent (Phase B).

    `brt_code` is the validated BRT the fix must make pass; it is baked into
    run_brt so the agent cannot alter the success criterion.
    """
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        EditFileTool(session),
        RunBrtTool(session, brt_code),
        RunTestsTool(session),
        RevertTool(session),
        SubmitFixTool(),
    ]
    assert_fix_gold_free(tools)
    return tools


def reactive_fix_agent_tools(session: ContainerSession) -> list[Tool]:
    """The reactive baseline's toolset (RQ1 Arm A): `fix_agent_tools` minus run_brt.

    Same tools, same order, same budgets. The agent may still run existing repo
    tests; it is simply under no obligation to observe a failing test first,
    which is what "reactive" means here. Descriptions that would name run_brt or
    a BRT use their reactive variants.
    """
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        EditFileTool(session),
        RunTestsTool(session),
        RevertTool(session, reactive=True),
        SubmitFixTool(reactive=True),
    ]
    assert_reactive_fix_gold_free(tools)
    return tools


def critic_agent_tools(session: ContainerSession) -> list[Tool]:
    """Read-only gold-free toolset for the Phase A Critic (BRT specificity review).

    The critic inspects the issue + candidate test and may use run_python to
    verify what the *correct* value should be — but it cannot write a test, edit
    source, or see the gold-fixed tree. It ends by calling submit_review.
    """
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        RunPythonTool(session),
        SubmitReviewTool(),
    ]
    assert_role_gold_free(tools)
    return tools


def reviewer_agent_tools(session: ContainerSession) -> list[Tool]:
    """Read-only gold-free toolset for the Phase B Reviewer (fix minimality review).

    The reviewer sees the candidate fix (already applied in the shared tree, so
    read_file shows the edited source) and may run a targeted existing test file
    as a regression check — but it cannot edit source or see the held-out tests.
    It ends by calling submit_review.
    """
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        RunTestsTool(session),
        SubmitReviewTool(),
    ]
    assert_role_gold_free(tools)
    return tools


def selector_agent_tools(session: ContainerSession) -> list[Tool]:
    """Read-only gold-free toolset for the Selector (best-of-N BRT choice).

    The selector reads the issue + candidate tests (with their buggy-tree
    outcomes) and may use run_python to check the correct value, then calls
    submit_selection. It cannot write tests, edit source, or see the fixed tree.
    """
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        RunPythonTool(session),
        SubmitSelectionTool(),
    ]
    assert_selector_gold_free(tools)
    return tools


def breaker_agent_tools(session: ContainerSession) -> list[Tool]:
    """Read-only gold-free toolset for the post-fix Breaker (verification gate).

    The candidate fix is already applied in the shared session, so read_file shows
    the fixed source and run_python probes the fixed behaviour. The breaker cannot
    edit source, write/run a BRT, or see the held-out tests. It ends by calling
    submit_break.
    """
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        RunPythonTool(session),
        SubmitBreakTool(),
    ]
    assert_breaker_gold_free(tools)
    return tools
