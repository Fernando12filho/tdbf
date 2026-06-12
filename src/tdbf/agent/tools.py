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

from ..harness.runner import (
    _REPO_PREAMBLES,
    _parse_failure_reason,
    _parse_pytest_outcome,
    brt_test_path,
    inject_setup_preamble,
)
from .session import ContainerSession
from .types import Tool, ToolResult

_MAX_OUTPUT = 2500  # chars of tool output fed back to the model


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped, {len(text) - limit} more chars]"


class _SessionTool(Tool):
    """Base for tools sharing one container session."""

    def __init__(self, session: ContainerSession) -> None:
        self.session = session


class ReadFileTool(_SessionTool):
    name = "read_file"
    description = "Read a source file from the (buggy) repository checkout. Path is relative to the repo root."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path relative to repo root, e.g. 'astropy/table/table.py'."}},
        "required": ["path"],
    }

    def execute(self, arguments: dict) -> ToolResult:
        path = str(arguments.get("path", "")).strip()
        if not path:
            return ToolResult("read_file requires a non-empty 'path'.", is_error=True)
        res = self.session.read_file(path, max_bytes=20000)
        if res.returncode != 0:
            return ToolResult(f"Could not read {path!r}: {_clip(res.output, 400)}", is_error=True)
        return ToolResult(_clip(res.output))


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

    def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        if not str(code).strip():
            return ToolResult("run_python requires non-empty 'code'.", is_error=True)
        preamble = _REPO_PREAMBLES.get(self.session.repo, "")
        self.session.write_file("/tmp/tdbf_probe.py", preamble + code)
        res = self.session.exec("python /tmp/tdbf_probe.py", timeout=90)
        body = res.output.strip()
        if not body:
            body = "(no output)" if res.returncode == 0 else "(no output; non-zero exit)"
        tag = "" if res.returncode == 0 else f"[exit {res.returncode}] "
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

    def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        if not str(code).strip():
            return ToolResult("run_test requires non-empty 'code'.", is_error=True)
        rel_path = brt_test_path(self.session.repo)
        full_code = inject_setup_preamble(self.session.repo, code)
        self.session.write_file(f"/testbed/{rel_path}", full_code)
        self.session.ensure_pytest()
        res = self.session.exec(
            f"python -m pytest {rel_path} -x --tb=short -q",
            timeout=180,
        )
        outcome = _parse_pytest_outcome(res.returncode, res.output)
        reason = _parse_failure_reason(res.output) if outcome == "fail" else None
        header = f"outcome={outcome}" + (f" failure_reason={reason}" if reason else "")
        return ToolResult(f"{header}\n\n{_clip(res.output)}")


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

    def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        if not str(code).strip():
            return ToolResult("submit_test requires non-empty 'code'.", is_error=True)
        return ToolResult(str(code))


# Names allowed in a BRT agent's registry. Any tool outside this set — in
# particular anything that could touch the gold/fixed tree — is rejected.
_GOLD_FREE_TOOL_NAMES = frozenset(
    {"read_file", "grep", "glob", "run_python", "run_test", "submit_test"}
)


def assert_gold_free(tools: list[Tool]) -> None:
    """Guard the tool surface against a fixed-tree / gold tool slipping in.

    This is the programmatic side of the validity invariant: the agent must
    never get a tool that observes the gold-fixed outcome.
    """
    for t in tools:
        if t.name not in _GOLD_FREE_TOOL_NAMES:
            raise ValueError(
                f"Tool {t.name!r} is not in the gold-free allowlist {sorted(_GOLD_FREE_TOOL_NAMES)}. "
                "No agent tool may run against the gold/fixed tree."
            )


def brt_agent_tools(session: ContainerSession) -> list[Tool]:
    """The standard gold-free toolset for a single BRT agent (Phase A)."""
    tools: list[Tool] = [
        ReadFileTool(session),
        GrepTool(session),
        GlobTool(session),
        RunPythonTool(session),
        RunTestTool(session),
        SubmitTestTool(),
    ]
    assert_gold_free(tools)
    return tools
