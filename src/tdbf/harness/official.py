"""Run a test file the way the SWE-bench grader runs tests (procedure agent, D4).

The baseline agent runs every non-Django test with bare pytest from the repo root
(`runner.brt_run_spec`). The procedure agent instead uses the grader's own command,
`MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]` from swebench 4.1.0, so our
verdict and SWT-bench's cannot disagree on mechanics:

    pytest repos    pytest -rA <path>
    django          ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 <label>
    sympy           bin/test -C --verbose <path>
    sphinx          tox --current-env -epy39 -v -- <path>

What we add or change, and why (each is recorded on the command):
  - the spec's `eval_commands`: one-off setup steps (Django's `locale-gen`) run once
    per session; `export` lines prefix every command, as in the grader's eval script.
  - astropy 0.x/1.x specs pass `--tb=no`, which hides the failing line and message
    (the grader only reads PASSED/FAILED lines). We use `--tb=short` instead: same
    tests, same verdicts, but the output shows the agent's own assertion. It shows
    nothing about the fix; the checkout is the buggy one.
  - the grader's `install` step (pip install -e .) is NOT re-run: the prebuilt image
    already has it.

When the official command does not work in an image, `fallback_command` is today's
path (bare pytest at the repo root; Django unchanged), marked `source="fallback"`.

Nothing in runner.py changes; its parsers are reused read-only.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from .runner import (
    _ANSI_RE,
    _PYTEST_EXTRA,
    _parse_failure_reason,
    _parse_pytest_outcome,
    _unittest_problems,
    parse_runtests_result,
)

# A repo-relative test path we are willing to put in a command line. It is also
# shell-quoted; the pattern keeps model-chosen paths from smuggling options in.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")

DJANGO = "django/django"
SYMPY = "sympy/sympy"


@dataclass
class TestCommand:
    """How to run one test file in one instance's image."""

    repo: str
    version: str | None
    base: str                    # the test command without the file
    source: str                  # "official" | "fallback"
    framework: str               # "pytest" | "django" | "sympy" | "tox-pytest"
    exports: list[str] = field(default_factory=list)   # "export X=Y" lines, prefixed
    setup: list[str] = field(default_factory=list)     # run once per session
    changes: list[str] = field(default_factory=list)   # what differs from the spec, why

    def directive(self, rel_path: str) -> str:
        """The argument naming `rel_path` for this runner (a Django label or a path)."""
        if not _SAFE_PATH_RE.match(rel_path or "") or ".." in rel_path:
            raise ValueError(f"unsafe test path: {rel_path!r}")
        return test_directive(self.repo, rel_path)

    def for_file(self, rel_path: str) -> str:
        """The full shell command (run from /testbed, testbed env active)."""
        prefix = "".join(f"{e}; " for e in self.exports)
        return f"{prefix}{self.base} {shlex.quote(self.directive(rel_path))}"

    def to_json(self) -> dict:
        return {"repo": self.repo, "version": self.version, "base": self.base,
                "source": self.source, "framework": self.framework,
                "exports": self.exports, "setup": self.setup, "changes": self.changes}


def test_directive(repo: str, rel_path: str) -> str:
    """Same mapping as swebench's get_test_directives: Django drops `tests/` and
    `.py` and uses dots; every other repo takes the path."""
    if repo != DJANGO:
        return rel_path
    d = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    d = d[len("tests/"):] if d.startswith("tests/") else d
    return d.replace("/", ".")


def _framework(repo: str, cmd: str) -> str:
    if repo == DJANGO or "runtests.py" in cmd:
        return "django"
    if repo == SYMPY or "bin/test" in cmd:
        return "sympy"
    if cmd.startswith("tox"):
        return "tox-pytest"
    return "pytest"


def official_command(repo: str, version: str | None) -> TestCommand | None:
    """The grader's command for (repo, version), or None if swebench has no spec."""
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except ImportError:  # pragma: no cover - swebench is a project dependency
        return None
    spec = MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(str(version)) if version else None
    if not spec or not isinstance(spec.get("test_cmd"), str):
        return None
    base = spec["test_cmd"].strip()
    changes: list[str] = []
    if "--tb=no" in base:
        base = base.replace("--tb=no", "--tb=short")
        changes.append("--tb=no -> --tb=short: show the failing assertion; verdicts unchanged")
    exports, setup = [], []
    for c in spec.get("eval_commands", []) or []:
        (exports if c.strip().startswith("export ") else setup).append(c.strip())
    return TestCommand(repo, str(version), base, "official", _framework(repo, base),
                       exports=exports, setup=setup, changes=changes)


def fallback_command(repo: str) -> TestCommand:
    """Today's baseline path (runner.brt_run_spec), but for a file at any path."""
    if repo == DJANGO:
        return TestCommand(repo, None,
                           "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
                           "fallback", "django")
    return TestCommand(repo, None,
                       f"python -m pytest --tb=short -q -rA{_PYTEST_EXTRA.get(repo, '')}",
                       "fallback", "pytest")


# --------------------------------------------------------------------------- #
# Reading the result                                                           #
# --------------------------------------------------------------------------- #

def test_statuses(repo: str, log: str) -> dict[str, str]:
    """{test name: PASSED|FAILED|ERROR|SKIPPED|XFAIL} from swebench's own log parser
    for this repo. The parsers used for the Lite repos ignore `test_spec`."""
    try:
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    except ImportError:  # pragma: no cover
        return {}
    parser = MAP_REPO_TO_PARSER.get(repo)
    if parser is None:
        return {}
    try:
        return dict(parser(_ANSI_RE.sub("", log or ""), None))
    except Exception:  # noqa: BLE001 - a parser crash means "no statuses"
        return {}


# sympy's bin/test: a FAILED test ends its traceback with AssertionError and is
# reported as "F"; any other exception is "E". The header of each block is
#   ____ sympy/core/tests/test_x.py:test_name ____
_SYMPY_BLOCK_RE = re.compile(r"^_{3,} (\S+\.py):(\S+) _{3,}\s*$", re.MULTILINE)
_SYMPY_SUMMARY_RE = re.compile(r"=+ tests finished: (.*?) =+", re.IGNORECASE)
_TRACE_EXC_RE = re.compile(r"^([A-Za-z_][\w.]*)(?::\s?(.*))?$")


def _sympy_blocks(log: str) -> list[tuple[str, str | None, str]]:
    """(test, exception class, message) for each failure block in bin/test output."""
    out = []
    heads = list(_SYMPY_BLOCK_RE.finditer(log))
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(log)
        body = log[m.end():end]
        exc, msg = None, ""
        for ln in body.splitlines():
            if not ln or ln[0].isspace() or ln.startswith(("Traceback", "=", "_")):
                continue
            em = _TRACE_EXC_RE.match(ln.strip())
            if em:
                exc, msg = em.group(1), em.group(2) or ""
        out.append((m.group(2), exc, msg))
    return out


_AUTHORING = frozenset({"ImportError", "ModuleNotFoundError", "NameError", "SyntaxError",
                        "IndentationError", "TabError"})


def parse_sympy_result(returncode: int, log: str) -> tuple[str, str | None]:
    """(outcome, failure_reason) from sympy's bin/test output.

    Same meaning as the pytest parser the baseline uses for sympy: "assertion" when
    an AssertionError fired; "runtime" for any other exception (the baseline does not
    count a non-Django exception as a reproduction)."""
    log = _ANSI_RE.sub("", log or "")
    # swebench keys a failed test as "<path>:<name>" and a passing one as "<name>"
    statuses = {t.split(":")[-1]: s for t, s in test_statuses(SYMPY, log).items()
                if t.split(":")[-1].startswith("test_")}
    summary = _SYMPY_SUMMARY_RE.search(log)
    if not statuses and summary is None:
        return "error", None
    if not statuses and summary is not None and "passed" not in summary.group(1):
        return "absent", None
    bad = [s for s in statuses.values() if s in ("FAILED", "ERROR")]
    if not bad:
        return ("pass", None) if returncode == 0 or statuses else ("error", None)
    blocks = _sympy_blocks(log)
    if any((exc or "").split(".")[-1] == "AssertionError" for _, exc, _ in blocks):
        return "fail", "assertion"
    return "fail", "runtime"


def parse_result(cmd: TestCommand, returncode: int, log: str) -> tuple[str, str | None]:
    """(outcome, failure_reason) for a run made with `cmd.for_file(...)`.

    outcome: pass | fail | error | absent. failure_reason (when fail): assertion |
    exception (Django only, as today) | runtime."""
    if cmd.framework == "django":
        return parse_runtests_result(returncode, log)
    if cmd.framework == "sympy":
        return parse_sympy_result(returncode, log)
    log = _ANSI_RE.sub("", log or "")
    # tox returns 1 for a failing pytest run, like pytest itself; anything it
    # could not even start comes back as another code and parses as "error".
    outcome = _parse_pytest_outcome(returncode, log)
    if outcome == "error":
        statuses = test_statuses(cmd.repo, log)
        if statuses and all(s == "PASSED" for s in statuses.values()):
            outcome = "pass"
        elif any(s in ("FAILED", "ERROR") for s in statuses.values()):
            outcome = "fail"
    reason = _parse_failure_reason(log) if outcome == "fail" else None
    if outcome == "fail" and reason is None:
        reason = "runtime"  # failed with no recognisable assertion: not evidence
    return outcome, reason


def unittest_problem_summary(log: str) -> list[dict]:
    """Django: the FAIL/ERROR blocks, for the run record."""
    return [{"kind": k, "test": h, "exception": e, "authoring": a}
            for k, h, e, a in _unittest_problems(_ANSI_RE.sub("", log or ""))]
