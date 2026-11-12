"""Tools of the procedure agent (design §3, §11).

The read-only tools (`read_file`, `grep`, `glob`) are the baseline's own classes,
reused unchanged. New here:

  run_python  same execution as the baseline's (Django under runtests.py), but every
              run gets an id (`r1`, `r2`, ...) and is kept, so evidence can cite a run
              and the controller can re-run it; output keeps its END as well as its
              start (the exception is at the end);
  run_test    runs the test at the test home chosen in Locate, with the official
              SWE-bench command; reports outcome, failure_reason and failing line;
  finish_*    one terminal tool per phase; it only checks the SHAPE of the result
              (the controller's gate checks the content).

Gold-free like the baseline's tools: everything runs on the buggy checkout.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Callable

from ..harness.official import TestCommand, parse_result
from ..harness.runner import clean_brt_output
from .preflight import DJANGO, _keep, run_snippet
from .procedure_gates import failing_line
from .session import ContainerSession
from .tools import GlobTool, GrepTool, ReadFileTool
from .types import Tool, ToolResult

TOOL_OUTPUT = 2500   # chars the model sees of a run (head and tail)
TEST_TIMEOUT = 180
TEST_FILE = "test_brt.py"


@dataclass
class RunLog:
    """Every run_python and run_test call of one procedure run, by id."""

    runs: dict[str, dict] = field(default_factory=dict)
    tests: list[dict] = field(default_factory=list)

    def add_run(self, code: str, module: str | None, rc: int, output: str) -> str:
        rid = f"r{len(self.runs) + 1}"
        self.runs[rid] = {"id": rid, "code": code, "module": module, "returncode": rc,
                          "output": _keep(output, 6000)}
        return rid


def _django_module(arguments: dict) -> str:
    m = str(arguments.get("module") or "").strip().strip("/").replace("/", ".")
    if m.startswith("tests."):
        m = m[len("tests."):]
    return m or "brt_app"


class ProcRunPython(Tool):
    name = "run_python"

    def __init__(self, session: ContainerSession, log: RunLog) -> None:
        self.session, self.log = session, log
        django = session.repo == DJANGO
        self.description = (
            "Run a Python snippet on the BUGGY code and return what it printed (or its traceback). "
            "Each run gets an id (r1, r2, ...) that you can cite as evidence."
            + (" Django: the snippet runs inside a TestCase under Django's test runner, with the "
               "contrib apps and a test database; pass `module` (a test package under tests/) to "
               "use that package's models." if django else ""))
        props = {"code": {"type": "string", "description": "Python source to execute."}}
        if django:
            props["module"] = {"type": "string",
                               "description": "Optional test package under tests/ whose models to use."}
        self.input_schema = {"type": "object", "properties": props, "required": ["code"]}

    def execute(self, arguments: dict) -> ToolResult:
        code = str(arguments.get("code") or "")
        if not code.strip():
            return ToolResult("run_python requires non-empty 'code'.", is_error=True)
        module = _django_module(arguments) if self.session.repo == DJANGO else None
        if module and module != "brt_app":
            res = self.session.exec('git cat-file -e "HEAD:tests/$M/__init__.py"',
                                    env={"M": module.replace(".", "/")})
            if res.returncode != 0:
                return ToolResult(f"tests/{module}/ is not a test package of this repository.",
                                  is_error=True)
        rc, out = run_snippet(self.session, self.session.repo, code, module or "brt_app")
        rid = self.log.add_run(code, module, rc, out)
        tag = "" if rc == 0 else f" [exit {rc}]"
        return ToolResult(f"run_id={rid}{tag}\n{_keep(out, TOOL_OUTPUT) or '(no output)'}")


@dataclass
class TestHome:
    test_dir: str
    module: str | None = None        # Django test package label
    source: str = "model"            # model | preflight_top | default
    style_notes: str = ""

    @property
    def rel_path(self) -> str:
        return f"{self.test_dir.rstrip('/')}/{TEST_FILE}" if self.test_dir not in ("", ".") else TEST_FILE


def run_test_at(session: ContainerSession, cmd: TestCommand, home: TestHome, code: str,
                timeout: int = TEST_TIMEOUT) -> dict:
    """Write `code` to the test home and run it with the official command."""
    session.write_file(f"/testbed/{home.rel_path}", code)
    if cmd.framework in ("pytest", "tox-pytest"):
        session.ensure_pytest()
    res = session.exec(cmd.for_file(home.rel_path), timeout=timeout)
    outcome, reason = parse_result(cmd, res.returncode, res.output)
    out = clean_brt_output(session.repo, res.output)
    return {"outcome": outcome, "reason": reason, "returncode": res.returncode,
            "failing_line": failing_line(out, home.rel_path), "output": _keep(out, 6000),
            "path": home.rel_path, "code": code}


class ProcRunTest(Tool):
    """Run a test at the test home. In strict mode the home is fixed by Locate; in
    soft mode the model names it with `test_dir` (and `module` on Django)."""

    name = "run_test"

    def __init__(self, session: ContainerSession, cmd: TestCommand, log: RunLog,
                 home: Callable[[dict], "TestHome | str"], *, home_args: bool = False) -> None:
        self.session, self.cmd, self.log, self._home = session, cmd, log, home
        django = session.repo == DJANGO
        self.description = (
            "Write the given test module to the test file and run it against the BUGGY code with the "
            "project's own test command. Returns outcome (pass/fail/error/absent), failure_reason "
            "(assertion" + (", exception" if django else "") + " or runtime) and the line of your test "
            "where it failed. A bug reproduction test must FAIL here at its assertion.")
        props = {"code": {"type": "string", "description": "Full test module source."}}
        if home_args:
            props["test_dir"] = {"type": "string",
                                 "description": "Directory (in the repository) where the test file goes."}
            if django:
                props["module"] = {"type": "string", "description": "Test package under tests/."}
        self.input_schema = {"type": "object", "properties": props, "required": ["code"]}

    def execute(self, arguments: dict) -> ToolResult:
        code = str(arguments.get("code") or "")
        if not code.strip():
            return ToolResult("run_test requires non-empty 'code'.", is_error=True)
        home = self._home(arguments)
        if isinstance(home, str):
            return ToolResult(home, is_error=True)
        run = run_test_at(self.session, self.cmd, home, code)
        run["id"] = f"t{len(self.log.tests) + 1}"
        self.log.tests.append(run)
        head = (f"test_id={run['id']} outcome={run['outcome']}"
                + (f" failure_reason={run['reason']}" if run["reason"] else "")
                + (f" failing_line={run['failing_line']}" if run["failing_line"] else ""))
        return ToolResult(f"{head}\n\n{_keep(run['output'], TOOL_OUTPUT)}")


class FinishTool(Tool):
    """A phase's terminal tool. `check(args)` returns an error for a malformed
    result (the model fixes it within the phase), else None; the arguments come back
    as JSON and are the phase's result."""

    terminal = True

    def __init__(self, name: str, description: str, schema: dict,
                 check: Callable[[dict], str | None] | None = None) -> None:
        self.name, self.description, self.input_schema = name, description, schema
        self._check = check

    def execute(self, arguments: dict) -> ToolResult:
        err = self._check(arguments) if self._check else None
        if err:
            return ToolResult(err, is_error=True)
        return ToolResult(json.dumps(arguments))


def read_only_tools(session: ContainerSession) -> list[Tool]:
    return [ReadFileTool(session), GrepTool(session), GlobTool(session)]


# --------------------------------------------------------------------------- #
# finish_* schemas and shape checks                                            #
# --------------------------------------------------------------------------- #

def _list_of(props: dict, required: list[str]) -> dict:
    return {"type": "array", "items": {"type": "object", "properties": props, "required": required}}


_CLAIM = {"value": {"type": "string", "description": "The behaviour, concretely (a value, output or exception)."},
          "quote": {"type": "string", "description": "Verbatim text from the issue that shows it."}}


def finish_understand() -> FinishTool:
    schema = {"type": "object", "properties": {
        "observed": _list_of(_CLAIM, ["value", "quote"]),
        "expected": _list_of(_CLAIM, ["value", "quote"]),
        "unknowns": _list_of({"question": {"type": "string"}, "why": {"type": "string"}}, ["question"]),
    }, "required": ["observed", "expected", "unknowns"]}

    def check(a: dict) -> str | None:
        for key in ("observed", "expected", "unknowns"):
            if not isinstance(a.get(key, []), list):
                return f"'{key}' must be a list."
        return None
    return FinishTool("finish_understand",
                      "Report the observed behaviour, the expected behaviour and the unknowns, "
                      "each observed/expected item with a verbatim quote from the issue.", schema, check)


def finish_reproduce(log: RunLog) -> FinishTool:
    schema = {"type": "object", "properties": {
        "run_id": {"type": "string", "description": "The run_python run (e.g. 'r3') that shows the bug."},
        "output_summary": {"type": "string", "description": "What that run printed that shows the bug."},
    }, "required": ["run_id", "output_summary"]}

    def check(a: dict) -> str | None:
        rid = str(a.get("run_id") or "").strip()
        if rid and rid not in log.runs:
            return f"run_id {rid!r} does not exist; your runs: {sorted(log.runs) or 'none'}."
        return None
    return FinishTool("finish_reproduce",
                      "Finish with the id of the run_python run that shows the bug (or an empty "
                      "run_id if you could not reproduce it) and what it printed.", schema, check)


SOURCES = ("docs", "existing_test", "observed_run", "analogy", "model")


def finish_pin() -> FinishTool:
    schema = {"type": "object", "properties": {"resolved": _list_of({
        "unknown_id": {"type": "string"},
        "value": {"type": "string", "description": "The correct expected value."},
        "source": {"type": "string", "enum": list(SOURCES)},
        "quote": {"type": "string", "description": "Verbatim text from the file, or from the run's output."},
        "where": {"type": "string", "description": "File path (docs/existing_test) or run id (observed_run/analogy)."},
    }, ["unknown_id", "value", "source"])}, "required": ["resolved"]}

    def check(a: dict) -> str | None:
        if not isinstance(a.get("resolved"), list):
            return "'resolved' must be a list (empty if you found no evidence)."
        bad = [r.get("source") for r in a["resolved"] if isinstance(r, dict) and r.get("source") not in SOURCES]
        return f"unknown source(s) {bad}; use one of {list(SOURCES)}." if bad else None
    return FinishTool("finish_pin",
                      "Report, for each unknown you found evidence for, the value, the source type, "
                      "the verbatim quote and where it is. Leave out unknowns without evidence.",
                      schema, check)


def finish_locate(django: bool) -> FinishTool:
    props = {"test_dir": {"type": "string", "description": "Existing directory where the test goes."},
             "style_notes": {"type": "string", "description": "How tests there are written."}}
    req = ["test_dir"]
    if django:
        props["module"] = {"type": "string", "description": "The test package name (tests/<module>/)."}
        req.append("module")
    return FinishTool("finish_locate", "Finish with the directory where the test should live.",
                      {"type": "object", "properties": props, "required": req})


def _parses(code: str) -> str | None:
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"the code does not parse: {e}"


def finish_write() -> FinishTool:
    schema = {"type": "object", "properties": {
        "code": {"type": "string", "description": "The full test module."},
        "assertion_map": _list_of({"line": {"type": "integer"},
                                   "evidence_id": {"type": "string"}}, ["line", "evidence_id"]),
    }, "required": ["code", "assertion_map"]}

    def check(a: dict) -> str | None:
        code = str(a.get("code") or "")
        if not code.strip():
            return "'code' is empty."
        if not isinstance(a.get("assertion_map"), list):
            return "'assertion_map' must be a list of {line, evidence_id}."
        return _parses(code)
    return FinishTool("finish_write",
                      "Finish with the test code and, for each assertion line, the id of the evidence "
                      "it relies on.", schema, check)


def submit_test_soft(django: bool) -> FinishTool:
    props = {"code": {"type": "string"}, "test_dir": {"type": "string"},
             "assertion_map": _list_of({"line": {"type": "integer"},
                                        "evidence_id": {"type": "string"}}, ["line", "evidence_id"])}
    req = ["code", "test_dir"]
    if django:
        props["module"] = {"type": "string"}
        req.append("module")

    def check(a: dict) -> str | None:
        code = str(a.get("code") or "")
        return "'code' is empty." if not code.strip() else _parses(code)
    return FinishTool("submit_test", "Submit the final test, where it lives, and its assertion map.",
                      {"type": "object", "properties": props, "required": req}, check)
