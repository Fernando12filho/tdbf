"""Smoke test for native (Django) BRT execution — PLAN.md Phase 6.

Offline (no Docker, no model). Checks that:
  1. every non-Django repo still gets exactly the old BRT file, preamble, command,
     prompt and tool descriptions (byte for byte);
  2. a Django BRT is placed in tests/<module>/test_brt.py and run through
     runtests.py, with its module carried as a header comment;
  3. the runtests.py output parser separates an assertion, an exception raised
     by the code under test, and a broken test — the Django gate rule of
     2026-10-07 (accept FAIL or ERROR, except test-authoring / setup errors);
  4. the tools, the gold-free gate and the retry feedback use that one path.

    python3 scripts/agent_django_brt_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent import pipeline  # noqa: E402
from tdbf.agent.session import ExecResult  # noqa: E402
from tdbf.agent.tools import (  # noqa: E402
    RunPythonTool,
    RunTestTool,
    SubmitTestTool,
    brt_agent_tools,
    fix_agent_tools,
)
from tdbf.harness.runner import (  # noqa: E402
    DJANGO_BRT_APP,
    brt_module,
    brt_run_spec,
    brt_test_path,
    clean_brt_output,
    inject_setup_preamble,
    parse_brt_result,
    parse_runtests_result,
    repo_test_hint,
    brt_env_hint,
    set_brt_module,
)

DJ = "django/django"
SEP = "=" * 70
DASH = "-" * 70


def _block(kind: str, test: str, traceback_tail: str, doc: str | None = None) -> str:
    lines = [SEP, f"{kind}: {test}"]
    if doc:
        lines.append(doc)
    lines += [DASH, "Traceback (most recent call last):",
              '  File "/testbed/tests/admin_views/test_brt.py", line 12, in test_x',
              "    do_it()", traceback_tail, ""]
    return "\n".join(lines)


def _out(blocks: list[str], ran: int = 1, ok: bool = False) -> str:
    head = ("Testing against Django installed in '/testbed/django'\n"
            "Importing application admin_views\nOperations to perform:\n"
            "  Synchronize unmigrated apps: admin_views, auth\n"
            "  Applying admin.0001_initial... OK\n"
            "Creating test database for alias 'default'...\n"
            "test_x (admin_views.test_brt.BrtTests) ... FAIL\n")
    tail = f"{DASH}\nRan {ran} tests in 0.004s\n\n" + ("OK\n" if ok else "FAILED (failures=1)\n")
    return head + "".join(b + "\n" for b in blocks) + tail


# ---------------------------------------------------------------- 1. non-Django
def test_non_django_unchanged() -> None:
    code = "def test_x():\n    assert 1 == 2\n"
    for repo in ("astropy/astropy", "sympy/sympy", "django/django-rest-framework"):
        spec = brt_run_spec(repo, code)
        rel = brt_test_path(repo)
        assert spec.files == {rel: inject_setup_preamble(repo, code)}, spec.files
        assert spec.command == f"python -m pytest {rel} -x --tb=short -q"
        assert spec.needs_pytest
        assert brt_env_hint(repo) == repo_test_hint(repo)
        assert pipeline.brt_system_prompt(repo) is pipeline.BRT_SYSTEM_PROMPT
        assert clean_brt_output(repo, "Applying x\nE boom") == "Applying x\nE boom"
    # pytest outcome parsing is the old one
    assert parse_brt_result("sympy/sympy", 1, "E   AssertionError: no\n") == ("fail", "assertion")
    assert parse_brt_result("sympy/sympy", 1, "E   ImportError: x\n") == ("fail", "runtime")
    # tool descriptions / schemas are the class defaults
    sess = FakeSession("sympy/sympy")
    for t in brt_agent_tools(sess):  # type: ignore[arg-type]
        if t.name in ("run_test", "submit_test", "run_python"):
            cls = type(t)
            assert t.description == cls.description and t.input_schema == cls.input_schema, t.name
    print("  [ok] non-Django repos: file, command, prompt, hint, tools unchanged")


# ---------------------------------------------------------- 2. placement/header
def test_django_placement() -> None:
    code = set_brt_module("from django.test import TestCase\n", "admin_views")
    assert code.startswith("# tdbf-brt-module: admin_views\n")
    assert brt_module(code) == "admin_views"
    assert set_brt_module(code, "forms_tests") .count("tdbf-brt-module") == 1
    spec = brt_run_spec(DJ, code)
    assert spec.files == {"tests/admin_views/test_brt.py": code}
    assert spec.command.endswith("runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 admin_views.test_brt")
    assert not spec.needs_pytest
    assert "django.setup()" not in code  # no preamble on the native path
    nested = brt_run_spec(DJ, set_brt_module("x\n", "gis_tests.geoapp"))
    assert "tests/gis_tests/geoapp/test_brt.py" in nested.files
    fallback = brt_run_spec(DJ, "from django.test import TestCase\n")
    assert set(fallback.files) == {f"tests/{DJANGO_BRT_APP}/test_brt.py", f"tests/{DJANGO_BRT_APP}/__init__.py"}
    assert brt_module("# tdbf-brt-module: ; rm -rf /\n") is None  # unsafe name never reaches a command
    print("  [ok] Django placement: tests/<module>/test_brt.py via runtests.py, header carries module")


# ---------------------------------------------------------------- 3. parser
def test_runtests_parser() -> None:
    fail = _out([_block("FAIL", "test_x (admin_views.test_brt.BrtTests)",
                        "AssertionError: None is not <class 'User'>")])
    assert parse_runtests_result(1, fail) == ("fail", "assertion")
    key = _out([_block("ERROR", "test_x (admin_views.test_brt.BrtTests)", "KeyError: 'model'")])
    assert parse_runtests_result(1, key) == ("fail", "exception"), "bug raising = reproduction"
    dotted = _out([_block("ERROR", "test_x (m.T)",
                          "django.db.utils.IntegrityError: FOREIGN KEY constraint failed")])
    assert parse_runtests_result(1, dotted) == ("fail", "exception")
    imp = _out([_block("ERROR", "test_x (m.T)", "ImportError: cannot import name 'nope'")])
    assert parse_runtests_result(1, imp) == ("fail", "runtime")
    cfg = _out([_block("ERROR", "test_x (m.T)",
                       "django.core.exceptions.ImproperlyConfigured: settings.DATABASES is improperly "
                       "configured. Please supply the NAME value.", doc="Docstring line.")])
    assert parse_runtests_result(1, cfg) == ("fail", "runtime"), "docstring + setup error"
    simple_db = _out([_block("FAIL", "test_x (m.T)",
                             "django.test.testcases.DatabaseOperationForbidden: Database queries to "
                             "'default' are not allowed in SimpleTestCase subclasses.")])
    assert parse_runtests_result(1, simple_db) == ("fail", "runtime"), "FAIL that is a setup error"
    app = _out([_block("ERROR", "test_x (m.T)",
                       "RuntimeError: Model class brt.models.X doesn't declare an explicit app_label "
                       "and isn't in an application in INSTALLED_APPS.")])
    assert parse_runtests_result(1, app) == ("fail", "runtime")
    no_table = _out([_block("ERROR", "test_x (brt_app.test_brt.T)",
                            "sqlite3.OperationalError: no such table: brt_app_gadget")])
    assert parse_runtests_result(1, no_table) == ("fail", "runtime"), "test-defined model, no table"
    mixed = _out([_block("ERROR", "test_a (m.T)", "NameError: name 'x' is not defined"),
                  _block("FAIL", "test_b (m.T)", "AssertionError: 1 != 2")], ran=2)
    assert parse_runtests_result(1, mixed) == ("fail", "assertion"), "assertion evidence wins"
    chained = _out([_block("ERROR", "test_x (m.T)",
                           "ImportError: x\n\nDuring handling of the above exception, another exception "
                           "occurred:\n\nTraceback (most recent call last):\n  File \"/testbed/django/a.py\""
                           ", line 1, in f\n    g()\nValueError: bad value")])
    assert parse_runtests_result(1, chained) == ("fail", "exception"), "last exception of a chain counts"
    load = _out([_block("ERROR", "test_brt (unittest.loader._FailedTest)",
                        "ImportError: Failed to import test module: test_brt")])
    assert parse_runtests_result(1, load) == ("error", None), "module that does not load = error (L2)"
    assert parse_runtests_result(0, _out([], ok=True)) == ("pass", None)
    assert parse_runtests_result(1, "SystemCheckError: System check identified some issues") == ("error", None)
    assert parse_runtests_result(5, _out([], ran=0, ok=True)) == ("absent", None)
    assert parse_runtests_result(124, "timeout after 180s") == ("error", None)
    shown = clean_brt_output(DJ, fail)
    assert "Applying admin" not in shown and "Creating test database" not in shown
    assert "AssertionError: None is not" in shown and "Ran 1 tests" in shown
    print("  [ok] runtests.py parser: assertion / exception / runtime / error / absent / pass")


# ---------------------------------------------------------------- 4. tools + gate
class FakeSession:
    """Records writes and commands; answers runtests.py with a canned run."""

    def __init__(self, repo: str = DJ, run_output: str | None = None, rc: int = 1) -> None:
        self.repo = repo
        self.instance_id = "django__django-1"
        self.writes: dict[str, str] = {}
        self.execs: list[tuple[str, dict | None]] = []
        self.run_output = run_output if run_output is not None else _out(
            [_block("ERROR", "test_x (admin_views.test_brt.BrtTests)", "KeyError: 'model'")])
        self.rc = rc
        self._pytest_ready = False

    def write_file(self, path, content) -> None:
        self.writes[path] = content

    tracked = ("admin_views", "gis_tests/geoapp")  # packages of the original checkout

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        self.execs.append((command, env))
        if command.startswith("git cat-file"):
            return ExecResult(0 if env and env.get("MODPATH") in self.tracked else 1, "")
        if "runtests.py" in command or "pytest" in command:
            return ExecResult(self.rc, self.run_output)
        return ExecResult(0, "")

    def ensure_pytest(self) -> None:
        self._pytest_ready = True

    def reset_tree(self) -> None:
        self.execs.append(("RESET", None))

    def read_file(self, path, *, max_bytes=0) -> ExecResult:
        return ExecResult(1, "")


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def test_tools_and_gate() -> None:
    sess = FakeSession()
    tools = brt_agent_tools(sess)  # type: ignore[arg-type]
    run_test, submit = _tool(tools, "run_test"), _tool(tools, "submit_test")
    assert "module" in run_test.input_schema["properties"] and "runtests.py" in run_test.description
    body = "from django.test import TestCase\n\nclass T(TestCase):\n    def test_x(self):\n        pass\n"

    res = run_test.execute({"code": body, "module": "admin_views"})
    assert not res.is_error and res.content.startswith("outcome=fail failure_reason=exception"), res.content
    assert "Applying admin" not in res.content
    written = sess.writes["/testbed/tests/admin_views/test_brt.py"]
    assert written.startswith("# tdbf-brt-module: admin_views\n") and "django.setup" not in written
    assert any("runtests.py" in c and c.endswith("admin_views.test_brt") for c, _ in sess.execs)
    assert not sess._pytest_ready, "runtests.py needs no pytest install"

    bad = run_test.execute({"code": body, "module": "no_such_pkg"})
    assert bad.is_error and "not a test package of this repository" in bad.content
    # a package the agent created itself (via run_python) is refused at submit too
    made_up = submit.execute({"code": body, "module": "brt_fk_order"})
    assert made_up.is_error, "agent-created packages vanish in the fix/label containers"
    unsafe = run_test.execute({"code": body, "module": "x; rm -rf /"})
    assert unsafe.is_error

    # submit without `module` lands where run_test last ran it
    sub = submit.execute({"code": body})
    assert not sub.is_error and brt_module(sub.content) == "admin_views", sub.content
    explicit = submit.execute({"code": body, "module": "tests/gis_tests/geoapp"})
    assert brt_module(explicit.content) == "gis_tests.geoapp", explicit.content

    # the gate consults the same path: an exception from the bug is a reproduction
    sess.execs.clear()
    gate = pipeline._score_on_buggy_in_session(sess, sub.content)  # type: ignore[arg-type]
    assert gate.outcome == "fail" and gate.failure_reason == "exception"
    assert sess.execs[0] == ("RESET", None), "gate verifies on the pristine buggy checkout"
    assert gate == "fail" and gate.failure_reason != "runtime"
    broken = FakeSession(run_output=_out([_block("ERROR", "test_x (m.T)", "NameError: name 'q'")]))
    g2 = pipeline._score_on_buggy_in_session(broken, sub.content)  # type: ignore[arg-type]
    assert g2.failure_reason == "runtime", "a broken test is still rejected"

    # the fix agent's run_brt and the fix loop's pass-check use the header's module
    fsess = FakeSession(run_output=_out([], ok=True), rc=0)
    run_brt = _tool(fix_agent_tools(fsess, sub.content), "run_brt")  # type: ignore[arg-type]
    out = run_brt.execute({})
    assert out.content.startswith("outcome=pass"), out.content
    assert "/testbed/tests/admin_views/test_brt.py" in fsess.writes
    assert pipeline._brt_passes_in_session(fsess, sub.content)  # type: ignore[arg-type]

    # run_python runs the snippet inside a TestCase under runtests.py
    psess = FakeSession(run_output="test_probe (admin_views.test_tdbf_probe.TdbfProbe) ... observed 42\nok\n"
                                   f"{DASH}\nRan 1 test in 0.01s\n\nOK\n", rc=0)
    probe = RunPythonTool(psess)  # type: ignore[arg-type]
    assert "module" in probe.input_schema["properties"]
    pr = probe.execute({"code": "print('observed', 42)", "module": "admin_views"})
    assert "observed 42" in pr.content and not pr.content.startswith("[the snippet raised")
    assert psess.writes["/tmp/tdbf_probe.py"] == "print('observed', 42)"
    assert "/testbed/tests/admin_views/test_tdbf_probe.py" in psess.writes
    assert any(c.endswith("admin_views.test_tdbf_probe") for c, _ in psess.execs)
    print("  [ok] tools + gate + run_brt + run_python share the runtests.py path; module carried")


def test_prompt_and_feedback() -> None:
    p = pipeline.brt_system_prompt(DJ)
    assert p is pipeline.DJANGO_BRT_SYSTEM_PROMPT
    filled = p.format(repo=DJ, base_commit="abc", env_hint=brt_env_hint(DJ))
    assert "failure_reason=assertion or exception" in filled and "runtests.py" in filled
    assert "Write plain pytest functions" not in filled
    native = pipeline._brt_retry_feedback("fail", "code", "runtime", native=True)
    assert "TEST ITSELF is broken" in native
    old = pipeline._brt_retry_feedback("fail", "code", "runtime")
    assert "CRASHED before reaching its assertion" in old, "non-native wording unchanged"
    l0 = pipeline._brt_retry_feedback(None, None, None, native=True)
    assert "assertion or exception" in l0 and "same module" in l0
    assert pipeline._brt_retry_feedback("pass", "c", None, native=True) == pipeline._brt_retry_feedback("pass", "c", None)
    print("  [ok] Django prompt + retry feedback; other repos' wording untouched")


def main() -> None:
    print("agent_django_brt_smoke")
    test_non_django_unchanged()
    test_django_placement()
    test_runtests_parser()
    test_tools_and_gate()
    test_prompt_and_feedback()
    print("ALL OK")


if __name__ == "__main__":
    main()
