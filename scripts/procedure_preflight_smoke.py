"""Offline smoke tests for the procedure agent's preflight (no Docker, no model).

Covers the pure parts of src/tdbf/agent/preflight.py and src/tdbf/harness/official.py:
official test commands for every SWE-bench Lite instance, test directives, result
parsing (pytest, sympy bin/test, Django), code-block extraction and classification,
doctest conversion, traceback parsing and path mapping, identifiers, message
fragments, candidate ranking, the probe/fallback decision with a fake session, and
the gold-free input.

Fixture issue texts are synthetic: none comes from the 114 evaluation bugs.

    python3 scripts/procedure_preflight_smoke.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent import preflight as P  # noqa: E402
from tdbf.agent.session import ExecResult  # noqa: E402
from tdbf.harness import official as O  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


# --------------------------------------------------------------------------- #
# official commands                                                            #
# --------------------------------------------------------------------------- #

def test_official_commands() -> None:
    from tdbf.data.instances import InstanceStore
    rows = InstanceStore("lite")._load()
    missing = [r["instance_id"] for r in rows if O.official_command(r["repo"], r.get("version")) is None]
    check(not missing, f"official command for all {len(rows)} Lite instances (missing: {missing[:3]})")
    fw = {}
    for r in rows:
        c = O.official_command(r["repo"], r.get("version"))
        if c:
            fw.setdefault(r["repo"], set()).add(c.framework)
    check(fw.get("django/django") == {"django"}, "django -> runtests framework")
    check(fw.get("sympy/sympy") == {"sympy"}, "sympy -> bin/test framework")
    check(fw.get("sphinx-doc/sphinx") == {"tox-pytest"}, "sphinx -> tox framework")
    check(fw.get("psf/requests") == {"pytest"}, "requests -> pytest framework")

    a = O.official_command("astropy/astropy", "1.3")
    check(a is not None and "--tb=no" not in a.base and "--tb=short" in a.base and a.changes,
          "astropy 1.3: --tb=no replaced by --tb=short and recorded")
    a5 = O.official_command("astropy/astropy", "5.0")
    check(a5 is not None and a5.base == "pytest -rA" and not a5.changes, "astropy 5.0 unchanged")

    d = O.official_command("django/django", "4.0")
    check(d.for_file("tests/utils_tests/test_x.py").endswith(" utils_tests.test_x"),
          "django file -> dotted label")
    dl = [O.official_command("django/django", v) for v in ("3.0", "3.1", "3.2", "4.0", "4.1", "5.0")]
    check(any(c.exports for c in dl if c), "django: eval_commands exports kept for some versions")
    check(all(not e.startswith("export") for c in dl if c for e in c.setup),
          "django: exports and setup steps separated")
    s = O.official_command("sympy/sympy", "1.7")
    check(s.for_file("sympy/core/tests/test_x.py").endswith("bin/test -C --verbose sympy/core/tests/test_x.py"),
          "sympy command takes the file path")
    for bad in ("../etc/passwd", "-k x", "a b.py", "x;rm -rf /"):
        try:
            s.for_file(bad)
            check(False, f"unsafe path rejected: {bad!r}")
        except ValueError:
            check(True, f"unsafe path rejected: {bad!r}")
    fb = O.fallback_command("sympy/sympy")
    check(fb.source == "fallback" and "python_functions=test_*" in fb.base,
          "fallback keeps today's sympy pytest option")


SYMPY_FAIL = """\
============================= test process starts ==============================
executable:         /opt/miniconda3/envs/testbed/bin/python  (3.9.19-final-0) [CPython]
architecture:       64-bit
sympy/core/tests/test_tdbf_probe.py[2]
test_tdbf_env_probe_pass ok
test_tdbf_env_probe_fail F                                                [FAIL]

________________________________________________________________________________
________ sympy/core/tests/test_tdbf_probe.py:test_tdbf_env_probe_fail __________
Traceback (most recent call last):
  File "/testbed/sympy/core/tests/test_tdbf_probe.py", line 5, in test_tdbf_env_probe_fail
    assert 1 == 2
AssertionError

============= tests finished: 1 passed, 1 failed, in 0.01 seconds ==============
DO *NOT* COMMIT!
"""

SYMPY_ERROR = """\
sympy/core/tests/test_tdbf_probe.py[1]
test_x E                                                                [FAIL]

________________________________________________________________________________
______________ sympy/core/tests/test_tdbf_probe.py:test_x ______________________
Traceback (most recent call last):
  File "/testbed/sympy/core/tests/test_tdbf_probe.py", line 2, in test_x
    foo()
NameError: name 'foo' is not defined

=========== tests finished: 0 passed, 1 exceptions, in 0.01 seconds ============
"""

SYMPY_PASS = """\
sympy/core/tests/test_tdbf_probe.py[1]
test_tdbf_env_probe_pass ok                                                  [OK]

================== tests finished: 1 passed, in 0.01 seconds ===================
"""

PYTEST_FAIL = """\
============================= test session starts ==============================
collected 1 item

tests/test_tdbf_probe.py F                                               [100%]

=================================== FAILURES ===================================
__________________________ test_tdbf_env_probe_fail ___________________________
tests/test_tdbf_probe.py:2: in test_tdbf_env_probe_fail
    assert 1 == 2
E   assert 1 == 2
=========================== short test summary info ============================
FAILED tests/test_tdbf_probe.py::test_tdbf_env_probe_fail - assert 1 == 2
============================== 1 failed in 0.02s ===============================
"""


def test_parsers() -> None:
    s = O.official_command("sympy/sympy", "1.7")
    check(O.parse_result(s, 1, SYMPY_FAIL) == ("fail", "assertion"), "sympy F -> fail/assertion")
    check(O.parse_result(s, 1, SYMPY_ERROR) == ("fail", "runtime"), "sympy E (NameError) -> fail/runtime")
    check(O.parse_result(s, 0, SYMPY_PASS) == ("pass", None), "sympy ok -> pass")
    check(O.parse_result(s, 1, "Traceback...\nImportError: x\n")[0] == "error", "sympy crash -> error")
    p = O.official_command("psf/requests", "2.10")
    check(O.parse_result(p, 1, PYTEST_FAIL) == ("fail", "assertion"), "pytest -rA fail -> assertion")
    st = O.test_statuses("psf/requests", PYTEST_FAIL)
    check(st.get("tests/test_tdbf_probe.py::test_tdbf_env_probe_fail") == "FAILED",
          "swebench parser reads the named test's status")
    d = O.official_command("django/django", "4.0")
    dj = ("test_tdbf_env_probe_fail (utils_tests.test_tdbf_probe.TdbfEnvProbe) ... FAIL\n\n"
          "======================================================================\n"
          "FAIL: test_tdbf_env_probe_fail (utils_tests.test_tdbf_probe.TdbfEnvProbe)\n"
          "----------------------------------------------------------------------\n"
          "Traceback (most recent call last):\n"
          '  File "/testbed/tests/utils_tests/test_tdbf_probe.py", line 6, in test_tdbf_env_probe_fail\n'
          "    self.assertEqual(1, 2)\n"
          "AssertionError: 1 != 2\n\n"
          "----------------------------------------------------------------------\n"
          "Ran 1 test in 0.001s\n\nFAILED (failures=1)\n")
    check(O.parse_result(d, 1, dj) == ("fail", "assertion"), "django FAIL -> assertion (existing parser)")


# --------------------------------------------------------------------------- #
# issue code                                                                   #
# --------------------------------------------------------------------------- #

ISSUE_MD = """\
Calling `Widget.render_html` with `escape=False` crashes.

```python
from widgetlib.core import Widget
w = Widget("a<b")
print(w.render_html(escape=False))
```

Output:

```
Traceback (most recent call last):
  File "repro.py", line 3, in <module>
    print(w.render_html(escape=False))
  File "/usr/lib/python3.9/site-packages/widgetlib/core/render.py", line 88, in render_html
    return self._fmt(text)
  File "/usr/lib/python3.9/site-packages/widgetlib/core/fmt.py", line 12, in _fmt
    raise ValueError("Unsupported markup in widget text: %r" % text)
ValueError: Unsupported markup in widget text: 'a<b'
```

Install with:
```bash
pip install widgetlib
```

I expected `a<b` to be returned unchanged, as in 1.2.
"""

ISSUE_TRAC = """\
The QuerySet.count_distinct() method returns 0 for empty strings.
{{{
#!python
from myapp.models import Thing
Thing.objects.create(name="")
print(Thing.objects.count_distinct("name"))
}}}
It should return 1.
"""

ISSUE_DOCTEST = """\
Simplification is wrong:

    >>> from symlib import simplify, sqrt
    >>> simplify(sqrt(4)
    ...          + 1)
    4
    >>> x = 2

The result should be 3.
"""

ISSUE_IPY = """\
In [1]: import numlib
In [2]: numlib.mean([1, 2])
Out[2]: 1
"""


def test_extraction() -> None:
    blocks = P.extract_blocks(ISSUE_MD)
    kinds = [P.classify_block(t, l)[0] for _, l, t in blocks]
    check(kinds == ["code", "traceback", "shell"], f"markdown blocks classified {kinds}")
    tb = P.parse_tracebacks(ISSUE_MD)
    check(len(tb) == 1 and len(tb[0].frames) == 3, "one traceback with 3 frames")
    check(tb[0].exc_type == "ValueError" and "Unsupported markup" in tb[0].message, "exception and message")
    tracked = {"widgetlib/core/render.py", "widgetlib/core/fmt.py", "tests/test_render.py"}
    check(P.map_to_repo(tb[0].frames[1].file, tracked) == "widgetlib/core/render.py",
          "site-packages path -> repo path")
    check(P.map_to_repo("repro.py", tracked) is None, "user script frame not mapped")
    check(P.message_fragments(tb) == ["Unsupported markup in widget text"], "message fragment")

    trac = P.extract_blocks(ISSUE_TRAC)
    check(len(trac) == 1 and trac[0][0] == "trac" and trac[0][1] == "python", "Trac {{{ #!python }}} block")
    check(P.classify_block(trac[0][2], trac[0][1])[0] == "code", "Trac block is runnable code")

    dt = P.extract_blocks(ISSUE_DOCTEST)
    check(len(dt) == 1 and dt[0][0] == "indented", "indented doctest block found")
    kind, script, _ = P.classify_block(dt[0][2], None)
    check(kind == "doctest" and script is not None, "doctest converted")
    ns: dict = {}
    import contextlib
    import io
    src = script.replace("from symlib import simplify, sqrt", "simplify = lambda v: v; sqrt = lambda v: v ** 0.5")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<t>", "exec"), ns)
    check(buf.getvalue().strip() == "3.0", f"doctest script echoes expression values ({buf.getvalue().strip()!r})")

    ipy = P.extract_blocks(ISSUE_IPY)
    check(ipy == [] or P.classify_block(ipy[0][2], None)[0] in ("doctest", "text"),
          "IPython prompts not mistaken for code")
    k2, s2, _ = P.classify_block(ISSUE_IPY, None)
    check(k2 == "doctest" and "numlib.mean" in s2, "IPython session converted")

    check(P.classify_block("Foo", None)[0] == "text", "lone name is prose")
    check(P.classify_block("$ python x.py", None)[0] == "shell", "shell prompt skipped")
    check(P.classify_block("x = [1, 2\n", None)[0] == "text", "unparsable block skipped")
    check(P.classify_block("<div></div>", "html")[0] == "text", "non-python language skipped")

    out = ("Traceback (most recent call last):\n  File \"s.py\", line 1, in <module>\n"
           "    f()\nKeyError: 'x'\n")
    check(P.exception_in_output(out) == "KeyError", "exception read from snippet output")
    pyt = "widgetlib/core/fmt.py:12: in _fmt\n    raise X\nwidgetlib/core/fmt.py:14: ValueError\n"
    tbs = P.parse_tracebacks(pyt)
    check(len(tbs) == 1 and tbs[0].exc_type == "ValueError", "pytest-style traceback parsed")


ISSUE_IPY_TB = """\
Poly with a domain fails
``` py
In [3]: Thing(1.5, kind='A[b]')
---------------------------------------------------------------------------
KindError                                 Traceback (most recent call last)
<ipython-input-3-abc> in <module>()
----> 1 Thing(1.5, kind='A[b]')

/home/u/src/thinglib/core.py in __new__(cls, value, **kw)
     40     def __new__(cls, value, **kw):
---> 41         opt = build(kw)

/home/u/src/thinglib/options.py in build(kw)
    118         raise KindError("expected a valid kind specification, got %s" % k)

KindError: expected a valid kind specification, got A[b]
```
"""

ISSUE_CRLF = ("Bad n\n```python\r\nfrom pkg import f\r\nf(3.)\r\n```\r\n```\r\n"
              "~/src/pkg/tree.pxi in pkg.tree.Heap.__init__()\r\n\r\n"
              "TypeError: 'float' object cannot be interpreted as an integer\r\n```\r\n")


def test_formats() -> None:
    b = P.extract_blocks(ISSUE_IPY_TB)
    kind, script, _ = P.classify_block(b[0][2], b[0][1])
    check(kind == "doctest" and "Thing(1.5" in script, "IPython session with traceback is run, not skipped")
    tb = P.parse_tracebacks(ISSUE_IPY_TB)
    check(len(tb) == 1 and [f.file for f in tb[0].frames] == ["/home/u/src/thinglib/core.py",
                                                              "/home/u/src/thinglib/options.py"],
          f"IPython frames parsed: {[(f.file, f.line) for f in tb[0].frames] if tb else tb}")
    check(tb and tb[0].frames[0].line == 41, "IPython frame line from the ---> arrow")
    check(tb and tb[0].exc_type == "KindError" and tb[0].message.startswith("expected a valid kind"),
          "IPython exception and message")
    c = P.extract_blocks(ISSUE_CRLF)
    check(len(c) == 2 and P.classify_block(c[0][2], c[0][1])[0] == "code", "CRLF fenced blocks extracted")
    tb2 = P.parse_tracebacks(ISSUE_CRLF)
    check(len(tb2) == 1 and tb2[0].frames[0].file.endswith("tree.pxi") and tb2[0].exc_type == "TypeError",
          "bare IPython frame (.pxi) with exception parsed")
    check(P.map_to_repo(tb2[0].frames[0].file, {"pkg/tree.pxi"}) == "pkg/tree.pxi", ".pxi frame mapped")
    ex = P.inline_expressions("evaluate `0**-oo` gives `0`; see `Pow` and `r.iter_content(16, decode_unicode=True)`"
                              " and ``discrete_features == 'auto'``")
    check(ex == ["0**-oo", "r.iter_content(16, decode_unicode=True)", "discrete_features == 'auto'"],
          f"inline expressions: {ex}")
    buf = __import__("io").StringIO()
    with __import__("contextlib").redirect_stdout(buf):
        exec(compile(P.inline_script(["1 + 1", "1 / 0"]), "<t>", "exec"), {})
    check(buf.getvalue().splitlines() == ["1 + 1 -> 2", "1 / 0 -> raised ZeroDivisionError: division by zero"],
          f"inline script output: {buf.getvalue().splitlines()}")
    ctx = P.session_context("sympy/sympy", ["a = 1"])
    check(ctx is not None and "isympy" in ctx[0] and "from sympy import *" in ctx[1] and "a = 1" in ctx[1],
          "sympy retry context: isympy namespace + earlier block")
    check(P.session_context("psf/requests", []) is None, "no retry context without earlier blocks")
    s = P.doctest_to_script(">>> print(q + 1)")
    buf = __import__("io").StringIO()
    with __import__("contextlib").redirect_stdout(buf):
        exec(compile("q = 41\n" + s, "<t>", "exec"), {"__name__": "__main__"})
    check(buf.getvalue().strip() == "42", "doctest script sees names from prepended context")
    defs = {"fit": [{"file": f"pkg/m{i}.py", "line": 1, "qualname": f"C{i}.fit"} for i in range(9)]
            + [{"file": "pkg/m0.py", "line": 9, "qualname": "D.fit"}],
            "Heap": [{"file": "pkg/heap.py", "line": 3, "qualname": "Heap"}]}
    ranked = P.rank_code_candidates([], defs, {}, ["fit", "Heap"])
    check(ranked[0].path == "pkg/heap.py", f"rare identifier beats a common one: {ranked[0].path}")
    obs = [P.Traceback([P.Frame("/testbed/pkg/core.py", 10, "g", "pkg/core.py")], "TypeError", "")]
    ranked = P.rank_code_candidates([], {}, {}, [], obs)
    check(ranked and ranked[0].path == "pkg/core.py" and "run of issue code" in ranked[0].reasons[0],
          "frames from preflight's own run become code candidates")
    code = P.rank_code_candidates([], {"f": [{"file": "examples/demo.py", "line": 1, "qualname": "f"},
                                             {"file": "pkg/f.py", "line": 3, "qualname": "f"}]}, {}, ["f"])
    check([x.path for x in code] == ["pkg/f.py"], "examples/ excluded from code candidates")


def test_identifiers_and_ranking() -> None:
    blocks = [t for _, l, t in P.extract_blocks(ISSUE_MD) if P.classify_block(t, l)[0] == "code"]
    ids = P.extract_identifiers(ISSUE_MD, blocks, "widgetlib")
    check("Widget.render_html" in ids[:3] or "render_html" in ids[:3], f"backticked method ranked high: {ids[:5]}")
    check("Widget" in ids, "imported class kept")
    check(not any(i in ("print", "self", "False") for i in ids), "builtins and keywords dropped")

    tb = P.parse_tracebacks(ISSUE_MD)
    tracked = {"widgetlib/core/render.py", "widgetlib/core/fmt.py", "widgetlib/core/tests/test_render.py",
               "widgetlib/core/tests/__init__.py", "tests/test_misc.py"}
    for f in tb[0].frames:
        f.repo_path = P.map_to_repo(f.file, tracked)
    defs = {"render_html": [{"file": "widgetlib/core/render.py", "line": 80, "qualname": "Widget.render_html"}],
            "Widget": [{"file": "widgetlib/core/render.py", "line": 10, "qualname": "Widget"}]}
    msg = {"Unsupported markup in widget text": ["widgetlib/core/fmt.py"]}
    code = P.rank_code_candidates(tb, defs, msg, ids)
    check([c.path for c in code[:2]] == ["widgetlib/core/fmt.py", "widgetlib/core/render.py"] or
          [c.path for c in code[:2]] == ["widgetlib/core/render.py", "widgetlib/core/fmt.py"],
          f"code candidates: {[(c.path, c.score) for c in code]}")
    check(all(c.reasons for c in code), "every code candidate carries its reasons")
    tests = P.rank_test_candidates("x/widgetlib", code, {"Widget": ["tests/test_misc.py"]},
                                   tracked, "widgetlib/core/tests")
    paths = [t.path for t in tests]
    check("widgetlib/core/tests/test_render.py" in paths, f"conventional neighbour found: {paths}")
    check("widgetlib/core/tests/" in paths, "default directory listed last-resort")
    check(tests[-1].path == "widgetlib/core/tests/", "default directory ranked last")

    dj = P.neighbour_tests("django/django", "django/utils/text.py",
                           {"tests/utils_tests/test_text.py", "tests/text/tests.py"})
    check(dj and dj[0] == "tests/utils_tests/test_text.py", f"django neighbour: {dj}")
    check(P.django_module("tests/utils_tests/test_text.py") == "utils_tests", "django module label")


# --------------------------------------------------------------------------- #
# environment decision with a fake session                                     #
# --------------------------------------------------------------------------- #

class FakeSession:
    """Answers commands from a table; records writes and commands."""

    def __init__(self, repo: str, official_ok: bool, fallback_ok: bool) -> None:
        self.repo = repo
        self.official_ok, self.fallback_ok = official_ok, fallback_ok
        self.files: dict[str, str] = {}
        self.commands: list[str] = []

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    def ensure_pytest(self) -> None:
        pass

    def exec(self, command: str, *, timeout: int = 120, activate: bool = True, env=None) -> ExecResult:
        self.commands.append(command)
        if command.startswith("python --version"):
            return ExecResult(0, "Python 3.9.19\n")
        if command.startswith("python -c 'import"):
            return ExecResult(0, "")
        if "bin/test" in command or "pytest" in command:
            ok = self.official_ok if "bin/test" in command else self.fallback_ok
            content = next(iter(v for k, v in self.files.items() if k.endswith(P.PROBE_FILE)), "")
            is_fail = "1 == 2" in content
            if not ok:
                return ExecResult(2, "ERROR: usage")
            if "bin/test" in command:
                return (ExecResult(1, SYMPY_FAIL.replace("test_tdbf_env_probe_pass ok\n", ""))
                        if is_fail else ExecResult(0, SYMPY_PASS))
            return (ExecResult(1, PYTEST_FAIL) if is_fail else
                    ExecResult(0, "PASSED sympy/core/tests/test_tdbf_probe.py::test_tdbf_env_probe_pass\n"
                                  "1 passed in 0.01s\n"))
        return ExecResult(0, "")


def test_environment_decision() -> None:
    pin = P.PreflightInput("sympy__sympy-1", "sympy/sympy", "abc", "issue", "1.7")
    tracked = ["sympy/core/basic.py", "sympy/core/tests/test_basic.py", "sympy/core/tests/__init__.py"]
    env = P.check_environment(FakeSession("sympy/sympy", True, True), pin, tracked)
    check(not env.invalid and env.command["source"] == "official", "official runner chosen when it works")
    check(env.probe_dir == "sympy/core/tests", "probe placed in the default test dir")
    env = P.check_environment(FakeSession("sympy/sympy", False, True), pin, tracked)
    check(not env.invalid and env.command["source"] == "fallback", "fallback when official fails")
    check(len([p for p in env.probes if "runner" in p]) == 2, "both probes recorded")
    env = P.check_environment(FakeSession("sympy/sympy", False, False), pin, tracked)
    check(env.invalid and env.command is None, "environment_invalid when neither works")


def test_gold_free_input() -> None:
    class Inst:
        instance_id, repo, base_commit, version = "x__y-1", "x/y", "abc", "1.0"
        problem_statement = "issue"
        gold_patch, test_patch = "GOLD", "TESTS"

        def context_safe_dict(self):
            return {"instance_id": self.instance_id, "repo": self.repo,
                    "base_commit": self.base_commit, "problem_statement": self.problem_statement}

    pin = P.PreflightInput.from_instance(Inst())
    vals = " ".join(str(v) for v in vars(pin).values())
    check("GOLD" not in vals and "TESTS" not in vals, "preflight input carries no gold patch or hidden tests")
    check(set(vars(pin)) == {"instance_id", "repo", "base_commit", "problem_statement", "version"},
          "preflight input fields are exactly the gold-free ones")


def main() -> int:
    for t in (test_official_commands, test_parsers, test_extraction, test_formats,
              test_identifiers_and_ranking, test_environment_decision, test_gold_free_input):
        print(f"--- {t.__name__}")
        t()
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
