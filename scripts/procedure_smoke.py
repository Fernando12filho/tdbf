"""Offline smoke tests for the procedure agent's controller, gates and soft mode.

No Docker, no model: a scripted client answers per phase (it sees which finish_* tool
it is offered), and the container steps (preflight, snippet runs, test runs, probes)
are replaced by stubs. Checks: the deterministic gates, phase routing and skips, the
send-backs (passed -> reproduce, no evidence -> pin, crash/unmapped -> write), D2
(unverified values only as comments, flagged), the 50-turn budget, the record, and the
soft variant. Issue texts are synthetic.

    python3 scripts/procedure_smoke.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent import procedure as PR  # noqa: E402
from tdbf.agent import procedure_tools as PT  # noqa: E402
from tdbf.agent import procedure_gates as G  # noqa: E402
from tdbf.agent.preflight import (Candidate, Candidates, EnvProfile, Preflight, Snippet)  # noqa: E402
from tdbf.agent.session import ExecResult  # noqa: E402
from tdbf.harness.official import official_command  # noqa: E402
from tdbf.models.messages import ToolCall, ToolUseResponse  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


# --------------------------------------------------------------------------- #
# gates (pure)                                                                 #
# --------------------------------------------------------------------------- #

def test_gates() -> None:
    issue = "Using it, evaluate `0**-oo` produces `0`.\r\nThe docs say it should return `zoo`."
    check(G.quote_in(issue, "evaluate 0**-oo produces 0."), "quote found despite backticks/CRLF")
    check(not G.quote_in(issue, "it returns infinity"), "invented quote rejected")
    check(not G.quote_in(issue, "it"), "too-short quote rejected")

    obs = [{"value": "`0`", "quote": "produces `0`"}]
    check(G.symptom_match(obs, "0**-oo -> 0")[0] == "matched", "value token matched")
    check(G.symptom_match(obs, "0**-oo -> zoo")[0] == "none", "absent value -> none")
    exc = [{"value": "raises KeyError", "quote": "q"}]
    tb = "Traceback (most recent call last):\n  File \"s.py\", line 1, in <module>\n    f()\nKeyError: 'a'\n"
    check(G.symptom_match(exc, tb)[0] == "matched", "named exception matched")
    tb2 = tb.replace("KeyError: 'a'", "ValueError: x")
    check(G.symptom_match(exc, tb2)[0] == "partial", "different exception -> partial")
    tb3 = tb.replace("KeyError: 'a'", "NameError: name 'f' is not defined")
    check(G.symptom_match(exc, tb3)[0] == "none", "authoring error -> none")
    check(G.symptom_match([{"value": "it is slow", "quote": "q"}], "x")[0] == "unchecked",
          "nothing concrete -> unchecked")

    code = ("from m import f\n\n\ndef test_brt():\n    x = f()\n    assert x == 3, (\n"
            "        'msg')\n    # UNVERIFIED: x.y == 2\n")
    check(G.assertion_statements(code) == [(6, 7)], f"assertion statements {G.assertion_statements(code)}")
    check(G.statement_at(code, 7) == (6, 7), "multi-line assert statement range")
    out_pytest = "tests/test_brt.py:6: in test_brt\n    assert x == 3\nE   assert 2 == 3\n"
    check(G.failing_line(out_pytest, "tests/test_brt.py") == 6, "failing line, pytest short")
    out_long = ">       assert x == 3\nE       assert 2 == 3\n\ntests/test_brt.py:6: AssertionError\n"
    check(G.failing_line(out_long, "tests/test_brt.py") == 6, "failing line, pytest long")
    out_std = 'Traceback:\n  File "/testbed/sympy/core/tests/test_brt.py", line 7, in test_brt\n'
    check(G.failing_line(out_std, "sympy/core/tests/test_brt.py") == 7, "failing line, unittest/sympy")
    ut = ("def t(self):\n    with self.assertRaises(ValueError):\n        f()\n"
          "    self.assertEqual(a, 1)\n    pytest.fail('x')\n")
    check(G.assertion_statements(ut) == [(2, 2), (4, 4), (5, 5)], "assertRaises/assertEqual/pytest.fail found")

    run = {"outcome": "fail", "reason": "assertion", "output": "E   assert 0 == 3", "failing_line": 6}
    check(not G._token_in("0**-oo -> zoo", "0") and G._token_in("0**-oo -> 0", "0")
          and G._token_in("AssertionError: 0 != zoo", "0"), "short tokens only as printed/compared values")
    ok = G.check_write(code, run, [{"line": 6, "evidence_id": "E1"}], {"E1"}, {"E1"},
                       accept_exception=False, symptom_tokens=["0"])
    check(ok.passed, f"write gate passes a mapped, evidenced failing assertion ({ok.cause})")
    r = G.check_write(code, {**run, "outcome": "pass"}, [], {"E1"}, {"E1"}, accept_exception=False,
                      symptom_tokens=None)
    check(r.cause == "passed_on_buggy", "pass on buggy -> passed_on_buggy")
    r = G.check_write(code, {**run, "reason": "runtime"}, [], {"E1"}, {"E1"}, accept_exception=False,
                      symptom_tokens=None)
    check(r.cause == "crashed", "runtime failure -> crashed")
    r = G.check_write(code, {**run, "reason": "exception"}, [{"line": 6, "evidence_id": "E1"}], {"E1"},
                      {"E1"}, accept_exception=True, symptom_tokens=None)
    check(r.passed, "Django: exception from the code accepted")
    r = G.check_write(code, run, [], {"E1"}, {"E1"}, accept_exception=False, symptom_tokens=None)
    check(r.cause == "unmapped_assertion", "unmapped assertion -> unmapped_assertion (D2)")
    r = G.check_write(code, run, [{"line": 6, "evidence_id": "U1"}], {"E1"}, {"E1", "U1"},
                      accept_exception=False, symptom_tokens=None)
    check(r.cause == "no_evidence", "assertion mapped to an unknown -> no_evidence")
    r = G.check_write(code, run, [{"line": 6, "evidence_id": "E9"}], {"E1"}, {"E1"},
                      accept_exception=False, symptom_tokens=None)
    check(r.cause == "bad_evidence_id", "made-up evidence id -> bad_evidence_id")
    r = G.check_write(code, run, [{"line": 6, "evidence_id": "E1"}], {"E1"}, {"E1"},
                      accept_exception=False, symptom_tokens=["KeyError"])
    check(r.cause == "symptom_mismatch", "failure without the reproduced token -> symptom_mismatch")
    bare = {**run, "output": 'File "/t/test_brt.py", line 6, in test_brt\n    assert x == 3\nAssertionError\n'}
    r = G.check_write(code, bare, [{"line": 6, "evidence_id": "E1"}], {"E1"}, {"E1"},
                      accept_exception=False, symptom_tokens=["KeyError"])
    check(r.passed and r.details.get("symptom") == "not_checkable",
          "bare AssertionError (sympy bin/test): symptom not checkable -> passes, flagged")
    check(G.assertion_detail("E   assert 0 == 3") and G.assertion_detail("AssertionError: 0 != zoo")
          and not G.assertion_detail("AssertionError\n"), "assertion detail detected")

    b, res = PR.phase_budgets(50)
    check(b == {"understand": 2, "reproduce": 12, "pin_expected": 10, "locate": 4, "write": 15} and res == 7,
          "D3 budgets 2/12/10/4/15 + reserve 7")
    b30, r30 = PR.phase_budgets(30)
    check(sum(b30.values()) + r30 == 30, "scaled budgets sum to the total")


# --------------------------------------------------------------------------- #
# controller with a scripted client and stubbed container                      #
# --------------------------------------------------------------------------- #

ISSUE = ("Power of zero is wrong\nUsing SymPy 1.5.1, evaluate `0**-oo` produces `0`.\n"
         "The documentation for the Pow class states that it should return `ComplexInfinity`, aka `zoo`.\n")
GOOD = "from sympy import S, oo, zoo\n\n\ndef test_brt():\n    assert S.Zero**-oo == zoo\n"
PASSING = "from sympy import S\n\n\ndef test_brt():\n    assert S.Zero == 0\n"
UNMAPPED = "from sympy import S, oo, zoo\n\n\ndef test_brt():\n    assert S.Zero**-oo == zoo\n    assert 1 == 1\n"


class Inst:
    instance_id, repo, base_commit, version = "sympy__sympy-99999", "sympy/sympy", "abc", "1.7"
    problem_statement = ISSUE
    gold_patch = "GOLD"

    def context_safe_dict(self):
        return {"instance_id": self.instance_id, "repo": self.repo, "base_commit": self.base_commit,
                "problem_statement": self.problem_statement}


class FakeSession:
    repo = "sympy/sympy"
    instance_id = "sympy__sympy-99999"

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = files or {}

    def exec(self, command, *, timeout=120, activate=True, env=None):
        if "git cat-file" in command:
            return ExecResult(0, "")
        return ExecResult(0, "")

    def read_file(self, path, *, max_bytes=0):
        if path in self.files:
            return ExecResult(0, self.files[path])
        return ExecResult(1, "No such file")

    def write_file(self, path, content):
        pass

    def ensure_pytest(self):
        pass

    def close(self):
        pass


def fake_preflight(snippet_output: str | None) -> Preflight:
    cmd = official_command("sympy/sympy", "1.7")
    env = EnvProfile("sympy/sympy", "1.7", python_version="3.9", command=cmd.to_json(),
                     default_test_dir="sympy/core/tests", probe_dir="sympy/core/tests")
    snips = []
    if snippet_output is not None:
        snips.append(Snippet(0, "inline", "inline", None, "0**-oo", script="print(0**-oo)",
                             runnable=True, ran=True, returncode=0, output=snippet_output))
    cands = Candidates(["Pow"], [Candidate("sympy/core/power.py", 5.0, ["defines Pow"])],
                       [Candidate("sympy/core/tests/test_power.py", 6.0, ["conventional test"])])
    return Preflight("sympy__sympy-99999", env, snips, [], cands)


def test_run(code: str) -> dict:
    """Stub of run_test_at: what the buggy sympy would do with these tests."""
    if "== zoo" in code:
        line = next(i for i, ln in enumerate(code.splitlines(), 1) if "== zoo" in ln)
        return {"outcome": "fail", "reason": "assertion", "returncode": 1, "failing_line": line,
                "output": f'File "/testbed/sympy/core/tests/test_brt.py", line {line}, in test_brt\n'
                          "AssertionError: 0 != zoo", "path": "sympy/core/tests/test_brt.py", "code": code}
    return {"outcome": "pass", "reason": None, "returncode": 0, "failing_line": None,
            "output": "test_brt ok", "path": "sympy/core/tests/test_brt.py", "code": code}


class ScriptedClient:
    """Answers by the finish tool it is offered: script[finish_name] is a list of visits,
    each a list of turns, each a list of (tool, args). Missing turns answer with no tool."""

    name = "scripted"

    def __init__(self, script: dict[str, list[list[list[tuple[str, dict]]]]]) -> None:
        self.script = script
        self.visit: dict[str, int] = {}
        self.turn: dict[str, int] = {}
        self.calls = 0
        self.seen_tools: list[list[str]] = []

    def run_tools(self, *, system, messages, tools, max_tokens=4096):
        self.calls += 1
        names = [t.name for t in tools]
        self.seen_tools.append(names)
        fin = next((n for n in names if n.startswith(("finish_", "submit_"))), None)
        if len(messages) == 1:  # a new visit starts with one user message
            self.visit[fin] = self.visit.get(fin, -1) + 1
            self.turn[fin] = 0
        visits = self.script.get(fin, [])
        v = self.visit.get(fin, 0)
        turns = visits[v] if v < len(visits) else []
        t = self.turn.get(fin, 0)
        self.turn[fin] = t + 1
        calls = turns[t] if t < len(turns) else []
        tcs = [ToolCall(f"c{self.calls}_{k}", n, a) for k, (n, a) in enumerate(calls)]
        return ToolUseResponse(text="", tool_calls=tcs, stop_reason="tool_use" if tcs else "end_turn",
                               input_tokens=100, output_tokens=10)


UNDERSTAND_OK = [[("finish_understand", {
    "observed": [{"value": "`0`", "quote": "evaluate `0**-oo` produces `0`."}],
    "expected": [{"value": "`zoo`", "quote": "it should return `ComplexInfinity`, aka `zoo`"}],
    "unknowns": []})]]


def run_proc(script, *, snippet="0**-oo -> 0", files=None, mode="strict", total=50, patch_test=test_run):
    PR.run_preflight = lambda session, inst: fake_preflight(snippet)
    PR.run_probe = lambda session, cmd, d: {"ok": True, "pass": {"outcome": "pass"},
                                           "fail": {"outcome": "fail", "reason": "assertion"}}

    def snippet_stub(session, repo, code, module="brt_app"):
        return (0, "0**-oo -> 0") if "0**-oo" in code else (1, "Traceback...\nNameError: name 'q' is not defined")
    PR.run_snippet = snippet_stub
    PT.run_snippet = snippet_stub
    stub = lambda session, cmd, home, code, timeout=180: patch_test(code)  # noqa: E731
    PR.run_test_at = stub
    PT.run_test_at = stub
    client = ScriptedClient(script)
    proc = PR.Procedure(Inst(), client, mode=mode, total_turns=total, session=FakeSession(files))
    return proc.run(), client


def test_happy_path() -> None:
    res, client = run_proc({
        "finish_understand": [UNDERSTAND_OK[0:1][0:1] and [UNDERSTAND_OK[0]]],
        "finish_write": [[[("run_test", {"code": GOOD})],
                          [("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "E1"}]})]]],
    })
    s = res.record
    phases = [p["phase"] for p in s["phases"]]
    check(phases == ["understand", "write"], f"only understand and write ran ({phases})")
    skipped = [g["phase"] for g in s["gates"] if g["cause"] == "skipped"]
    check(skipped == ["reproduce", "pin_expected", "locate"],
          f"reproduce (preflight matched), pin (no unknowns), locate (clear candidate) skipped: {skipped}")
    check(s["outcome"] == "test" and res.turns_used == 3, f"test accepted in 3 turns ({res.turns_used})")
    check(res.test_code.startswith("# Bug reproduction test") and "E1 (expected, issue)" in res.test_code,
          "evidence header written on the test")
    check(not s["flags"], f"no flags ({s['flags']})")
    check(s["test_home"]["source"] == "preflight_top" and s["test_home"]["test_dir"] == "sympy/core/tests",
          "test home from the clear preflight candidate")
    check(client.seen_tools[0] == ["finish_understand"], "understand offers no repository tools")
    check("run_python" not in client.seen_tools[-1] and "run_test" in client.seen_tools[-1],
          "write offers run_test, not run_python (design §3.6)")
    check(s["repro"]["source"] == "preflight", "reproduction taken from preflight")
    json.dumps(res.record)
    json.dumps(res.transcript, default=str)
    check(True, "record and transcript serialise to JSON")


def test_understand_retry() -> None:
    bad = [[("finish_understand", {"observed": [{"value": "0", "quote": "it gives zero always"}],
                                   "expected": [], "unknowns": []})]]
    res, _ = run_proc({
        "finish_understand": [bad, UNDERSTAND_OK],
        "finish_write": [[[("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "E1"}]})]]],
    })
    g = [x for x in res.record["gates"] if x["phase"] == "understand"]
    check([x["cause"] for x in g] == ["quote_missing", "ok"], f"understand repeated once on a bad quote: {[x['cause'] for x in g]}")
    check(res.record["visits"]["understand"] == 2 and res.record["outcome"] == "test", "then continued")


def test_send_backs() -> None:
    res, _ = run_proc({
        "finish_understand": [UNDERSTAND_OK],
        "finish_write": [
            [[("finish_write", {"code": PASSING, "assertion_map": [{"line": 5, "evidence_id": "E1"}]})]],
            [[("finish_write", {"code": UNMAPPED, "assertion_map": [{"line": 5, "evidence_id": "E1"}]})]],
            [[("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "E1"}]})]],
        ],
        "finish_reproduce": [[[("run_python", {"code": "print(0**-oo)"})],
                              [("finish_reproduce", {"run_id": "r1", "output_summary": "0"})]]],
    })
    causes = [(g["phase"], g["cause"]) for g in res.record["gates"]]
    check(("write", "passed_on_buggy") in causes and ("reproduce", "matched") in causes,
          "passed on buggy -> back to reproduce, which matched")
    check(("write", "unmapped_assertion") in causes, "unmapped assertion -> stays in write")
    check(res.record["outcome"] == "test" and res.record["visits"]["write"] == 3, "accepted on the 3rd write visit")
    check(res.turns_used <= 50, "within 50 turns")


def test_pin_and_d2() -> None:
    und = [[("finish_understand", {"observed": [{"value": "`0`", "quote": "evaluate `0**-oo` produces `0`."}],
                                   "expected": [], "unknowns": [{"question": "what should 0**-oo be?"}]})]]
    files = {"sympy/core/power.py": "    | 0**-oo | zoo | This is not strictly true |"}
    res, _ = run_proc({
        "finish_understand": [und],
        "finish_pin": [[[("finish_pin", {"resolved": [{"unknown_id": "U1", "value": "zoo", "source": "docs",
                                                       "quote": "| 0**-oo | zoo |", "where": "sympy/core/power.py"}]})]]],
        "finish_write": [[[("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "E1"}]})]]],
    }, files=files)
    pin = next(g for g in res.record["gates"] if g["phase"] == "pin_expected")
    check(pin["details"]["verified"] == ["U1"], "docs quote verified in the file -> unknown resolved")
    check(res.record["outcome"] == "test" and not res.record["unknowns"], "test accepted with pinned evidence")

    res2, _ = run_proc({
        "finish_understand": [und],
        "finish_pin": [[[("finish_pin", {"resolved": [{"unknown_id": "U1", "value": "zoo", "source": "model"}]})]],
                       [[("finish_pin", {"resolved": []})]]],
        "finish_write": [[[("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "U1"}]})]],
                         [[("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "U1"}]})]],
                         [[("finish_write", {"code": GOOD, "assertion_map": [{"line": 5, "evidence_id": "U1"}]})]]],
    }, files=files)
    causes = [(g["phase"], g["cause"]) for g in res2.record["gates"]]
    check(("write", "no_evidence") in causes, "assertion on an unknown -> no_evidence")
    check(sum(1 for p, _ in causes if p == "pin_expected") == 2, "sent back to pin_expected once more")
    check(res2.record["outcome"] == "forced_test" and "forced_submit" in res2.record["flags"],
          "never verified -> forced submission, flagged")
    check("unknown_expected" in res2.record["flags"] and "# UNVERIFIED: zoo" in res2.test_code,
          "D2: unknown kept as # UNVERIFIED comment and flagged")


def test_budget() -> None:
    res, client = run_proc({}, snippet=None)  # the model never calls a finish tool
    check(res.turns_used <= 50 and client.calls <= 50, f"never more than 50 turns ({res.turns_used}, {client.calls} calls)")
    check(res.record["outcome"] == "no_test", "no test without any run")
    v = res.record["visits"]
    check(all(v[p] <= PR.MAX_VISITS[p] for p in v), f"visit limits respected: {v}")
    check("unreproduced" in res.record["flags"], f"flags recorded: {res.record['flags']}")
    # a model that keeps working without finishing: every phase uses its whole allowance
    busy = {f: [[[("glob", {"pattern": "x"})]] * 60] * 3 for f in
            ("finish_understand", "finish_reproduce", "finish_pin", "finish_locate", "finish_write")}
    busy["finish_understand"] = [[[]]]
    res, client = run_proc(busy, snippet=None)
    check(res.turns_used == 50 and client.calls == 50, f"busy model stops at exactly 50 turns ({client.calls})")
    check("budget_exhausted" in res.record["flags"], f"budget_exhausted flagged: {res.record['flags']}")


def test_soft() -> None:
    res, client = run_proc({"submit_test": [[[("run_python", {"code": "print(0**-oo)"})],
                                             [("run_test", {"code": GOOD, "test_dir": "sympy/core/tests"})],
                                             [("submit_test", {"code": GOOD, "test_dir": "sympy/core/tests",
                                                               "assertion_map": []})]]]},
                           mode="soft")
    s = res.record
    check(s["mode"] == "agent_procedure_soft" and [p["phase"] for p in s["phases"]] == ["write"],
          "soft: one agent run")
    check(set(client.seen_tools[0]) == {"read_file", "grep", "glob", "run_python", "run_test", "submit_test"},
          "soft: all tools at once")
    check(s["outcome"] == "test" and res.turns_used == 3 and not s["gates"], "soft: no gates, test kept")


def main() -> int:
    for t in (test_gates, test_happy_path, test_understand_retry, test_send_backs, test_pin_and_d2,
              test_budget, test_soft):
        print(f"--- {t.__name__}")
        t()
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
