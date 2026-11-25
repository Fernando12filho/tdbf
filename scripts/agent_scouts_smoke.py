"""Offline smoke for the scouts architecture (three scouts in parallel, organizer,
BRT agent with gate 2 = the failure must match the issue).

No Docker, no LLM, no budget. Fake clients drive every role against a fake session;
the gold barrier is faked. Asserts:

  - scout / organizer / judge tool registries are gold-free and complete, and the
    guards reject a source-editing or fixed-tree tool;
  - submit_match validates its verdict, and the judge fails OPEN to 'match';
  - gate 2: mismatch -> retry with the judge's feedback -> match; the final test is
    the retried one and the record carries every verdict;
  - the match budget is respected (always 'mismatch' stops after match_retries);
  - a match retry that fails gate 1 falls back to the last gate-passing test;
  - with no brief and no judge the BRT phase record is unchanged (no new keys);
  - run_scouts_brt_phase: each role uses its own client, the organizer's brief
    reaches the test writer, a crashing scout does not sink the others, and a
    missing brief falls back to the raw scout reports;
  - gold_patch / test_patch never reach any agent or record.

    python3 scripts/agent_scouts_smoke.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_pipeline_smoke import PipelineFakeSession  # noqa: E402

from tdbf.agent import pipeline, scouts  # noqa: E402
from tdbf.agent.roles import _parse_match  # noqa: E402
from tdbf.agent.runtime import AgentResult  # noqa: E402
from tdbf.agent.session import ExecResult  # noqa: E402
from tdbf.agent.tools import (  # noqa: E402
    CheckTestTool,
    EditFileTool,
    GuardedSubmitTestTool,
    MatchLoopState,
    RunBrtTool,
    RunTestTool,
    assert_writer_v2_gold_free,
    writer_v2_agent_tools,
    SubmitMatchTool,
    assert_match_gold_free,
    assert_organizer_gold_free,
    assert_scout_gold_free,
    match_agent_tools,
    organizer_agent_tools,
    scout_agent_tools,
)
from tdbf.data.instances import SWEBenchInstance  # noqa: E402
from tdbf.harness.runner import ContainerResult  # noqa: E402
from tdbf.models.messages import ToolCall, ToolUseResponse  # noqa: E402

_INST = SWEBenchInstance(
    instance_id="demo__pkg-1", repo="demo/pkg", base_commit="abc123",
    gold_patch="GOLD MUST NOT BE USED", test_patch="TEST_PATCH MUST NOT BE USED",
    problem_statement="compute() returns 4 but should return 5.",
)
_TEST_A = "def test_brt():\n    assert compute() == 6  # invented value\n"
_TEST_B = "def test_brt():\n    assert compute() == 5  # the issue's value\n"
_TEST_PASSES = "def test_brt():\n    assert True  # PASSES_ON_BUGGY\n"
_LEAK = "MUST NOT BE USED"


def _resp(name: str, args: dict, tag: str, tin: int = 1000, tout: int = 50) -> ToolUseResponse:
    return ToolUseResponse(text=tag, tool_calls=[ToolCall(tag, name, args)],
                           stop_reason="tool_use", input_tokens=tin, output_tokens=tout)


class RoleClient:
    """A fake client for ONE role. `script` is what it submits, call by call
    (the last entry repeats). Records every task it was given."""

    def __init__(self, role: str, script: list) -> None:
        self.role = role
        self.name = f"fake:{role}"
        self.script = script
        self.calls = 0
        self.tasks: list[str] = []
        self.threads: set[str] = set()

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        self.tasks.append(messages[0].content)
        names = {t.name for t in tools}
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        if "submit_test" in names:
            return _resp("submit_test", {"code": item}, f"w{self.calls}", 8000, 200)
        if "submit_match" in names:
            verdict, feedback = item
            return _resp("submit_match", {"verdict": verdict, "feedback": feedback},
                         f"j{self.calls}", 2000, 60)
        terminal = next(n for n in names if n.startswith("submit_"))
        field = "brief" if terminal == "submit_brief" else "report"
        return _resp(terminal, {field: item}, f"{self.role}{self.calls}", 3000, 100)


class GateSession(PipelineFakeSession):
    """pytest fails (assertion) unless the written test contains PASSES_ON_BUGGY."""

    def __init__(self, repo: str = "demo/pkg") -> None:
        super().__init__(repo)
        self.last_test = ""

    def write_file(self, path, content) -> None:
        super().write_file(path, content)
        if "test_brt" in path:
            self.last_test = content

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        if "pytest" in command and "PASSES_ON_BUGGY" in self.last_test:
            self.execs.append(command)
            return ExecResult(0, "1 passed in 0.1s")
        return super().exec(command, timeout=timeout, activate=activate, env=env)


def _fake_gold() -> None:
    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        if gold_patch is None:
            return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
        return ContainerResult("pass", output="1 passed")
    pipeline.run_test_in_container = fake_run_test
    pipeline.ContainerSession = lambda instance, keep=False: GateSession(instance.repo)
    scouts.ContainerSession = lambda instance, keep=False: GateSession(instance.repo)


def _no_leak(*objs) -> None:
    blob = json.dumps(objs, default=str)
    assert _LEAK not in blob, "gold/test_patch leaked!"


# ── tool surfaces ─────────────────────────────────────────────────────────────

def test_registries_gold_free() -> None:
    sess = PipelineFakeSession()
    tpl = {t.name for t in scout_agent_tools(sess, "template")}
    loc = {t.name for t in scout_agent_tools(sess, "localizer")}
    rep = {t.name for t in scout_agent_tools(sess, "reproducer")}
    assert tpl == {"read_file", "grep", "glob", "run_python", "run_test", "submit_template"}, tpl
    assert loc == {"read_file", "grep", "glob", "run_python", "submit_locations"}, loc
    assert rep == {"read_file", "grep", "glob", "run_python", "submit_reproduction"}, rep
    assert {t.name for t in organizer_agent_tools()} == {"submit_brief"}
    assert {t.name for t in match_agent_tools()} == {"submit_match"}
    for guard, bad in ((assert_scout_gold_free, EditFileTool(sess)),
                       (assert_scout_gold_free, RunBrtTool(sess, "x")),
                       (assert_organizer_gold_free, EditFileTool(sess)),
                       (assert_match_gold_free, RunBrtTool(sess, "x"))):
        try:
            guard([bad])
            raise AssertionError(f"{guard.__name__} must reject {bad.name}")
        except ValueError:
            pass


def test_submit_match_and_failopen() -> None:
    t = SubmitMatchTool()
    assert not t.execute({"verdict": "match"}).is_error
    assert t.execute({"verdict": "mismatch"}).is_error  # needs feedback
    assert t.execute({"verdict": "maybe"}).is_error
    ok = t.execute({"verdict": "mismatch", "feedback": "assert 5"})
    assert json.loads(ok.content) == {"verdict": "mismatch", "feedback": "assert 5"}
    assert _parse_match(AgentResult("", None, "natural", [], [], 0, 0, 0)) == ("match", "")


# ── gate 2 inside run_brt_phase ───────────────────────────────────────────────

def test_match_retry_then_match() -> None:
    _fake_gold()
    writer = RoleClient("writer", [_TEST_A, _TEST_B])
    judge = RoleClient("judge", [("mismatch", "The issue says 5, not 6."), ("match", "")])
    phase = pipeline.run_brt_phase(_INST, writer, max_iterations=4, max_total_tokens=300_000,
                                   match_client=judge, match_retries=3)
    assert phase.submission == _TEST_B, phase.submission
    assert [c["verdict"] for c in phase.match_checks] == ["mismatch", "match"]
    assert "The issue says 5, not 6." in writer.tasks[1], "retry must carry the judge's feedback"
    assert "NOT for the reason the issue describes" in writer.tasks[1]
    assert len(phase.prior_attempts) == 1 and phase.prior_attempts[0][1] == _TEST_A
    assert phase.level == 4
    rec = pipeline.brt_run_record(_INST, writer.name, split="lite", flags={}, phase=phase)
    assert [c["verdict"] for c in rec["attempts"][0]["match_checks"]] == ["mismatch", "match"]
    # the judge saw the issue, the test and the buggy output, never gold
    assert "returns 4 but should return 5" in judge.tasks[0] and "AssertionError" in judge.tasks[0]
    _no_leak(rec, writer.tasks, judge.tasks)


def test_match_budget() -> None:
    _fake_gold()
    writer = RoleClient("writer", [_TEST_A, _TEST_B, _TEST_A])
    judge = RoleClient("judge", [("mismatch", "still wrong")])
    phase = pipeline.run_brt_phase(_INST, writer, max_iterations=4, max_total_tokens=300_000,
                                   match_client=judge, match_retries=1)
    assert writer.calls == 2, writer.calls          # original + 1 match retry
    assert len(phase.match_checks) == 2             # every gate-passing test judged
    assert phase.submission == _TEST_B              # keeps the last (most refined) one


def test_fallback_to_last_gate_pass() -> None:
    _fake_gold()
    writer = RoleClient("writer", [_TEST_A, _TEST_PASSES])
    judge = RoleClient("judge", [("mismatch", "assert 5")])
    phase = pipeline.run_brt_phase(_INST, writer, max_iterations=4, max_total_tokens=300_000,
                                   brt_retries=0, match_client=judge, match_retries=2)
    assert phase.submission == _TEST_A, "must fall back to the last gate-passing test"
    assert phase.reproduces_on_buggy
    assert [p[1] for p in phase.prior_attempts] == [_TEST_PASSES], phase.prior_attempts


def test_baseline_record_unchanged() -> None:
    _fake_gold()
    writer = RoleClient("writer", [_TEST_B])
    phase = pipeline.run_brt_phase(_INST, writer, max_iterations=4, max_total_tokens=300_000)
    rec = pipeline.brt_run_record(_INST, writer.name, split="lite", flags={}, phase=phase)
    assert "match_checks" not in rec["attempts"][0] and "match_cost_usd" not in rec["attempts"][0]
    assert "brt_retries" not in rec
    assert pipeline.BRIEF_HEADER.strip()[:20] not in writer.tasks[0]
    assert phase.match_checks == [] and phase.match_cost_usd is None and phase.match_loop is None
    assert "match_loop" not in rec["attempts"][0]


# ── the full scouts flow ──────────────────────────────────────────────────────

def _clients(**override) -> dict:
    base = {
        "template": RoleClient("template", ["TEMPLATE: def test_t(): assert compute() is not None"]),
        "localizer": RoleClient("localizer", ["LOCATIONS: demo/pkg/core.py:3 compute()"]),
        "reproducer": RoleClient("reproducer", ["REPRO: compute() -> 4; issue expects 5"]),
        "organizer": RoleClient("organizer", ["BRIEF: call compute(); assert == 5 (issue)"]),
        "writer": RoleClient("writer", [_TEST_B]),
        "judge": RoleClient("judge", [("match", "")]),
    }
    base.update(override)
    return base


def test_scouts_flow() -> None:
    _fake_gold()
    cl = _clients()
    sres, phase = scouts.run_scouts_brt_phase(_INST, cl, max_iterations=4,
                                              max_total_tokens=300_000, max_tokens_per_turn=4096)
    for role in scouts.ROLES:
        assert cl[role].calls >= 1, f"{role} never ran"
    assert sres.brief == "BRIEF: call compute(); assert == 5 (issue)"
    w = cl["writer"].tasks[0]
    assert pipeline.BRIEF_HEADER.strip()[:20] in w and sres.brief in w
    org = cl["organizer"].tasks[0]
    for text in ("TEMPLATE:", "LOCATIONS:", "REPRO:"):
        assert text in org, f"organizer did not get {text}"
    assert {r.model for r in sres.scouts.values()} == {"fake:template", "fake:localizer",
                                                        "fake:reproducer"}
    threads = cl["template"].threads | cl["localizer"].threads | cl["reproducer"].threads
    assert "MainThread" not in threads, "scouts must run in worker threads"
    assert phase.submission == _TEST_B and phase.level == 4
    _no_leak([r.record() for r in sres.scouts.values()], sres.organizer.record(),
             sres.brief, *(c.tasks for c in cl.values()))


def test_scout_crash_and_no_brief() -> None:
    _fake_gold()
    cl = _clients(localizer=RoleClient("localizer", [RuntimeError("provider down")]),
                  organizer=RoleClient("organizer", [None]))
    # organizer returns an empty brief -> submit_brief errors -> nudges -> no submission
    cl["organizer"].script = [""]
    sres, phase = scouts.run_scouts_brt_phase(_INST, cl, max_iterations=4,
                                              max_total_tokens=300_000, max_tokens_per_turn=4096)
    assert sres.scouts["localizer"].report is None and "provider down" in sres.scouts["localizer"].error
    assert sres.scouts["template"].report and sres.scouts["reproducer"].report
    assert "not finish" in cl["organizer"].tasks[0], "organizer must be told a report is missing"
    assert sres.brief and "[template]" in sres.brief and "[reproducer]" in sres.brief
    assert sres.notes, "the raw-report fallback must be noted"
    assert phase.submission == _TEST_B


def test_match_only_arm() -> None:
    _fake_gold()
    cl = _clients()
    sres, phase = scouts.run_scouts_brt_phase(_INST, cl, max_iterations=4,
                                              max_total_tokens=300_000, max_tokens_per_turn=4096,
                                              use_scouts=False)
    assert sres is None
    for role in scouts.SCOUT_ROLES + ("organizer",):
        assert cl[role].calls == 0, f"{role} must not run in the match-only arm"
    assert pipeline.BRIEF_HEADER.strip()[:20] not in cl["writer"].tasks[0]
    assert [c["verdict"] for c in phase.match_checks] == ["match"]


# ── v2: the judge inside the writer's loop ───────────────────────────────────

class ScriptedWriter:
    """Writer that makes the given tool calls in order (name, args)."""

    name = "fake:writer"

    def __init__(self, steps: list) -> None:
        self.steps = steps
        self.calls = 0
        self.system = ""
        self.results: list[str] = []

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.system = system
        if messages and messages[-1].role == "tool":
            self.results.append(str(messages[-1].content))
        name, args = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        assert name in {t.name for t in tools}, (name, sorted(t.name for t in tools))
        return _resp(name, args, f"s{self.calls}", 5000, 100)


def test_writer_v2_registry() -> None:
    sess = PipelineFakeSession()
    st = MatchLoopState(lambda c, o: ("match", ""))
    names = {t.name for t in writer_v2_agent_tools(sess, st)}
    assert names == {"read_file", "grep", "glob", "run_python", "check_test", "submit_test"}, names
    for bad in (RunTestTool(sess), RunBrtTool(sess, "x"), EditFileTool(sess)):
        try:
            assert_writer_v2_gold_free([bad])
            raise AssertionError(f"writer v2 guard must reject {bad.name}")
        except ValueError:
            pass


def test_check_test_and_guarded_submit() -> None:
    sess = GateSession()
    verdicts = iter([("mismatch", "the issue says 5")])
    st = MatchLoopState(lambda code, out: next(verdicts, ("match", "")), max_mismatches=3)
    check, submit = CheckTestTool(sess, st), GuardedSubmitTestTool(sess, st)
    r = check.execute({"code": _TEST_A})
    assert "verdict=MISMATCH (1/3)" in r.content and "the issue says 5" in r.content, r.content
    refused = submit.execute({"code": _TEST_A})
    assert refused.is_error and "MISMATCH" in refused.content
    r = check.execute({"code": _TEST_B})
    assert "verdict=match" in r.content
    ok = submit.execute({"code": _TEST_B})
    assert not ok.is_error and ok.content == _TEST_B and not st.submitted_unverified
    # a test that passes on buggy code is never shown to the judge
    seen = []
    st2 = MatchLoopState(lambda c, o: seen.append(c) or ("match", ""))
    r = CheckTestTool(sess, st2).execute({"code": _TEST_PASSES})
    assert "not consulted" in r.content and not seen
    # refusals are bounded so forced closing turns can still submit (flagged)
    st3 = MatchLoopState(lambda c, o: ("match", ""), max_refusals=2)
    sub3 = GuardedSubmitTestTool(sess, st3)
    assert sub3.execute({"code": _TEST_A}).is_error and sub3.execute({"code": _TEST_A}).is_error
    assert not sub3.execute({"code": _TEST_A}).is_error and st3.submitted_unverified


def test_writer_loop_in_run_brt_phase() -> None:
    _fake_gold()
    writer = ScriptedWriter([("check_test", {"code": _TEST_A}), ("check_test", {"code": _TEST_B}),
                             ("submit_test", {"code": _TEST_B})])
    judge = RoleClient("judge", [("mismatch", "The issue says 5, not 6."), ("match", "")])
    phase = pipeline.run_brt_phase(_INST, writer, max_iterations=6, max_total_tokens=300_000,
                                   writer_judge_client=judge, judge_context="BRIEF: expect 5")
    assert phase.submission == _TEST_B and phase.level == 4
    assert [c["verdict"] for c in phase.match_checks] == ["mismatch", "match"]
    assert phase.match_loop == {"attempts": 1, "checks": 2, "mismatches": 1, "refusals": 0,
                                "final_matched": True, "submitted_unverified": False}, phase.match_loop
    assert "check_test" in writer.system and "run_test" not in writer.system
    assert "Investigators' brief" in judge.tasks[0] and "BRIEF: expect 5" in judge.tasks[0]
    assert "The issue says 5, not 6." in writer.results[0], "the verdict must reach the writer"
    rec = pipeline.brt_run_record(_INST, writer.name, split="lite", flags={}, phase=phase)
    assert rec["attempts"][0]["match_loop"]["final_matched"] is True
    _no_leak(rec, judge.tasks, writer.results)
    try:
        pipeline.run_brt_phase(_INST, writer, max_iterations=2, max_total_tokens=1000,
                               writer_judge_client=judge, match_client=judge)
        raise AssertionError("in-loop and outer judge together must be refused")
    except ValueError:
        pass


# ── blackboard ──────────────────────────────────────────────────────────────

def test_board_basics() -> None:
    import threading
    import time as _t
    from tdbf.agent.blackboard import (Board, PostFindingTool, ReadBoardTool, WaitForBoardTool,
                                       AnswerTool, NoticeTool, board_tools)
    b = Board()
    post = PostFindingTool(b, "localizer")
    assert post.execute({"section": "locations", "text": "x"}).is_error        # evidence required
    assert post.execute({"section": "questions", "text": "x", "evidence": "y"}).is_error
    assert not post.execute({"section": "locations", "text": "core.py:3", "evidence": "read"}).is_error
    assert b.unseen("writer") == 1 and b.unseen("localizer") == 0              # own posts not unread
    r = ReadBoardTool(b, "writer").execute({})
    assert "core.py:3" in r.content and b.unseen("writer") == 0
    # notices: appended to non-terminal results; scouts are told when the writer is done
    inner = ReadBoardTool(b, "reproducer")
    b.post("writer", "questions", "what does compute() return?")
    n = NoticeTool(inner, b, "reproducer", scout=True).execute({})
    assert "Open questions: #2" in n.content, n.content
    a = AnswerTool(b, "reproducer")
    assert a.execute({"question_id": 1, "text": "x", "evidence": "y"}).is_error   # #1 is not a question
    assert not a.execute({"question_id": 2, "text": "4", "evidence": "run"}).is_error
    assert b.summary()["answered"] == 1
    # wait_for_board wakes on a new post, and returns at once when the board closes
    w = WaitForBoardTool(b, "template")
    b.read("template")
    threading.Timer(0.2, lambda: b.post("writer", "test_status", "fails", "run")).start()
    t0 = _t.monotonic()
    assert "1 new entry" in w.execute({"seconds": 5}).content and _t.monotonic() - t0 < 2
    b.close()
    assert "FINISHED" in NoticeTool(inner, b, "reproducer", scout=True).execute({}).content
    names = {t.name for t in board_tools(b, "writer", scout=False)}
    assert names == {"read_board", "post_finding", "ask_board"}, names


class BoardScout:
    """Fake scout: posts one finding, then finishes."""

    def __init__(self, role: str, section: str, text: str) -> None:
        self.name = f"fake:{role}"
        self.calls = 0
        self.section, self.text = section, text
        self.tool_names: set[str] = set()

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.calls += 1
        self.tool_names = {t.name for t in tools}
        if self.calls == 1:
            return _resp("post_finding", {"section": self.section, "text": self.text,
                                          "evidence": "observed"}, f"p{self.calls}", 1000, 30)
        return _resp("finish", {"summary": "posted"}, f"f{self.calls}", 1000, 20)


class SlowWriter(ScriptedWriter):
    def run_tools(self, **kw):
        import time as _t
        if self.calls == 0:
            _t.sleep(0.5)  # let the scouts post first (deterministic test)
        return super().run_tools(**kw)


def test_blackboard_flow() -> None:
    from tdbf.agent import blackboard
    _fake_gold()
    blackboard.ContainerSession = lambda instance, keep=False: GateSession(instance.repo)
    scouts_ = {"template": BoardScout("template", "test_conventions", "plain pytest"),
               "localizer": BoardScout("localizer", "entry_point", "compute()"),
               "reproducer": BoardScout("reproducer", "observed_behaviour", "compute() -> 4")}
    writer = SlowWriter([("read_board", {}), ("check_test", {"code": _TEST_B}),
                         ("submit_test", {"code": _TEST_B})])
    judge = RoleClient("judge", [("match", "")])
    clients = {**scouts_, "writer": writer, "judge": judge}
    bb, phase = blackboard.run_blackboard_brt_phase(
        _INST, clients, max_iterations=6, max_total_tokens=300_000, max_tokens_per_turn=4096,
        scout_join_timeout=10)
    assert phase.submission == _TEST_B and phase.level == 4
    s = bb.board.summary()
    assert s["entries"] == 3 and set(s["by_author"]) == {"template", "localizer", "reproducer"}, s
    assert "compute() -> 4" in writer.results[0], "the writer must see the scouts' findings"
    assert "compute() -> 4" in judge.tasks[0], "the judge's context is the live board"
    assert "Shared board" in writer.system and "check_test" in writer.system
    for sc in scouts_.values():
        assert {"post_finding", "read_board", "answer", "wait_for_board", "finish"} <= sc.tool_names
        assert not ({"edit_file", "run_brt", "submit_test", "check_test"} & sc.tool_names)
    assert all(r.agent and r.agent.stop_cause == "submit" for r in bb.scouts.values())
    assert bb.board.closed
    _no_leak(bb.board.log(), writer.results, judge.tasks)


def main() -> int:
    tests = [test_registries_gold_free, test_submit_match_and_failopen,
             test_match_retry_then_match, test_match_budget, test_fallback_to_last_gate_pass,
             test_baseline_record_unchanged, test_scouts_flow, test_scout_crash_and_no_brief,
             test_match_only_arm, test_writer_v2_registry, test_check_test_and_guarded_submit,
             test_writer_loop_in_run_brt_phase, test_board_basics, test_blackboard_flow]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nScouts architecture: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
