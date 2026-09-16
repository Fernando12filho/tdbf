"""Offline smoke for the Phase C multi-agent orchestration (Critic + Reviewer).

No Docker, no LLM, no budget. Fake clients drive the worker + role agents against
a fake session and both gold barriers are faked. Asserts:

  - the review-role tool registries are gold-free + complete (submit_review present,
    no test-writing / source-editing / fixed-tree tool), and the guard rejects a
    mutating tool;
  - SubmitReviewTool validates verdict/feedback; _parse_verdict fails OPEN (a role
    with no clean verdict never blocks the worker);
  - run_brt_phase with a Critic runs the revise loop: worker -> critic(revise) ->
    worker -> critic(accept); the FINAL (revised) test is what gets scored, tokens
    aggregate across worker + critic runs, and the record/transcript carry the
    critic metadata;
  - run_fix_phase with a Reviewer runs the same revise loop and resolves;
  - neither phase leaks gold_patch / test_patch into any agent run or transcript.

    python3 scripts/agent_multiagent_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_pipeline_smoke import PipelineFakeSession  # edit/pytest/git-diff fake

from tdbf.agent import pipeline
from tdbf.agent.roles import Review, _parse_verdict, run_critic
from tdbf.agent.runtime import AgentResult
from tdbf.agent.tools import (
    EditFileTool,
    SubmitReviewTool,
    _ROLE_TOOL_NAMES,
    assert_role_gold_free,
    critic_agent_tools,
    reviewer_agent_tools,
)
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import ContainerResult, HarnessResult
from tdbf.models.messages import ToolCall, ToolUseResponse

_WEAK_TEST = "def test_brt():\n    assert compute() == 4  # asserts the BUGGY value\n"
_STRONG_TEST = "def test_brt():\n    assert compute() == 5  # asserts the CORRECT value\n"

_INST = SWEBenchInstance(
    instance_id="demo__pkg-1",
    repo="demo/pkg",
    base_commit="abc123",
    gold_patch="GOLD MUST NOT BE USED",
    test_patch="TEST_PATCH MUST NOT BE USED",
    problem_statement="compute() returns 4 but should return 5.",
)


# ── fake client: worker + critic + reviewer, branch on the offered tools ──────

class MultiFakeClient:
    name = "anthropic:claude-sonnet-4-6"

    def __init__(self) -> None:
        self.brt_worker = self.critic = self.fix_worker = self.reviewer = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        names = {t.name for t in tools}
        if "submit_test" in names:  # ---- BRT worker: weak first, then revised ----
            self.brt_worker += 1
            code = _WEAK_TEST if self.brt_worker == 1 else _STRONG_TEST
            return ToolUseResponse(
                text="submit test", tool_calls=[ToolCall(f"b{self.brt_worker}", "submit_test", {"code": code})],
                stop_reason="tool_use", input_tokens=8000, output_tokens=200,
            )
        if "submit_review" in names and "run_python" in names:  # ---- Critic ----
            self.critic += 1
            args = ({"verdict": "revise", "feedback": "You asserted the buggy value 4; assert the correct value 5."}
                    if self.critic == 1 else {"verdict": "accept", "feedback": ""})
            return ToolUseResponse(
                text="critique", tool_calls=[ToolCall(f"c{self.critic}", "submit_review", args)],
                stop_reason="tool_use", input_tokens=3000, output_tokens=80,
            )
        if "submit_review" in names and "run_tests" in names:  # ---- Reviewer ----
            self.reviewer += 1
            args = ({"verdict": "revise", "feedback": "Drop the unrelated edit; keep the fix minimal."}
                    if self.reviewer == 1 else {"verdict": "accept", "feedback": ""})
            return ToolUseResponse(
                text="review", tool_calls=[ToolCall(f"rv{self.reviewer}", "submit_review", args)],
                stop_reason="tool_use", input_tokens=3200, output_tokens=90,
            )
        if "submit_fix" in names:  # ---- fix worker: edit -> submit; revise -> submit ----
            self.fix_worker += 1
            if self.fix_worker == 1:
                return ToolUseResponse(
                    text="edit", tool_calls=[ToolCall("e1", "edit_file",
                        {"path": "demo/pkg/core.py", "old_str": "return 4", "new_str": "return 5"})],
                    stop_reason="tool_use", input_tokens=9000, output_tokens=150,
                )
            return ToolUseResponse(
                text="submit fix", tool_calls=[ToolCall(f"f{self.fix_worker}", "submit_fix", {})],
                stop_reason="tool_use", input_tokens=9100, output_tokens=60,
            )
        raise AssertionError(f"unexpected tool set offered: {sorted(names)}")


# ── unit checks on the role tool surface ─────────────────────────────────────

def test_role_registries_gold_free() -> None:
    sess = PipelineFakeSession()
    critic = critic_agent_tools(sess)
    reviewer = reviewer_agent_tools(sess)
    assert {t.name for t in critic} <= set(_ROLE_TOOL_NAMES)
    assert {t.name for t in reviewer} <= set(_ROLE_TOOL_NAMES)
    assert "submit_review" in {t.name for t in critic}
    # roles cannot write tests, edit source, or run the fixed tree
    for banned in ("submit_test", "edit_file", "run_brt", "submit_fix"):
        assert banned not in {t.name for t in critic + reviewer}, banned
    # the guard rejects a mutating tool sneaking in
    try:
        assert_role_gold_free([EditFileTool(sess)])
        raise AssertionError("assert_role_gold_free must reject edit_file")
    except ValueError:
        pass


def test_submit_review_validation_and_failopen() -> None:
    t = SubmitReviewTool()
    assert not t.execute({"verdict": "accept"}).is_error
    assert t.execute({"verdict": "revise"}).is_error          # revise needs feedback
    assert t.execute({"verdict": "maybe"}).is_error           # bad verdict
    ok = t.execute({"verdict": "revise", "feedback": "do X"})
    assert json.loads(ok.content) == {"verdict": "revise", "feedback": "do X"}
    # fail-open: a role that never submitted a verdict defaults to accept
    empty = AgentResult("", None, "natural", [], [], 0, 0, 0)
    assert _parse_verdict(empty) == ("accept", "")


# ── orchestration: BRT phase with a Critic ───────────────────────────────────

def _patch_brt_gold(monkey_l4: bool = True) -> None:
    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        if gold_patch is None:
            return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
        return ContainerResult("pass", output="1 passed")
    pipeline.run_test_in_container = fake_run_test


def test_brt_phase_with_critic() -> None:
    pipeline.ContainerSession = lambda instance, keep=False: PipelineFakeSession(instance.repo)
    _patch_brt_gold()
    client = MultiFakeClient()

    phase = pipeline.run_brt_phase(
        _INST, client, max_iterations=6, max_total_tokens=300_000,
        critic_client=client, critic_rounds=2, critic_max_iterations=4,
    )
    assert phase is not None
    assert [r.verdict for r in phase.reviews] == ["revise", "accept"], phase.reviews
    # the FINAL, revised test is what was submitted + scored
    assert phase.submission == _STRONG_TEST, phase.submission
    assert phase.level == 4
    # tokens aggregate worker1 + critic1 + worker2 + critic2 (not just one run)
    assert phase.total_input_tokens == 8000 + 3000 + 8000 + 3000, phase.total_input_tokens

    # record + transcript carry critic metadata; no gold leak
    rec = pipeline.brt_run_record(_INST, client.name, split="lite", flags={"multi": True}, phase=phase)
    att = rec["attempts"][0]
    assert att["critic_rounds"] == 2 and att["critic_verdicts"] == ["revise", "accept"]
    assert att["input_tokens"] == phase.total_input_tokens
    tr = pipeline.brt_transcript_dict(_INST, client.name, phase=phase)
    assert tr["reviews"] == [
        {"verdict": "revise", "feedback": "You asserted the buggy value 4; assert the correct value 5."},
        {"verdict": "accept", "feedback": ""},
    ]
    assert "MUST NOT BE USED" not in json.dumps(rec) + json.dumps(tr), "gold/test_patch leaked!"


# ── orchestration: fix phase with a Reviewer ─────────────────────────────────

def test_fix_phase_with_reviewer() -> None:
    pipeline.ContainerSession = lambda instance, keep=False: PipelineFakeSession(instance.repo)
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline._verify_scored = lambda *a, **k: None  # stubbed harness: no report on disk
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True,
        fail_to_pass_passed=["test_brt"], fail_to_pass_failed=[],
        pass_to_pass_passed=["test_existing"], pass_to_pass_failed=[], applied=True,
    )
    client = MultiFakeClient()

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        phase = pipeline.run_fix_phase(
            _INST, _STRONG_TEST, client, split="lite",
            max_iterations=6, max_total_tokens=300_000, run_dir=Path(td),
            reviewer_client=client, reviewer_rounds=2, reviewer_max_iterations=4,
        )
    assert phase is not None
    assert [r.verdict for r in phase.reviews] == ["revise", "accept"], phase.reviews
    assert phase.resolved is True
    assert "return 5" in phase.fix_patch
    # tokens aggregate worker(edit+submit) + reviewer1 + worker(submit) + reviewer2
    assert phase.total_input_tokens == (9000 + 9100) + 3200 + 9100 + 3200, phase.total_input_tokens

    rec = pipeline.fix_run_record(_INST, client.name, split="lite", flags={"multi": True}, phase=phase)
    att = rec["attempts"][0]
    assert att["reviewer_rounds"] == 2 and att["reviewer_verdicts"] == ["revise", "accept"]
    tr = pipeline.fix_transcript_dict(_INST, client.name, phase=phase)
    assert tr["reviews"][0]["verdict"] == "revise"
    assert "MUST NOT BE USED" not in json.dumps(rec) + json.dumps(tr), "gold/test_patch leaked!"


def main() -> int:
    tests = [
        test_role_registries_gold_free,
        test_submit_review_validation_and_failopen,
        test_brt_phase_with_critic,
        test_fix_phase_with_reviewer,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nMulti-agent (Phase C) glue: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
