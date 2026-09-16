"""Offline smoke for the post-fix adversarial Breaker (Phase C, verification gate).

No Docker, no LLM, no budget. A fake client drives the fix worker + breaker against
a fake session; the gold barrier is faked (fix -> resolved). Asserts:

  - the breaker tool registry is gold-free + complete (submit_break present; no
    edit / test-write / run_brt / fixed-gold tool), and the guard rejects a mutating
    tool sneaking in;
  - SubmitBreakTool validates verdict/feedback ('broken' requires feedback);
    _parse_break fails OPEN to 'robust' (a flaky breaker never destroys a good fix);
  - run_fix_phase with a breaker runs the post-fix loop: fix -> breaker(broken) ->
    fix -> breaker(robust) -> submit; the FINAL (re-fixed) patch is scored, tokens
    aggregate across fix + breaker runs, and the record/transcript carry breaker
    metadata;
  - the breaker phase leaks no gold_patch / test_patch into any agent run or transcript.

    python3 scripts/agent_breaker_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_pipeline_smoke import PipelineFakeSession  # edit/pytest/git-diff fake

from tdbf.agent import pipeline
from tdbf.agent.roles import Break, _parse_break, run_breaker  # noqa: F401 (import = contract)
from tdbf.agent.runtime import AgentResult
from tdbf.agent.tools import (
    EditFileTool,
    SubmitBreakTool,
    _BREAKER_TOOL_NAMES,
    assert_breaker_gold_free,
    breaker_agent_tools,
)
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import HarnessResult
from tdbf.models.messages import ToolCall, ToolUseResponse

_STRONG_TEST = "def test_brt():\n    assert compute() == 5\n"

_INST = SWEBenchInstance(
    instance_id="demo__pkg-1",
    repo="demo/pkg",
    base_commit="abc123",
    gold_patch="GOLD MUST NOT BE USED",
    test_patch="TEST_PATCH MUST NOT BE USED",
    problem_statement="compute() returns 4 but should return 5.",
)


# ── fake client: fix worker + breaker, branch on the offered tools ────────────

class BreakerFakeClient:
    name = "anthropic:claude-sonnet-4-6"

    def __init__(self) -> None:
        self.fix_worker = self.breaker = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        names = {t.name for t in tools}
        if "submit_break" in names:  # ---- Breaker: broken first, then robust ----
            self.breaker += 1
            args = ({"verdict": "broken",
                     "feedback": "The fix passes the BRT but compute() is still wrong on another path."}
                    if self.breaker == 1 else {"verdict": "robust", "feedback": ""})
            return ToolUseResponse(
                text="probe", tool_calls=[ToolCall(f"bk{self.breaker}", "submit_break", args)],
                stop_reason="tool_use", input_tokens=3500, output_tokens=100,
            )
        if "submit_fix" in names:  # ---- fix worker: edit -> submit; re-fix -> submit ----
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


# ── unit checks on the breaker tool surface ──────────────────────────────────

def test_breaker_registry_gold_free() -> None:
    sess = PipelineFakeSession()
    tools = breaker_agent_tools(sess)
    names = {t.name for t in tools}
    assert names <= set(_BREAKER_TOOL_NAMES), names
    assert "submit_break" in names and "run_python" in names
    # the breaker cannot edit source, write/run a BRT, or run the held-out tests
    for banned in ("edit_file", "run_brt", "submit_fix", "submit_test", "run_test", "run_tests"):
        assert banned not in names, banned
    # the guard rejects a mutating tool sneaking in
    try:
        assert_breaker_gold_free([EditFileTool(sess)])
        raise AssertionError("assert_breaker_gold_free must reject edit_file")
    except ValueError:
        pass


def test_submit_break_validation_and_failopen() -> None:
    t = SubmitBreakTool()
    assert not t.execute({"verdict": "robust"}).is_error
    assert t.execute({"verdict": "broken"}).is_error          # broken needs feedback
    assert t.execute({"verdict": "maybe"}).is_error           # bad verdict
    ok = t.execute({"verdict": "broken", "feedback": "case X still fails"})
    assert json.loads(ok.content) == {"verdict": "broken", "feedback": "case X still fails"}
    # fail-open: a breaker that never submitted a verdict defaults to robust
    empty = AgentResult("", None, "natural", [], [], 0, 0, 0)
    assert _parse_break(empty) == ("robust", "")


# ── orchestration: fix phase with a post-fix Breaker ─────────────────────────

def test_fix_phase_with_breaker() -> None:
    pipeline.ContainerSession = lambda instance, keep=False: PipelineFakeSession(instance.repo)
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline._verify_scored = lambda *a, **k: None  # stubbed harness: no report on disk
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True,
        fail_to_pass_passed=["test_brt"], fail_to_pass_failed=[],
        pass_to_pass_passed=["test_existing"], pass_to_pass_failed=[], applied=True,
    )
    client = BreakerFakeClient()

    with tempfile.TemporaryDirectory() as td:
        phase = pipeline.run_fix_phase(
            _INST, _STRONG_TEST, client, split="lite",
            max_iterations=6, max_total_tokens=300_000, run_dir=Path(td),
            breaker_client=client, breaker_rounds=2, breaker_max_iterations=4,
        )
    assert phase is not None
    assert [b.verdict for b in phase.breaks] == ["broken", "robust"], phase.breaks
    assert phase.resolved is True
    assert "return 5" in phase.fix_patch
    # tokens aggregate fix(edit+submit) + breaker1 + fix(submit) + breaker2
    assert phase.total_input_tokens == (9000 + 9100) + 3500 + 9100 + 3500, phase.total_input_tokens

    rec = pipeline.fix_run_record(_INST, client.name, split="lite", flags={"breaker": True}, phase=phase)
    att = rec["attempts"][0]
    assert att["breaker_rounds"] == 2 and att["breaker_verdicts"] == ["broken", "robust"]
    tr = pipeline.fix_transcript_dict(_INST, client.name, phase=phase)
    assert tr["breaks"][0]["verdict"] == "broken"
    assert "MUST NOT BE USED" not in json.dumps(rec) + json.dumps(tr), "gold/test_patch leaked!"


def main() -> int:
    tests = [
        test_breaker_registry_gold_free,
        test_submit_break_validation_and_failopen,
        test_fix_phase_with_breaker,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAdversarial breaker glue: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
