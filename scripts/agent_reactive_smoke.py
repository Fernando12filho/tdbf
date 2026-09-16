"""Offline checks for the reactive baseline (RQ1 Arm A).

RQ1 claims the arms differ ONLY in the reproduction phase and the tool that runs
its output. These checks hold the code to that claim:

  - the reactive toolset is exactly the fix toolset minus run_brt, same order
  - nothing the reactive agent sees names run_brt, a BRT, or a reproduction test
  - the rules the two prompts share are verbatim in both
  - the test-first prompt is byte-identical to the one that produced the stored
    Sonnet reference results (pinned by hash)
  - run_fix_phase(inst, None, ...) runs the fix agent with no BRT anywhere in its
    task, still scores the patch, and never leaks gold_patch / test_patch
  - the BRT-dependent options are refused rather than silently ignored

No Docker, no LLM.

    python3 scripts/agent_reactive_smoke.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_pipeline_smoke import PipelineFakeSession

from tdbf.agent import pipeline
from tdbf.agent.tools import fix_agent_tools, reactive_fix_agent_tools
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import HarnessResult
from tdbf.models.messages import ToolCall, ToolUseResponse

# sha256 of FIX_SYSTEM_PROMPT as it stood when the stored Sonnet runs were made.
# If this fails, the test-first arm changed and is no longer comparable to them.
TEST_FIRST_PROMPT_SHA256 = "84735a24d026dfd3e37cadd78b146c45be98961a132343e9f299d96dfdc0c4f2"

# Rules both prompts must state identically (prompt text is joined across the
# backslash line continuations, so compare on the rendered string).
SHARED_RULES = [
    "You are an autonomous bug-fixing agent. Your single goal is to FIX the bug described in a "
    "GitHub issue by editing the source code of `{repo}` (checked out at the buggy commit `{base_commit}`).",
    "Use run_tests on the SPECIFIC existing test file(s) closest to your change (e.g. the test module "
    "for the file you edited) to check you did not break related behaviour. Do NOT run whole test "
    "directories — they are slow and full of failures unrelated to your change.",
    "Submit once your fix addresses the issue (see the submit rules below).",
    "- Keep the change minimal and targeted; do not reformat unrelated code.",
    "- IMPORTANT: existing tests in this repo may already fail for reasons unrelated to you "
    "(environment, network, pre-existing flakiness). A failure reported by run_tests does NOT by "
    "itself mean your fix is wrong. Only treat it as a regression if you can point to a specific "
    "test that your edit plausibly broke.",
    "- Call submit_fix EXACTLY ONCE.",
    "- Be efficient — you have a limited number of steps and a token budget.",
]
BRT_WORDS = ("run_brt", "BRT", "reproduction test", "bug-reproduction")


def _inst() -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id="demo__pkg-1", repo="demo/pkg", base_commit="abc123",
        gold_patch="GOLD MUST NOT BE USED", test_patch="TEST_PATCH MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )


class ReactiveFakeClient:
    """edit_file -> submit_fix, recording everything the agent was shown."""

    name = "anthropic:claude-sonnet-4-6"

    def __init__(self) -> None:
        self.turn = 0
        self.tools_seen: list[list[str]] = []
        self.systems: list[str] = []
        self.first_user: str | None = None

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.turn += 1
        self.tools_seen.append([t.name for t in tools])
        self.systems.append(system)
        self.descriptions = [t.description for t in tools]
        if self.first_user is None:
            self.first_user = messages[0].content
        if self.turn == 1:
            return ToolUseResponse(
                text="Editing the root cause.",
                tool_calls=[ToolCall("e1", "edit_file", {"path": "demo/pkg/core.py",
                                                         "old_str": "return 4", "new_str": "return 5"})],
                stop_reason="tool_use", input_tokens=9000, output_tokens=200)
        return ToolUseResponse(
            text="Submitting.", tool_calls=[ToolCall("s1", "submit_fix", {})],
            stop_reason="tool_use", input_tokens=9100, output_tokens=40)


def _stub_scoring(calls: list) -> None:
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda pred, split, run_id, ids: calls.append(run_id) or 0
    pipeline._verify_scored = lambda *a, **k: None  # stubbed harness: no report on disk
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True,
        fail_to_pass_passed=["t"], fail_to_pass_failed=[],
        pass_to_pass_passed=[], pass_to_pass_failed=[], applied=True)


def test_toolset_is_fix_minus_run_brt() -> None:
    s = PipelineFakeSession()
    first = [t.name for t in fix_agent_tools(s, "def test_x(): assert 0")]
    reactive = [t.name for t in reactive_fix_agent_tools(s)]
    assert reactive == [n for n in first if n != "run_brt"], (first, reactive)
    assert "run_brt" in first and "run_brt" not in reactive


def test_nothing_reactive_names_a_brt() -> None:
    texts = [t.description for t in reactive_fix_agent_tools(PipelineFakeSession())]
    texts.append(pipeline.REACTIVE_FIX_SYSTEM_PROMPT)
    for text in texts:
        for w in BRT_WORDS:
            assert w not in text, f"reactive arm mentions {w!r}: {text[:90]!r}"


def test_shared_rules_are_verbatim_in_both_prompts() -> None:
    for rule in SHARED_RULES:
        assert rule in pipeline.FIX_SYSTEM_PROMPT, f"test-first prompt lost: {rule[:60]!r}"
        assert rule in pipeline.REACTIVE_FIX_SYSTEM_PROMPT, f"reactive prompt lost: {rule[:60]!r}"


def test_test_first_prompt_is_unchanged() -> None:
    got = hashlib.sha256(pipeline.FIX_SYSTEM_PROMPT.encode()).hexdigest()
    assert got == TEST_FIRST_PROMPT_SHA256, (
        "FIX_SYSTEM_PROMPT changed: the test-first arm is no longer the one that produced "
        "the stored reference results. Update the pin only if that is intended.")


def test_reactive_run_has_no_brt_and_still_scores() -> None:
    started, scored = [], []
    pipeline.ContainerSession = lambda instance, keep=False: (
        started.append(1) or PipelineFakeSession(instance.repo))
    _stub_scoring(scored)
    client = ReactiveFakeClient()
    with tempfile.TemporaryDirectory() as td:
        res = pipeline.run_fix_phase(_inst(), None, client, split="lite",
                                     max_iterations=8, max_total_tokens=300_000,
                                     run_dir=Path(td))
    assert res is not None
    assert all("run_brt" not in names for names in client.tools_seen), client.tools_seen
    assert all(s.startswith("You are an autonomous bug-fixing agent") for s in client.systems)
    assert all(w not in client.first_user for w in BRT_WORDS), client.first_user
    assert client.first_user.rstrip().endswith("should return 5."), client.first_user
    assert "MUST NOT BE USED" not in client.first_user + "".join(client.systems)
    assert res.fix_patch.strip(), "the edit should have produced a patch"
    assert res.resolved is True and len(scored) == 1, (res.resolved, scored)
    assert len(started) == 1, "one container, no Phase A"
    assert res.task == client.first_user


def test_test_first_task_still_carries_the_brt() -> None:
    """The switch must not disturb Arm B: its task keeps the BRT section verbatim."""
    brt = "def test_brt():\n    assert compute() == 5\n"
    pipeline.ContainerSession = lambda instance, keep=False: PipelineFakeSession(instance.repo)
    _stub_scoring([])
    seen = {}

    class _Client(ReactiveFakeClient):
        def run_tools(self, *, system, messages, tools, max_tokens=4096):
            seen.setdefault("task", messages[0].content)
            seen.setdefault("tools", [t.name for t in tools])
            return super().run_tools(system=system, messages=messages, tools=tools)

    with tempfile.TemporaryDirectory() as td:
        pipeline.run_fix_phase(_inst(), brt, _Client(), split="lite",
                               max_iterations=8, max_total_tokens=300_000, run_dir=Path(td))
    assert seen["task"].endswith(
        "should return 5.\n\n--- Bug-reproduction test (currently FAILS; make it PASS by "
        "fixing the source) ---\n" + brt), seen["task"]
    assert "run_brt" in seen["tools"]


def test_brt_dependent_options_refused() -> None:
    for kw in ({"max_attempts": 2}, {"reviewer_client": object()}, {"breaker_client": object()}):
        try:
            pipeline.run_fix_phase(_inst(), None, ReactiveFakeClient(), split="lite",
                                   max_iterations=8, max_total_tokens=300_000, **kw)
        except ValueError as e:
            assert "no BRT" in str(e), e
        else:
            raise AssertionError(f"reactive arm should refuse {kw}")


def main() -> int:
    tests = [
        test_toolset_is_fix_minus_run_brt,
        test_nothing_reactive_names_a_brt,
        test_shared_rules_are_verbatim_in_both_prompts,
        test_test_first_prompt_is_unchanged,
        test_reactive_run_has_no_brt_and_still_scores,
        test_test_first_task_still_carries_the_brt,
        test_brt_dependent_options_refused,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nReactive baseline (Arm A): all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
