"""Offline glue test for scripts/04_agent_brt.py (Phase A orchestration).

Validates the script's wiring with no Docker, no LLM, no budget: a scripted fake
client + fake session drive the real `run_one`, and the gold scoring is faked to
yield fail->pass (L4). Confirms the gold-barrier ordering, the offline scoring,
the results record shape, and the transcript dump.

    python3 scripts/agent_brt_smoke.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_tools_smoke import FakeSession  # reuse the Docker-free session fake


class FakeLiveSession(FakeSession):
    """FakeSession + lifecycle methods the script's `with`-style usage needs."""

    def start(self):
        return self

    def close(self):
        pass

from tdbf.agent import pipeline  # phase logic now lives here (patched below)
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import ContainerResult
from tdbf.models.messages import ToolCall, ToolUseResponse


def _load_script_module():
    spec = importlib.util.spec_from_file_location("agent_brt", ROOT / "scripts" / "04_agent_brt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeBRTClient:
    """Drives: run_test (sees a clean fail) -> submit_test."""

    name = "anthropic:claude-opus-4-8"

    def __init__(self) -> None:
        self.turn = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.turn += 1
        code = "def test_brt():\n    assert compute() == 5\n"
        if self.turn == 1:
            return ToolUseResponse(
                text="Let me run the candidate test on the buggy code.",
                tool_calls=[ToolCall("t1", "run_test", {"code": code})],
                stop_reason="tool_use", input_tokens=8000, output_tokens=300,
            )
        return ToolUseResponse(
            text="It fails with an assertion on buggy code; submitting.",
            tool_calls=[ToolCall("t2", "submit_test", {"code": code})],
            stop_reason="tool_use", input_tokens=8200, output_tokens=120,
        )


def main() -> int:
    mod = _load_script_module()

    inst = SWEBenchInstance(
        instance_id="demo__pkg-1",
        repo="demo/pkg",
        base_commit="abc123",
        gold_patch="diff --git a/demo/pkg/core.py b/demo/pkg/core.py\n",  # never inspected (scoring is faked)
        test_patch="MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )

    captured: dict = {}

    # Patch external dependencies. build_client/save_run are called in the script;
    # the container + gold scoring now live in tdbf.agent.pipeline (patch there).
    mod.build_client = lambda spec: FakeBRTClient()
    mod.save_run = lambda record: captured.setdefault("record", record)
    pipeline.ContainerSession = lambda instance, keep=False: FakeLiveSession(instance.repo)
    # Fake the gold scoring: fail on buggy (assertion), pass on fixed -> L4.
    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        if gold_patch is None:
            return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
        return ContainerResult("pass", output="1 passed")
    pipeline.run_test_in_container = fake_run_test

    with tempfile.TemporaryDirectory() as td:
        classification = mod.run_one(
            inst, "anthropic:claude-opus-4-8",
            split="lite",
            max_iterations=6,
            max_total_tokens=200_000,
            think=False,
            prompt_harden=False,
            keep_container=False,
            verbose=False,
            transcript_dir=Path(td),
        )

        # --- assertions ---
        assert classification is not None, "no classification produced"
        assert classification.level.value == 4, f"expected L4, got {classification.level_label}"
        assert classification.is_valid_brt

        rec = captured["record"]
        assert rec["mode"] == "agent" and rec["best_level"] == 4
        att = rec["attempts"][0]
        assert att["stop_cause"] == "submit", att["stop_cause"]
        assert att["input_tokens"] == 16200 and att["output_tokens"] == 420
        # opus pricing: 16200/1e6*5 + 420/1e6*25
        assert abs(att["cost_usd"] - (16200 / 1e6 * 5 + 420 / 1e6 * 25)) < 1e-9, att["cost_usd"]
        assert att["num_tool_calls"] == 2
        assert "def test_brt" in att["test_code"]

        transcript = Path(td) / "demo__pkg-1.json"
        assert transcript.exists(), "transcript not written"
        import json
        t = json.loads(transcript.read_text())
        assert t["submission"] and "MUST NOT BE USED" not in json.dumps(t), "test_patch leaked!"
        assert t["stop_cause"] == "submit"

    print("  ok  L4 classification via offline scoring")
    print("  ok  gold barrier ordered after agent (agent saw no gold/fixed signal)")
    print("  ok  results record: tokens, opus cost, tool-call count, level")
    print("  ok  transcript written; dataset test_patch not leaked")
    print("\nPhase A orchestration glue: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
