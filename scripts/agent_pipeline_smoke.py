"""Offline glue test for scripts/06_agent_pipeline.py (end-to-end BRT -> fix).

No Docker, no LLM, no budget. One scripted fake client drives both phases against
a fake session, and both gold barriers are faked (BRT -> L4, fix -> resolved).
Confirms the chaining contract:
  - Phase A produces an L4 BRT; its in-memory test_code is handed to Phase B (no
    results.jsonl round-trip);
  - Phase B resolves; two records are written (mode `agent`, then `fix_agent`)
    with the same shapes 04/05 emit;
  - the gate logic (L4-only) holds;
  - neither phase leaks gold_patch / test_patch into the agent run or transcript.

    python3 scripts/agent_pipeline_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_tools_smoke import FakeSession  # Docker-free session fake

from tdbf.agent import pipeline  # both phases live here (patched below)
from tdbf.agent.session import ExecResult
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import ContainerResult, HarnessResult
from tdbf.models.messages import ToolCall, ToolUseResponse

_BRT = "def test_brt():\n    assert compute() == 5\n"
_BUGGY_SRC = "def compute():\n    return 4\n"
_FIXED_SRC = "def compute():\n    return 5\n"
_FAKE_DIFF = (
    "diff --git a/demo/pkg/core.py b/demo/pkg/core.py\n"
    "--- a/demo/pkg/core.py\n+++ b/demo/pkg/core.py\n"
    "@@ -1,2 +1,2 @@\n def compute():\n-    return 4\n+    return 5\n"
)


class PipelineFakeSession(FakeSession):
    """Serves both phases. pytest fails (assertion) until a source edit is made,
    then passes — so BRT run_test sees a clean fail and fix run_brt sees a pass."""

    def __init__(self, repo: str = "demo/pkg") -> None:
        super().__init__(repo)
        self.edited = False
        self.reverted = False

    def start(self):
        return self

    def close(self):
        pass

    def read_file(self, path, *, max_bytes=0) -> ExecResult:
        if path.endswith("core.py"):
            return ExecResult(0, _FIXED_SRC if self.edited else _BUGGY_SRC)
        return ExecResult(1, f"cat: {path}: No such file or directory")

    def write_file(self, path, content) -> None:
        super().write_file(path, content)
        if path.endswith("core.py"):  # edit_file touched the source
            self.edited = True

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        self.execs.append(command)
        if "git diff" in command:
            return ExecResult(0, _FAKE_DIFF if self.edited else "")
        if "git checkout" in command:
            return ExecResult(0, "")
        if "pytest" in command:
            if self.edited:
                return ExecResult(0, "1 passed in 0.1s")
            return ExecResult(1, "E   AssertionError: assert 4 == 5\n1 failed in 0.1s")
        return super().exec(command, timeout=timeout, activate=activate, env=env)

    def reset_tree(self) -> None:
        self.reverted = True

    def ensure_pytest(self) -> None:
        self._pytest_ready = True


class FakePipelineClient:
    """One client, both phases. Branches on the tools offered:
    BRT phase (submit_test): run_test -> submit_test.
    fix phase (submit_fix):  edit_file -> run_brt -> submit_fix."""

    name = "anthropic:claude-sonnet-4-6"

    def __init__(self) -> None:
        self.brt_turn = 0
        self.fix_turn = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        names = {t.name for t in tools}
        if "submit_test" in names:  # ---- Phase A ----
            self.brt_turn += 1
            if self.brt_turn == 1:
                return ToolUseResponse(
                    text="Run the candidate test on the buggy code.",
                    tool_calls=[ToolCall("t1", "run_test", {"code": _BRT})],
                    stop_reason="tool_use", input_tokens=8000, output_tokens=300,
                )
            return ToolUseResponse(
                text="Clean assertion fail; submitting the BRT.",
                tool_calls=[ToolCall("t2", "submit_test", {"code": _BRT})],
                stop_reason="tool_use", input_tokens=8200, output_tokens=120,
            )
        # ---- Phase B (submit_fix present) ----
        self.fix_turn += 1
        if self.fix_turn == 1:
            return ToolUseResponse(
                text="Editing the root cause.",
                tool_calls=[ToolCall("e1", "edit_file", {"path": "demo/pkg/core.py", "old_str": "return 4", "new_str": "return 5"})],
                stop_reason="tool_use", input_tokens=9000, output_tokens=200,
            )
        if self.fix_turn == 2:
            return ToolUseResponse(
                text="Running the BRT.",
                tool_calls=[ToolCall("r1", "run_brt", {})],
                stop_reason="tool_use", input_tokens=9100, output_tokens=80,
            )
        return ToolUseResponse(
            text="BRT passes; submitting the fix.",
            tool_calls=[ToolCall("s1", "submit_fix", {})],
            stop_reason="tool_use", input_tokens=9200, output_tokens=60,
        )


def _load_script_module():
    spec = importlib.util.spec_from_file_location("agent_pipeline", ROOT / "scripts" / "06_agent_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: 06 defines a dataclass under `from __future__ import
    # annotations`, and the dataclass machinery resolves the module via sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_end_to_end_glue() -> None:
    mod = _load_script_module()
    inst = SWEBenchInstance(
        instance_id="demo__pkg-1",
        repo="demo/pkg",
        base_commit="abc123",
        gold_patch="GOLD MUST NOT BE USED",
        test_patch="TEST_PATCH MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )
    records: list[dict] = []

    # build_client/save_run are used by the script; container + gold scoring live
    # in tdbf.agent.pipeline (patch there).
    mod.build_client = lambda spec, **kw: FakePipelineClient()
    mod.save_run = lambda record: records.append(record)
    pipeline.ContainerSession = lambda instance, keep=False: PipelineFakeSession(instance.repo)

    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        # BRT gold scoring: fail (assertion) on buggy, pass on gold-fixed -> L4.
        if gold_patch is None:
            return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
        return ContainerResult("pass", output="1 passed")
    pipeline.run_test_in_container = fake_run_test
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True,
        fail_to_pass_passed=["test_brt"], fail_to_pass_failed=[],
        pass_to_pass_passed=["test_existing"], pass_to_pass_failed=[], applied=True,
    )

    with tempfile.TemporaryDirectory() as td:
        brt_dir, fix_dir = Path(td) / "brt", Path(td) / "fix"
        outcome = mod.run_one(
            inst, "anthropic:claude-sonnet-4-6",
            split="lite", max_iterations=8, max_total_tokens=300_000,
            think=False, prompt_harden=False, brt_only=False,
            keep_container=False, verbose=False, num_ctx=None,
            brt_dir=brt_dir, fix_dir=fix_dir,
        )

        # --- pipeline outcome ---
        assert outcome is not None
        assert outcome.brt_level == 4 and outcome.is_valid_brt, outcome
        assert outcome.resolved is True, outcome
        assert outcome.cost_usd > 0, outcome.cost_usd

        # --- two records, in order, with the 04/05 shapes ---
        assert len(records) == 2, [r["mode"] for r in records]
        brt_rec, fix_rec = records
        assert brt_rec["mode"] == "agent" and brt_rec["best_level"] == 4
        assert brt_rec["flags"]["pipeline"] is True
        assert brt_rec["attempts"][0]["stop_cause"] == "submit"
        assert "def test_brt" in brt_rec["attempts"][0]["test_code"]
        assert fix_rec["mode"] == "fix_agent" and fix_rec["resolved"] is True
        assert fix_rec["flags"]["pipeline"] is True
        assert fix_rec["attempts"][0]["applied"] is True
        assert "return 5" in fix_rec["attempts"][0]["fix_patch"]

        # --- transcripts written in both dirs, no gold/test_patch leak ---
        brt_t = json.loads((brt_dir / "demo__pkg-1.json").read_text())
        fix_t = json.loads((fix_dir / "demo__pkg-1.json").read_text())
        blob = json.dumps(brt_t) + json.dumps(fix_t)
        assert "MUST NOT BE USED" not in blob, "gold_patch / test_patch leaked into an agent run!"
        assert brt_t["submission"] and fix_t["resolved"] is True


def test_gate_skips_fix_when_brt_not_l4() -> None:
    """If Phase A is not L4, the fix phase must not run (only the BRT record is written)."""
    mod = _load_script_module()
    inst = SWEBenchInstance(
        instance_id="demo__pkg-2", repo="demo/pkg", base_commit="abc123",
        gold_patch="GOLD MUST NOT BE USED", test_patch="TEST_PATCH MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )
    records: list[dict] = []
    started: list[str] = []

    def _session(instance, keep=False):
        started.append(instance.instance_id)
        return PipelineFakeSession(instance.repo)

    mod.build_client = lambda spec, **kw: FakePipelineClient()
    mod.save_run = lambda record: records.append(record)
    pipeline.ContainerSession = _session

    # BRT gold scoring: fail on buggy AND fail on fixed -> L3 (not valid).
    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
    pipeline.run_test_in_container = fake_run_test

    with tempfile.TemporaryDirectory() as td:
        outcome = mod.run_one(
            inst, "anthropic:claude-sonnet-4-6",
            split="lite", max_iterations=8, max_total_tokens=300_000,
            think=False, prompt_harden=False, brt_only=False,
            keep_container=False, verbose=False, num_ctx=None,
            brt_dir=Path(td) / "brt", fix_dir=Path(td) / "fix",
        )
        assert outcome is not None and not outcome.is_valid_brt
        assert outcome.resolved is None, "fix phase should be skipped for a non-L4 BRT"
        assert len(records) == 1 and records[0]["mode"] == "agent"
        assert len(started) == 1, "fix phase started a container despite the gate"


def main() -> int:
    tests = [test_end_to_end_glue, test_gate_skips_fix_when_brt_not_l4]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nEnd-to-end pipeline glue: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
