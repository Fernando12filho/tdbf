"""Offline glue test for scripts/07_deployable_pipeline.py (gold-free pipeline).

No Docker, no LLM, no budget. Reuses the fakes from agent_pipeline_smoke and
asserts the HONEST contract that distinguishes 07 from 06:
  - the BRT phase scores the BUGGY tree ONLY — it verifies the submitted test in
    the live session and never reaches the patch-based container at all, so no
    gold_patch can be applied (no L4 testing, gold never consulted);
  - the gold-free gate ACCEPTS a reproducing test (fail-on-buggy = candidate,
    which may secretly be L3) and proceeds to the fix phase;
  - the gold-free gate REJECTS a non-reproducing test (pass-on-buggy = L1) and
    skips the fix phase (no fix container is started);
  - neither phase leaks gold_patch / test_patch into an agent run or transcript.

    python3 scripts/agent_deployable_smoke.py
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

from agent_pipeline_smoke import FakePipelineClient, PipelineFakeSession

from tdbf.agent import pipeline
from tdbf.agent.session import ExecResult
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import ContainerResult, HarnessResult


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "deployable_pipeline", ROOT / "scripts" / "07_deployable_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class GatePytestSession(PipelineFakeSession):
    """A fake session whose pytest run returns a scripted (returncode, output)
    while the tree is unedited, so a gate test can drive the BUGGY-side outcome.

    The deployable gate verifies the submitted BRT *in the session* rather than
    in a fresh container, so the buggy-side outcome is produced here — stubbing
    run_test_in_container no longer reaches it. Once a source edit lands the
    scripted result stops applying and the normal fail->pass behaviour resumes,
    which is what the fix phase needs.
    """

    def __init__(self, repo: str, returncode: int, output: str) -> None:
        super().__init__(repo)
        self._rc, self._out = returncode, output

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        if "pytest" in command and not self.edited:
            self.execs.append(command)
            return ExecResult(self._rc, self._out)
        return super().exec(command, timeout=timeout, activate=activate, env=env)


def _inst(iid: str) -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id=iid, repo="demo/pkg", base_commit="abc123",
        gold_patch="GOLD MUST NOT BE USED", test_patch="TEST_PATCH MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )


def _patch_fix_scoring(resolved: bool = True) -> None:
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline._verify_scored = lambda *a, **k: None  # stubbed harness: no report on disk
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=resolved,
        fail_to_pass_passed=["test_brt"] if resolved else [],
        fail_to_pass_failed=[] if resolved else ["test_brt"],
        pass_to_pass_passed=["test_existing"], pass_to_pass_failed=[], applied=True,
    )


def test_gate_accepts_reproducing_and_runs_fix() -> None:
    """A test that fails on buggy code (candidate; may secretly be L3) clears the
    gold-free gate and drives the fix phase. The BRT phase consults NO gold."""
    mod = _load_script_module()
    inst = _inst("demo__pkg-1")
    records: list[dict] = []
    gold_args: list = []

    mod.build_client = lambda spec, **kw: FakePipelineClient()
    mod.save_run = lambda record: records.append(record)
    pipeline.ContainerSession = lambda instance, keep=False: PipelineFakeSession(instance.repo)

    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        gold_args.append(gold_patch)  # capture whether gold was ever applied
        return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
    pipeline.run_test_in_container = fake_run_test
    # The gate verifies in-session (PipelineFakeSession fails with an assertion
    # while unedited), so the deployable path should not reach run_test_in_container
    # at all — which is a stronger form of "no gold was consulted".
    _patch_fix_scoring(resolved=True)

    with tempfile.TemporaryDirectory() as td:
        brt_dir, fix_dir = Path(td) / "brt", Path(td) / "fix"
        outcome = mod.run_one(
            inst, "anthropic:claude-sonnet-4-6",
            split="lite", max_iterations=8, max_total_tokens=300_000,
            think=False, prompt_harden=False, brt_only=False,
            keep_container=False, verbose=False, num_ctx=None,
            brt_dir=brt_dir, fix_dir=fix_dir,
        )

        # --- the gold-free invariant: BRT phase scored buggy-only, no gold ever ---
        assert gold_args == [], (
            "the deployable BRT phase must verify in-session and never reach the "
            f"patch-based container; run_test_in_container gold_patch args = {gold_args}")

        # --- gate accepted, fix ran and resolved ---
        assert outcome is not None
        assert outcome.reproduces_on_buggy is True and outcome.gate_accepted is True, outcome
        assert outcome.resolved is True, outcome
        assert outcome.cost_usd > 0, outcome.cost_usd

        # --- two records: gold-free BRT (no L4 level) then fix ---
        assert len(records) == 2, [r["mode"] for r in records]
        brt_rec, fix_rec = records
        assert brt_rec["mode"] == "agent_deployable"
        assert brt_rec["best_level"] is None, "deployable pipeline must NOT assign an L4 level"
        assert brt_rec["reproduces_on_buggy"] is True and brt_rec["gate_accepted"] is True
        assert brt_rec["outcome_buggy"] == "fail"
        assert brt_rec["flags"]["pipeline"] == "deployable"
        assert "def test_brt" in brt_rec["attempts"][0]["test_code"]
        assert fix_rec["mode"] == "fix_agent" and fix_rec["resolved"] is True
        assert fix_rec["flags"]["pipeline"] == "deployable"

        # --- no gold/test_patch leak into transcripts ---
        brt_t = json.loads((brt_dir / "demo__pkg-1.json").read_text())
        fix_t = json.loads((fix_dir / "demo__pkg-1.json").read_text())
        blob = json.dumps(brt_t) + json.dumps(fix_t)
        assert "MUST NOT BE USED" not in blob, "gold_patch / test_patch leaked into an agent run!"


def test_gate_rejects_nonreproducing() -> None:
    """A test that PASSES on buggy code (L1, non-reproducing) is rejected by the
    gold-free gate; the fix phase must not run and no fix container is started."""
    mod = _load_script_module()
    inst = _inst("demo__pkg-2")
    records: list[dict] = []
    started: list[str] = []

    # BRT passes on buggy => does not reproduce => L1 => gate must reject.
    def _session(instance, keep=False):
        started.append(instance.instance_id)
        return GatePytestSession(instance.repo, 0, "1 passed in 0.1s")

    mod.build_client = lambda spec, **kw: FakePipelineClient()
    mod.save_run = lambda record: records.append(record)
    pipeline.ContainerSession = _session

    with tempfile.TemporaryDirectory() as td:
        outcome = mod.run_one(
            inst, "anthropic:claude-sonnet-4-6",
            split="lite", max_iterations=8, max_total_tokens=300_000,
            think=False, prompt_harden=False, brt_only=False,
            keep_container=False, verbose=False, num_ctx=None,
            brt_dir=Path(td) / "brt", fix_dir=Path(td) / "fix",
        )
        assert outcome is not None
        assert outcome.reproduces_on_buggy is False and outcome.gate_accepted is False, outcome
        assert outcome.resolved is None, "fix phase should be skipped for a non-reproducing BRT"
        assert len(records) == 1 and records[0]["mode"] == "agent_deployable"
        assert records[0]["gate_accepted"] is False and records[0]["best_level"] is None
        assert len(started) == 1, "fix phase started a container despite the gate rejecting"


def test_gate_rejects_crashing_test() -> None:
    """A test that FAILS on buggy code but crashed before reaching its assertion
    (failure_reason='runtime') is NOT a reproduction: it would fail identically on
    fixed code. pytest exit code 1 covers both cases, so the gate must consult
    failure_reason. Regression test — the gate previously accepted this."""
    mod = _load_script_module()
    inst = _inst("demo__pkg-3")
    records: list[dict] = []
    started: list[str] = []

    # Fails on buggy (exit 1) but via AttributeError, never reaching an assert.
    def _session(instance, keep=False):
        started.append(instance.instance_id)
        return GatePytestSession(
            instance.repo, 1,
            "E   AttributeError: 'Thing' object has no attribute 'nope'\n1 failed in 0.1s")

    mod.build_client = lambda spec, **kw: FakePipelineClient()
    mod.save_run = lambda record: records.append(record)
    pipeline.ContainerSession = _session

    with tempfile.TemporaryDirectory() as td:
        outcome = mod.run_one(
            inst, "anthropic:claude-sonnet-4-6",
            split="lite", max_iterations=8, max_total_tokens=300_000,
            think=False, prompt_harden=False, brt_only=False,
            keep_container=False, verbose=False, num_ctx=None,
            brt_dir=Path(td) / "brt", fix_dir=Path(td) / "fix",
        )
        assert outcome is not None
        assert outcome.reproduces_on_buggy is False, \
            "a crashing test must NOT count as reproducing the bug"
        assert outcome.gate_accepted is False, outcome
        assert outcome.resolved is None, "fix phase must be skipped for a crashing BRT"
        assert records[0]["failure_reason_buggy"] == "runtime", records[0]
        assert len(started) == 1, "fix phase started a container despite the gate rejecting"


def main() -> int:
    tests = [test_gate_accepts_reproducing_and_runs_fix, test_gate_rejects_nonreproducing,
             test_gate_rejects_crashing_test]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nDeployable pipeline glue: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
