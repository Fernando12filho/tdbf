"""Offline checks for the paired sweep driver (scripts/11_paired_sweep.py).

Both arms really run — through run_fix_phase and script 07's run_one — against
the shared fake client and fake container; only the harness is stubbed. Checks:

  - an AB unit writes Arm A's reactive record, Arm B's records and one
    sweep_unit, all sharing one pair_id, with the order recorded
  - the per-instance order is a seeded coin flip: stable across runs, and both
    orders actually occur
  - re-running resumes: completed units are skipped, nothing is re-recorded
  - a unit whose arm fails leaves NO records, and the sweep moves on
  - repeated failures stop the sweep; a quota (429) error stops it at once
  - --arms B runs the test-first arm alone (the capability probe)
  - free OpenRouter models default to 16k/turn and 1M total; others keep 07's
  - the daily call tally is kept and a sweep will not start a unit past it

No Docker, no LLM.

    python3 scripts/agent_sweep_smoke.py
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
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import ContainerResult, HarnessResult


def _load_sweep():
    spec = importlib.util.spec_from_file_location("paired_sweep", ROOT / "scripts" / "11_paired_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SW = _load_sweep()


def _inst(n: int) -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id=f"demo__pkg-{n}", repo="demo/pkg", base_commit="abc123",
        gold_patch="GOLD MUST NOT BE USED", test_patch="TEST_PATCH MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )


def _stub_world(*, client_factory=FakePipelineClient, broken: set[str] = frozenset()) -> None:
    """Fake client + container for both arms; stubbed harness."""
    SW.build_client = lambda spec, **kw: client_factory()
    SW.m07().build_client = lambda spec, **kw: client_factory()

    def session(instance, keep=False):
        if instance.instance_id in broken:
            raise RuntimeError("Docker image not found (simulated)")
        return PipelineFakeSession(instance.repo)
    pipeline.ContainerSession = session
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline._verify_scored = lambda *a, **k: None  # stubbed harness: no report on disk
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True, fail_to_pass_passed=["t"],
        fail_to_pass_failed=[], pass_to_pass_passed=[], pass_to_pass_failed=[], applied=True)
    # offline L0-L4 label: fails on buggy, passes on gold
    pipeline.run_test_in_container = lambda inst, tp, *, gold_patch=None, test_path=None, **k: (
        ContainerResult("pass") if gold_patch else ContainerResult("fail", "assertion", "E assert 4 == 5"))


class _Store:
    """results.jsonl stand-in: save() appends lines the resume check can read."""

    def __init__(self, root: Path) -> None:
        self.path = root / "results.jsonl"

    def save(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


def _cfg(root: Path, **kw):
    base = dict(model="openrouter:demo/model:free", arms="AB", run_group="smoke",
                max_iterations=8, out_dir=root / "sweep")
    base.update(kw)
    return SW.Config(**base)


def _sweep(root, store, cfg, insts):
    return SW.run_sweep(cfg, insts, save=store.save, results_path=store.path,
                        usage_path=root / "usage.json")


def test_ab_unit_records_share_a_pair_id() -> None:
    _stub_world()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        s = _sweep(root, store, _cfg(root), [_inst(1)])
        assert s.status == SW.EXIT_OK and len(s.done) == 1, s
        recs = store.records()
        modes = [(r["mode"], (r.get("flags") or {}).get("arm")) for r in recs]
        assert ("fix_agent", "reactive") in modes, modes
        assert ("agent_deployable", "deployable") in modes, modes
        assert ("fix_agent", "deployable") in modes, modes
        assert modes[-1][0] == "sweep_unit", "the unit summary is written last"
        pair_ids = {r["flags"]["pair_id"] for r in recs}
        assert len(pair_ids) == 1, pair_ids
        unit = recs[-1]
        assert unit["flags"]["order"] in ("AB", "BA")
        assert unit["arm_a"]["resolved"] is True and unit["arm_b"]["resolved"] is True
        assert unit["arm_b"]["gate_accepted"] is True and unit["arm_b"]["brt_level"] == 4
        assert unit["calls"] > 0
        # both arms see identical budgets, recorded everywhere
        for r in recs:
            assert r["flags"]["max_tokens_per_turn"] == 4096, r["flags"]
        # the reactive record never mentions a BRT; gold never leaks anywhere
        blob = json.dumps(recs)
        assert "MUST NOT BE USED" not in blob
        reactive = next(r for r in recs if r["flags"].get("arm") == "reactive")
        transcript = json.loads((root / "sweep" / "smoke" / "A" / "demo__pkg-1.json").read_text())
        assert "run_brt" not in transcript["system"] and "BRT" not in transcript["task"]
        assert reactive["resolved"] is True


def test_order_is_a_seeded_coin_flip() -> None:
    ids = [f"repo__x-{i}" for i in range(40)]
    first = [SW.arm_order(0, i) for i in ids]
    assert first == [SW.arm_order(0, i) for i in ids], "must be stable across runs"
    assert set(first) == {"AB", "BA"}, "both orders must occur"
    assert first != [SW.arm_order(1, i) for i in ids], "the seed must matter"


def test_resume_skips_completed_units() -> None:
    _stub_world()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        cfg = _cfg(root)
        _sweep(root, store, cfg, [_inst(1), _inst(2)])
        n = len(store.records())
        s = _sweep(root, store, cfg, [_inst(1), _inst(2), _inst(3)])
        assert s.skipped == 2 and len(s.done) == 1, s
        assert len(store.records()) == n + 4, "only instance 3 is new (3 records + unit)"
        # a different run group or arm set is a different sweep
        s2 = _sweep(root, store, _cfg(root, run_group="other"), [_inst(1)])
        assert s2.skipped == 0 and len(s2.done) == 1


def test_failed_unit_leaves_no_records() -> None:
    _stub_world(broken={"demo__pkg-2"})
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        s = _sweep(root, store, _cfg(root), [_inst(1), _inst(2), _inst(3)])
        assert s.status == SW.EXIT_OK, s
        assert [iid for iid, _ in s.failed] == ["demo__pkg-2"], s.failed
        assert all(r["instance_id"] != "demo__pkg-2" for r in store.records())
        assert {u["instance_id"] for u in s.done} == {"demo__pkg-1", "demo__pkg-3"}


def test_repeated_failures_stop_the_sweep() -> None:
    _stub_world(broken={f"demo__pkg-{i}" for i in range(1, 6)})
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        s = _sweep(root, store, _cfg(root), [_inst(i) for i in range(1, 6)])
        assert s.status == SW.EXIT_FAILURES and len(s.failed) == 3, s
        assert store.records() == []


class _QuotaError(Exception):
    status_code = 429


class QuotaClient(FakePipelineClient):
    def run_tools(self, **kw):
        raise _QuotaError("Rate limit exceeded: free-models-per-day")


def test_quota_error_stops_immediately() -> None:
    _stub_world(client_factory=QuotaClient)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        s = _sweep(root, store, _cfg(root), [_inst(1), _inst(2)])
        assert s.status == SW.EXIT_QUOTA and "quota" in s.stop_reason, s
        assert s.failed == [] and s.done == [] and store.records() == []


def test_b_only_is_the_capability_probe() -> None:
    _stub_world()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        s = _sweep(root, store, _cfg(root, arms="B"), [_inst(1)])
        assert len(s.done) == 1
        recs = store.records()
        assert all((r.get("flags") or {}).get("arm") != "reactive" for r in recs), recs
        unit = recs[-1]
        assert unit["arm_a"] is None and unit["arm_b"]["resolved"] is True
        assert unit["flags"]["order"] == "B"


def test_default_budgets() -> None:
    assert SW.default_budgets("openrouter:google/gemma-4-31b-it:free") == (16_000, 1_000_000)
    assert SW.default_budgets("openrouter:google/gemma-4-31b-it") == (4_096, 300_000)
    assert SW.default_budgets("anthropic:claude-sonnet-4-6") == (4_096, 300_000)
    assert SW.default_budgets("ollama:qwen2.5-coder:14b@http://h:11434") == (4_096, 300_000)


def test_daily_budget_is_respected() -> None:
    _stub_world()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        usage = SW.Usage(root / "usage.json", daily_budget=100)
        usage.add(95)  # earlier today
        s = _sweep(root, store, _cfg(root, daily_budget=100, min_remaining=10), [_inst(1)])
        assert s.status == SW.EXIT_QUOTA and "daily budget" in s.stop_reason, s
        assert store.records() == []
        # with room left, the unit runs and its calls are added to today's tally
        s2 = _sweep(root, store, _cfg(root, daily_budget=1000, min_remaining=10), [_inst(1)])
        assert len(s2.done) == 1
        assert SW.Usage(root / "usage.json", 1000).used_today() == 95 + s2.done[0]["calls"]


class _UpstreamError(Exception):
    status_code = 429
    body = {"metadata": {"limit_source": "upstream_provider_shared_pool"}}


class UpstreamClient(FakePipelineClient):
    def run_tools(self, **kw):
        raise _UpstreamError("google/gemma-4-31b-it:free is temporarily rate-limited upstream")


def test_upstream_throttle_fails_the_unit_not_the_sweep() -> None:
    """Live 2026-09-16: a shared-pool throttle stopped the whole sweep as if it
    were our daily cap. It must fail only the unit (retried next run)."""
    _stub_world(client_factory=UpstreamClient)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        s = _sweep(root, store, _cfg(root), [_inst(1)])
        assert s.status == SW.EXIT_OK, s
        assert [iid for iid, _ in s.failed] == ["demo__pkg-1"], s.failed
        assert store.records() == []


def test_env_gate_drops_instances_not_marked_valid() -> None:
    """Frozen-sample rule: an instance whose environment is not confirmed valid is
    excluded BEFORE any arm runs. Nothing enforced this, so matplotlib-23913
    (gold_resolved: false) was run and scored as an ordinary unresolved result."""
    with tempfile.TemporaryDirectory() as td:
        rec = Path(td) / "env.json"
        rec.write_text(json.dumps({"instances": {
            "demo__pkg-1": {"state": "valid"},
            "demo__pkg-2": {"state": "pending_recheck"},
            "demo__pkg-3": {"state": "deferred"},
        }}))
        ids = ["demo__pkg-1", "demo__pkg-2", "demo__pkg-3", "demo__pkg-4"]
        keep, dropped = SW.drop_invalid_envs(ids, rec)
        # unknown to the record (pkg-4) is left alone; only explicit non-valid drops
        assert keep == ["demo__pkg-1", "demo__pkg-4"], keep
        assert dropped == [("demo__pkg-2", "pending_recheck"),
                           ("demo__pkg-3", "deferred")], dropped
        # a missing record must not silently drop the whole sample
        assert SW.drop_invalid_envs(ids, Path(td) / "nope.json") == (ids, [])


def test_failed_unit_still_counts_its_calls_against_the_quota() -> None:
    """A unit that dies mid-phase spent real calls; the provider does not refund
    them. usage.json used to be written only on the success path, so the three
    503-killed instances of 2026-09-22 left no trace in the day's tally."""
    _stub_world(broken={"demo__pkg-2"})
    from tdbf.models import llm

    with tempfile.TemporaryDirectory() as td:
        root = Path(td); store = _Store(root)
        usage_path = root / "usage.json"
        # the sweep reads the process-wide counter; simulate the broken unit
        # burning 7 calls before it raised
        real = llm.calls_served_total
        ticks = iter([0, 7, 7, 7, 7, 7, 7, 7, 7])
        SW.calls_served_total = lambda: next(ticks, 7)
        try:
            s = _sweep(root, store, _cfg(root), [_inst(2)])
        finally:
            SW.calls_served_total = real
        assert [iid for iid, _ in s.failed] == ["demo__pkg-2"], s.failed
        assert s.failed_calls == 7, s.failed_calls
        assert json.loads(usage_path.read_text()) != {}, "failed unit was not billed"


def main() -> int:
    tests = [
        test_ab_unit_records_share_a_pair_id,
        test_env_gate_drops_instances_not_marked_valid,
        test_failed_unit_still_counts_its_calls_against_the_quota,
        test_order_is_a_seeded_coin_flip,
        test_resume_skips_completed_units,
        test_failed_unit_leaves_no_records,
        test_repeated_failures_stop_the_sweep,
        test_quota_error_stops_immediately,
        test_upstream_throttle_fails_the_unit_not_the_sweep,
        test_b_only_is_the_capability_probe,
        test_default_budgets,
        test_daily_budget_is_respected,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nPaired sweep driver: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
