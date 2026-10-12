"""Offline checks for the Results-page numbers (tdbf.metrics.build_results) and
the RQ1 paired script (scripts/14_paired_analysis.py). No API, no Docker.

Four made-up instances, one per outcome, so every expected value can be worked
out by hand:

  i1  both arms resolve                    test L4, passes the gate
  i2  only test-first resolves             test L3, passes the gate
  i3  only reactive resolves               test L2 (crashes), gate rejects
  i4  neither; test-first's hidden F2P tests still fail, reactive broke a P2P
      test                                 test L4, passes the gate

plus one development record that must never leak into the Results numbers.

    python3 scripts/metrics_smoke.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import metrics as M  # noqa: E402


def _unit(iid, group, *, arm_a=None, arm_b=None, calls=10, day="2026-09-21"):
    return {"instance_id": iid, "repo": "demo/pkg", "mode": "sweep_unit",
            "model": "openrouter:demo:free", "timestamp": f"{day}T00:00:00",
            "flags": {"run_group": group, "max_iterations": 30,
                      "max_tokens_per_turn": 16000, "max_total_tokens": 1000000},
            "arm_a": arm_a, "arm_b": arm_b, "calls": calls}


def _fix(iid, group, arm, *, resolved, stop="submit", f2p=(), p2p=(), applied=True):
    return {"instance_id": iid, "repo": "demo/pkg", "mode": "fix_agent",
            "timestamp": "2026-09-22T00:00:00", "resolved": resolved,
            "flags": {"run_group": group, "arm": arm}, "best_attempt": 1,
            "attempts": [{"attempt": 1, "resolved": resolved, "stop_cause": stop,
                          "applied": applied, "fail_to_pass_failed": list(f2p),
                          "pass_to_pass_failed": list(p2p)}]}


def _test(iid, *, level, gated):
    return {"instance_id": iid, "repo": "demo/pkg", "mode": "agent_deployable",
            "timestamp": "2026-09-21T00:00:00", "best_level": level,
            "gate_accepted": gated, "flags": {"run_group": "rq1-sweep"}}


def _records() -> list[dict]:
    tf, rx = "rq1-sweep", "rq1-armA"
    b = lambda ok, lvl, gated: {"resolved": ok, "brt_level": lvl, "gate_accepted": gated}
    a = lambda ok: {"resolved": ok}
    return [
        # test-first arm (calls 30, 40, 50, 60 -> mean 45, median 45)
        _unit("i1", tf, arm_b=b(True, 4, True), calls=30),
        _unit("i2", tf, arm_b=b(True, 3, True), calls=40),
        _unit("i3", tf, arm_b=b(False, 2, False), calls=50),
        _unit("i4", tf, arm_b=b(False, 4, True), calls=60, day="2026-09-24"),
        _test("i1", level=4, gated=True), _test("i2", level=3, gated=True),
        _test("i3", level=2, gated=False), _test("i4", level=4, gated=True),
        _fix("i1", tf, "deployable", resolved=True),
        _fix("i2", tf, "deployable", resolved=True, stop="submit_forced"),
        _fix("i4", tf, "deployable", resolved=False, f2p=["t_bug"]),
        # reactive arm (calls 10, 20, 30, 40 -> mean 25, median 25)
        _unit("i1", rx, arm_a=a(True), calls=10, day="2026-09-25"),
        _unit("i2", rx, arm_a=a(False), calls=20),
        _unit("i3", rx, arm_a=a(True), calls=30),
        _unit("i4", rx, arm_a=a(False), calls=40, day="2026-09-28"),
        _fix("i1", rx, "reactive", resolved=True),
        _fix("i2", rx, "reactive", resolved=False, stop="max_iterations", f2p=["t_bug"]),
        _fix("i3", rx, "reactive", resolved=True, stop="submit_forced"),
        _fix("i4", rx, "reactive", resolved=False, p2p=["t_old"]),
        # a development record: must not appear anywhere in the Results numbers
        _unit("dev1", "gate-fix", arm_b=b(True, 4, True), calls=999),
    ]


def test_pairing_and_resolved_rates() -> None:
    r = M.build_results(_records())
    assert r["n"] == 4, r["n"]
    tf, rx = r["arms"]
    assert tf["resolved"]["label"] == "2/4 = 0.50" and rx["resolved"]["label"] == "2/4 = 0.50"
    p = r["paired"]
    assert (p["both"], p["tf_only"], p["rx_only"], p["neither"]) == (["i1"], ["i2"], ["i3"], ["i4"])


def test_calls_dates_and_stopping() -> None:
    tf, rx = M.build_results(_records())["arms"]
    assert (tf["calls_mean"], tf["calls_median"], tf["calls_total"]) == (45, 45, 180)
    assert (rx["calls_mean"], rx["calls_median"], rx["calls_total"]) == (25, 25, 100)
    assert (tf["first_day"], tf["last_day"]) == ("2026-09-21", "2026-09-24")
    assert (rx["first_day"], rx["last_day"]) == ("2026-09-21", "2026-09-28")
    # test-first: 3 fix runs (i3 never reached fixing); 2 stopped themselves
    assert tf["fix_runs"] == 3 and tf["stopped_itself"]["label"] == "2/3 = 0.67"
    assert rx["stopped_itself"]["label"] == "2/4 = 0.50"
    assert rx["ran_out_of_turns"]["label"] == "2/4 = 0.50"


def test_funnel_levels_c2p_valid() -> None:
    r = M.build_results(_records())
    f = r["funnel"]
    assert f["gate_rate"]["label"] == "3/4 = 0.75"          # GateRate = gated / N
    assert f["resolve_at_gated"]["label"] == "2/3 = 0.67"   # resolved / gated
    assert f["pipeline_resolve"]["label"] == "2/4 = 0.50"   # resolved / N
    assert r["levels"] == [0, 0, 1, 1, 2]
    assert r["c2p"]["label"] == "2/3 = 0.67"                # L4 / (L3 + L4)
    assert r["valid_brt"]["label"] == "2/4 = 0.50"          # L4 / N


def test_failure_kinds_and_stages() -> None:
    r = M.build_results(_records())
    count = lambda arm: {x["key"]: x["count"] for x in r["failure"][arm]}
    assert count("test-first") == {"resolved": 2, "f2p_fail": 1, "p2p_fail": 0,
                                   "not_applied": 0, "gate_rejected": 1, "other": 0}
    assert count("reactive") == {"resolved": 2, "f2p_fail": 1, "p2p_fail": 1,
                                 "not_applied": 0, "other": 0}
    assert [s["count"] for s in r["stages"]] == [4, 4, 3, 2]


def test_left_out_and_dev_isolation() -> None:
    r = M.build_results(_records(), sample_ids=["i1", "i2", "i3", "i4", "x9"],
                        env_states={"x9": "deferred"})
    assert r["left_out"] == [{"instance_id": "x9", "state": "deferred"}]
    assert "dev1" not in sum(r["paired"].values(), [])
    assert r["arms"][0]["calls_total"] == 180  # the dev record's 999 calls stay out


def test_substitution_precedence() -> None:
    """rq1-djv2 replaces test-first by group precedence, not by date, and a
    substituted instance takes its fix/test records from rq1-djv2 only."""
    sub = "rq1-djv2"
    b = lambda ok, lvl, gated: {"resolved": ok, "brt_level": lvl, "gate_accepted": gated}
    recs = _records() + [
        # older than the originals, still wins: i2 now fails its hidden test
        _unit("i2", sub, arm_b=b(False, 4, True), day="2026-09-01"),
        {**_test("i2", level=4, gated=True), "flags": {"run_group": sub}},
        _fix("i2", sub, "deployable", resolved=False, f2p=["t_bug"]),
        # gate rejects now: no fix record in rq1-djv2, so the old group's i4
        # fix (an F2P failure) must NOT be borrowed
        _unit("i4", sub, arm_b=b(False, 2, False), day="2026-09-01"),
        {**_test("i4", level=2, gated=False), "flags": {"run_group": sub}},
    ]
    r = M.build_results(recs)
    assert r["paired"] == {"both": ["i1"], "tf_only": [], "rx_only": ["i3"],
                           "neither": ["i2", "i4"]}
    fail = {d["key"]: d["count"] for d in r["failure"]["test-first"]}
    assert (fail["resolved"], fail["f2p_fail"], fail["gate_rejected"]) == (1, 1, 2)
    assert r["levels"] == [0, 0, 2, 0, 2]   # i3, i4 L2; i1, i2 L4
    orig = M.build_results(recs, arms_def=M.EXPERIMENT_ARMS_ORIGINAL)
    assert orig["paired"] == M.build_results(_records())["paired"]


def test_paired_script_mcnemar() -> None:
    spec = importlib.util.spec_from_file_location(
        "paired", ROOT / "scripts" / "14_paired_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.mcnemar_exact(0, 0) is None
    assert mod.mcnemar_exact(1, 1) == 1.0
    assert abs(mod.mcnemar_exact(3, 5) - 186 / 256) < 1e-12   # 2*(1+8+28+56)/256
    assert abs(mod.mcnemar_exact(0, 8) - 2 / 256) < 1e-12
    obs, lo, hi = mod.bootstrap_ci([(True, True)] * 3 + [(False, False)] * 3)
    assert obs == lo == hi == 0.0   # no discordant pairs -> no spread


def main() -> int:
    tests = [
        test_pairing_and_resolved_rates,
        test_calls_dates_and_stopping,
        test_funnel_levels_c2p_valid,
        test_failure_kinds_and_stages,
        test_left_out_and_dev_isolation,
        test_substitution_precedence,
        test_paired_script_mcnemar,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nResults metrics: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
