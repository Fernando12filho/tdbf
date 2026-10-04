"""Offline checks for the per-instance status table (tdbf.ledger). No API, no Docker.

Seven made-up instances, one per status plus one edge case:

  a__a-1  in the sample, both arms finished             -> paired
  a__a-2  in the sample, only test-first finished       -> one_arm
  a__a-3  in the sample, valid environment, no runs     -> pending
  a__a-4  in the sample, deferred (no image)            -> excluded
  a__a-5  in the sample, deferred, but both arms ran    -> paired (a result wins)
  b__b-1  development set                               -> development
  b__b-2  never drawn, only a non-experiment run         -> unused

    python3 scripts/ledger_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import ledger as L  # noqa: E402
from tdbf.metrics import EXPERIMENT_ARMS  # noqa: E402

TF, RX = (gs[0] for _, gs, _ in EXPERIMENT_ARMS)


def _unit(iid, group, arms=None):
    """arms: "A", "B" or "AB"; default from the base sample's one-arm groups."""
    arms = arms or {TF: "B", RX: "A"}.get(group, "B")
    return {"instance_id": iid, "mode": "sweep_unit", "model": "m",
            "timestamp": "2026-09-22T00:00:00",
            "flags": {"run_group": group, "arms": arms},
            "arm_a": {"resolved": True} if "A" in arms else None,
            "arm_b": {"resolved": True} if "B" in arms else None}


def _run(iid, group=None, model="m"):
    return {"instance_id": iid, "mode": "fix_agent", "model": model,
            "timestamp": "2026-09-23T00:00:00", "flags": {"run_group": group}}


def main() -> int:
    records = [
        _unit("a__a-1", TF), _unit("a__a-1", RX), _run("a__a-1", TF),
        _unit("a__a-2", TF),
        _unit("a__a-5", TF), _unit("a__a-5", RX),
        _unit("e__e-1", "rq1-ext", arms="AB"),   # extension: both arms in one unit
        _run("b__b-1", model="dev"),
        _run("b__b-2", group="probe", model="other"),
        _unit("b__b-2", "probe"),          # a non-experiment unit counts for nothing
    ]
    samples = [{"name": "s", "excluded_development_set": ["b__b-1"],
                "instances": ["a__a-1", "a__a-2", "a__a-3", "a__a-4", "a__a-5"]},
               {"name": "ext", "instances": ["e__e-1"]}]
    env = {"a__a-4": {"state": "deferred", "reason": "no image"},
           "a__a-5": {"state": "deferred", "reason": "late"},
           "a__a-3": {"state": "valid"}}
    lite = ["a__a-1", "a__a-2", "a__a-3", "a__a-4", "a__a-5", "b__b-1", "b__b-2", "c__c-9",
            "e__e-1"]

    led = L.build_ledger(lite_ids=lite, records=records, samples=samples,
                         env_states=env, images={"a__a-1", "zz__zz-1"})
    got = {r["instance_id"]: r["status"] for r in led["rows"]}
    want = {"a__a-1": "paired", "a__a-2": "one_arm", "a__a-3": "pending",
            "a__a-4": "excluded", "a__a-5": "paired", "b__b-1": "development",
            "b__b-2": "unused", "c__c-9": "unused", "e__e-1": "paired"}
    assert got == want, got

    rows = {r["instance_id"]: r for r in led["rows"]}
    assert rows["a__a-1"]["arms_done"] == ["reactive", "test-first"], rows["a__a-1"]
    assert rows["a__a-1"]["runs"] == 1 and rows["a__a-1"]["image"] is True
    assert rows["a__a-2"]["image"] is False
    assert rows["a__a-4"]["env_reason"] == "no image"
    assert rows["b__b-2"]["runs"] == 1 and rows["b__b-2"]["models"] == ["other"]
    assert rows["c__c-9"]["runs"] == 0 and rows["c__c-9"]["last_run"] is None

    assert led["counts"] == {"paired": 3, "one_arm": 1, "pending": 1, "excluded": 1,
                             "development": 1, "unused": 2}, led["counts"]
    assert led["total"] == 9 and led["complete_index"]
    assert led["images_on_disk"] == 2 and led["images_orphaned"] == ["zz__zz-1"]
    a = next(t for t in led["repos"] if t["repo"] == "a/a")
    assert (a["total"], a["paired"], a["images"]) == (5, 2, 1), a

    # No Lite index and no Docker: the table still builds from what it has.
    led2 = L.build_ledger(lite_ids=None, records=records, samples=samples,
                          env_states=env, images=None)
    assert not led2["complete_index"] and "c__c-9" not in {r["instance_id"] for r in led2["rows"]}
    assert led2["images_on_disk"] is None and all(r["image"] is None for r in led2["rows"])

    assert L.repo_from_id("scikit-learn__scikit-learn-12471") == "scikit-learn/scikit-learn"
    assert L.repo_from_id("pylint-dev__pylint-7114") == "pylint-dev/pylint"
    print("ledger smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
