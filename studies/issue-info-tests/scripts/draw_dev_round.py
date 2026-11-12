#!/usr/bin/env python3
"""Draw the development bugs for one round of Phase 1 (design §10c).

Pool: SWT-bench Lite (SWE-bench Lite minus the cases in SWT-bench's
dataset/filter_cases_lite.txt) minus the 114 evaluation bugs
(sample_tests_swt114_frozen.json). The pool is shuffled ONCE with the fixed seed;
round k takes positions 10(k-1)..10k-1 of that order. Rounds are therefore
reproducible and never reuse a bug (drawn without replacement).

Bugs whose environment turns out invalid are not replaced here: the round
record lists them and the next unused position is taken by hand-free rule
`--extra n`, which appends the next n bugs of the same order.

    python3 studies/issue-info-tests/scripts/draw_dev_round.py --round 1 \
        --swt-filter ~/swt-bench/dataset/filter_cases_lite.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.data.instances import InstanceStore  # noqa: E402

SEED = 20261112
ROUND_SIZE = 10
OUT_DIR = ROOT / "studies" / "issue-info-tests" / "data" / "dev"


def pool(swt_filter: Path) -> list[str]:
    frozen = json.loads((ROOT / "sample_tests_swt114_frozen.json").read_text())
    ids = frozen["instance_ids"] if isinstance(frozen, dict) and "instance_ids" in frozen else [
        x if isinstance(x, str) else x["instance_id"]
        for x in (frozen if isinstance(frozen, list) else frozen.get("instances", []))]
    eval114 = set(ids)
    assert len(eval114) == 114, len(eval114)
    excluded = set(swt_filter.read_text().split())
    lite = sorted(r["instance_id"] for r in InstanceStore("lite")._load())
    return [i for i in lite if i not in excluded and i not in eval114]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--extra", type=int, default=0,
                    help="also take the next n bugs of the order (replacements for invalid envs)")
    ap.add_argument("--swt-filter", type=Path, required=True)
    args = ap.parse_args()

    order = pool(args.swt_filter)
    n_pool = len(order)
    random.Random(SEED).shuffle(order)
    start = ROUND_SIZE * (args.round - 1)
    picked = order[start:start + ROUND_SIZE]
    extra = order[start + ROUND_SIZE:start + ROUND_SIZE + args.extra] if args.extra else []
    rec = {
        "round": args.round,
        "seed": SEED,
        "pool": "SWT-bench Lite minus sample_tests_swt114_frozen.json",
        "pool_size": n_pool,
        "positions": [start, start + ROUND_SIZE + len(extra) - 1],
        "instance_ids": picked,
        "replacements": extra,
        "note": "replacements are the next positions of the same order; used only for "
                "bugs whose environment is invalid",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"round{args.round}.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
