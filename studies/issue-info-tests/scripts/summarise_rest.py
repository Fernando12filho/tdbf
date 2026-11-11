#!/usr/bin/env python3
"""Summary of LLM labels for bugs without human labels: errors, cost, how often each
information type is present (run 1), and run-to-run consistency (no human reference).

    python3 summarise_rest.py --ids data/rest_ids.txt --llm results/rest_spans_v3.part*.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

GROUPS = {
    "expected_behaviour": lambda l: l == "BUG_EXPECTED_BEHAVIOUR",
    "location": lambda l: l.startswith("BUG_AFFECTED_AREA"),
    "suggestion": lambda l: l.startswith("BUG_SUGGESTED_FIX"),
    "reproduction_steps": lambda l: l == "BUG_REPRODUCTION_STEPS",
    "observed_behaviour": lambda l: l == "BUG_OBSERVED_BEHAVIOUR",
    "environment": lambda l: l == "BUG_ENVIRONMENT",
    "reference": lambda l: l == "BUG_REFERENCE",
}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--llm", nargs="+", required=True)
    a = ap.parse_args()
    ids = [i.strip() for i in Path(a.ids).read_text().splitlines() if i.strip()]
    recs = [json.loads(l) for p in a.llm for l in Path(p).read_text().splitlines() if l]
    errs = [(r["instance_id"], r["run"], r["error"]) for r in recs if r.get("error")]
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in recs)
    runs = {(r["instance_id"], r["run"]): {s["label"] for s in r["spans"]} for r in recs if not r.get("error")}
    print(f"records {len(recs)} (expected {3 * len(ids)}), errors {len(errs)}, cost ${cost:.3f}")
    for e in errs:
        print("  error:", e)
    print(f"\n{'type':22} {'present (run 1)':>16} {'kappa r1-r2':>12} {'r1-r3':>7} {'r2-r3':>7}")
    for name, test in GROUPS.items():
        pres = {r: [int(any(test(l) for l in runs.get((i, r), ()))) for i in ids] for r in (1, 2, 3)}
        ks = [kappa(pres[x], pres[y]) for x, y in itertools.combinations((1, 2, 3), 2)]
        fmt = lambda k: "  n/a" if k is None else f"{k:5.2f}"
        print(f"{name:22} {sum(pres[1]):>9}/{len(ids):<6} {fmt(ks[0]):>12} {fmt(ks[1]):>7} {fmt(ks[2]):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
