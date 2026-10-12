#!/usr/bin/env python3
"""RQ1 paired analysis: test-first vs reactive on the same instances.

Pure aggregation of runs already logged (NO API, NO Docker). The pairing comes
from `tdbf.metrics.build_results`, the same function behind the webapp Results
page, so this script and the page always agree on which instances are paired.

On top of the pairing it runs the two analyses the frozen sample pre-registered:

  McNemar's exact test on the discordant pairs
      b = only reactive resolved, c = only test-first resolved
      p = min(1, 2 * sum_{i=0..min(b,c)} C(b+c, i) / 2^(b+c))

  Paired bootstrap 95% CI of the difference in resolved rate
      d = (resolved_test_first - resolved_reactive) / N
      resample the N pairs with replacement, 10,000 times (seed 20260916);
      the CI is the 2.5th to 97.5th percentile of the resampled d.

  python3 scripts/14_paired_analysis.py              # Django test-first = rq1-djv2
  python3 scripts/14_paired_analysis.py --original   # without that substitution
"""

from __future__ import annotations

import argparse
import random
from math import comb
from pathlib import Path

from tdbf import ledger, metrics

SEED = 20260916  # the frozen sample's seed


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value; None when there are no discordant pairs."""
    d = b + c
    if d == 0:
        return None
    k = min(b, c)
    return min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2 ** d)


def bootstrap_ci(pairs: list[tuple[bool, bool]], *, reps: int = 10_000,
                 seed: int = SEED) -> tuple[float, float, float]:
    """(observed d, CI low, CI high) for d = test-first rate - reactive rate.
    Each pair is (test_first_resolved, reactive_resolved) and is resampled whole."""
    n = len(pairs)
    obs = (sum(t for t, _ in pairs) - sum(r for _, r in pairs)) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append((sum(t for t, _ in s) - sum(r for _, r in s)) / n)
    diffs.sort()
    return obs, diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="results.jsonl")
    ap.add_argument("--original", action="store_true",
                    help="original Django test-first records (no rq1-djv2 substitution)")
    args = ap.parse_args()

    # Environment states for every frozen sample (ledger.SAMPLE_FILES), so an
    # instance its record marks not-valid is never counted.
    _, env = ledger.load_samples(Path(args.file).resolve().parent)
    states = {k: (v or {}).get("state", "?") for k, v in env.items()}
    arms_def = metrics.EXPERIMENT_ARMS_ORIGINAL if args.original else metrics.EXPERIMENT_ARMS
    res = metrics.build_results(metrics.load_records(args.file), env_states=states,
                                arms_def=arms_def)
    n = res["n"]
    if not n:
        print("No instance has a completed unit in both arms yet.")
        return 1

    p = res["paired"]
    both, tf, rx, neither = (len(p[k]) for k in ("both", "tf_only", "rx_only", "neither"))
    # One (test_first_resolved, reactive_resolved) per instance, in instance-id
    # order, so a given seed always draws the same resamples.
    tf_won = set(p["both"]) | set(p["tf_only"])
    rx_won = set(p["both"]) | set(p["rx_only"])
    ids = sorted(p["both"] + p["tf_only"] + p["rx_only"] + p["neither"])
    pairs = [(i in tf_won, i in rx_won) for i in ids]

    arms = {a["label"]: a for a in res["arms"]}
    print("view: " + ("ORIGINAL (no Django substitution)" if args.original
                      else "CORRECTED (Django test-first = rq1-djv2)"))
    print(f"paired instances: {n}\n")
    print("                      test-first")
    print("                    resolved   not")
    print(f"  reactive  resolved  {both:5d}   {rx:5d}")
    print(f"            not       {tf:5d}   {neither:5d}\n")
    print(f"  test-first resolved {arms['test-first']['resolved']['label']}")
    print(f"  reactive   resolved {arms['reactive']['resolved']['label']}\n")

    pv = mcnemar_exact(rx, tf)
    print(f"  discordant: test-first only {tf}, reactive only {rx}")
    print("  McNemar exact two-sided p = "
          + ("n/a (no discordant pairs)" if pv is None else f"{pv:.4f}"))
    obs, lo, hi = bootstrap_ci(pairs)
    print(f"  difference (test-first - reactive) = {obs:+.3f}"
          f"   95% CI [{lo:+.3f}, {hi:+.3f}]  (10,000 paired resamples, seed {SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
