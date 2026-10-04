"""Draw and freeze the RQ1 sample — once, before any sweep result exists.

Stratified random sample of SWE-bench Lite by repository, proportional to each
repo's share of the eligible pool (largest-remainder allocation), excluding the
development set: every instance that already has a record in results.jsonl, i.e.
the instances the pipeline was built and tuned on.

The output file is committed to git before the sweep starts. The commit is the
evidence that the list — and the analysis rules written into it — were fixed
before any result was seen. Re-running with the same seed reproduces the same
list; the script refuses to overwrite an existing file.

    python3 scripts/12_freeze_sample.py --size 60 --seed 20260916

An EXTENSION sample is drawn the same way from what is left: pass every earlier
sample with --exclude-sample (its instances are excluded whether or not they
ever produced a record -- an unfinished or deferred instance has none), and
--extension-of to write the extension's rules.

    python3 scripts/12_freeze_sample.py --size 90 --seed 20261004 \
        --exclude-sample sample_lite_frozen.json --extension-of sample_lite_frozen.json \
        --run-group rq1-ext --out sample_lite_ext_frozen.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.data.instances import InstanceStore
from tdbf.results import RESULTS_FILE

RULES = [
    "Both arms run on every instance, back to back; the order per instance is "
    "11_paired_sweep.arm_order(seed=0, instance_id).",
    "The unit of analysis is a sweep_unit with arms == 'AB'. Arm B counts as not "
    "resolved when its gold-free gate rejects.",
    "Environment-invalid instances are excluded BEFORE any arm runs: an instance whose "
    "Docker image cannot be built, or whose GOLD patch does not resolve in our harness. "
    "This is decided from the gold check alone, never from agent results, and the "
    "excluded instances and their count are reported.",
    "An instance with no completed unit at the data freeze is reported as incomplete, "
    "not silently dropped. Units that fail for infrastructure reasons are retried.",
    "Primary analysis: McNemar's test on discordant pairs, and a paired bootstrap 95% "
    "CI (10,000 resamples) of the difference in resolved rate (test-first minus reactive).",
    "The model is chosen by the capability probe, which runs on the development set only "
    "and never touches this sample.",
    "Budgets are identical across arms and recorded per record: for free OpenRouter "
    "models 16,000 tokens per turn, 1,000,000 total, 30 iterations per phase.",
]


def extension_rules(base: str, run_group: str) -> list[str]:
    return [
        f"This sample EXTENDS {base}: same dataset, same eligibility, same stratified "
        "method, drawn from the instances neither in that sample nor in the development "
        "set. Its instances never overlap an earlier sample.",
        f"Both arms run as one AB unit per instance (run group {run_group!r}), back to "
        "back, in the order 11_paired_sweep.arm_order(seed=0, instance_id) — the same "
        "unit the base sample's arms were run as, recorded together instead of in two "
        "blocks.",
        "Code is frozen at the commit recorded in `code_commit` (git tag "
        "extension-freeze). Any later change to agent or harness behaviour is reported "
        "as a departure.",
        "Primary RQ1 analysis: the base and extension samples POOLED (every completed "
        "paired instance of both), with the base sample's tests: McNemar exact on "
        "discordant pairs and a paired bootstrap 95% CI (10,000 resamples). The "
        "extension alone and the base alone are also reported.",
        "Known departure in the base sample, decided before this draw: 19 of its "
        "Django test-first units ran before run_tests used Django's runner (commit "
        "7bc3d80). It is reported, not corrected.",
        "Environment-invalid instances (no prebuilt image, or the gold patch does not "
        "resolve) are excluded before any arm runs, decided from the gold check alone; "
        "they and their count are reported. Units that fail for infrastructure reasons "
        "are retried; an instance with no completed unit at the data freeze is "
        "reported as incomplete.",
        "Budgets identical across arms and recorded per record: for free OpenRouter "
        "models 16,000 tokens per turn, 1,000,000 total, 30 iterations per phase.",
    ]


def code_commit() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True)
    except OSError:
        return None
    return p.stdout.strip() or None


def development_set(results_path: Path) -> list[str]:
    ids = set()
    for line in results_path.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["instance_id"])
    return sorted(ids)


def allocate(eligible: dict[str, int], size: int) -> dict[str, int]:
    """Proportional allocation, largest remainder; ties broken by repo name."""
    total = sum(eligible.values())
    quotas = {r: n * size / total for r, n in eligible.items()}
    alloc = {r: int(q) for r, q in quotas.items()}
    short = size - sum(alloc.values())
    for r in sorted(quotas, key=lambda r: (-(quotas[r] - alloc[r]), r))[:short]:
        alloc[r] += 1
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--out", default="sample_lite_frozen.json")
    ap.add_argument("--exclude-sample", nargs="*", default=[], metavar="FILE",
                    help="earlier frozen samples whose instances are not eligible")
    ap.add_argument("--extension-of", metavar="FILE",
                    help="write extension rules (pooled analysis) relative to this sample")
    ap.add_argument("--run-group", default="rq1-ext",
                    help="with --extension-of: the run group the extension runs under")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"{out.name} already exists — a frozen sample is never redrawn.")

    store = InstanceStore(args.split)
    rows = list(store.iter_instances())
    dev = development_set(RESULTS_FILE)
    earlier: set[str] = set()
    for f in args.exclude_sample:
        earlier |= set(json.loads((ROOT / f).read_text())["instances"])
    fingerprint = hashlib.sha256(
        "\n".join(sorted(f"{r.instance_id}:{r.base_commit}" for r in rows)).encode()
    ).hexdigest()

    pool: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        if r.instance_id not in dev and r.instance_id not in earlier:
            pool[r.repo].append(r.instance_id)
    all_repos = sorted({r.repo for r in rows})
    eligible = {repo: len(pool.get(repo, [])) for repo in all_repos}
    alloc = allocate({r: n for r, n in eligible.items() if n}, args.size)

    rng = random.Random(args.seed)
    chosen: list[str] = []
    for repo in sorted(alloc):  # fixed iteration order keeps the draw reproducible
        chosen += rng.sample(sorted(pool[repo]), alloc[repo])

    record = {
        "name": (f"tdbf-rq1-{args.split}-ext-n{args.size}" if args.extension_of
                 else f"tdbf-rq1-{args.split}-n{args.size}"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": store.dataset_name,
        "dataset_instances": len(rows),
        "dataset_sha256": fingerprint,
        "seed": args.seed,
        "size": args.size,
        "method": "stratified random by repository; proportional allocation over the "
                  "eligible pool, largest remainder, ties by repo name; per repo "
                  "random.Random(seed).sample over sorted ids, repos in sorted order",
        "excluded_development_set": dev,
        **({"excluded_earlier_samples": {f: sorted(json.loads((ROOT / f).read_text())
                                                   ["instances"]) for f in args.exclude_sample}}
           if args.exclude_sample else {}),
        **({"extension_of": args.extension_of, "run_group": args.run_group,
            "code_commit": code_commit()} if args.extension_of else {}),
        "allocation": {repo: {"in_lite": sum(1 for r in rows if r.repo == repo),
                              "eligible": eligible[repo],
                              "drawn": alloc.get(repo, 0)}
                       for repo in all_repos},
        "rules": (extension_rules(args.extension_of, args.run_group)
                  if args.extension_of else RULES),
        "instances": sorted(chosen),
    }
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"Wrote {out.name}: {len(chosen)} instances, seed {args.seed}")
    print(f"Excluded development set: {len(dev)} instances")
    for repo in all_repos:
        a = record["allocation"][repo]
        print(f"  {repo:28s} lite={a['in_lite']:3d}  eligible={a['eligible']:3d}  drawn={a['drawn']:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
