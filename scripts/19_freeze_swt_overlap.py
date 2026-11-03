#!/usr/bin/env python3
"""Freeze the instance set of the test-agent studies: the RQ1 experiment's paired
instances that are also in SWT-bench Lite (the 114 our nemotron tests were
re-scored on).

Writes two files, in the same format as the RQ1 samples:

  sample_tests_swt114_frozen.json   the ids, how they were chosen, from what
  sample_tests_swt114_env.json      each instance's environment state, copied from
                                    the RQ1 environment records (all valid: gold
                                    checked there), so a batch does not redo it

The ids are the paired test-first instances (metrics.experiment_tests, the same
pairing as the Results page) that have a stored SWT-bench verdict in study
rq1-nemotron. The harness scored every paired instance that is in SWT-bench Lite
and skipped the rest (scripts/17, 18), so that set is the overlap. Refuses to
overwrite an existing frozen file.

    python3 scripts/19_freeze_swt_overlap.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import ledger, metrics  # noqa: E402

NAME = "tdbf-tests-swt-lite-overlap-n114"
FROZEN = ROOT / "sample_tests_swt114_frozen.json"
ENV = ROOT / "sample_tests_swt114_env.json"
SOURCE_ENVS = ("sample_lite_env.json", "sample_lite_ext_env.json")
EXPECTED = 114


def main() -> int:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN.name} exists: a frozen sample is never rewritten")
    recs = metrics.load_records(ROOT / "results.jsonl")
    _, env = ledger.load_samples(ROOT)
    states = {k: (v or {}).get("state", "?") for k, v in env.items()}
    paired = set(metrics.experiment_tests(recs, env_states=states))
    swt = metrics.load_records(ROOT / "swt_results.jsonl")
    scored = {r["instance_id"] for r in swt if metrics.swt_study(r) == metrics.EXPERIMENT_STUDY}
    ids = sorted(paired & scored)
    if len(ids) != EXPECTED or scored - paired:
        raise SystemExit(f"expected {EXPECTED} paired+scored ids, got {len(ids)} "
                         f"(scored but not paired: {sorted(scored - paired)})")

    source_env: dict[str, dict] = {}
    for name in SOURCE_ENVS:
        source_env.update(json.loads((ROOT / name).read_text())["instances"])
    missing = [i for i in ids if (source_env.get(i) or {}).get("state") != "valid"]
    if missing:
        raise SystemExit(f"not valid in the RQ1 environment records: {missing}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    repo_of = {r["instance_id"]: r["repo"] for r in swt}
    by_repo = collections.Counter(repo_of[i] for i in ids)
    FROZEN.write_text(json.dumps({
        "name": NAME,
        "created_utc": now,
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "method": "the RQ1 experiment's paired instances (metrics.experiment_tests, "
                  "rq1-nemotron) that are in SWT-bench Lite, i.e. that have a stored "
                  "SWT-bench verdict in study rq1-nemotron (swt_results.jsonl)",
        "derived_from": ["sample_lite_frozen.json", "sample_lite_ext_frozen.json"],
        "code_commit": commit,
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "size": len(ids),
        "by_repo": dict(sorted(by_repo.items())),
        "instances": ids,
    }, indent=2) + "\n")
    rows = {i: {**source_env[i], "copied_from_rq1_env_utc": now} for i in ids}
    ENV.write_text(json.dumps({
        "sample": NAME, "written_utc": now, "counts": {"valid": len(ids)},
        "instances": rows}, indent=2) + "\n")
    print(f"{len(ids)} instances -> {FROZEN.name}, {ENV.name}")
    for repo, n in sorted(by_repo.items()):
        print(f"  {repo:28s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
