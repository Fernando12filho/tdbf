#!/usr/bin/env python3
"""Cross-check our BRT labels against the official SWT-bench harness.

Subcommands, no API and no Docker:

  export   turn the stored test-first BRTs (the tests behind the Results page)
           into an SWT-bench predictions file, plus a sidecar with our own label
           for each instance
  compare  read the SWT-bench reports for those predictions and put the two
           verdicts side by side
  ingest   append the verified verdicts to swt_results.jsonl (the store the
           webapp /swt page reads) and copy the raw evidence to runs/swt/

The harness itself runs elsewhere (scripts/18_swt_score.py, on an x86_64 host).

Each prediction's `model_patch` adds exactly the files our scorer ran
(`brt_run_spec`): tests/<module>/test_brt.py for Django, test_brt.py with the
setup preamble at the repo root for every other repo.

SWT-bench verdicts (src/grading.py, figures/util.py in the harness):
  S    at least one test goes fail->pass and none fails after the fix (strict);
       our L4 should correspond to this
  F->P at least one test goes fail->pass, others may still fail (loose)

  python3 scripts/17_swt_predictions.py export --ids django__django-11099 sympy__sympy-20590
  python3 scripts/17_swt_predictions.py export --all
  python3 scripts/17_swt_predictions.py compare --logs ~/swt-bench/run_instance_swt_logs
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tdbf import ledger, metrics
from tdbf.harness.runner import brt_run_spec, make_test_patch

# Where a BRT must live for the repo's OWN runner (the one SWT-bench uses) to
# collect it, when that differs from where our scorer ran it. sympy's bin/test
# only collects files inside the sympy package, so a root-level test_brt.py is
# run as "0 tests" (trial 2026-10-27, sympy-20590). The content is unchanged.
SWT_PLACEMENT = {"sympy/sympy": "sympy/core/tests"}

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "swt"
MODEL_NAME = "tdbf__nemotron-3-super"
RUN_ID = "tdbf-test-first"
STORE = ROOT / "swt_results.jsonl"  # append-only, tracked; kept apart from results.jsonl


def export(ids: list[str] | None, out: Path, model_name: str) -> None:
    recs = metrics.load_records(ROOT / "results.jsonl")
    _, env = ledger.load_samples(ROOT)
    states = {k: (v or {}).get("state", "?") for k, v in env.items()}
    tests = metrics.experiment_tests(recs, env_states=states)
    if ids:
        missing = sorted(set(ids) - set(tests))
        if missing:
            raise SystemExit(f"not in the paired experiment: {' '.join(missing)}")
        tests = {i: tests[i] for i in ids}

    out.parent.mkdir(parents=True, exist_ok=True)
    labels = {}
    with out.open("w") as f:
        for iid, rec in tests.items():
            attempt = metrics._best_attempt(rec)
            spec = brt_run_spec(rec["repo"], attempt.get("test_code") or "")
            subdir = SWT_PLACEMENT.get(rec["repo"])
            if subdir:
                spec.files = {f"{subdir}/{Path(p).name}": c for p, c in spec.files.items()}
            patch = "".join(make_test_patch(p, c) for p, c in spec.files.items())
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": model_name,
                                "model_patch": patch}) + "\n")
            labels[iid] = {
                "repo": rec["repo"],
                "run_group": metrics._flag(rec, "run_group"),
                "level": attempt.get("level"),
                "outcome_buggy": attempt.get("outcome_buggy"),
                "outcome_fixed": attempt.get("outcome_fixed"),
                "failure_reason_buggy": attempt.get("failure_reason_buggy"),
                "files": sorted(spec.files),
            }
    sidecar = out.with_suffix(".labels.json")
    sidecar.write_text(json.dumps(labels, indent=1))
    print(f"{len(labels)} predictions -> {out}\nour labels  -> {sidecar}")
    for iid, lab in labels.items():
        print(f"  {iid:40s} L{lab['level']}  {', '.join(lab['files'])}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _added(report: dict, key: str) -> bool:
    """Did our test add a `key` transition over the base run (the same subtraction
    SWT-bench's figures/util.py makes)?"""
    pred = set(report.get("tests_pred", {}).get(key, []))
    base = set(report.get("tests_base", {}).get(key, []))
    return bool(pred - base)


def swt_metrics(report: dict) -> dict:
    """SWT-bench's per-instance metrics from one report.json entry."""
    applied = bool(report.get("patch_successfully_applied"))
    return {
        "applied": applied,                                    # W
        "success": bool(report.get("resolved")),               # S (strict)
        "f2p": applied and _added(report, "FAIL_TO_PASS"),     # F->P (loose)
        "f2x": applied and (_added(report, "FAIL_TO_PASS") or _added(report, "FAIL_TO_FAIL")),
        "p2p": applied and _added(report, "PASS_TO_PASS"),
        "coverage_pred": report.get("coverage_pred"),
        "coverage_delta_pred": report.get("coverage_delta_pred"),
        "tests_pred": report.get("tests_pred"),
    }


def ingest(logs: Path, predictions: Path, run_id: str, model_name: str,
           store: Path, allow_no_meta: bool) -> None:
    """Append one verified record per scored instance to `store`, and copy the
    raw evidence to runs/swt/<run_id>/<instance>/.

    A report is refused unless the harness scored exactly the patch we exported
    (sha256 of its model_patch.diff) during this run (written after the run's
    start in the 18_swt_score.py metadata). The harness skips any instance whose
    report already exists for the same run id, the trap that left 20 stale fix
    verdicts in September, so a reused run id must never pass silently."""
    patches = {p["instance_id"]: p["model_patch"] for p in
               (json.loads(l) for l in predictions.read_text().splitlines() if l.strip())}
    labels = json.loads(predictions.with_suffix(".labels.json").read_text())
    have = {(r["instance_id"], r["harness"]["run_id"], r["source"]["patch_sha256"])
            for r in metrics.load_records(store)}
    model_dir = model_name.replace("/", "__")
    meta_root = logs.parent / "tdbf_run_meta" / run_id
    added = skipped = 0
    with store.open("a") as out:
        for iid, lab in labels.items():
            rdir = logs / run_id / model_dir / iid
            report_path = rdir / "report.json"
            if not report_path.exists():
                print(f"  {iid:40s} no report, skipped")
                skipped += 1
                continue
            sha = _sha(patches[iid])
            scored = rdir / "model_patch.diff"
            if not scored.exists() or _sha(scored.read_text()) != sha:
                print(f"  {iid:40s} REFUSED: the harness scored a different patch")
                skipped += 1
                continue
            meta_path = meta_root / f"{iid}.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            written = datetime.fromtimestamp(report_path.stat().st_mtime, timezone.utc)
            if meta is None and not allow_no_meta:
                print(f"  {iid:40s} REFUSED: no run metadata (rerun with 18_swt_score.py)")
                skipped += 1
                continue
            if meta is not None and written < datetime.fromisoformat(meta["started"]):
                print(f"  {iid:40s} REFUSED: report predates this run (stale, reused run id?)")
                skipped += 1
                continue
            if (iid, run_id, sha) in have:
                print(f"  {iid:40s} already stored")
                continue
            report = json.loads(report_path.read_text()).get(iid, {})
            evidence = ROOT / "runs" / "swt" / run_id / iid
            for d in (logs / run_id).glob(f"*/{iid}"):
                shutil.copytree(d, evidence / d.parent.name, dirs_exist_ok=True)
            rec = {
                "instance_id": iid,
                "repo": lab["repo"],
                "kind": "tdbf_test_first",
                "source": {"run_group": lab["run_group"], "patch_sha256": sha,
                           "files": lab["files"]},
                "ours": {k: lab[k] for k in ("level", "outcome_buggy", "outcome_fixed",
                                             "failure_reason_buggy")},
                "swt": swt_metrics(report),
                "harness": {"run_id": run_id, "model_name": model_name,
                            "commit": (meta or {}).get("harness_commit"),
                            "image": (meta or {}).get("image"),
                            "repo_digests": (meta or {}).get("repo_digests"),
                            "gold_exit": (meta or {}).get("gold_exit"),
                            "started": (meta or {}).get("started"),
                            "report_written": written.isoformat(),
                            "meta_missing": meta is None},
                "evidence": str(evidence.relative_to(ROOT)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(rec) + "\n")
            have.add((iid, run_id, sha))
            added += 1
            m = rec["swt"]
            print(f"  {iid:40s} stored: ours L{lab['level']}  S={m['success']}  F->P={m['f2p']}  W={m['applied']}")
    print(f"\n{added} stored, {skipped} skipped -> {store}")


def _swt_verdict(report: dict) -> str:
    """'S', 'F->P only', 'no F->P' or 'not applied', from one instance's report."""
    if not report.get("patch_successfully_applied"):
        return "not applied"
    if report.get("resolved"):
        return "S"
    pred = set(report.get("tests_pred", {}).get("FAIL_TO_PASS", []))
    base = set(report.get("tests_base", {}).get("FAIL_TO_PASS", []))
    return "F->P only" if pred - base else "no F->P"


def compare(logs: Path, predictions: Path, run_id: str, model_name: str) -> None:
    labels = json.loads(predictions.with_suffix(".labels.json").read_text())
    rows, table = [], collections.Counter()
    for iid, lab in labels.items():
        path = logs / run_id / model_name.replace("/", "__") / iid / "report.json"
        if not path.exists():
            verdict = "no report"
        else:
            verdict = _swt_verdict(json.loads(path.read_text()).get(iid, {}))
        ours = f"L{lab['level']}"
        table[(ours, verdict)] += 1
        agree = (lab["level"] == 4) == (verdict == "S")
        rows.append((iid, ours, verdict, "" if agree or verdict == "no report" else "DISAGREE"))

    for row in rows:
        print("  {:40s} ours {:3s}  SWT-bench {:12s} {}".format(*row))
    print("\n  ours \\ SWT-bench   " + "  ".join(f"{v:>11s}" for v in
          ("S", "F->P only", "no F->P", "not applied", "no report")))
    for lvl in ("L4", "L3", "L2", "L1", "L0"):
        counts = [table[(lvl, v)] for v in ("S", "F->P only", "no F->P", "not applied", "no report")]
        if any(counts):
            print(f"  {lvl:17s}  " + "  ".join(f"{c:11d}" for c in counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    pick = ex.add_mutually_exclusive_group(required=True)
    pick.add_argument("--ids", nargs="+")
    pick.add_argument("--all", action="store_true")
    ex.add_argument("--out", type=Path, default=OUT_DIR / f"{RUN_ID}.jsonl")
    ex.add_argument("--model-name", default=MODEL_NAME)
    cp = sub.add_parser("compare")
    cp.add_argument("--logs", type=Path, required=True, help="the harness's run_instance_swt_logs dir")
    cp.add_argument("--predictions", type=Path, default=OUT_DIR / f"{RUN_ID}.jsonl")
    cp.add_argument("--run-id", default=RUN_ID)
    cp.add_argument("--model-name", default=MODEL_NAME)
    ig = sub.add_parser("ingest")
    ig.add_argument("--logs", type=Path, required=True, help="the harness's run_instance_swt_logs dir")
    ig.add_argument("--predictions", type=Path, default=OUT_DIR / f"{RUN_ID}.jsonl")
    ig.add_argument("--run-id", required=True)
    ig.add_argument("--model-name", default=MODEL_NAME)
    ig.add_argument("--store", type=Path, default=STORE)
    ig.add_argument("--allow-no-meta", action="store_true",
                    help="accept a report without 18_swt_score.py metadata (patch hash still checked)")
    args = ap.parse_args()
    if args.cmd == "export":
        export(None if args.all else args.ids, args.out, args.model_name)
    elif args.cmd == "ingest":
        ingest(args.logs, args.predictions, args.run_id, args.model_name, args.store, args.allow_no_meta)
    else:
        compare(args.logs, args.predictions, args.run_id, args.model_name)


if __name__ == "__main__":
    main()
