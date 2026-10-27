#!/usr/bin/env python3
"""Cross-check our BRT labels against the official SWT-bench harness.

Two subcommands, no API and no Docker:

  export   turn the stored test-first BRTs (the tests behind the Results page)
           into an SWT-bench predictions file, plus a sidecar with our own label
           for each instance
  compare  read the SWT-bench reports for those predictions and put the two
           verdicts side by side

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
import json
from pathlib import Path

from tdbf import ledger, metrics
from tdbf.harness.runner import brt_run_spec, make_test_patch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "swt"
MODEL_NAME = "tdbf__nemotron-3-super"
RUN_ID = "tdbf-test-first"


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
    args = ap.parse_args()
    if args.cmd == "export":
        export(None if args.all else args.ids, args.out, args.model_name)
    else:
        compare(args.logs, args.predictions, args.run_id, args.model_name)


if __name__ == "__main__":
    main()
