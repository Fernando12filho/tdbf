"""Step 1 (needs Docker + `pip install swebench datasets`):
Validate the harness against the GOLD patch before trusting any
generated test.

Logic: if you feed the harness the human's known-correct patch for an
instance, it MUST report that instance resolved (all FAIL_TO_PASS now
pass, all PASS_TO_PASS still pass). If your harness invocation cannot
even score the known-good answer correctly, nothing built on top of it
is trustworthy — and the proposal's whole evaluation depends on this
acceptance check being correct.

This script also prints the harness's output log tree, which is what you
use to implement `harness.runner.read_report` against the REAL schema
instead of a guessed one.

Usage:
  pip install swebench datasets
  python scripts/01_validate_gold.py --split lite             # first instance
  python scripts/01_validate_gold.py --split lite --id <inst> # specific one
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.data.instances import InstanceStore
from tdbf.harness.runner import PredictionRecord, run_harness, write_predictions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first)")
    ap.add_argument("--run-id", default="gold_validation")
    args = ap.parse_args()

    store = InstanceStore(args.split)
    inst = store.get(args.id) if args.id else store.first()
    print(f"Instance:    {inst.instance_id}")
    print(f"Repo:        {inst.repo}")
    print(f"Base commit: {inst.base_commit}")
    print(f"FAIL_TO_PASS: {len(inst.fail_to_pass)}  "
          f"PASS_TO_PASS: {len(inst.pass_to_pass)}")

    # Feed the harness the GOLD patch. Expected verdict: resolved.
    rec = PredictionRecord(
        instance_id=inst.instance_id,
        model_name_or_path="GOLD",
        model_patch=inst.gold_patch,
        
    )
    preds = write_predictions(
        [rec], Path("runs") / args.run_id / "predictions.jsonl"
    )
    print(f"\nWrote predictions -> {preds}")
    print("Invoking official SWE-bench harness on the GOLD patch...\n")

    rc = run_harness(
        predictions_path=preds,
        dataset_split=args.split,
        run_id=args.run_id,
        instance_ids=[inst.instance_id],
        max_workers=1,
        build_locally=True,
    )

    print(f"\nHarness exit code: {rc}")
    print("\nHarness output tree (use this to implement read_report):")
    logs = Path("logs")
    if logs.exists():
        subprocess.run(["find", "logs", "-maxdepth", "5", "-type", "f"])
    else:
        print("  (no logs/ dir — harness likely failed before scoring; "
              "read the stderr above)")

    print(
        "\nNEXT: open the report JSON shown above, confirm GOLD is "
        "'resolved', then implement harness.runner.read_report against "
        "that exact schema. Only after gold validates do you trust the "
        "pipeline on generated tests."
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
