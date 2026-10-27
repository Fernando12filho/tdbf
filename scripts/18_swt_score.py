#!/usr/bin/env python3
"""Score SWT-bench predictions on the prebuilt SWE-bench images.

Run this from the SWT-bench harness checkout (github.com/logic-star-ai/swt-bench),
with that checkout's venv. It works on an x86_64 host and on Apple Silicon
(Colima): the harness picks arm64 images on an arm64 host and would then build
them from scratch, so this script makes it see x86_64 and run the prebuilt
amd64 images under emulation, as our own pipeline does.

  cd ~/swt-bench && source .venv/bin/activate
  python ~/tdbf/scripts/18_swt_score.py \
      --predictions ~/tdbf/runs/swt/trial.jsonl \
      --ids django__django-11999 sympy__sympy-20590 pytest-dev__pytest-5227

Standalone on purpose: it imports the harness's own `src` package to compute
image names, not tdbf.

Why not let the harness build its images: it builds base/env/instance images
from scratch (conda solves against today's package versions, ~120 GB
recommended), which is the dependency drift the prebuilt SWE-bench images avoid.
It only checks that an image with the expected name exists, so for each instance
this script pulls the prebuilt image and tags it under the three names the
harness looks for:

  exec.base.x86_64:latest
  exec.env.x86_64.<env_hash>:latest
  exec.eval.x86_64.<env_hash>.<instance_hash>:latest

The harness deletes an env or instance image that is older than the image it is
based on. All three tags point at one image, so their creation times are equal
and nothing is deleted. The base tag is shared by every instance, which is why
instances run strictly one at a time and the tags are removed after each.

Per instance:
  1. pull the prebuilt image (linux/amd64)
  2. tag it under the three harness names
  3. gold run (--predictions_path gold): checks the image grades the gold tests
     as success, and gives the coverage baseline
  4. prediction run on our test
  5. remove the three tags; with --rm-image also the prebuilt image
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Lite"
GOLD_RUN_ID = "validate-lite-gold-1"  # the name src.report looks for
RUN_ID = "tdbf-test-first"


def registry_image(instance_id: str) -> str:
    # same naming as tdbf.harness.runner.registry_image
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}:latest"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


# Runs src.main with platform.machine() reporting x86_64, so make_exec_spec
# names the x86_64 images we tagged (it reads the host arch, see src/exec_spec.py).
_AS_X86 = ("import platform, runpy, sys; platform.machine = lambda: 'x86_64'; "
           "sys.argv[0] = 'src.main'; runpy.run_module('src.main', run_name='__main__')")


def harness(predictions: str, instance_id: str, run_id: str, workers: int) -> int:
    return sh([sys.executable, "-c", _AS_X86,
               "--dataset_name", DATASET, "--filter_swt",
               "--predictions_path", predictions,
               "--instance_ids", instance_id,
               "--run_id", run_id,
               "--max_workers", str(workers),
               "--cache_level", "instance", "--clean", "False"]).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--ids", nargs="*", help="default: every instance in the predictions file")
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--skip-gold", action="store_true", help="no gold run (no coverage, no image check)")
    ap.add_argument("--rm-image", action="store_true", help="delete the prebuilt image after each instance")
    ap.add_argument("--max-workers", type=int, default=1)
    args = ap.parse_args()

    if not Path("src/exec_spec.py").exists():
        raise SystemExit("run this from the SWT-bench checkout (it imports the harness's src package)")
    sys.path.insert(0, ".")
    import platform
    platform.machine = lambda: "x86_64"  # same override as _AS_X86
    from src.dataset import load_swebench_dataset
    from src.exec_spec import make_exec_spec

    preds = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    ids = args.ids or [p["instance_id"] for p in preds]
    dataset = {i["instance_id"]: i for i in load_swebench_dataset(DATASET, "test", filter_swt=True)}
    missing = [i for i in ids if i not in dataset]
    if missing:
        raise SystemExit(f"not in SWT-bench Lite: {' '.join(missing)}")

    summary = {}
    for iid in ids:
        spec = make_exec_spec(dataset[iid])
        tags =[spec.base_image_key, spec.env_image_key, spec.instance_image_key]
        src_image = registry_image(iid)
        print(f"\n=== {iid}\n    {src_image}\n    -> " + "\n    -> ".join(tags), flush=True)

        local = sh(["docker", "image", "inspect", src_image], capture_output=True).returncode == 0
        if not local and sh(["docker", "pull", "--platform", "linux/amd64", src_image]).returncode != 0:
            summary[iid] = "pull failed"
            continue
        try:
            for t in tags:
                sh(["docker", "tag", src_image, t], check=True)
            gold = None if args.skip_gold else harness("gold", iid, GOLD_RUN_ID, args.max_workers)
            pred = harness(str(args.predictions.resolve()), iid, args.run_id, args.max_workers)
            summary[iid] = f"gold exit {gold}, prediction exit {pred}"
        finally:
            for t in tags:
                sh(["docker", "rmi", t], capture_output=True)
            if args.rm_image and not local:  # never delete an image that was here before
                sh(["docker", "rmi", src_image], capture_output=True)

    print("\nsummary")
    for iid, s in summary.items():
        print(f"  {iid:40s} {s}")
    print("\nreports: run_instance_swt_logs/<run_id>/<model>/<instance>/report.json"
          "\nthen, from the tdbf checkout:"
          "\n  python3 scripts/17_swt_predictions.py compare --logs <swt-bench>/run_instance_swt_logs"
          f" --predictions {args.predictions}")


if __name__ == "__main__":
    main()
