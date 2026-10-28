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
from datetime import datetime, timezone
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Lite"
GOLD_RUN_ID = "validate-lite-gold-1"  # the name src.report looks for
RUN_ID = "tdbf-test-first"
PULL_TRIES = 3
META_DIR = "tdbf_run_meta"  # inside the harness checkout, next to run_instance_swt_logs
TDBF_ROOT = Path(__file__).resolve().parent.parent  # for --ingest


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


def ingest(args, instance_id: str) -> None:
    """Store one finished instance in the tdbf repo. Ingest re-checks the patch
    hash and the report's freshness and never stores a verdict twice, so calling
    it per instance is safe. A failure is reported, not fatal: the reports stay
    in run_instance_swt_logs and a later manual ingest picks them up.
    Committing swt_results.jsonl stays a manual step."""
    script = TDBF_ROOT / "scripts" / "17_swt_predictions.py"
    rc = sh([args.ingest_python, str(script), "ingest",
             "--logs", str(Path("run_instance_swt_logs").resolve()),
             "--predictions", str(args.predictions.resolve()),
             "--run-id", args.run_id, "--ids", instance_id]).returncode
    if rc != 0:
        print(f"    ingest of {instance_id} failed (exit {rc}); run 17 ingest by hand later", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--ids", nargs="*", help="default: every instance in the predictions file")
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--skip-gold", action="store_true", help="no gold run (no coverage, no image check)")
    ap.add_argument("--rm-image", action="store_true", help="delete the prebuilt image after each instance")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--ingest", action="store_true",
                    help="after each instance, store its verdict in the tdbf repo's swt_results.jsonl "
                         "(17_swt_predictions.py ingest), so the webapp /swt page updates during the run")
    ap.add_argument("--ingest-python", default="python3",
                    help="a python with the tdbf package installed (the harness venv does not have it)")
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
        print(f"skipped, not in SWT-bench Lite ({len(missing)}): {' '.join(missing)}")
    ids = [i for i in ids if i in dataset]
    # images already on disk first, so a long run makes progress before any pull
    def on_disk(iid):
        return sh(["docker", "image", "inspect", registry_image(iid)], capture_output=True).returncode == 0
    ids.sort(key=lambda i: not on_disk(i))

    summary = {}
    for iid in ids:
        spec = make_exec_spec(dataset[iid])
        tags =[spec.base_image_key, spec.env_image_key, spec.instance_image_key]
        src_image = registry_image(iid)
        print(f"\n=== {iid}\n    {src_image}\n    -> " + "\n    -> ".join(tags), flush=True)

        local = sh(["docker", "image", "inspect", src_image], capture_output=True).returncode == 0
        # a dropped connection ("short read ... unexpected EOF") is common on big
        # layers; docker keeps finished layers, so a retry resumes
        if not local and not any(
                sh(["docker", "pull", "--platform", "linux/amd64", src_image]).returncode == 0
                for _ in range(PULL_TRIES)):
            summary[iid] = "pull failed"
            continue
        digest = sh(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", src_image],
                    capture_output=True, text=True).stdout.strip()
        started = datetime.now(timezone.utc).isoformat()
        try:
            for t in tags:
                sh(["docker", "tag", src_image, t], check=True)
            gold = None if args.skip_gold else harness("gold", iid, GOLD_RUN_ID, args.max_workers)
            pred = harness(str(args.predictions.resolve()), iid, args.run_id, args.max_workers)
            summary[iid] = f"gold exit {gold}, prediction exit {pred}"
            # read by `17_swt_predictions.py ingest`: which image graded this
            # instance, and when, so a report older than this run is refused
            meta = Path(META_DIR) / args.run_id / f"{iid}.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps({
                "instance_id": iid, "run_id": args.run_id, "image": src_image,
                "repo_digests": json.loads(digest or "[]"), "tags": tags,
                "gold_exit": gold, "prediction_exit": pred, "started": started,
                "finished": datetime.now(timezone.utc).isoformat(),
                "harness_commit": sh(["git", "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            }, indent=1))
            if args.ingest:
                ingest(args, iid)
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
