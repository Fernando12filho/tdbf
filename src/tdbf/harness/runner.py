"""Thin wrapper around `swebench.harness.run_evaluation`.

We do NOT reimplement container provisioning or test scoring. The
official harness already: builds an isolated per-instance Docker image,
checks out base_commit, applies a candidate patch, runs the instance's
FAIL_TO_PASS / PASS_TO_PASS commands, and writes a structured report.

This module's only jobs:
  1. Write the predictions.jsonl the harness expects.
  2. Invoke the harness scoped to one or a few instance_ids.
  3. Read back the per-instance report JSON.

A generated test is delivered to the harness as a *patch that adds a
test file* (not as a code fix). Applying that patch on the buggy tree
and on the gold-patched tree is what produces the fail->pass signal the
BRT validator interprets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def make_test_patch(test_path: str, test_code: str) -> str:
    """Build a unified diff that ADDS a new test file.

    Using /dev/null as the old file makes this a pure addition, so it
    applies cleanly on both the buggy tree and the gold-patched tree
    (the gold patch touches product code, not this new file).
    """
    body = test_code if test_code.endswith("\n") else test_code + "\n"
    lines = body.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    hunk = "".join(f"+{ln}\n" for ln in lines)
    return (
        f"diff --git a/{test_path} b/{test_path}\n"
        f"new file mode 100644\n"
        f"--- /dev/null\n"
        f"+++ b/{test_path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{hunk}"
    )


@dataclass
class PredictionRecord:
    instance_id: str
    model_name_or_path: str
    model_patch: str

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


def write_predictions(
    records: list[PredictionRecord], path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json()) + "\n")
    return path


@dataclass
class HarnessResult:
    instance_id: str
    resolved: bool
    fail_to_pass_passed: list[str]
    fail_to_pass_failed: list[str]
    pass_to_pass_passed: list[str]
    pass_to_pass_failed: list[str]
    applied: bool
    error: str | None = None


def run_harness(
    predictions_path: Path,
    dataset_split: str,
    run_id: str,
    instance_ids: list[str],
    *,
    max_workers: int = 1,
    build_locally: bool = True,
) -> int:
    """Invoke the official harness. Returns the process exit code.

    `build_locally=True` adds `--namespace ''`, which the SWE-bench docs
    specify to build evaluation images locally instead of pulling the
    prebuilt registry images.
    """
    dataset_name = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
    }[dataset_split]

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--instance_ids",
        *instance_ids,
    ]
    if build_locally:
        cmd += ["--namespace", ""]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:])
    return proc.returncode


def read_report(
    run_id: str,
    model_name: str,
    instance: "object",
) -> HarnessResult:
    """Parse the harness's per-instance report.

    The harness writes a run-level report and per-instance logs under
    `logs/run_evaluation/<run_id>/<model>/<instance_id>/`. The exact
    schema has shifted across harness versions, so this reads defensively
    and records what it could and could not find rather than assuming a
    fixed layout. Wire to the concrete path after the first real run
    (scripts/01_validate_gold.py exercises this and prints the tree).
    """
    raise NotImplementedError(
        "Implement against the real report tree after the first "
        "harness run. scripts/01_validate_gold.py is built to surface "
        "that tree so this is filled in against ground truth, not a "
        "guessed schema."
    )
