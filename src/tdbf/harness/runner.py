"""Thin wrapper around `swebench.harness.run_evaluation`.

We do NOT reimplement container provisioning or test scoring. The
official harness already: builds an isolated per-instance Docker image,
checks out base_commit, applies a candidate patch, runs the instance's
FAIL_TO_PASS / PASS_TO_PASS commands, and writes a structured report.

This module's jobs:
  1. Write the predictions.jsonl the harness expects.
  2. Invoke the harness scoped to one or a few instance_ids.
  3. Read back the per-instance report JSON.
  4. Run a generated test patch directly inside the instance container
     (bypassing eval.sh) to get the fail->pass signal for BRT classification.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Generated test is dropped at the repo root so pytest can discover it
# without needing to know the per-repo test directory layout.
BRT_TEST_PATH = "test_brt.py"


def make_test_patch(test_path: str, test_code: str) -> str:
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
    cache_level: str = "instance",
) -> int:
    # cache_level="instance" keeps the eval Docker image after the run so
    # run_test_in_container can reuse it. Default "env" would remove it.

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
        "--cache_level",
        cache_level,
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
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    report_path = (
        Path("logs") / "run_evaluation" / run_id / model_name / instance_id / "report.json"
    )

    if not report_path.exists():
        return HarnessResult(
            instance_id=instance_id,
            resolved=False,
            fail_to_pass_passed=[],
            fail_to_pass_failed=[],
            pass_to_pass_passed=[],
            pass_to_pass_failed=[],
            applied=False,
            error=f"report not found: {report_path}",
        )

    data = json.loads(report_path.read_text())
    inst = data[instance_id]
    ftp = inst.get("tests_status", {}).get("FAIL_TO_PASS", {})
    ptp = inst.get("tests_status", {}).get("PASS_TO_PASS", {})

    return HarnessResult(
        instance_id=instance_id,
        resolved=inst.get("resolved", False),
        fail_to_pass_passed=ftp.get("success", []),
        fail_to_pass_failed=ftp.get("failure", []),
        pass_to_pass_passed=ptp.get("success", []),
        pass_to_pass_failed=ptp.get("failure", []),
        applied=inst.get("patch_successfully_applied", False),
    )


def _parse_pytest_outcome(returncode: int, output: str) -> str:
    # git apply failure — patch couldn't be applied to the tree
    if "error: patch failed" in output or "patch does not apply" in output:
        return "absent"
    # pytest exit codes: 0=pass, 1=fail, 5=no tests collected
    if returncode == 0:
        return "pass"
    if returncode == 1:
        return "fail"
    if returncode == 5:
        return "absent"
    # exit 2 (interrupted), 3 (internal error), 4 (bad CLI), or
    # docker/git errors all land here
    return "error"


def run_test_in_container(
    instance: "object",
    test_patch: str,
    *,
    gold_patch: str | None = None,
    timeout: int = 120,
    verbose: bool = False,
) -> str:
    """Apply test_patch inside the instance's Docker image and run pytest.

    Runs twice from the pipeline:
      - gold_patch=None  → buggy tree  → expected "fail"
      - gold_patch=<str> → fixed tree  → expected "pass"

    Returns one of: "pass", "fail", "error", "absent".

    The Docker image must already exist (built by run_harness / gold
    validation). Colima users: ensure DOCKER_HOST is set in the shell
    before calling this, or the subprocess docker call will fail.
    """
    instance_id: str = instance.instance_id  # type: ignore[attr-defined]
    image_name = f"sweb.eval.x86_64.{instance_id}:latest"

    # Colima (macOS Docker) only mounts ~/ into the VM by default.
    # /tmp on macOS is separate from the VM's /tmp, so always use home dir.
    with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "test.patch").write_text(test_patch)

        script_lines = [
            "set -e",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
        ]
        if gold_patch is not None:
            (tmp / "gold.patch").write_text(gold_patch)
            script_lines.append("git apply /patches/gold.patch")
        script_lines.append("git apply /patches/test.patch")
        script_lines.append(f"pytest {BRT_TEST_PATH} -x --tb=short -q")
        script = "\n".join(script_lines)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/patches:ro",
            image_name,
            "bash", "-c", script,
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return "error"

        if "Unable to find image" in proc.stderr:
            raise RuntimeError(
                f"Docker image {image_name!r} not found. "
                "Run scripts/01_validate_gold.py first to build it."
            )

        outcome = _parse_pytest_outcome(
            proc.returncode, proc.stdout + proc.stderr
        )
        if verbose or outcome == "error":
            label = "FIXED" if gold_patch is not None else "BUGGY"
            sys.stderr.write(
                f"\n--- container output [{label}] (rc={proc.returncode}) ---\n"
                + (proc.stdout + proc.stderr)[-3000:]
                + "\n---\n"
            )
        return outcome
