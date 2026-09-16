"""Offline checks for the offline-scoring guard (the gold barrier's last step).

The agent smokes stub the SWE-bench harness and switch the verifier off, so this
file exercises the REAL `_verify_scored` and `report_dir` against files on disk.
Two bugs it pins down, both found 2026-09-16:

  - The harness skips an instance whose report.json already exists for the same
    run id + model. With a fixed run id every re-run re-read the FIRST run's
    verdict: 20 of 64 stored fix verdicts belonged to a different patch, and a
    reactive-vs-test-first pair would always have tied.
  - The harness names the model directory with "/" replaced by "__"; reading the
    raw name found nothing for any OpenRouter id and a missing report reads as
    "not resolved".

    python3 scripts/agent_scoring_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.agent import pipeline
from tdbf.harness.runner import read_report, report_dir

MODEL = "openrouter:nvidia/nemotron-3-super-120b-a12b:free"
IID = "demo__pkg-1"
PATCH = "diff --git a/pkg/core.py b/pkg/core.py\n-    return 4\n+    return 5\n"


def _write_report(run_id: str, *, patch: str = PATCH, resolved: bool = True,
                  mtime: float | None = None) -> Path:
    d = report_dir(run_id, MODEL, IID)
    d.mkdir(parents=True, exist_ok=True)
    (d / "patch.diff").write_text(patch)
    rep = d / "report.json"
    rep.write_text(json.dumps({IID: {
        "resolved": resolved, "patch_successfully_applied": True,
        "tests_status": {"FAIL_TO_PASS": {"success": ["t"], "failure": []},
                         "PASS_TO_PASS": {"success": [], "failure": []}},
    }}))
    if mtime is not None:
        os.utime(rep, (mtime, mtime))
    return d


def test_model_dir_uses_harness_naming() -> None:
    d = report_dir("r1", MODEL, IID)
    assert "/" not in d.parts[3], d            # the model segment
    assert d.parts[3] == "openrouter:nvidia__nemotron-3-super-120b-a12b:free", d.parts[3]


def test_read_report_finds_a_slashed_model() -> None:
    _write_report("r_slash", resolved=True)
    h = read_report("r_slash", MODEL, SimpleNamespace(instance_id=IID))
    assert h.error is None, h.error
    assert h.resolved is True


def test_verify_accepts_a_fresh_report_for_this_patch() -> None:
    started = time.time()
    _write_report("r_ok")
    assert pipeline._verify_scored("r_ok", MODEL, IID, PATCH, started) is None


def test_verify_rejects_a_missing_report() -> None:
    err = pipeline._verify_scored("r_none", MODEL, IID, PATCH, time.time())
    assert err and "no report" in err, err


def test_verify_rejects_a_stale_report() -> None:
    # the harness skipped: the report on disk was written by an earlier run
    _write_report("r_stale", mtime=time.time() - 86_400)
    err = pipeline._verify_scored("r_stale", MODEL, IID, PATCH, time.time())
    assert err and "predates" in err, err


def test_verify_rejects_a_different_patch() -> None:
    started = time.time()
    _write_report("r_other", patch=PATCH.replace("return 5", "return 6"))
    err = pipeline._verify_scored("r_other", MODEL, IID, PATCH, started)
    assert err and "not the submitted patch" in err, err


def test_score_fix_patch_never_reuses_a_run_id() -> None:
    """Two scorings of the same instance must never share a report directory."""
    seen = []
    saved = (pipeline.run_harness, pipeline.read_report, pipeline.write_predictions,
             pipeline._verify_scored)
    try:
        pipeline.write_predictions = lambda recs, path: Path(path)
        pipeline.run_harness = lambda pred, split, run_id, ids: seen.append(run_id) or 0
        pipeline.read_report = lambda run_id, model, inst: SimpleNamespace(resolved=True)
        pipeline._verify_scored = lambda *a, **k: None
        inst = SimpleNamespace(instance_id=IID)
        for _ in range(2):
            pipeline.score_fix_patch(inst, MODEL, PATCH, split="lite", run_dir=Path("runs"))
    finally:
        (pipeline.run_harness, pipeline.read_report, pipeline.write_predictions,
         pipeline._verify_scored) = saved
    assert len(set(seen)) == 2, seen


def main() -> int:
    tests = [
        test_model_dir_uses_harness_naming,
        test_read_report_finds_a_slashed_model,
        test_verify_accepts_a_fresh_report_for_this_patch,
        test_verify_rejects_a_missing_report,
        test_verify_rejects_a_stale_report,
        test_verify_rejects_a_different_patch,
        test_score_fix_patch_never_reuses_a_run_id,
    ]
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)  # report_dir is relative to cwd ("logs/run_evaluation/...")
        try:
            for t in tests:
                t()
                print(f"  ok  {t.__name__}")
        finally:
            os.chdir(cwd)
    print(f"\nOffline scoring guard: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
