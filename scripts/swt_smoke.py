"""Offline checks for the SWT-bench re-scoring path: `17_swt_predictions.py
ingest`, tdbf.metrics.build_swt and the webapp /swt page. No API, no Docker.

Five made-up instances, with fake harness reports:

  i1  ours L4, harness S            agree
  i2  ours L4, harness no F->P      disagree (e.g. the runner collected nothing)
  i3  ours L3, harness S            disagree (per-test vs. per-file)
  i4  the harness scored a different patch       ingest must refuse it
  i5  the report predates this run (stale id)    ingest must refuse it

plus one published tool that solved i1 and i2.

    python3 scripts/swt_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

from tdbf import metrics as M  # noqa: E402

RUN, MODEL = "smoke-run", "tdbf__smoke"


def _load_17():
    spec = importlib.util.spec_from_file_location("swt17", ROOT / "scripts" / "17_swt_predictions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(iid, *, f2p=(), f2f=(), resolved=False, applied=True):
    tests = {"FAIL_TO_PASS": list(f2p), "PASS_TO_PASS": [], "FAIL_TO_FAIL": list(f2f),
             "PASS_TO_FAIL": [], "UNMATCHED": []}
    empty = {k: [] for k in tests}
    return {iid: {"patch_successfully_applied": applied, "resolved": resolved,
                  "coverage_pred": 1.0, "coverage_delta_pred": 0.5,
                  "tests_pred": tests, "tests_base": empty, "tests_gold": empty}}


def main() -> None:
    m17 = _load_17()
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        m17.ROOT = tmp  # evidence goes under the temp dir, never the real runs/
        cases = {
            "i1": dict(level=4, report=_report("i1", f2p=["t"], resolved=True)),
            "i2": dict(level=4, report=_report("i2")),
            "i3": dict(level=3, report=_report("i3", f2p=["t"], resolved=True)),
            "i4": dict(level=4, report=_report("i4", f2p=["t"], resolved=True), wrong_patch=True),
            "i5": dict(level=4, report=_report("i5", f2p=["t"], resolved=True), stale=True),
        }
        preds = tmp / "preds.jsonl"
        logs = tmp / "harness" / "run_instance_swt_logs"
        labels = {}
        with preds.open("w") as f:
            for iid, c in cases.items():
                patch = f"diff --git a/test_brt.py b/test_brt.py\n+# {iid}\n"
                f.write(json.dumps({"instance_id": iid, "model_name_or_path": MODEL,
                                    "model_patch": patch}) + "\n")
                labels[iid] = {"repo": "demo/pkg", "run_group": "g", "level": c["level"],
                               "outcome_buggy": "fail", "outcome_fixed": "pass",
                               "failure_reason_buggy": "assertion", "files": ["test_brt.py"]}
                rdir = logs / RUN / MODEL / iid
                rdir.mkdir(parents=True)
                (rdir / "report.json").write_text(json.dumps(c["report"]))
                (rdir / "model_patch.diff").write_text(patch + ("# changed\n" if c.get("wrong_patch") else ""))
                meta = tmp / "harness" / "tdbf_run_meta" / RUN / f"{iid}.json"
                meta.parent.mkdir(parents=True, exist_ok=True)
                started = "2999-01-01T00:00:00+00:00" if c.get("stale") else "2000-01-01T00:00:00+00:00"
                meta.write_text(json.dumps({"started": started, "harness_commit": "abc1234",
                                            "image": "img", "repo_digests": [], "gold_exit": 0}))
        preds.with_suffix(".labels.json").write_text(json.dumps(labels))
        store = tmp / "swt_results.jsonl"

        print("ingest")
        m17.ingest(logs, preds, RUN, MODEL, store, allow_no_meta=False)
        recs = M.load_records(store)
        check("stored", sorted(r["instance_id"] for r in recs), ["i1", "i2", "i3"])
        check("evidence copied", (tmp / "runs" / "swt" / RUN / "i1" / MODEL / "report.json").exists(), True)
        m17.ingest(logs, preds, RUN, MODEL, store, allow_no_meta=False)
        check("second ingest adds nothing", len(M.load_records(store)), 3)

        print("build_swt")
        published = [{"tool": "Demo", "model": "m", "metric": "S", "instances": ["i1", "i2"]}]
        res = M.build_swt(recs, published)
        check("n", res["n"], 3)
        check("S", res["success"]["label"], "2/3 = 0.67")
        check("F->P", res["f2p"]["label"], "2/3 = 0.67")
        check("ours L4", res["ours_l4"]["label"], "2/3 = 0.67")
        check("agreement", res["agreement"]["label"], "1/3 = 0.33")
        check("disagreements", [x["instance_id"] for x in res["disagreements"]], ["i2", "i3"])
        check("cross L4", next(r for r in res["cross"] if r["level"] == "L4")["counts"], [1, 0, 1, 0])
        t = res["tools"][0]
        check("published", (t["theirs"]["num"], t["both"], t["only_ours"], t["only_theirs"]), (2, 1, 1, 1))
        check("empty store", M.build_swt([], published), {"n": 0})

        print("webapp /swt")
        cwd = os.getcwd()
        os.chdir(ROOT / "webapp")
        try:
            import app as webapp
        finally:
            os.chdir(cwd)
        webapp.SWT_FILE, webapp.PUBLISHED_FILE = store, tmp / "none.jsonl"
        client = webapp.app.test_client()
        page = client.get("/swt")
        html = page.get_data(as_text=True)
        check("status", page.status_code, 200)
        check("lists the disagreement", "per-test vs. per-file" in html, True)
        webapp.SWT_FILE = tmp / "empty.jsonl"
        check("empty page renders", client.get("/swt").status_code, 200)

    if failures:
        raise SystemExit(f"SWT smoke: {len(failures)} check(s) failed: {', '.join(failures)}")
    print("SWT smoke: all checks passed.")


if __name__ == "__main__":
    main()
