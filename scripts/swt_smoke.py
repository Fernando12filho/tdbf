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

from tdbf import metrics as M, studies  # noqa: E402

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
                labels[iid] = {"repo": "demo/pkg", "run_group": "rq1-sweep",
                               "model": studies.NEMOTRON_FREE, "level": c["level"],
                               "outcome_buggy": "fail", "outcome_fixed": "pass",
                               "failure_reason_buggy": "assertion", "files": ["test_brt.py"]}
                # the harness's real layout: report.json in <run>/<model>/<id>/,
                # the applied patch in <run>/pred_pre__<model>/<id>/ and pred_post__
                rdir = logs / RUN / MODEL / iid
                rdir.mkdir(parents=True)
                (rdir / "report.json").write_text(json.dumps(c["report"]))
                for phase in ("pred_pre", "pred_post"):
                    pdir = logs / RUN / f"{phase}__{MODEL}" / iid
                    pdir.mkdir(parents=True)
                    wrong = c.get("wrong_patch") and phase == "pred_post"
                    (pdir / "model_patch.diff").write_text(patch + ("# changed\n" if wrong else ""))
                meta = tmp / "harness" / "tdbf_run_meta" / RUN / f"{iid}.json"
                meta.parent.mkdir(parents=True, exist_ok=True)
                started = "2999-01-01T00:00:00+00:00" if c.get("stale") else "2000-01-01T00:00:00+00:00"
                meta.write_text(json.dumps({"started": started, "harness_commit": "abc1234",
                                            "image": "img", "repo_digests": [], "gold_exit": 0}))
        preds.with_suffix(".labels.json").write_text(json.dumps(labels))
        store = tmp / "swt_results.jsonl"

        print("ingest one instance (18_swt_score.py --ingest)")
        one = tmp / "one.jsonl"
        m17.ingest(logs, preds, RUN, MODEL, one, allow_no_meta=False, ids=["i3"])
        check("--ids stores only that instance", [r["instance_id"] for r in M.load_records(one)], ["i3"])

        print("ingest")
        m17.ingest(logs, preds, RUN, MODEL, store, allow_no_meta=False)
        recs = M.load_records(store)
        check("stored", sorted(r["instance_id"] for r in recs), ["i1", "i2", "i3"])
        check("evidence copied", (tmp / "runs" / "swt" / RUN / "i1" / MODEL / "report.json").exists(), True)
        m17.ingest(logs, preds, RUN, MODEL, store, allow_no_meta=False)
        check("second ingest adds nothing", len(M.load_records(store)), 3)

        print("ingest refuses a verdict whose run group / model fit no study")
        bad = tmp / "bad.jsonl"
        lab = json.loads(preds.with_suffix(".labels.json").read_text())
        lab["i3"] = {**lab["i3"], "run_group": "not-registered"}
        bad_preds = tmp / "bad_preds.jsonl"
        bad_preds.write_text(preds.read_text())
        bad_preds.with_suffix(".labels.json").write_text(json.dumps(lab))
        m17.ingest(logs, bad_preds, RUN, MODEL, bad, allow_no_meta=False, ids=["i3"])
        check("unregistered group refused", M.load_records(bad), [])
        lab["i3"] = {**lab["i3"], "run_group": "rq1-sweep", "model": "openrouter:other/model@x"}
        bad_preds.with_suffix(".labels.json").write_text(json.dumps(lab))
        m17.ingest(logs, bad_preds, RUN, MODEL, bad, allow_no_meta=False, ids=["i3"])
        check("wrong model for the study refused", M.load_records(bad), [])
        check("stored verdicts carry study + model",
              {(r["study"], r["agent_model"]) for r in recs},
              {("rq1-nemotron", studies.NEMOTRON_FREE)})

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
        check("empty store", M.build_swt([], published)["n"], 0)

        print("one study per view")
        other = [{**r, "study": "test-agent-gpt-4o-mini-pilot",
                  "swt": {**r["swt"], "success": False, "f2p": False},
                  "timestamp": "2999-01-01T00:00:00"} for r in recs]
        mixed = recs + other  # newer verdicts from another model on the same instances
        check("default view untouched by another study", M.build_swt(mixed)["success"]["label"],
              "2/3 = 0.67")
        check("other study on its own", M.build_swt(mixed, study="test-agent-gpt-4o-mini-pilot")
              ["success"]["label"], "0/3 = 0.00")
        check("both studies listed", [x["name"] for x in M.build_swt(mixed)["studies"]],
              ["rq1-nemotron", "test-agent-gpt-4o-mini-pilot"])
        legacy = [{k: v for k, v in r.items() if k != "study"} for r in recs]
        check("pre-registry verdicts placed by their run group", M.build_swt(legacy)["n"], 3)

        print("coverage delta, aggregated like the harness's report")
        def rec(iid, success, delta):
            return {"instance_id": iid, "kind": "tdbf_test_first", "repo": "demo/pkg",
                    "study": "rq1-nemotron",
                    "ours": {"level": 4 if success else 3},
                    "swt": {"applied": True, "success": success, "f2p": success,
                            "coverage_delta_pred": delta}}
        cov = M.build_swt([rec("a", True, 0.8), rec("b", True, 0.4),
                           rec("c", False, 0.3), rec("d", False, None)])["coverage"]
        # d has no executable changed lines: out of the denominator, not a zero
        check("delta all", (round(cov["all"]["value"], 4), cov["all"]["den"]), (0.5, 3))
        check("delta S", (round(cov["s"]["value"], 4), cov["s"]["den"]), (0.6, 2))
        check("delta not S", (round(cov["not_s"]["value"], 4), cov["not_s"]["den"]), (0.3, 1))

        print("leaderboard")
        board = [{"tool": "Hi", "success_pct": 90.0, "coverage_delta_pct": 50.0, "n": 276},
                 {"tool": "Lo", "success_pct": 10.0, "coverage_delta_pct": 5.0, "n": 276}]
        lb = M.build_swt(recs, published, board)["leaderboard"]
        check("ours inserted by S", [b["tool"] for b in lb],
              ["Hi", "TDBF test-first (ours)", "Lo"])
        check("our row on our N", next(b for b in lb if b["ours"])["n"], 3)
        check("no leaderboard, no rows", M.build_swt(recs, published)["leaderboard"], [])

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
        webapp.LEADERBOARD_FILE = tmp / "board.jsonl"
        webapp.LEADERBOARD_FILE.write_text("\n".join(json.dumps(x) for x in board) + "\n")
        html = client.get("/swt").get_data(as_text=True)
        store.write_text(store.read_text() + "".join(json.dumps(x) + "\n" for x in other))
        page2 = client.get("/swt?study=test-agent-gpt-4o-mini-pilot").get_data(as_text=True)
        check("study tabs", "nav-tabs" in page2 and "test-agent-gpt-4o-mini-pilot" in page2, True)
        check("leaderboard section", "SWT-bench Lite leaderboard" in html and "TDBF test-first (ours)" in html, True)
        check("coverage rows", "Δᶜ S" in html, True)
        webapp.SWT_FILE = tmp / "empty.jsonl"
        check("empty page renders", client.get("/swt").status_code, 200)

    if failures:
        raise SystemExit(f"SWT smoke: {len(failures)} check(s) failed: {', '.join(failures)}")
    print("SWT smoke: all checks passed.")


if __name__ == "__main__":
    main()
