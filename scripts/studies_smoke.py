"""Offline checks for the study registry (src/tdbf/studies.py) and the write guard.

No API, no Docker. Also checks the REAL results.jsonl against the registry, so a
record that slipped in under the wrong model or an unregistered group fails here.

    python3 scripts/studies_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import metrics, results, studies  # noqa: E402
from tdbf.studies import GPT_4O_MINI, NEMOTRON_FREE, StudyError  # noqa: E402


def _refused(fn, *args, contains: str) -> None:
    try:
        fn(*args)
    except StudyError as e:
        assert contains in str(e), (contains, str(e))
    else:
        raise AssertionError(f"expected a refusal mentioning {contains!r}")


def test_real_log_fits_the_registry() -> None:
    recs = metrics.load_records(ROOT / "results.jsonl")
    bad = studies.violations(recs)
    assert not bad, f"{len(bad)} record(s) outside the registry:\n  " + "\n  ".join(bad[:10])
    assert recs, "results.jsonl is empty?"


def test_every_run_group_in_one_study() -> None:
    seen = {}
    for s in studies.STUDIES:
        for g in s.run_groups:
            assert g not in seen, (g, seen.get(g), s.name)
            seen[g] = s.name
    assert set(studies.get("rq1-nemotron").run_groups) == metrics.EXPERIMENT_RUN_GROUPS


def test_check_write() -> None:
    _refused(studies.check_write, None, NEMOTRON_FREE, contains="needs a run group")
    _refused(studies.check_write, "made-up-group", NEMOTRON_FREE, contains="not registered")
    _refused(studies.check_write, "rq1-ext-v2", NEMOTRON_FREE, contains="frozen")
    _refused(studies.check_write, "tests-4omini", NEMOTRON_FREE, contains="One study, one model")
    assert studies.check_write("tests-4omini", GPT_4O_MINI).name == "test-agent-gpt-4o-mini"


def test_records_of_refuses_a_foreign_model() -> None:
    ok = {"instance_id": "a", "model": NEMOTRON_FREE, "flags": {"run_group": "rq1-sweep"}}
    other_study = {"instance_id": "a", "model": GPT_4O_MINI,
                   "flags": {"run_group": "swt-4omini-pilot"}}
    assert studies.records_of("rq1-nemotron", [ok, other_study]) == [ok]
    intruder = {**ok, "model": GPT_4O_MINI, "mode": "agent_deployable", "timestamp": "x"}
    _refused(studies.records_of, "rq1-nemotron", [ok, intruder], contains="another model")
    # and the experiment's metrics refuse it too, instead of pairing it in
    _refused(metrics.experiment_tests, [ok, intruder], contains="another model")


def test_save_run_guard() -> None:
    with tempfile.TemporaryDirectory() as td:
        saved_file, saved_default = results.RESULTS_FILE, results._DEFAULT_RUN_GROUP
        results.RESULTS_FILE = Path(td) / "results.jsonl"
        results._try_post = lambda record: None  # never reach a running webapp
        try:
            rec = {"instance_id": "a", "model": GPT_4O_MINI, "flags": {}}
            results._DEFAULT_RUN_GROUP = None
            _refused(results.save_run, rec, contains="needs a run group")
            _refused(results.save_run, {**rec, "flags": {"run_group": "rq1-sweep"}},
                     contains="frozen")
            assert not results.RESULTS_FILE.exists(), "a refused record was written"

            # use_run_group checks up front, then tags records that carry no group
            _refused(results.use_run_group, "rq1-sweep", NEMOTRON_FREE, contains="frozen")
            results.use_run_group("tests-4omini", GPT_4O_MINI)
            results.save_run(rec)
            written = [json.loads(l) for l in results.RESULTS_FILE.read_text().splitlines()]
            assert [r["flags"]["run_group"] for r in written] == ["tests-4omini"], written
            assert rec["flags"] == {}, "save_run must not mutate the caller's record"
        finally:
            results.RESULTS_FILE, results._DEFAULT_RUN_GROUP = saved_file, saved_default


def test_load_best_brt_stays_in_its_study() -> None:
    def rec(model, group, level, ts):
        return {"instance_id": "x__x-1", "model": model, "timestamp": ts,
                "flags": {"run_group": group},
                "attempts": [{"level": level, "test_code": f"# {model} {group}"}]}
    with tempfile.TemporaryDirectory() as td:
        saved = results.RESULTS_FILE
        results.RESULTS_FILE = Path(td) / "results.jsonl"
        try:
            results.RESULTS_FILE.write_text("".join(json.dumps(r) + "\n" for r in [
                rec(NEMOTRON_FREE, "rq1-sweep", 4, "2026-09-01"),
                rec(GPT_4O_MINI, "tests-4omini", 4, "2026-11-04"),       # newer, other model
                rec(NEMOTRON_FREE, "rq1-ext", 3, "2026-10-01"),          # same study, L3
            ]))
            s = studies.get("rq1-nemotron")
            got = results.load_best_brt("x__x-1", run_groups=s.run_groups, models=s.models)
            assert got["model"] == NEMOTRON_FREE and got["run_group"] == "rq1-sweep", got
            other = results.load_best_brt("x__x-1", run_groups=("tests-4omini",))
            assert other["model"] == GPT_4O_MINI, other
            assert results.load_best_brt("x__x-1", run_groups=("nowhere",)) is None
        finally:
            results.RESULTS_FILE = saved


def main() -> int:
    tests = [test_real_log_fits_the_registry, test_every_run_group_in_one_study,
             test_check_write, test_records_of_refuses_a_foreign_model, test_save_run_guard,
             test_load_best_brt_stays_in_its_study]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nStudy registry: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
