"""Build and check the environment for every instance in the frozen sample.

For each instance, with no model calls and no experimental result:

  1. GOLD check — score the instance's own gold patch through the verified scorer.
     This builds the Docker image (kept: cache_level=instance) and confirms our
     harness can score the known-correct answer as resolved. An instance that
     fails here is ENVIRONMENT-INVALID and, per the rule frozen with the sample,
     is excluded before any arm runs.
  2. BRT-mechanics check — in a live session, the path the gold-free gate uses:
     a trivially failing test must come back fail/assertion and a trivially
     passing one pass. Ten of the sample's repos had never been exercised by the
     agents; a repo whose test layout breaks the BRT path must be found now, not
     as a column of spurious L0s in the sweep.

Status is written after every instance to runs/build_sample/status.json;
re-running skips instances already checked. Run it alone or alongside a sweep —
it never writes results.jsonl.

    python3 scripts/13_build_sample.py                     # all of sample_lite_frozen.json
    python3 scripts/13_build_sample.py --limit 1           # validate on one first
    python3 scripts/13_build_sample.py --retry-invalid     # re-check the failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.agent import pipeline
from tdbf.data.instances import InstanceStore

FAILING = "def test_tdbf_env_probe_fails():\n    assert 1 == 2\n"
PASSING = "def test_tdbf_env_probe_passes():\n    assert 1 == 1\n"


def check_instance(inst, split: str, run_dir: Path) -> dict:
    t0 = time.time()
    status: dict = {"instance_id": inst.instance_id, "repo": inst.repo}

    harness, err = pipeline.score_fix_patch(inst, "gold", inst.gold_patch,
                                            split=split, run_dir=run_dir, tag="gold")
    status["gold_resolved"] = bool(harness.resolved) if not err else None
    status["gold_error"] = err

    status["brt_fail_ok"] = status["brt_pass_ok"] = False
    status["brt_error"] = None
    if status["gold_resolved"]:
        try:
            session = pipeline.ContainerSession(inst).start()
        except RuntimeError as e:
            status["brt_error"] = f"session: {e}"
        else:
            try:
                bad = pipeline._score_on_buggy_in_session(session, FAILING)
                good = pipeline._score_on_buggy_in_session(session, PASSING)
                status["brt_fail_ok"] = bad == "fail" and bad.failure_reason == "assertion"
                status["brt_pass_ok"] = good == "pass"
                if not (status["brt_fail_ok"] and status["brt_pass_ok"]):
                    status["brt_error"] = (f"failing test -> {bad} ({bad.failure_reason}); "
                                           f"passing test -> {good}; tail: "
                                           f"{(good.output or bad.output or '')[-300:]!r}")
            finally:
                session.close()

    status["valid"] = bool(status["gold_resolved"] and status["brt_fail_ok"]
                           and status["brt_pass_ok"])
    status["seconds"] = round(time.time() - t0, 1)
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="sample_lite_frozen.json")
    ap.add_argument("--status", default="runs/build_sample/status.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="+", help="check only these sample instances")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="leave these for later (e.g. one whose build exhausts host memory)")
    ap.add_argument("--retry-invalid", action="store_true")
    args = ap.parse_args()

    sample = json.loads((ROOT / args.sample).read_text())
    split = "lite" if "Lite" in sample["dataset"] else "verified"
    status_path = Path(args.status)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    todo = [i for i in sample["instances"]
            if i not in status or (args.retry_invalid and not status[i]["valid"])]
    if args.ids:
        unknown = set(args.ids) - set(sample["instances"])
        if unknown:
            raise SystemExit(f"not in the frozen sample: {sorted(unknown)}")
        todo = [i for i in todo if i in args.ids]
    todo = [i for i in todo if i not in args.skip]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(sample['instances'])} in sample, {len(status)} already checked, "
          f"{len(todo)} to check now")

    # One builder at a time: each rewrites the whole status file after every
    # instance, so two concurrent builders would silently erase each other's work.
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock = status_path.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"another builder holds {lock} (pid {lock.read_text().strip()}); "
                         "if it is not running, delete the lock file and retry")
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        return _build(todo, sample, split, status, status_path)
    finally:
        lock.unlink(missing_ok=True)


def _build(todo, sample, split, status, status_path) -> int:
    store = InstanceStore(split)
    for n, iid in enumerate(todo, 1):
        inst = store.get(iid)
        print(f"\n[{n}/{len(todo)}] {iid} ({inst.repo}) ...", flush=True)
        st = check_instance(inst, split, status_path.parent / "predictions")
        status[iid] = st
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2))
        verdict = "VALID" if st["valid"] else "INVALID"
        print(f"  {verdict}  gold_resolved={st['gold_resolved']}  "
              f"brt fail/pass ok={st['brt_fail_ok']}/{st['brt_pass_ok']}  {st['seconds']}s"
              + (f"\n  gold_error: {st['gold_error']}" if st["gold_error"] else "")
              + (f"\n  brt_error: {st['brt_error']}" if st["brt_error"] else ""), flush=True)

    checked = [status[i] for i in sample["instances"] if i in status]
    invalid = [s for s in checked if not s["valid"]]
    print(f"\n{len(checked)}/{len(sample['instances'])} checked: "
          f"{len(checked) - len(invalid)} valid, {len(invalid)} environment-invalid")
    for s in invalid:
        print(f"  INVALID {s['instance_id']}: gold={s['gold_resolved']} "
              f"{s['gold_error'] or ''} {s['brt_error'] or ''}"[:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
