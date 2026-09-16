"""Re-score stored fix verdicts that belong to a different patch.

Until 2026-09-16 every fix was scored under the fixed run id `agent_fix_<id>`.
The SWE-bench harness skips an instance whose report already exists for that run
id and model, so every re-run of an instance silently inherited the FIRST run's
verdict. The audit that day found 20 of 64 stored fix verdicts belonging to an
older, different patch.

This re-scores exactly those records — no model calls, Docker only — under a
fresh, verified run id, and updates them in place. The superseded value is kept
beside the new one (`resolved_stale`) so the correction stays auditable, and git
holds the file's previous state.

**Run it alone.** It rewrites the whole of results.jsonl after each re-score, so a
pipeline that appends a record meanwhile would have that record overwritten.

    python3 scripts/rescore_stale.py --dry-run          # list what is stale
    python3 scripts/rescore_stale.py --limit 1          # validate on one first
    python3 scripts/rescore_stale.py                    # all of them
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent.pipeline import score_fix_patch
from tdbf.data.instances import InstanceStore
from tdbf.results import RESULTS_FILE

LEGACY_RUN = "agent_fix_{iid}"


def _legacy_scored_patch(rec: dict) -> str | None:
    """The patch the harness actually scored under the old fixed run id."""
    iid = rec["instance_id"]
    d = (Path("logs") / "run_evaluation" / LEGACY_RUN.format(iid=iid)
         / rec["model"].replace("/", "__") / iid)
    p = d / "patch.diff"
    return p.read_text().strip() if p.exists() else None


def _is_stale(rec: dict) -> bool:
    if rec.get("mode") != "fix_agent" or rec.get("rescored_at"):
        return False
    patch = (rec["attempts"][0].get("fix_patch") or "").strip()
    if not patch:
        return False  # nothing was scored; "not resolved" is correct
    return _legacy_scored_patch(rec) != patch


def _save(lines: list[dict]) -> None:
    tmp = RESULTS_FILE.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in lines))
    tmp.replace(RESULTS_FILE)  # atomic: a crash never leaves a half-written store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--id", default=None, help="only this instance_id")
    args = ap.parse_args()

    records = [json.loads(l) for l in RESULTS_FILE.read_text().splitlines() if l.strip()]
    todo = [i for i, r in enumerate(records)
            if _is_stale(r) and (args.id is None or r["instance_id"] == args.id)]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} stale fix verdict(s) to re-score")

    stores: dict[str, InstanceStore] = {}
    changed = 0
    for n, i in enumerate(todo, 1):
        rec = records[i]
        grp = (rec.get("flags") or {}).get("run_group") or "-"
        print(f"\n[{n}/{len(todo)}] {rec['instance_id']}  {rec['model']}  group={grp}  "
              f"stored resolved={rec.get('resolved')}")
        if args.dry_run:
            continue
        split = rec.get("split") or "lite"
        store = stores.setdefault(split, InstanceStore(split))
        inst = store.get(rec["instance_id"])
        patch = rec["attempts"][0]["fix_patch"]
        harness, err = score_fix_patch(inst, rec["model"], patch, split=split,
                                       run_dir=Path("runs") / "rescore", tag="rescore")
        if err:
            print(f"  [SKIPPED] could not verify: {err}")
            continue
        old = rec.get("resolved")
        rec["resolved_stale"] = old
        rec["resolved"] = harness.resolved
        rec["best_level"] = 4 if harness.resolved else 0
        rec["rescored_at"] = datetime.now(timezone.utc).isoformat()
        att = rec["attempts"][0]
        att["resolved"] = harness.resolved
        att["applied"] = harness.applied
        att["fail_to_pass_failed"] = harness.fail_to_pass_failed
        att["pass_to_pass_failed"] = harness.pass_to_pass_failed
        _save(records)
        flip = "  <-- CHANGED" if bool(old) != bool(harness.resolved) else ""
        changed += bool(old) != bool(harness.resolved)
        print(f"  resolved: {old} -> {harness.resolved}{flip}")

    if not args.dry_run:
        print(f"\nDone. {changed} verdict(s) changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
