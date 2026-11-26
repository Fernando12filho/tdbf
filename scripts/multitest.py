"""Multi-test reproduction tests study (studies/multi-test/plan.md), on the 50
development bugs with gpt-5-mini: the single agent told to write a main test plus as
many related-case tests as it judges the bug needs (run group mt-brt), with the
offline L0-L4 label. Test writing only: no fix runs (user decision). Scoring against
the stored one-test tests: scripts/20_final_eval.py --set dev --systems ... multi.
Resumes by run group; stops at --max-usd.

    python3 scripts/multitest.py --limit 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf import metrics  # noqa: E402
from tdbf.data.instances import InstanceStore  # noqa: E402
from tdbf.models.llm import build_client  # noqa: E402
from tdbf.results import RESULTS_FILE, save_run, use_run_group  # noqa: E402

MODEL = "openrouter:openai/gpt-5-mini@openai"
BUGS = [ROOT / "studies" / "scouts" / "data" / f for f in
        ("dev_bugs.json", "dev_round4.json", "dev_round5.json")]
SINGLE_GROUPS = ("v2-base", "proc-dev-baseline")
_PROC_REF = "agent/procedure:results.jsonl"


def dev_ids() -> list[str]:
    out = []
    for f in BUGS:
        out += json.loads(f.read_text())["instance_ids"]
    return out


def records() -> list[dict]:
    path = RESULTS_FILE if RESULTS_FILE.is_absolute() else ROOT / RESULTS_FILE
    recs = metrics.load_records(path)
    proc = subprocess.run(["git", "-C", str(ROOT), "show", _PROC_REF],
                          capture_output=True, text=True, check=True).stdout
    return recs + [json.loads(l) for l in proc.splitlines() if '"proc-dev-baseline"' in l]


def latest(recs: list[dict], groups) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in reversed(list(groups)):  # earlier groups in the tuple win
        for r in recs:
            if (r.get("flags") or {}).get("run_group") == g:
                out[r["instance_id"]] = r
    return out


def gate_ok(rec: dict) -> bool:
    a = metrics._best_attempt(rec)
    return a.get("outcome_buggy") == "fail" and a.get("failure_reason_buggy") != "runtime"


def run_brt(inst, client, args, out_dir: Path) -> float | None:
    from tdbf.agent.pipeline import brt_run_record, brt_transcript_dict, run_brt_phase
    phase = run_brt_phase(inst, client, max_iterations=50, max_total_tokens=1_000_000,
                          max_tokens_per_turn=8192, multi_test=True, verbose=args.verbose)
    if phase is None:
        return None
    flags = {"arm": "multi", "multi_test": True, "max_iterations": 50, "max_total_tokens": 1_000_000,
             "max_tokens_per_turn": 8192, "submit_nudge_limit": phase.submit_nudge_limit,
             "run_group": args.run_group}
    rec = brt_run_record(inst, client.name, split="lite", flags=flags, phase=phase)
    rec["gate_accepted"] = phase.reproduces_on_buggy
    save_run(rec)
    (out_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(brt_transcript_dict(inst, client.name, phase), indent=2, default=str))
    print(f"-> L{phase.level} gate={phase.reproduces_on_buggy} cost=${phase.cost_usd}", flush=True)
    return phase.cost_usd



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=4.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    args.run_group = "mt-brt"

    client = build_client(MODEL)
    use_run_group(args.run_group, client.name)
    recs = records()
    ids = args.ids or dev_ids()
    done = set(latest(recs, (args.run_group,)))
    todo = [i for i in ids if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"group={args.run_group} todo={len(todo)}/{len(ids)} "
          f"max_usd={args.max_usd}", flush=True)
    store = InstanceStore("lite")
    out_dir = ROOT / "runs" / "multitest" / args.run_group
    out_dir.mkdir(parents=True, exist_ok=True)
    spent, t0 = 0.0, time.monotonic()
    for iid in todo:
        if spent >= args.max_usd:
            print(f"cost cap reached (${spent:.2f}); stopping.")
            break
        print(f"\n{'=' * 60}\n{iid}", flush=True)
        try:
            inst = store.get(iid)
            spent += run_brt(inst, client, args, out_dir) or 0.0
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            print(f"{iid}: CRASH {type(exc).__name__}: {msg[:300]}", flush=True)
            if "Error code: 402" in msg or "Error code: 403" in msg:
                print("STOP: OpenRouter credits or key limit exhausted")
                return 2
    print(f"\nbatch done: ${spent:.4f}, {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
