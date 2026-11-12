"""Run the procedure agent (strict or soft) or the unchanged baseline on instances.

Phase 1 of the issue-information study (design/procedure_agent.md §9, §10c). Every
agent gets the same model and the same turn budget per attempt (D3: 50; D7:
gpt-5-mini). The baseline is `pipeline.run_brt_phase`, called unchanged with
max_iterations=50 (its own gate retries included, as always).

Each finished instance is labelled offline (L0-L4, gold patch, after the agent),
saved to results.jsonl under its run group (which must be registered in
src/tdbf/studies.py), and its transcript written to runs/procedure/<group>/<id>.json.
Instances already saved under the run group are skipped (resume). A cost cap stops
the batch before the next instance once the billed total reaches --max-usd.

    python3 scripts/procedure_brt.py --agent strict --round-file studies/issue-info-tests/data/dev/round1.json \\
        --model openrouter:openai/gpt-5-mini@openai --run-group proc-dev-r1-strict
    python3 scripts/procedure_brt.py ... --dry-run          # plan only, no Docker, no calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.data.instances import InstanceStore  # noqa: E402
from tdbf.models.llm import build_client  # noqa: E402
from tdbf.results import RESULTS_FILE, save_run, use_run_group  # noqa: E402


def done_ids(run_group: str) -> set[str]:
    path = ROOT / RESULTS_FILE if not RESULTS_FILE.is_absolute() else RESULTS_FILE
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("flags") or {}).get("run_group") == run_group:
                out.add(r["instance_id"])
    return out


def procedure_record(inst, model: str, split: str, flags: dict, res, label) -> dict:
    buggy, fixed, cls = label if label else (None, None, None)
    s = res.record
    attempt = {
        "attempt": 1, "mode": s["mode"],
        "level": cls.level.value if cls else 0,
        "l3_subtype": cls.l3_subtype if cls else None,
        "stop_cause": res.stop_cause, "iterations": res.turns_used,
        "truncated_turns": sum(p["truncated_turns"] for p in s["phases"]),
        "num_tool_calls": sum(len(st["tools"]) for p in s["phases"] for st in p["steps"]),
        "outcome_buggy": res.outcome_buggy, "failure_reason_buggy": res.failure_reason_buggy,
        "outcome_buggy_label": str(buggy) if buggy is not None else None,
        "outcome_fixed": str(fixed) if fixed is not None else None,
        "test_code": res.test_code, "test_path": res.home.rel_path if res.home else None,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "cache_write_tokens": res.cache_write_tokens, "cache_read_tokens": res.cache_read_tokens,
        "cost_usd": res.cost_usd, "duration_seconds": s["seconds"],
    }
    return {"instance_id": inst.instance_id, "repo": inst.repo, "model": model, "split": split,
            "mode": s["mode"], "flags": flags, "best_level": attempt["level"], "best_attempt": 1,
            "attempts": [attempt], "procedure": s}


def run_one(inst, args, client, out_dir: Path) -> float | None:
    print(f"\n{'=' * 60}\n{inst.instance_id}  ({inst.repo})  agent={args.agent}", flush=True)
    flags = {"agent": args.agent, "turn_budget": args.total_turns,
             "max_tokens_per_turn": args.max_tokens_per_turn, "max_total_tokens": args.max_tokens}
    if args.agent == "baseline":
        from tdbf.agent.pipeline import brt_run_record, brt_transcript_dict, run_brt_phase
        phase = run_brt_phase(inst, client, max_iterations=args.total_turns,
                              max_total_tokens=args.max_tokens,
                              max_tokens_per_turn=args.max_tokens_per_turn, verbose=args.verbose)
        if phase is None:
            return None
        (out_dir / f"{inst.instance_id}.json").write_text(
            json.dumps(brt_transcript_dict(inst, client.name, phase), indent=2))
        flags.update({"max_iterations": args.total_turns, "submit_nudge_limit": phase.submit_nudge_limit})
        save_run(brt_run_record(inst, client.name, split=args.split, flags=flags, phase=phase))
        print(f"-> L{phase.level}  cost=${phase.cost_usd}", flush=True)
        return phase.cost_usd

    from tdbf.agent.procedure import run_procedure
    from tdbf.agent.procedure_score import label_offline
    res = run_procedure(inst, client, mode=args.agent, total_turns=args.total_turns,
                        max_total_tokens=args.max_tokens,
                        max_tokens_per_turn=args.max_tokens_per_turn, verbose=args.verbose)
    # =================== GOLD BARRIER: offline label only =================== #
    label = None
    if res.test_code and res.home and res.command and not args.no_label:
        label = label_offline(inst, res.command, res.home.rel_path, res.test_code)
    tr = dict(res.transcript)
    if label:
        tr["label"] = {"buggy": str(label[0]), "buggy_reason": label[0].failure_reason,
                       "fixed": str(label[1]), "level": label[2].level_label}
    (out_dir / f"{inst.instance_id}.json").write_text(json.dumps(tr, indent=2, default=str))
    save_run(procedure_record(inst, client.name, args.split, flags, res, label))
    print(f"-> {res.record['outcome']}  {label[2].level_label if label else 'no label'}  "
          f"turns={res.turns_used}  flags={res.record['flags']}  cost=${res.cost_usd}", flush=True)
    return res.cost_usd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["strict", "soft", "baseline"], required=True)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--round-file", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--total-turns", type=int, default=50, help="turns per attempt (D3: 50)")
    ap.add_argument("--max-tokens", type=int, default=1_000_000, help="token safety ceiling per run")
    ap.add_argument("--max-tokens-per-turn", type=int, default=4096)
    ap.add_argument("--max-usd", type=float, default=6.0, help="stop the batch once billed cost reaches this")
    ap.add_argument("--no-label", action="store_true", help="skip the offline gold label")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    ids = list(args.ids)
    if args.round_file:
        ids += json.loads(args.round_file.read_text())["instance_ids"]
    client = build_client(args.model)
    use_run_group(args.run_group, client.name)   # refuses an unregistered group BEFORE any call
    todo = [i for i in ids if i not in done_ids(args.run_group)]
    print(f"agent={args.agent} model={client.name} group={args.run_group} "
          f"turns={args.total_turns} todo={len(todo)}/{len(ids)} max_usd={args.max_usd}")
    if args.dry_run:
        print("\n".join(todo))
        return 0
    store = InstanceStore(args.split)
    out_dir = ROOT / "runs" / "procedure" / args.run_group
    out_dir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    t0 = time.monotonic()
    for iid in todo:
        if spent >= args.max_usd:
            print(f"cost cap reached (${spent:.2f} >= ${args.max_usd}); stopping.")
            break
        try:
            spent += run_one(store.get(iid), args, client, out_dir) or 0.0
        except Exception as exc:  # noqa: BLE001 - report, continue with the next instance
            print(f"{iid}: CRASH {type(exc).__name__}: {exc}", flush=True)
    print(f"\nbatch done: ${spent:.4f}, {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
