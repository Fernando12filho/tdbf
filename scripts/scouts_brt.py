"""Run the scouts architecture's Phase A on instances (development study scouts-dev).

Two arms, both with gate 2 (the test's failure must match the issue, judged
gold-free, retried with the judge's feedback):

  --arm match   the unchanged BRT agent + gate 2 (no scouts, no brief)
  --arm full    Template Generator, Edit Localizer and Issue Reproducer in
                parallel, the Organizer's brief, then the BRT agent + gate 2

v2 (studies/scouts/plan_v2.md): the judge (v2) runs INSIDE the writer's loop
(check_test + guarded submit_test) instead of the outer retry loop:

  --arm base    the old BRT agent, unchanged (the reference on new bugs)
  --arm loop    the BRT agent with the in-loop judge, no scouts
  --arm full2   scouts + organizer, then the BRT agent with the in-loop judge
                (give the scouts and organizer a cheaper model with --role-model)
  --arm bb      blackboard (studies/scouts/plan_blackboard.md): the three scouts and
                the in-loop-judge writer work at the same time and share findings
                on a board; no organizer
  --arm rbb     reactive blackboard (plan_reactive_blackboard.md): entries pushed into
                tool results, check_test results auto-posted, problems routed to
                scouts, a soft start

Every role gets its own client. By default all roles use --model; override one
with --role-model ROLE=SPEC (roles: template localizer reproducer organizer
writer judge). The run group must be registered in src/tdbf/studies.py, and every
role's model must be one of its study's models.

Each finished instance is labelled offline (L0-L4, gold patch, after the agents),
saved to results.jsonl under its run group, and its transcript written to
runs/scouts/<group>/<id>.json. Instances already saved under the run group are
skipped (resume). The batch stops before the next instance once the billed total
reaches --max-usd.

    python3 scripts/scouts_brt.py --arm match --model openrouter:openai/gpt-5-mini@openai \\
        --run-group scouts-dev-match
    python3 scripts/scouts_brt.py --arm full ... --run-group scouts-dev-full --dry-run
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

from tdbf import studies  # noqa: E402
from tdbf.data.instances import InstanceStore  # noqa: E402
from tdbf.models.llm import build_client  # noqa: E402
from tdbf.results import RESULTS_FILE, save_run, use_run_group  # noqa: E402

DEFAULT_BUGS = ROOT / "studies" / "scouts" / "data" / "dev_bugs.json"


def done_ids(run_group: str) -> set[str]:
    path = RESULTS_FILE if RESULTS_FILE.is_absolute() else ROOT / RESULTS_FILE
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


def role_specs(default: str, overrides: list[str]) -> dict[str, str]:
    from tdbf.agent.scouts import ROLES
    specs = {r: default for r in ROLES}
    for item in overrides:
        role, sep, spec = item.partition("=")
        if not sep or role not in ROLES or not spec:
            raise SystemExit(f"--role-model expects ROLE=SPEC with ROLE in {ROLES}; got {item!r}")
        specs[role] = spec
    return specs


def _total(*costs) -> float:
    return round(sum(c for c in costs if c is not None), 6)


def run_one(inst, args, clients: dict, out_dir: Path) -> float | None:
    from tdbf.agent.pipeline import brt_run_record, brt_transcript_dict, serialise_messages
    from tdbf.agent.scouts import run_scouts_brt_phase

    from tdbf.agent.pipeline import run_brt_phase

    print(f"\n{'=' * 60}\n{inst.instance_id}  ({inst.repo})  arm={args.arm}", flush=True)
    t0 = time.monotonic()
    bb = None
    if args.arm in ("bb", "rbb"):
        from tdbf.agent.blackboard import run_blackboard_brt_phase
        sres = None
        bb, phase = run_blackboard_brt_phase(
            inst, clients, max_iterations=args.total_turns, max_total_tokens=args.max_tokens,
            max_tokens_per_turn=args.max_tokens_per_turn, scout_max_iterations=args.scout_turns,
            max_mismatches=args.max_mismatches, reactive=args.arm == "rbb",
            score_fixed=not args.no_label, verbose=args.verbose)
    elif args.arm == "base":
        sres = None
        phase = run_brt_phase(inst, clients["writer"], max_iterations=args.total_turns,
                              max_total_tokens=args.max_tokens,
                              max_tokens_per_turn=args.max_tokens_per_turn,
                              score_fixed=not args.no_label, verbose=args.verbose)
    else:
        sres, phase = run_scouts_brt_phase(
            inst, clients, max_iterations=args.total_turns, max_total_tokens=args.max_tokens,
            max_tokens_per_turn=args.max_tokens_per_turn, scout_max_iterations=args.scout_turns,
            match_retries=args.match_retries, use_scouts=args.arm in ("full", "full2"),
            writer_loop=args.arm in ("loop", "full2"), scout_max_tokens=args.scout_max_tokens,
            max_mismatches=args.max_mismatches, verbose=args.verbose)
    if phase is None:
        return None
    total = _total(phase.cost_usd, phase.match_cost_usd, sres.cost_usd if sres else None,
                   bb.scouts_cost_usd if bb else None)
    flags = {"arm": args.arm, "pipeline": "scouts", "turn_budget": args.total_turns,
             "scout_turn_budget": args.scout_turns if (sres or bb) else None,
             "match_retries": args.match_retries if args.arm in ("match", "full") else None,
             "writer_loop": args.arm in ("loop", "full2", "bb", "rbb"),
             "max_mismatches": (args.max_mismatches if args.arm in ("loop", "full2", "bb", "rbb")
                                else None),
             "blackboard": args.arm in ("bb", "rbb"),
             "reactive": args.arm == "rbb",
             "scout_max_tokens": args.scout_max_tokens if sres else None,
             "max_tokens_per_turn": args.max_tokens_per_turn,
             "max_total_tokens": args.max_tokens, "max_iterations": args.total_turns,
             "submit_nudge_limit": phase.submit_nudge_limit,
             "role_models": {r: c.name for r, c in clients.items()}}
    rec = brt_run_record(inst, clients["writer"].name, split=args.split, flags=flags, phase=phase)
    if args.no_label:  # no offline L0-L4 label: record None, never a misleading L0
        flags["offline_label"] = False
        rec["best_level"] = None
        for a in rec["attempts"]:
            a["level"] = None
    rec["cost_total_usd"] = total
    rec["seconds"] = round(time.monotonic() - t0, 1)
    if sres is not None:
        rec["scouts"] = {"roles": [r.record() for r in sres.scouts.values()],
                         "organizer": sres.organizer.record(), "brief_chars": len(sres.brief or ""),
                         "cost_usd": sres.cost_usd, "seconds": round(sres.seconds, 1),
                         "notes": sres.notes}
    if bb is not None:
        rec["blackboard"] = {"board": bb.board.summary(),
                             "roles": [r.record() for r in bb.scouts.values()],
                             "cost_usd": bb.scouts_cost_usd, "seconds": round(bb.seconds, 1)}
    save_run(rec)

    tr = brt_transcript_dict(inst, clients["writer"].name, phase)
    tr["match_checks"] = phase.match_checks
    tr["match_loop"] = phase.match_loop
    if bb is not None:
        tr["board"] = bb.board.log()
        tr["bb_scouts"] = {r.role: {"error": r.error,
                                    "messages": serialise_messages(r.agent.messages) if r.agent else []}
                           for r in bb.scouts.values()}
    if sres is not None:
        tr["scouts"] = {r.role: {"report": r.report, "error": r.error,
                                 "messages": serialise_messages(r.agent.messages) if r.agent else []}
                        for r in sres.scouts.values()}
        org = sres.organizer
        tr["organizer"] = {"brief": org.report,
                           "messages": serialise_messages(org.agent.messages) if org.agent else []}
        tr["brief_used"] = sres.brief
    (out_dir / f"{inst.instance_id}.json").write_text(json.dumps(tr, indent=2, default=str))
    verdicts = ",".join(c["verdict"] for c in phase.match_checks) or "-"
    print(f"-> L{phase.level}  match={verdicts}  cost=${total}  ({rec['seconds']:.0f}s)", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["match", "full", "base", "loop", "full2", "bb", "rbb"], required=True)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--bugs-file", type=Path, nargs="+", default=[DEFAULT_BUGS],
                    help="JSON file(s) with instance_ids (default: the 30 development bugs)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N instances to do")
    ap.add_argument("--model", required=True, help="default model spec for every role")
    ap.add_argument("--role-model", action="append", default=[], metavar="ROLE=SPEC")
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--split", default="lite")
    ap.add_argument("--total-turns", type=int, default=50, help="turns per writer attempt")
    ap.add_argument("--scout-turns", type=int, default=20, help="turns per scout")
    ap.add_argument("--match-retries", type=int, default=3, help="v1 outer judge loop")
    ap.add_argument("--max-mismatches", type=int, default=3,
                    help="v2: judge mismatches after which submit_test is accepted (flagged)")
    ap.add_argument("--scout-max-tokens", type=int, default=None,
                    help="token ceiling per scout (default: --max-tokens)")
    ap.add_argument("--max-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-tokens-per-turn", type=int, default=8192)
    ap.add_argument("--max-usd", type=float, default=6.0)
    ap.add_argument("--no-label", action="store_true",
                    help="skip the offline L0-L4 label (gold-fixed run) after the agents; the "
                         "level is recorded as None")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    ids = list(args.ids)
    if not ids:
        for f in args.bugs_file:
            ids += json.loads(f.read_text())["instance_ids"]
    specs = role_specs(args.model, args.role_model)
    clients = {role: build_client(spec) for role, spec in specs.items()}  # one object per role
    use_run_group(args.run_group, clients["writer"].name)  # refuses before any model call
    study = studies.of_group(args.run_group)
    bad = {r: c.name for r, c in clients.items() if c.name not in study.models}
    if bad:
        raise SystemExit(f"roles {bad} use models outside study {study.name} {study.models}; "
                         "register them first")
    todo = [i for i in ids if i not in done_ids(args.run_group)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"arm={args.arm} group={args.run_group} roles={specs} turns={args.total_turns} "
          f"scout_turns={args.scout_turns} match_retries={args.match_retries} "
          f"todo={len(todo)}/{len(ids)} max_usd={args.max_usd}")
    if args.dry_run:
        print("\n".join(todo))
        return 0
    store = InstanceStore(args.split)
    out_dir = ROOT / "runs" / "scouts" / args.run_group
    out_dir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    t0 = time.monotonic()
    for iid in todo:
        if spent >= args.max_usd:
            print(f"cost cap reached (${spent:.2f} >= ${args.max_usd}); stopping.")
            break
        try:
            spent += run_one(store.get(iid), args, clients, out_dir) or 0.0
        except Exception as exc:  # noqa: BLE001 - report, continue with the next instance
            print(f"{iid}: CRASH {type(exc).__name__}: {exc}", flush=True)
    print(f"\nbatch done: ${spent:.4f}, {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
