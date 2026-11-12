#!/usr/bin/env python3
"""Review a development round: per bug and agent, the L0-L4 label, turns, cost and
flags, then the totals the stopping rule uses (design §10c): valid tests (L4),
wrong-reason tests (L3), and cost per valid test.

Reads results.jsonl (records of run groups proc-dev-r<k>-{strict,soft,baseline}).
Turns: procedure agents count every model turn; the baseline's `iterations` are the
turns of its last attempt, and its earlier gate retries are added from `brt_retries`.

    python3 studies/issue-info-tests/scripts/review_round.py --rounds 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGENTS = ("baseline", "strict", "soft")


def load(rounds: list[int]) -> dict[tuple[str, str], dict]:
    groups = {f"proc-dev-r{r}-{a}": a for r in rounds for a in AGENTS}
    out: dict[tuple[str, str], dict] = {}
    for line in (ROOT / "results.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        a = groups.get((r.get("flags") or {}).get("run_group"))
        if a:
            out[(r["instance_id"], a)] = r  # the last record wins (re-runs)
    return out


def row(r: dict | None) -> dict:
    if r is None:
        return {"level": None, "turns": None, "cost": None, "flags": "", "note": "missing"}
    att = r["attempts"][0]
    retries = r.get("brt_retries") or []
    turns = (att.get("iterations") or 0) + sum(x.get("iterations") or 0 for x in retries)
    p = r.get("procedure") or {}
    note = p.get("outcome") or (f"{len(retries)} retr{'y' if len(retries) == 1 else 'ies'}" if retries else "")
    l3 = att.get("l3_subtype")
    return {"level": r.get("best_level"), "l3": l3, "turns": turns, "cost": att.get("cost_usd"),
            "flags": ",".join(p.get("flags") or []), "note": note,
            "reason": att.get("failure_reason_buggy")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    recs = load(args.rounds)
    ids = []
    for r in args.rounds:
        f = ROOT / "studies" / "issue-info-tests" / "data" / "dev" / f"round{r}.json"
        ids += json.loads(f.read_text())["instance_ids"]
    print("| bug | " + " | ".join(f"{a} L / turns / $ / notes" for a in AGENTS) + " |")
    print("|---|" + "---|" * len(AGENTS))
    tot = {a: {"L4": 0, "L3": 0, "cost": 0.0, "turns": 0, "n": 0} for a in AGENTS}
    for iid in ids:
        cells = []
        for a in AGENTS:
            x = row(recs.get((iid, a)))
            if x["level"] is not None:
                t = tot[a]
                t["n"] += 1
                t["L4"] += x["level"] == 4
                t["L3"] += x["level"] == 3
                t["cost"] += x["cost"] or 0
                t["turns"] += x["turns"] or 0
            lv = "-" if x["level"] is None else f"L{x['level']}" + (f"({x['l3']})" if x.get("l3") else "")
            cost = "-" if x["cost"] is None else f"{x['cost']:.3f}"
            cells.append(f"{lv} / {x['turns'] if x['turns'] is not None else '-'} / {cost} / "
                         f"{x['note']}{' [' + x['flags'] + ']' if x['flags'] else ''}")
        print(f"| {iid} | " + " | ".join(cells) + " |")
    print()
    print("| agent | bugs | valid (L4) | wrong reason (L3) | total $ | $ per valid test | mean turns |")
    print("|---|---|---|---|---|---|---|")
    for a in AGENTS:
        t = tot[a]
        cpv = f"{t['cost'] / t['L4']:.3f}" if t["L4"] else "-"
        mt = f"{t['turns'] / t['n']:.1f}" if t["n"] else "-"
        print(f"| {a} | {t['n']} | {t['L4']} | {t['L3']} | {t['cost']:.3f} | {cpv} | {mt} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
