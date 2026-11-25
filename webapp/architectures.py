"""Data for the /architectures page: every Phase A architecture tried in development,
on the same bugs, against the old single BRT agent.

Live from results.jsonl (run groups of studies scouts-dev, scouts-dev-v2,
blackboard-dev). Two references come from elsewhere and are named as such:
- the old agent on the 30 development bugs (`proc-dev-baseline`) is the compact
  copy in studies/scouts/data/baseline_proc_dev.jsonl (agent/procedure@c797b2f);
- the procedure agent's totals are quoted from that branch's
  studies/issue-info-tests/results/dev_comparison.md (no per-bug records here).
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV30 = ROOT / "studies" / "scouts" / "data" / "dev_bugs.json"
FRESH = [ROOT / "studies" / "scouts" / "data" / f"dev_round{k}.json" for k in (4, 5)]
BASELINE30 = ROOT / "studies" / "scouts" / "data" / "baseline_proc_dev.jsonl"

# (key, label, run group, short description, models)
ARMS_DEV30 = [
    ("base", "Old agent (single BRT agent)", None,
     "One agent, one conversation: read code, run the issue's code, write the test, run it, "
     "revise; gate 1 (fails on buggy at its assertion), up to 2 retries.", "gpt-5-mini"),
    ("match", "Old agent + judge (v1)", "scouts-dev-match",
     "As the old agent, then a judge checks the failure matches the issue; mismatch → new "
     "conversation with feedback (up to 3).", "gpt-5-mini"),
    ("full1", "Scouts v1 (parallel → organizer)", "scouts-dev-full-s100",
     "Template Generator, Edit Localizer and Issue Reproducer alone in parallel; an organizer "
     "merges their reports into a brief; writer + v1 judge.", "gpt-5-mini everywhere"),
    ("bb", "Blackboard", "bb-dev30",
     "Scouts and writer work at the same time and share findings on a structured board; "
     "writer with the in-loop judge v2; no organizer.", "nano scouts · mini writer/judge"),
    ("rbb", "Reactive blackboard", "rbb-dev30",
     "The blackboard with an automatic flow: new findings pushed into each agent's tool results, "
     "the writer's test results auto-posted, rejections and crashes routed to the right scout as "
     "questions, a 90 s soft start.", "nano scouts · mini writer/judge"),
]
ARMS_FRESH = [
    ("base", "Old agent (single BRT agent)", "v2-base", ARMS_DEV30[0][3], "gpt-5-mini"),
    ("loop", "Writer loop + judge v2", "v2-loop",
     "The old agent with the judge INSIDE its loop: check_test returns the verdict; "
     "submit only after 'match'.", "gpt-5-mini"),
    ("full2", "Scouts v2 (nano scouts → organizer)", "v2-full-1m",
     "Scouts v1's structure with gpt-5-nano scouts and organizer, then the writer loop.",
     "nano scouts · mini writer/judge"),
    ("bb", "Blackboard", "bb-fresh", ARMS_DEV30[3][3], "nano scouts · mini writer/judge"),
    ("rbb", "Reactive blackboard", "rbb-fresh", ARMS_DEV30[4][3], "nano scouts · mini writer/judge"),
]

# Quoted from agent/procedure@c797b2f studies/issue-info-tests/results/dev_comparison.md.
PROCEDURE_ROWS = [
    {"label": "Procedure agent, strict (phases + gates)", "l4": 15, "l3": 11, "other": 4,
     "cost": 1.39, "models": "gpt-5-mini"},
    {"label": "Procedure agent, soft (same steps as one prompt)", "l4": 20, "l3": 6, "other": 4,
     "cost": 1.57, "models": "gpt-5-mini"},
]


def _ids(path: Path) -> list[str]:
    return json.loads(path.read_text())["instance_ids"]


def mcnemar(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def _cost(r: dict) -> float:
    if r.get("cost_total_usd") is not None:
        return float(r["cost_total_usd"])
    return float(sum((a.get("cost_usd") or 0) for a in r.get("attempts", [])))


def _by_group(records: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in records:
        g = (r.get("flags") or {}).get("run_group")
        if g:
            out.setdefault(g, {})[r["instance_id"]] = r   # last record per instance wins
    return out


def _arm_rows(ids: list[str], arms: list, levels: dict[str, dict[str, int]],
              costs: dict[str, dict[str, float]], extra: dict[str, dict]) -> list[dict]:
    base = levels["base"]
    base_cost = sum(costs["base"].get(i, 0) for i in ids)
    base_l4 = sum(base.get(i) == 4 for i in ids)
    base_l3 = sum(base.get(i) == 3 for i in ids)
    base_pv = base_cost / base_l4 if base_l4 else None
    rows = []
    for key, label, group, desc, models in arms:
        lv = levels.get(key, {})
        done = [i for i in ids if i in lv]
        l4 = sum(lv[i] == 4 for i in done)
        l3 = sum(lv[i] == 3 for i in done)
        cost = sum(costs.get(key, {}).get(i, 0) for i in done)
        pv = cost / l4 if l4 else None
        row = {"key": key, "label": label, "group": group, "desc": desc, "models": models,
               "n": len(done), "total": len(ids), "l4": l4, "l3": l3, "other": len(done) - l4 - l3,
               "cost": cost, "per_valid": pv, **extra.get(key, {})}
        if key != "base" and done:
            gain = [i for i in done if lv[i] == 4 and base.get(i) != 4]
            loss = [i for i in done if lv[i] != 4 and base.get(i) == 4]
            b_l4 = sum(base.get(i) == 4 for i in done)
            b_l3 = sum(base.get(i) == 3 for i in done)
            b_cost = sum(costs["base"].get(i, 0) for i in done)
            b_pv = b_cost / b_l4 if b_l4 else None
            ratio = (pv / b_pv) if (pv and b_pv) else None
            row.update(gain=gain, loss=loss, p=mcnemar(len(gain), len(loss)),
                       base_l4_same=b_l4, ratio=ratio,
                       adopted=(len(done) == len(ids) and l4 > b_l4 and l3 <= b_l3
                                and ratio is not None and ratio <= 1.5))
        rows.append(row)
    rows[0].update(base_l4=base_l4, base_l3=base_l3, per_valid=base_pv)
    return rows


def _judge_stats(recs: dict[str, dict]) -> dict | None:
    checks = mism = bugs_mm = ended = 0
    seen = False
    for r in recs.values():
        a = r["attempts"][0]
        ml, mc = a.get("match_loop"), a.get("match_checks") or []
        if ml is None and not mc:
            continue
        seen = True
        checks += len(mc)
        m = sum(c["verdict"] == "mismatch" for c in mc)
        mism += m
        if m:
            bugs_mm += 1
            if (ml or {}).get("final_matched") or (mc and mc[-1]["verdict"] == "match"):
                ended += 1
    return {"checks": checks, "mismatches": mism, "bugs_with_mismatch": bugs_mm,
            "ended_match": ended} if seen else None


def _board_stats(recs: dict[str, dict]) -> dict | None:
    bbs = [r["blackboard"] for r in recs.values() if r.get("blackboard")]
    if not bbs:
        return None
    n = len(bbs)
    ent = sum(b["board"]["entries"] for b in bbs)
    q = sum(b["board"]["questions"] for b in bbs)
    ans = sum(b["board"]["answered"] for b in bbs)
    by_author: dict[str, int] = {}
    for b in bbs:
        for k, v in b["board"]["by_author"].items():
            by_author[k] = by_author.get(k, 0) + v
    writer_reads = sum(b["board"]["reads"].get("writer", 0) for b in bbs)
    routed = sum(b["board"].get("routed", 0) for b in bbs)
    routed_ans = sum(b["board"].get("routed_answered", 0) for b in bbs)
    return {"bugs": n, "entries_per_bug": ent / n, "questions": q, "answered": ans,
            "routed": routed, "routed_answered": routed_ans,
            "by_author": {k: v / n for k, v in sorted(by_author.items())},
            "writer_reads_per_bug": writer_reads / n,
            "scouts_cost": sum((b.get("cost_usd") or 0) for b in bbs)}


def build(results_file: Path) -> dict:
    recs = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
    G = _by_group(recs)
    dev30, fresh = _ids(DEV30), _ids(FRESH[0]) + _ids(FRESH[1])

    base30 = {json.loads(l)["instance_id"]: json.loads(l)
              for l in BASELINE30.read_text().splitlines() if l.strip()}
    lv30 = {"base": {i: r["best_level"] for i, r in base30.items()}}
    c30 = {"base": {i: r["cost_usd"] for i, r in base30.items()}}
    extra30, extraF = {}, {}
    for key, _, g, *_ in ARMS_DEV30[1:]:
        lv30[key] = {i: r["best_level"] for i, r in G.get(g, {}).items()}
        c30[key] = {i: _cost(r) for i, r in G.get(g, {}).items()}
        extra30[key] = {"judge": _judge_stats(G.get(g, {})), "board": _board_stats(G.get(g, {}))}
    lvF, cF = {}, {}
    for key, _, g, *_ in ARMS_FRESH:
        lvF[key] = {i: r["best_level"] for i, r in G.get(g, {}).items()}
        cF[key] = {i: _cost(r) for i, r in G.get(g, {}).items()}
        extraF[key] = {"judge": _judge_stats(G.get(g, {})), "board": _board_stats(G.get(g, {}))}

    def matrix(ids, arms, lv):
        return [{"id": i, "cells": [lv.get(k, {}).get(i) for k, *_ in arms]} for i in ids]

    return {
        "dev30": {"ids": dev30, "rows": _arm_rows(dev30, ARMS_DEV30, lv30, c30, extra30),
                  "procedure": PROCEDURE_ROWS, "matrix": matrix(dev30, ARMS_DEV30, lv30),
                  "arms": ARMS_DEV30},
        "fresh": {"ids": fresh, "rows": _arm_rows(fresh, ARMS_FRESH, lvF, cF, extraF),
                  "matrix": matrix(fresh, ARMS_FRESH, lvF), "arms": ARMS_FRESH},
    }
