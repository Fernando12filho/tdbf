"""Paired A/B over results.jsonl: the same instances under two run groups.

Re-running instances that are already in the store answers a question the
aggregate report cannot: on THIS instance, did the changed pipeline decide
differently? Metrics dedup per (arm, instance) keeps the best run, so the two
groups must be separated by `flags.run_group` — that is what this reads.

Reports, per instance and in aggregate:
  gate      — did the gold-free gate accept the BRT
  level     — offline L0-L4 label, when --score-brt produced one (never gates)
  resolved  — held-out verdict from the SWE-bench harness
  cost      — priced through tdbf.pricing, so an unpriced model can't read $0

    python3 scripts/10_gate_ab.py --before "" --after gate-fix
    python3 scripts/10_gate_ab.py --before "" --after gate-fix --model sonnet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.metrics import attempt_cost, load_records, short_model


def _group_of(rec: dict) -> str:
    return (rec.get("flags") or {}).get("run_group") or ""


def _collect(records: list[dict], group: str, model_sub: str) -> dict[str, dict]:
    """instance_id -> {gate, level, resolved, cost} for one run group.

    Exactly ONE run per (instance, mode) — the latest by timestamp. Summing every
    matching record instead would make cost incomparable across groups: a group
    where instances were run six times during development would appear six times
    more expensive than one run of the same pipeline, which is a property of how
    often we re-ran it, not of what it costs.
    """
    latest: dict[tuple[str, str], dict] = {}
    for r in records:
        if _group_of(r) != group or model_sub not in r["model"]:
            continue
        mode = r.get("mode")
        if mode not in ("agent_deployable", "fix_agent"):
            continue
        key = (r["instance_id"], mode)
        prev = latest.get(key)
        if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
            latest[key] = r

    out: dict[str, dict] = {}
    for (iid, mode), r in latest.items():
        row = out.setdefault(iid, {
            "gate": None, "level": None, "resolved": None, "cost": 0.0, "runs": 0})
        att = (r.get("attempts") or [{}])[0]
        row["cost"] += attempt_cost(att, r.get("model", ""))
        row["runs"] += 1
        if mode == "agent_deployable":
            row["gate"] = bool(r.get("gate_accepted"))
            if r.get("best_level") is not None:
                row["level"] = r["best_level"]
        else:
            row["resolved"] = bool(r.get("resolved"))
    return out


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="", help="run_group of the baseline ('' = untagged)")
    ap.add_argument("--after", required=True, help="run_group of the new runs")
    ap.add_argument("--model", default="sonnet", help="substring of the model id to compare")
    ap.add_argument("--results", default="results.jsonl")
    args = ap.parse_args()

    records = load_records(args.results)
    before = _collect(records, args.before, args.model)
    after = _collect(records, args.after, args.model)
    shared = sorted(set(before) & set(after))
    if not shared:
        print(f"No instances appear in BOTH groups "
              f"({args.before or '<untagged>'!r} n={len(before)}, {args.after!r} n={len(after)}).")
        return 1

    b_label = args.before or "<untagged>"
    print(f"\nPaired comparison on {len(shared)} instance(s)   "
          f"[{b_label}] -> [{args.after}]   model~{args.model}\n")
    head = f"  {'instance':30s} {'gate':>12s} {'level':>12s} {'resolved':>14s}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    changed = []
    for iid in shared:
        b, a = before[iid], after[iid]
        marks = ""
        if b["gate"] != a["gate"]:
            marks += " GATE"
        if b["resolved"] != a["resolved"]:
            marks += " RESOLVED"
        if marks:
            changed.append((iid, marks.strip()))
        print(f"  {iid:30s} "
              f"{_fmt(b['gate']) + ' -> ' + _fmt(a['gate']):>12s} "
              f"{_fmt(b['level']) + ' -> ' + _fmt(a['level']):>12s} "
              f"{_fmt(b['resolved']) + ' -> ' + _fmt(a['resolved']):>14s}"
              + (f"   <-{marks}" if marks else ""))

    def agg(rows: dict, key: str) -> str:
        vals = [rows[i][key] for i in shared if rows[i][key] is not None]
        return f"{sum(1 for v in vals if v)}/{len(vals)}" if vals else "—"

    b_cost = sum(before[i]["cost"] for i in shared)
    a_cost = sum(after[i]["cost"] for i in shared)
    print(f"\n  {'':30s} {b_label:>18s}   {args.after:>18s}")
    print(f"  {'gate accepted':30s} {agg(before, 'gate'):>18s}   {agg(after, 'gate'):>18s}")
    print(f"  {'resolved':30s} {agg(before, 'resolved'):>18s}   {agg(after, 'resolved'):>18s}")
    print(f"  {'cost (paired instances)':30s} {'$%.2f' % b_cost:>18s}   {'$%.2f' % a_cost:>18s}")
    if b_cost:
        print(f"  {'cost change':30s} {'':>18s}   {(a_cost - b_cost) / b_cost * 100:>17.0f}%")

    print(f"\n  instances that changed: {len(changed)}")
    for iid, what in changed:
        print(f"    {iid}  ({what})")
    if not changed:
        print("    none — the pipeline decided identically on every paired instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
