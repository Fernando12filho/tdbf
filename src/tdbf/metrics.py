"""Metric aggregation over results.jsonl — the single source of truth for the
paper/advisor numbers, shared by the CLI report (scripts/08_metrics_report.py)
and the webapp Metrics page.

Pure functions: give them a list of run records (dicts, as written to
results.jsonl) and get back a structured report. No I/O beyond `load_records`,
no API, no Docker.

Metric suite:
  BRT quality (Phase A):   C2P = L4/(L3+L4);  Valid-BRT = L4/N;  L0..L4 hist.
  Fix quality (Phase B):   Resolve@BRT = resolved/N  (given a validated BRT).
  Whole pipeline (funnel): Pipeline resolve = resolved/N_A
                           = GateRate (L4/N_A) x Resolve@Gated.
  Efficiency:              $/instance, $/resolved.

Dedup: per (arm, instance) keep ONE canonical run — BRT: highest level;
Fix: a resolved run if any, else the latest. `runs` and `instances` are both
reported so the dedup is transparent.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path


# ── loading & record helpers ────────────────────────────────────────────────

def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _flag(rec: dict, name: str):
    return (rec.get("flags") or {}).get(name)


def _best_attempt(rec: dict) -> dict:
    atts = rec.get("attempts") or []
    if not atts:
        return {}
    bi = rec.get("best_attempt")
    for a in atts:
        if a.get("attempt") == bi:
            return a
    return max(atts, key=lambda a: (a.get("level") or 0))


def short_model(m: str) -> str:
    return (m or "?").split("@")[0].replace("anthropic:", "").replace("ollama:", "")


def repo_of(rec: dict) -> str:
    return (rec.get("repo") or "?").split("/")[-1]


def is_local(model: str) -> bool:
    return "ollama:" in (model or "")


# ── per-instance dedup ───────────────────────────────────────────────────────

def brt_by_instance(recs: list[dict]) -> dict[str, dict]:
    """instance_id -> record with the highest BRT level."""
    best: dict[str, dict] = {}
    for r in recs:
        iid = r["instance_id"]
        lvl = r.get("best_level")
        lvl = -1 if lvl is None else lvl
        cur = best.get(iid)
        cur_lvl = -1 if (cur is None or cur.get("best_level") is None) else cur["best_level"]
        if cur is None or lvl > cur_lvl:
            best[iid] = r
    return best


def fix_by_instance(recs: list[dict]) -> dict[str, dict]:
    """instance_id -> a resolved run if any, else the latest by timestamp."""
    out: dict[str, dict] = {}
    for r in sorted(recs, key=lambda x: x.get("timestamp", "")):
        iid = r["instance_id"]
        prev = out.get(iid)
        if prev is None or (r.get("resolved") and not prev.get("resolved")) or not prev.get("resolved"):
            out[iid] = r
    return out


# ── small rate helper ────────────────────────────────────────────────────────

def rate(n: int, d: int) -> dict:
    """A rate as both numbers and a preformatted label."""
    return {
        "num": n,
        "den": d,
        "value": (n / d) if d else None,
        "pct": (f"{n / d:.0%}" if d else "—"),
        "label": (f"{n}/{d} = {n / d:.2f}" if d else f"{n}/0 = —"),
    }


# ── metric blocks ────────────────────────────────────────────────────────────

def brt_arm(label: str, recs: list[dict], *, local: bool = False) -> dict:
    by = brt_by_instance(recs)
    hist = collections.Counter(
        (r.get("best_level") if r.get("best_level") is not None else -1)
        for r in by.values()
    )
    L = {k: hist.get(k, 0) for k in range(5)}
    n = len(by)
    return {
        "label": label,
        "local": local,
        "runs": len(recs),
        "instances": n,
        "levels": [L[0], L[1], L[2], L[3], L[4]],
        "unlabeled": hist.get(-1, 0),
        "c2p": rate(L[4], L[3] + L[4]),
        "valid": rate(L[4], n),
    }


def fix_arm(label: str, recs: list[dict], *, local: bool = False) -> dict:
    by = fix_by_instance(recs)
    n = len(by)
    resolved = sum(1 for r in by.values() if r.get("resolved"))
    cost = 0.0
    for r in by.values():
        c = _best_attempt(r).get("cost_usd")
        if c:
            cost += c
    return {
        "label": label,
        "local": local,
        "runs": len(recs),
        "instances": n,
        "resolved": resolved,
        "resolve_at_brt": rate(resolved, n),
        "cost": cost,
        "cost_per_inst": (cost / n) if n else 0.0,
        "cost_per_resolved": (cost / resolved) if resolved else None,
    }


# ── the full report ──────────────────────────────────────────────────────────

def build_report(records: list[dict]) -> dict:
    agent = [r for r in records if r.get("mode") == "agent"]
    fix = [r for r in records if r.get("mode") == "fix_agent"]
    deploy = [r for r in records if r.get("mode") == "agent_deployable"]
    baseline = [r for r in records if r.get("mode") is None]

    # [1] BRT arms, grouped by (model, prompt_harden)
    brt_groups: dict = collections.defaultdict(list)
    for r in agent:
        brt_groups[(short_model(r["model"]), bool(_flag(r, "prompt_harden")), is_local(r["model"]))].append(r)
    brt_arms = []
    for (model, harden, local), rs in brt_groups.items():
        label = model + ("  +harden" if harden else "")
        brt_arms.append(brt_arm(label, rs, local=local))
    brt_arms.sort(key=lambda a: (a["local"], -a["instances"], a["label"]))

    # [2] Fix arms, grouped by model
    fix_groups: dict = collections.defaultdict(list)
    for r in fix:
        fix_groups[(short_model(r["model"]), is_local(r["model"]))].append(r)
    fix_arms = [fix_arm(m, rs, local=local) for (m, local), rs in fix_groups.items()]
    fix_arms.sort(key=lambda a: (a["local"], -a["instances"]))

    # [3] Whole-pipeline funnel (assembled ceiling: L4 gate -> Phase B), per model
    pipeline = []
    for model in sorted({short_model(r["model"]) for r in agent}):
        a = [r for r in agent if short_model(r["model"]) == model]
        f = [r for r in fix if short_model(r["model"]) == model]
        if not f:
            continue
        a_by = brt_by_instance(a)
        f_by = fix_by_instance(f)
        n_a = len(a_by)
        gated = {i for i, r in a_by.items() if r.get("best_level") == 4}
        resolved_gated = sum(1 for i in gated if f_by.get(i, {}).get("resolved"))
        pipeline.append({
            "model": model,
            "local": is_local(next(r["model"] for r in a)),
            "n_a": n_a,
            "gate_rate": rate(len(gated), n_a),
            "resolve_at_gated": rate(resolved_gated, len(gated)),
            "pipeline_resolve": rate(resolved_gated, n_a),
        })

    # [4] Deployable pipeline (07), gold-free gate — pilot
    deployable = []
    for model in sorted({short_model(r["model"]) for r in deploy}):
        d = [r for r in deploy if short_model(r["model"]) == model]
        insts = {r["instance_id"] for r in d}
        gate = sum(1 for r in d if r.get("gate_accepted") or r.get("reproduces_on_buggy"))
        f_by = fix_by_instance([r for r in fix if r["instance_id"] in insts
                                and short_model(r["model"]) == model])
        resolved = sum(1 for r in f_by.values() if r.get("resolved"))
        deployable.append({
            "model": model,
            "local": is_local(d[0]["model"]),
            "instances": len(insts),
            "gate_accept": gate,
            "resolved": rate(resolved, len(f_by) or gate),
        })

    # [5] Repo breakdown (Claude Sonnet)
    repo_rows = []
    csa = [r for r in agent if "sonnet" in r["model"]]
    csf = [r for r in fix if "sonnet" in r["model"]]
    for repo in sorted({repo_of(r) for r in csa}):
        a_by = brt_by_instance([r for r in csa if repo_of(r) == repo])
        f_by = fix_by_instance([r for r in csf if repo_of(r) == repo])
        l4 = sum(1 for r in a_by.values() if r.get("best_level") == 4)
        l3 = sum(1 for r in a_by.values() if r.get("best_level") == 3)
        res = sum(1 for r in f_by.values() if r.get("resolved"))
        repo_rows.append({
            "repo": repo,
            "c2p": rate(l4, l3 + l4),
            "valid": rate(l4, len(a_by)),
            "resolve_at_brt": rate(res, len(f_by)) if f_by else None,
        })

    # [6] Non-agentic baseline (02/03 workflow), for contrast
    base_groups: dict = collections.defaultdict(list)
    for r in baseline:
        base_groups[(short_model(r["model"]), is_local(r["model"]))].append(r)
    baseline_arms = [brt_arm(m, rs, local=local) for (m, local), rs in base_groups.items()]
    baseline_arms.sort(key=lambda a: (a["local"], -a["instances"]))

    return {
        "counts": {
            "records": len(records),
            "brt_agent": len(agent),
            "fix_agent": len(fix),
            "deployable": len(deploy),
            "baseline": len(baseline),
        },
        "brt_arms": brt_arms,
        "fix_arms": fix_arms,
        "pipeline": pipeline,
        "deployable": deployable,
        "repo_breakdown": repo_rows,
        "baseline_arms": baseline_arms,
    }


# ── text formatter (for the CLI) ─────────────────────────────────────────────

def format_text(rep: dict) -> str:
    c = rep["counts"]
    out = []
    bar = "=" * 68
    out.append(bar)
    out.append("TDBF metric report  —  agentic pipeline")
    out.append(f"({c['records']} records: {c['brt_agent']} BRT-agent, "
               f"{c['fix_agent']} fix-agent, {c['deployable']} deployable, "
               f"{c['baseline']} non-agentic baseline)")
    out.append(bar)

    out.append("\n[1] BRT QUALITY  (Phase A — mode=agent)")
    for a in rep["brt_arms"]:
        L = a["levels"]
        out.append(f"  {a['label']}")
        out.append(f"    runs={a['runs']}  instances={a['instances']}   "
                   f"levels L0..L4 = {L[0]},{L[1]},{L[2]},{L[3]},{L[4]}")
        out.append(f"    C2P (L4/(L3+L4))   = {a['c2p']['label']}")
        out.append(f"    Valid-BRT (L4/N)   = {a['valid']['label']}")

    out.append("\n[2] FIX QUALITY  (Phase B — mode=fix_agent)")
    for a in rep["fix_arms"]:
        out.append(f"  {a['label']}")
        out.append(f"    runs={a['runs']}  instances={a['instances']}")
        out.append(f"    Resolve@BRT        = {a['resolve_at_brt']['label']}")
        if a["cost"]:
            cpr = f"${a['cost_per_resolved']:.3f}/resolved" if a["cost_per_resolved"] else "—/resolved"
            out.append(f"    cost               = ${a['cost']:.2f} total  |  "
                       f"${a['cost_per_inst']:.3f}/inst  |  {cpr}")

    out.append("\n[3] WHOLE PIPELINE  (assembled ceiling: L4 gate -> Phase B)")
    for p in rep["pipeline"]:
        out.append(f"  {p['model']}")
        out.append(f"    N (Phase A instances)     = {p['n_a']}")
        out.append(f"    GateRate (L4/N)           = {p['gate_rate']['label']}")
        out.append(f"    Resolve@Gated             = {p['resolve_at_gated']['label']}")
        out.append(f"    Pipeline resolve (res/N_A)= {p['pipeline_resolve']['label']}")

    if rep["deployable"]:
        out.append("\n[4] DEPLOYABLE PIPELINE  (07 — gold-free gate; pilot)")
        for d in rep["deployable"]:
            out.append(f"  {d['model']}: instances={d['instances']}  "
                       f"gate-accept={d['gate_accept']}  "
                       f"resolved(of gated)={d['resolved']['label']}")

    out.append("\n[5] REPO BREAKDOWN  (Claude Sonnet)")
    for r in rep["repo_breakdown"]:
        res = r["resolve_at_brt"]["label"] if r["resolve_at_brt"] else "n/a"
        out.append(f"  {r['repo']:12s} BRT: C2P {r['c2p']['label']}  "
                   f"Valid {r['valid']['label']}   | Fix: Resolve@BRT {res}")

    if rep["baseline_arms"]:
        out.append("\n[6] NON-AGENTIC BASELINE  (02/03 --attempts workflow, for contrast)")
        for a in rep["baseline_arms"]:
            L = a["levels"]
            out.append(f"  {a['label']}")
            out.append(f"    runs={a['runs']}  instances={a['instances']}   "
                       f"levels L0..L4 = {L[0]},{L[1]},{L[2]},{L[3]},{L[4]}")
            out.append(f"    C2P (L4/(L3+L4))   = {a['c2p']['label']}")
            out.append(f"    Valid-BRT (L4/N)   = {a['valid']['label']}")

    out.append("\n" + bar)
    out.append("Note: per-instance dedup = best-of (BRT) / resolved-first (fix). "
               "Freeze canonical\nruns before final paper numbers. Local (ollama:*) "
               "runs are free; cost = Claude only.")
    out.append(bar)
    return "\n".join(out)
