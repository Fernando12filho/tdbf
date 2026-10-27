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

from .pricing import cost_usd


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


def attempt_cost(attempt: dict, model: str) -> float:
    """Cost of one attempt, recomputed from tokens when it wasn't recorded.

    `cost_usd` returns None for a model absent from the pricing table, and that
    None was stored — 21 fable runs sit in the log at $0 despite recording their
    token counts. Recomputing here (rather than rewriting the append-only log)
    repairs those rows and any future pricing gap, and costs nothing when the
    stored value is already right.
    """
    stored = attempt.get("cost_usd")
    if stored:
        return float(stored)
    recomputed = cost_usd(
        model,
        attempt.get("input_tokens") or 0,
        attempt.get("output_tokens") or 0,
        cache_write_tok=attempt.get("cache_write_tokens") or 0,
        cache_read_tok=attempt.get("cache_read_tokens") or 0,
    )
    return float(recomputed or 0.0)


def short_model(m: str) -> str:
    return ((m or "?").split("@")[0]
            .replace("anthropic:", "").replace("ollama:", "").replace("openrouter:", ""))


def repo_of(rec: dict) -> str:
    return (rec.get("repo") or "?").split("/")[-1]


def is_local(model: str) -> bool:
    return "ollama:" in (model or "")


# Arm identity. `flags.arm` is written by current runs; older records are
# classified from the `flags.pipeline` spelling they happen to carry:
#   "deployable" -> 07 gold-free gate    True -> 06 L4-gated ceiling
#   absent/None  -> 05 fix-phase-only
# Every metric that compares arms must group on this, never on model alone.
_LEGACY_ARM = {"deployable": "deployable", True: "ceiling", None: "fix-only"}


def arm_of(rec: dict) -> str:
    """Arm identity, qualified by `flags.run_group` when the run carries one.

    The run group matters because the per-instance dedup keeps the BEST run of
    an (arm, instance): re-running the same instances through a changed pipeline
    would otherwise merge old and new results and report the max of the two,
    hiding exactly the difference the re-run exists to measure.
    """
    flags = rec.get("flags") or {}
    arm = flags.get("arm") or _LEGACY_ARM.get(
        flags.get("pipeline"), str(flags.get("pipeline")))
    group = flags.get("run_group")
    return f"{arm}/{group}" if group else str(arm)


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
        cost += attempt_cost(_best_attempt(r), r.get("model", ""))
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
    # A fix whose harness verdict could not be verified as belonging to its own
    # patch (`scoring_error`) is not a result and must not count either way.
    fix = [r for r in records if r.get("mode") == "fix_agent" and not r.get("scoring_error")]
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

    # [2] Fix arms, grouped by (ARM, model). Grouping by model alone pools the
    # 05 / 06 / 07 runs of the same instance into one row — the ceiling arm's
    # fixes land in the deployable arm's rate — which stays invisible while the
    # arms agree and silently merges them the moment they diverge.
    fix_groups: dict = collections.defaultdict(list)
    for r in fix:
        fix_groups[(arm_of(r), short_model(r["model"]), is_local(r["model"]))].append(r)
    fix_arms = [fix_arm(f"{m}  [{a}]", rs, local=local)
                for (a, m, local), rs in fix_groups.items()]
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
    #
    # Fix runs are scoped to the DEPLOYABLE arm, not just to (instance, model).
    # The same instance has been fixed by 05, 06 and 07 under one model (13 such
    # pairs exist in the current store), so an unscoped filter pools the L4-gated
    # ceiling's fixes into the gold-free arm's resolved count. That is invisible
    # while the arms agree and silently merges them the moment they don't.
    deployable = []
    # Keyed by (arm, model) like the fix arms: grouping on model alone merges
    # runs of the same instances made under different pipeline versions, which
    # is exactly what a re-run is meant to tell apart.
    for arm, model in sorted({(arm_of(r), short_model(r["model"])) for r in deploy}):
        d = [r for r in deploy if short_model(r["model"]) == model and arm_of(r) == arm]
        d_by = {r["instance_id"]: r for r in d}  # per instance, not per run
        gate_by_inst = {
            iid: any(x.get("gate_accepted") or x.get("reproduces_on_buggy")
                     for x in d if x["instance_id"] == iid)
            for iid in d_by
        }
        f_by = fix_by_instance([r for r in fix
                                if r["instance_id"] in d_by
                                and short_model(r["model"]) == model
                                and arm_of(r) == arm])
        resolved = sum(1 for r in f_by.values() if r.get("resolved"))
        gated = sum(1 for v in gate_by_inst.values() if v)
        # BRT quality for this arm. The deployable records carry the offline
        # L0-L4 label whenever the run used --score-brt, but block [1] only sees
        # mode=="agent", so a 07-only model (every sweep model) reported no C2P
        # at all. The label is ours to read here; the GATE still never sees it.
        # `--score-brt` is optional, so denominators count LABELLED instances
        # and the unlabelled remainder is reported rather than folded in.
        lab = brt_by_instance(d)
        hist = collections.Counter(
            (r.get("best_level") if r.get("best_level") is not None else -1)
            for r in lab.values()
        )
        L = {k: hist.get(k, 0) for k in range(5)}
        labelled = sum(L.values())
        deployable.append({
            "model": f"{model}  [{arm}]",
            "local": is_local(d[0]["model"]),
            "runs": len(d),
            "instances": len(d_by),
            "gate_accept": gated,
            "gate_rate": rate(gated, len(d_by)),
            "scored": len(f_by),
            "resolved": rate(resolved, len(f_by)),
            "pipeline_resolve": rate(resolved, len(d_by)),
            "levels": [L[0], L[1], L[2], L[3], L[4]],
            "labelled": labelled,
            "unlabelled": hist.get(-1, 0),
            "c2p": rate(L[4], L[3] + L[4]),
            "valid": rate(L[4], labelled),
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


# ── the main experiment (webapp Results page) ───────────────────────────────
# One model, the frozen samples (base + extension, pooled), two arms run over the
# same instances. Every number below is over those instances only; development
# runs never mix in.

# (label, run groups in precedence order, flags.arm of the fix-phase record).
# The base sample ran each arm as its own block (rq1-sweep = test-first,
# rq1-armA = reactive); the extension runs both arms as one unit per instance,
# so rq1-ext and rq1-ext-v2 serve both. rq1-ext-v2 is the rest of the extension,
# run with the Django-native BRT code (identical to rq1-ext's code elsewhere).
#
# rq1-djv2 SUBSTITUTES test-first on Django: the original Django BRTs ran under
# a preamble with no INSTALLED_APPS, an environment fault, so test-first was
# re-run on all 40 paired Django instances (reactive kept). When an instance has
# units in several of an arm's groups, the LATER-listed group wins (never the
# later timestamp), and its fix/test records come from that same group.
EXPERIMENT_ARMS = (
    ("test-first", ("rq1-sweep", "rq1-ext", "rq1-ext-v2", "rq1-djv2"), "deployable"),
    ("reactive", ("rq1-armA", "rq1-ext", "rq1-ext-v2"), "reactive"),
)
# The same experiment without the Django substitution (original test-first
# records), reported alongside the corrected numbers.
EXPERIMENT_ARMS_ORIGINAL = (
    ("test-first", ("rq1-sweep", "rq1-ext", "rq1-ext-v2"), "deployable"),
    ("reactive", ("rq1-armA", "rq1-ext", "rq1-ext-v2"), "reactive"),
)
EXPERIMENT_RUN_GROUPS = frozenset(g for _, gs, _ in EXPERIMENT_ARMS for g in gs)
_UNIT_SIDE = {"test-first": "arm_b", "reactive": "arm_a"}


def unit_arms(rec: dict, arms=EXPERIMENT_ARMS) -> set[str]:
    """Experiment arms a `sweep_unit` record completed: its run group must belong
    to the arm AND the unit must carry that arm's outcome (an AB unit carries both)."""
    if rec.get("mode") != "sweep_unit":
        return set()
    group = _flag(rec, "run_group")
    return {label for label, groups, _ in arms
            if group in groups and rec.get(_UNIT_SIDE[label]) is not None}

# Why an attempt did not resolve, decided from the scorer's own lists of
# hidden tests that failed. SWE-bench records WHICH tests failed, not how many
# there were, so a partial fix (9 of 10) and a total miss look the same here.
FAILURE_KINDS = (
    ("resolved", "Resolved"),
    ("f2p_fail", "The bug's hidden tests (F2P) still fail"),
    ("p2p_fail", "Bug's tests pass, but an existing test (P2P) broke"),
    ("not_applied", "Patch did not apply"),
    ("gate_rejected", "Gate rejected the test, so no fix was attempted"),
    ("other", "Not resolved, no detail recorded"),
)


def _latest(recs: list[dict]) -> dict[str, dict]:
    """instance_id -> the most recent record (results.jsonl is append-only)."""
    out: dict[str, dict] = {}
    for r in sorted(recs, key=lambda x: x.get("timestamp", "")):
        out[r["instance_id"]] = r
    return out


def _preferred(recs: list[dict], groups: tuple[str, ...]) -> dict[str, dict]:
    """instance_id -> its record from the highest-precedence run group (the
    later-listed one); the most recent record breaks ties within a group."""
    rank = {g: k for k, g in enumerate(groups)}
    out: dict[str, dict] = {}
    for r in sorted(recs, key=lambda x: (rank.get(_flag(x, "run_group"), -1),
                                         x.get("timestamp", ""))):
        out[r["instance_id"]] = r
    return out


def _same_group(recs: list[dict], units: dict[str, dict]) -> dict[str, dict]:
    """instance_id -> the latest of `recs` from the run group of that instance's
    chosen unit, so a substituted unit never borrows an older run's records."""
    return _latest([r for r in recs if r["instance_id"] in units
                    and _flag(r, "run_group") == _flag(units[r["instance_id"]], "run_group")])


def _failure_kind(fix_rec: dict | None, *, gated: bool = True) -> str:
    if not gated:
        return "gate_rejected"
    if fix_rec is None:
        return "other"
    a = _best_attempt(fix_rec)
    if fix_rec.get("resolved") or a.get("resolved"):
        return "resolved"
    if a.get("applied") is False:
        return "not_applied"
    if a.get("fail_to_pass_failed"):
        return "f2p_fail"
    if a.get("pass_to_pass_failed"):
        return "p2p_fail"
    return "other"


def _stop_kind(stop_cause: str | None) -> str:
    """How an agent ended: it called its submit tool itself, or it hit the
    iteration limit and was made to hand in what it had."""
    if stop_cause == "submit":
        return "stopped_itself"
    if stop_cause in ("submit_forced", "max_iterations"):
        return "ran_out_of_turns"
    return "other"


def _median(xs: list[float]) -> float | None:
    s = sorted(xs)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def _experiment_units(records: list[dict], arms_def=EXPERIMENT_ARMS
                      ) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """Each arm's chosen unit per instance, and the test-first Phase A record
    from the same run group as that instance's test-first unit."""
    units = {
        label: _preferred([r for r in records if label in unit_arms(r, arms_def)], groups)
        for label, groups, _ in arms_def
    }
    tests = _same_group([r for r in records if r.get("mode") == "agent_deployable"],
                        units["test-first"])
    return units, tests


def experiment_tests(records: list[dict], *, env_states: dict[str, str] | None = None,
                     arms_def=EXPERIMENT_ARMS) -> dict[str, dict]:
    """instance_id -> the test-first Phase A record behind the Results page, for
    every paired instance (same pairing and precedence as build_results). Its
    single attempt holds the test the pipeline handed to the fix agent."""
    units, tests = _experiment_units(records, arms_def)
    states = env_states or {}
    return {i: tests[i] for i in sorted(set(units["test-first"]) & set(units["reactive"]))
            if states.get(i, "valid") == "valid" and i in tests}


def build_results(records: list[dict], *, sample_ids: list[str] | None = None,
                  env_states: dict[str, str] | None = None,
                  arms_def=EXPERIMENT_ARMS) -> dict:
    """Numbers for the Results page: the frozen-sample experiment only.

    Instances are those with a completed unit in BOTH arms (paired), so every
    rate shares one denominator N. `sample_ids` / `env_states` (the frozen
    sample file and the environment record) are optional and only used to
    report what was left out and why. `arms_def` picks the arm definition
    (EXPERIMENT_ARMS_ORIGINAL = without the Django substitution).
    """
    units, tests = _experiment_units(records, arms_def)
    fixes = {
        label: _same_group([r for r in records if r.get("mode") == "fix_agent"
                            and _flag(r, "arm") == arm], units[label])
        for label, _, arm in arms_def
    }
    arm_flag = {label: arm for label, _, arm in arms_def}
    by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        if r.get("mode") != "sweep_unit" and _flag(r, "pair_id"):
            by_pair[_flag(r, "pair_id")].append(r)

    def arm_calls(label: str, unit: dict) -> int:
        """A one-arm unit's `calls` is that arm's; an AB unit's covers both arms,
        so count this arm's own records (same pair_id), as script 11 does."""
        if _flag(unit, "arms") != "AB":
            return unit.get("calls") or 0
        return sum(sum(((_flag(r, "served_by") or {}).values()))
                   for r in by_pair.get(_flag(unit, "pair_id"), [])
                   if _flag(r, "arm") == arm_flag[label])

    tf, rx = units["test-first"], units["reactive"]
    # The frozen rule excludes an environment-invalid instance before any arm
    # runs. The sweep enforces that, but a result must not depend on the driver
    # having done so: anything the environment record marks other than valid is
    # left out here too (instances it does not know are kept).
    states = env_states or {}
    paired = sorted(i for i in set(tf) & set(rx) if states.get(i, "valid") == "valid")
    n = len(paired)
    if not n:
        return {"n": 0}

    def resolved(label: str, iid: str) -> bool:
        u = units[label][iid]
        side = u.get("arm_b") if label == "test-first" else u.get("arm_a")
        return bool((side or {}).get("resolved"))

    # ── per arm: fixed, calls, dates, how the fix agent stopped ─────────────
    arms = []
    for label, groups, _ in arms_def:
        calls = [arm_calls(label, units[label][i]) for i in paired]
        stamps = sorted(units[label][i].get("timestamp", "") for i in paired)
        stops = collections.Counter(
            _stop_kind(_best_attempt(fixes[label][i]).get("stop_cause"))
            for i in paired if i in fixes[label])
        n_fix = sum(stops.values())
        arms.append({
            "label": label,
            "run_group": ", ".join(groups),
            "resolved": rate(sum(resolved(label, i) for i in paired), n),
            "calls_total": sum(calls),
            "calls_mean": sum(calls) / n,
            "calls_median": _median(calls),
            "first_day": stamps[0][:10] if stamps else None,
            "last_day": stamps[-1][:10] if stamps else None,
            "fix_runs": n_fix,
            "stopped_itself": rate(stops["stopped_itself"], n_fix),
            "ran_out_of_turns": rate(stops["ran_out_of_turns"], n_fix),
        })

    # ── paired outcomes: who fixed what ──────────────────────────────────────
    both, tf_only, rx_only, neither = [], [], [], []
    for i in paired:
        t, r = resolved("test-first", i), resolved("reactive", i)
        (both if t and r else tf_only if t else rx_only if r else neither).append(i)

    # ── test-first pipeline: gate and BRT levels ─────────────────────────────
    gated = [i for i in paired if tests.get(i, {}).get("gate_accepted")]
    levels = collections.Counter(int(tests[i].get("best_level") or 0)
                                 for i in paired if i in tests)
    L = [levels.get(k, 0) for k in range(5)]
    tf_resolved = sum(resolved("test-first", i) for i in paired)

    # ── why attempts did not resolve ─────────────────────────────────────────
    failure = {}
    for label, _, _ in arms_def:
        c = collections.Counter(
            _failure_kind(fixes[label].get(i),
                          gated=(label != "test-first" or i in gated))
            for i in paired)
        failure[label] = [{"key": k, "label": text, "count": c.get(k, 0)}
                          for k, text in FAILURE_KINDS
                          if not (k == "gate_rejected" and label != "test-first")]

    # ── per repository ───────────────────────────────────────────────────────
    by_repo: dict[str, list[str]] = collections.defaultdict(list)
    for i in paired:
        by_repo[repo_of(units["test-first"][i])].append(i)
    repos = [{
        "repo": repo,
        "n": len(ids),
        "gated": sum(i in gated for i in ids),
        "tf_resolved": sum(resolved("test-first", i) for i in ids),
        "rx_resolved": sum(resolved("reactive", i) for i in ids),
    } for repo, ids in sorted(by_repo.items(), key=lambda kv: (-len(kv[1]), kv[0]))]

    # ── where the test-first run stops (RQ3 stages) ─────────────────────────
    usable = [i for i in paired if tests.get(i, {}).get("best_level", 0) >= 1]
    stages = [
        {"label": "Attempted", "count": n},
        {"label": "Wrote a usable test (level L1 or higher)", "count": len(usable)},
        {"label": "Test reproduces the bug (passed the gate)", "count": len(gated)},
        {"label": "Bug resolved", "count": tf_resolved},
    ]

    # ── what the frozen sample left out ─────────────────────────────────────
    left_out = []
    if sample_ids is not None:
        for iid in sample_ids:
            if iid in paired:
                continue
            state = (env_states or {}).get(iid, "not run")
            left_out.append({"instance_id": iid, "state": state})

    first = units["test-first"][paired[0]]
    return {
        "n": n,
        "model": first.get("model"),
        "budget": {k: _flag(first, k) for k in
                   ("max_iterations", "max_tokens_per_turn", "max_total_tokens")},
        "sample_size": len(sample_ids) if sample_ids is not None else None,
        "left_out": left_out,
        "arms": arms,
        "paired": {"both": both, "tf_only": tf_only, "rx_only": rx_only,
                   "neither": neither},
        "funnel": {
            "gate_rate": rate(len(gated), n),
            "resolve_at_gated": rate(tf_resolved, len(gated)),
            "pipeline_resolve": rate(tf_resolved, n),
        },
        "levels": L,
        "c2p": rate(L[4], L[3] + L[4]),
        "valid_brt": rate(L[4], n),
        "failure": failure,
        "repos": repos,
        "stages": stages,
    }


# ── SWT-bench re-scoring (webapp /swt page) ─────────────────────────────────
# Our stored test-first BRTs scored again by the official SWT-bench harness
# (scripts/17_swt_predictions.py + 18_swt_score.py). Records live in
# swt_results.jsonl, apart from results.jsonl; published per-instance lists of
# other tools live in published_swt_lite.jsonl.

SWT_VERDICTS = (
    ("S", "S: success (strict)"),
    ("f2p_only", "F→P only: a test went fail→pass, another still fails after the fix"),
    ("no_f2p", "No F→P: no test went fail→pass"),
    ("not_applied", "Patch did not apply"),
)

# Most likely cause of each kind of disagreement, from how the two scorers differ.
_SWT_DISAGREEMENT_HINTS = {
    ("L4", "not_applied"): "conversion: the exported patch did not apply",
    ("L4", "no_f2p"): "harness ran no failing test: placement, the repo's runner, or a "
                      "missing package in its environment (e.g. pytest)",
    ("L4", "f2p_only"): "a test still fails after the fix under the harness's runner",
    ("L3", "S"): "per-test vs. per-file: the harness counts tests, we count the file",
    ("low", "S"): "the harness counts a test missing or erroring on buggy code as failing; "
                  "we label it L2",
}


def swt_verdict(rec: dict) -> str:
    m = rec.get("swt") or {}
    if not m.get("applied"):
        return "not_applied"
    if m.get("success"):
        return "S"
    return "f2p_only" if m.get("f2p") else "no_f2p"


def build_swt(records: list[dict], published: list[dict] | None = None) -> dict:
    """Numbers for the /swt page. The latest stored record per instance counts."""
    latest = _latest([r for r in records if r.get("kind") == "tdbf_test_first"])
    rows = []
    for iid, r in sorted(latest.items()):
        level = (r.get("ours") or {}).get("level")
        ours = f"L{level}" if level is not None else "L?"
        verdict = swt_verdict(r)
        agree = (level == 4) == (verdict == "S")
        key = ("low" if level is not None and level <= 2 else ours, verdict)
        rows.append({
            "instance_id": iid, "repo": r.get("repo"), "ours": ours, "verdict": verdict,
            "agree": agree, "hint": None if agree else _SWT_DISAGREEMENT_HINTS.get(key),
            "files": (r.get("source") or {}).get("files") or [],
            "coverage_delta": (r.get("swt") or {}).get("coverage_delta_pred"),
            "run_id": (r.get("harness") or {}).get("run_id"),
            "evidence": r.get("evidence"),
        })
    n = len(rows)
    if not n:
        return {"n": 0}
    count = collections.Counter(x["verdict"] for x in rows)
    m = [latest[x["instance_id"]].get("swt") or {} for x in rows]
    cov = [x["coverage_delta"] for x in rows if x["coverage_delta"] is not None]

    cross = collections.Counter((x["ours"], x["verdict"]) for x in rows)
    levels = [lvl for lvl in ("L4", "L3", "L2", "L1", "L0", "L?")
              if any(x["ours"] == lvl for x in rows)]

    by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for x in rows:
        by_repo[x["repo"]].append(x)
    repos = [{
        "repo": repo, "n": len(xs),
        "ours_l4": sum(x["ours"] == "L4" for x in xs),
        "success": sum(x["verdict"] == "S" for x in xs),
        "f2p": sum(x["verdict"] in ("S", "f2p_only") for x in xs),
    } for repo, xs in sorted(by_repo.items(), key=lambda kv: (-len(kv[1]), kv[0]))]

    # published tools, on exactly the instances we have re-scored
    scored = {x["instance_id"] for x in rows}
    ours_s = {x["instance_id"] for x in rows if x["verdict"] == "S"}
    tools = []
    for p in published or []:
        theirs = set(p.get("instances") or []) & scored
        tools.append({
            "tool": p.get("tool"), "model": p.get("model"), "metric": p.get("metric"),
            "note": p.get("note"), "source": p.get("source"),
            "theirs": rate(len(theirs), n), "both": len(theirs & ours_s),
            "only_ours": len(ours_s - theirs), "only_theirs": len(theirs - ours_s),
        })

    return {
        "n": n,
        "run_ids": sorted({x["run_id"] for x in rows if x["run_id"]}),
        "applied": rate(sum(bool(x.get("applied")) for x in m), n),
        "success": rate(count["S"], n),
        "f2p": rate(count["S"] + count["f2p_only"], n),
        "p2p": rate(sum(bool(x.get("p2p")) for x in m), n),
        "ours_l4": rate(sum(x["ours"] == "L4" for x in rows), n),
        "agreement": rate(sum(x["agree"] for x in rows), n),
        "coverage_delta_mean": (sum(cov) / len(cov)) if cov else None,
        "verdicts": SWT_VERDICTS,
        "cross": [{"level": lvl, "counts": [cross[(lvl, v)] for v, _ in SWT_VERDICTS]}
                  for lvl in levels],
        "repos": repos,
        "rows": rows,
        "disagreements": [x for x in rows if not x["agree"]],
        "tools": tools,
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
            out.append(f"  {d['model']}: runs={d['runs']}  instances={d['instances']}")
            out.append(f"    GateRate (gold-free)      = {d['gate_rate']['label']}")
            if d["labelled"]:
                L = d["levels"]
                out.append(f"    levels L0..L4             = "
                           f"{L[0]},{L[1]},{L[2]},{L[3]},{L[4]}"
                           + (f"   (+{d['unlabelled']} unlabelled)"
                              if d["unlabelled"] else ""))
                out.append(f"    C2P (L4/(L3+L4))          = {d['c2p']['label']}")
                out.append(f"    Valid-BRT (L4/labelled)   = {d['valid']['label']}")
            elif d["unlabelled"]:
                out.append(f"    BRT levels                = "
                           f"unlabelled ({d['unlabelled']}) — run with --score-brt")
            out.append(f"    Resolve@Gated             = {d['resolved']['label']}"
                       f"   (scored fixes: {d['scored']})")
            out.append(f"    Pipeline resolve (res/N)  = {d['pipeline_resolve']['label']}")

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
