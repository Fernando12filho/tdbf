"""Paired sweep for RQ1: reactive (Arm A) vs test-first (Arm B), one instance at a time.

Each instance is one unit: both arms run back to back, in an order drawn from a
seeded coin flip per instance, and are scored with the verified scorer. Why this
shape (decided 2026-09-16):

  - Back to back, not "all A then all B": free endpoints drift (load, throttling,
    provider or silent model changes). Blocking the arms would confound arm with
    time; pairing puts both arms of an instance under the same conditions.
  - A unit is written only when BOTH arms finished. If the daily cap hits or a
    model disappears mid-sweep, every recorded unit is still a complete pair —
    McNemar only uses complete pairs. A failed unit leaves no records at all and
    is simply retried on the next run.
  - Random order within the pair (recorded) cancels any first-mover effect.
  - Sequential: the binding limit is requests per day, not wall clock. One lane
    uses a day's free quota in a few hours; parallel lanes would only add 429s
    and container contention that can land unevenly on the arms.

Arm B goes through `07_deployable_pipeline.run_one` itself, so the sweep's
test-first arm cannot drift from script 07. Arm A is `run_fix_phase(inst, None,
...)` — the same fix agent without run_brt and without Phase A.

Records: Arm A writes a `fix_agent` record (flags.arm="reactive"); Arm B writes
its usual `agent_deployable` (+ `fix_agent` if the gate accepted) records
(flags.arm="deployable"); then one `sweep_unit` summary. All share
flags.pair_id. The RQ1 analysis reads `sweep_unit` records with arms == "AB".
Arm B counts as not resolved when its gate rejects — the pipeline declined to fix.

Budgets default per model: free OpenRouter models get 16k tokens per turn and
1M total (they reason at length and cannot cache); others keep 07's 4,096 /
300k. Both arms always get the same values, recorded on every record.

    # tomorrow's capability probe: test-first arm only, one model, five instances
    python3 scripts/11_paired_sweep.py --model openrouter:google/gemma-4-31b-it:free \\
        --arms B --run-group probe-0917 --ids astropy__astropy-12907 django__django-11099

    # the RQ1 sweep over the frozen sample
    python3 scripts/11_paired_sweep.py --model openrouter:<chosen>:free \\
        --arms AB --run-group rq1 --sample sample_lite_frozen.json

Exit status: 0 finished, 3 stopped at the daily quota, 4 stopped after repeated
failures. Re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.agent.pipeline import fix_run_record, fix_transcript_dict, run_fix_phase, take_served
from tdbf.data.instances import InstanceStore
from tdbf.models.llm import build_client, calls_served_total, is_upstream_throttle
from tdbf.results import RESULTS_FILE, save_run

FREE_BUDGETS = (16_000, 1_000_000)      # (tokens per turn, total tokens) — decided 09-16
DEFAULT_BUDGETS = (4_096, 300_000)      # script 07's defaults

EXIT_OK, EXIT_QUOTA, EXIT_FAILURES = 0, 3, 4

_M07 = None


def m07():
    """Script 07 as a module (its file name is not importable)."""
    global _M07
    if _M07 is None:
        spec = importlib.util.spec_from_file_location(
            "deployable_pipeline", ROOT / "scripts" / "07_deployable_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _M07 = mod
    return _M07


# ── small pieces ──────────────────────────────────────────────────────────────

def default_budgets(model_spec: str) -> tuple[int, int]:
    if model_spec.startswith("openrouter:") and model_spec.endswith(":free"):
        return FREE_BUDGETS
    return DEFAULT_BUDGETS


def arm_order(seed: int, instance_id: str) -> str:
    """'AB' or 'BA' — a coin flip seeded by (seed, instance), so it does not
    depend on processing order and a resumed sweep draws the same order."""
    return "AB" if random.Random(f"{seed}:{instance_id}").random() < 0.5 else "BA"


def is_quota_error(exc: BaseException) -> bool:
    """A 429 that survived the client's own retries and is OURS — the daily cap
    or account throttling. An upstream shared-pool throttle is not: the client
    already waited it out several times, and if it persists the unit fails and is
    retried later rather than ending the sweep (first live run, 2026-09-16, was
    stopped by exactly this on gemma-4-31b's Google AI Studio pool)."""
    if is_upstream_throttle(exc):
        return False
    if getattr(exc, "status_code", None) == 429:
        return True
    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, openai.RateLimitError)


class UnitFailed(Exception):
    """An arm could not produce a trustworthy outcome (infrastructure, not model)."""


def calls_in(records: list[dict]) -> int:
    """Model calls a unit used: exact from OpenRouter provenance when present,
    otherwise the recorded turns (which miss BRT retry attempts)."""
    n = 0
    for r in records:
        served = (r.get("flags") or {}).get("served_by")
        if served:
            n += sum(served.values())
        else:
            n += sum(len(a.get("step_summary") or []) for a in r.get("attempts") or [])
    return n


class Usage:
    """Model calls per UTC day, persisted, so a resumed sweep knows today's spend.
    Whether OpenRouter resets at UTC midnight is not documented — treat the
    budget as a soft guard; the quota error is the hard stop."""

    def __init__(self, path: Path, daily_budget: int) -> None:
        self.path, self.daily_budget = path, daily_budget
        self.data = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def used_today(self) -> int:
        return int(self.data.get(self._today(), 0))

    def remaining(self) -> int:
        return self.daily_budget - self.used_today()

    def add(self, n: int) -> None:
        self.data[self._today()] = self.used_today() + n
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def env_states(env_path: Path) -> dict[str, str]:
    """{instance_id: state} from the environment record written by 13_build_sample.

    States are 'valid' / 'pending_recheck' / 'deferred'. Missing file -> {}.
    """
    if not env_path.exists():
        return {}
    data = json.loads(env_path.read_text())
    ins = data.get("instances", data) if isinstance(data, dict) else data
    if isinstance(ins, dict):
        return {k: (v or {}).get("state", "?") for k, v in ins.items()}
    out = {}
    for x in ins:
        if isinstance(x, dict) and x.get("instance_id"):
            out[x["instance_id"]] = x.get("state", "?")
    return out


def drop_invalid_envs(ids: list[str], env_path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Split ids into (runnable, [(excluded id, state), ...]).

    The frozen sample's pre-registered rule: an instance whose image cannot be
    built, or whose GOLD patch does not resolve in our harness, is excluded
    BEFORE any arm runs -- deciding that from the gold check alone, never from
    agent results. Nothing enforced it: the sweep read the id list and ran
    whatever was in it, so matplotlib-23913 (gold_resolved: false) was scored as
    an ordinary unresolved result. An instance we cannot score correctly must be
    reported as excluded, not counted as a failure the agent caused.
    """
    states = env_states(env_path)
    if not states:
        return ids, []
    keep, dropped = [], []
    for i in ids:
        state = states.get(i)
        if state is None or state == "valid":
            keep.append(i)          # unknown to the record -> caller's problem, not ours
        else:
            dropped.append((i, state))
    return keep, dropped


class Tee:
    """Mirror stdout/stderr into a log file.

    The 2026-09-21/22 sweep printed its 503s to the terminal and nothing else, so
    when three instances failed there was no way afterwards to see why. Anything
    the sweep says about a failure has to survive the session.
    """

    def __init__(self, stream, handle) -> None:
        self._stream, self._handle = stream, handle

    def write(self, text: str) -> int:
        self._handle.write(text)
        self._handle.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        self._handle.flush()
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def completed_units(results_path: Path, model_name: str, run_group: str, arms: str) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        f = r.get("flags") or {}
        if (r.get("mode") == "sweep_unit" and r.get("model") == model_name
                and f.get("run_group") == run_group and f.get("arms") == arms):
            done.add(r["instance_id"])
    return done


# ── the two arms ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    model: str
    arms: str
    run_group: str
    split: str = "lite"
    seed: int = 0
    max_iterations: int = 30
    max_tokens_per_turn: int = 4_096
    max_total_tokens: int = 300_000
    num_ctx: int | None = None
    score_brt: bool = True
    verbose: bool = False
    daily_budget: int = 950
    min_remaining: int = 60
    max_consecutive_failures: int = 3
    out_dir: Path = field(default_factory=lambda: Path("runs") / "sweep")


def _budget_flags(cfg: Config) -> dict:
    return {"max_iterations": cfg.max_iterations, "max_total_tokens": cfg.max_total_tokens,
            "max_tokens_per_turn": cfg.max_tokens_per_turn}


def run_arm_a(inst, cfg: Config, common: dict, records: list[dict]) -> dict:
    """Reactive baseline: the fix agent without Phase A and without run_brt."""
    client = build_client(cfg.model, num_ctx=cfg.num_ctx)
    base = cfg.out_dir / cfg.run_group
    fix = run_fix_phase(
        inst, None, client, split=cfg.split,
        max_iterations=cfg.max_iterations, max_total_tokens=cfg.max_total_tokens,
        max_tokens_per_turn=cfg.max_tokens_per_turn, verbose=cfg.verbose,
        run_dir=base / "predictions",
    )
    if fix is None:
        raise UnitFailed("Arm A: container could not start")
    if fix.resolved is None:
        raise UnitFailed(f"Arm A: unverified verdict ({fix.scoring_error})")
    (base / "A").mkdir(parents=True, exist_ok=True)
    (base / "A" / f"{inst.instance_id}.json").write_text(
        json.dumps(fix_transcript_dict(inst, client.name, fix), indent=2))
    flags = {**_budget_flags(cfg), "pipeline": "reactive", "arm": "reactive",
             "run_group": cfg.run_group, **common}
    served = take_served(client)
    if served is not None:
        flags["served_by"] = served
    records.append(fix_run_record(inst, client.name, split=cfg.split, flags=flags, phase=fix))
    r = fix.agent
    return {"resolved": bool(fix.resolved), "stop_cause": r.stop_cause,
            "iterations": r.iterations, "truncated_turns": r.truncated_turns,
            "patch_chars": len(fix.fix_patch), "cost_usd": fix.cost_usd}


def run_arm_b(inst, cfg: Config, common: dict, records: list[dict]) -> dict:
    """Test-first: script 07's run_one, unchanged."""
    base = cfg.out_dir / cfg.run_group
    out = m07().run_one(
        inst, cfg.model, split=cfg.split,
        max_iterations=cfg.max_iterations, max_total_tokens=cfg.max_total_tokens,
        max_tokens_per_turn=cfg.max_tokens_per_turn,
        think=False, prompt_harden=False, brt_only=False, keep_container=False,
        score_brt=cfg.score_brt, run_group=cfg.run_group, verbose=cfg.verbose,
        num_ctx=cfg.num_ctx, brt_dir=base / "B_brt", fix_dir=base / "B_fix",
        extra_flags=common, save=records.append,
    )
    if out is None:
        raise UnitFailed("Arm B: container could not start for the BRT phase")
    if out.gate_accepted and out.resolved is None:
        raise UnitFailed("Arm B: fix phase did not produce a verified verdict")
    return {"gate_accepted": out.gate_accepted, "reproduces": out.reproduces_on_buggy,
            "brt_level": out.brt_level,
            # A rejected gate means the pipeline declined to fix: not resolved.
            "resolved": bool(out.resolved) if out.gate_accepted else False,
            "cost_usd": out.cost_usd}


def run_unit(inst, cfg: Config) -> tuple[dict, list[dict]]:
    """Run the configured arm(s) on one instance. Returns (unit, records).
    Raises UnitFailed / quota errors; nothing is persisted here."""
    pair_id = uuid.uuid4().hex[:12]
    order = arm_order(cfg.seed, inst.instance_id) if cfg.arms == "AB" else cfg.arms
    common = {"pair_id": pair_id, "order": order, "arms": cfg.arms}
    records: list[dict] = []
    outcomes: dict[str, dict] = {}
    t0 = time.time()
    for arm in order:
        print(f"\n=== {inst.instance_id}: Arm {arm} "
              f"({'reactive' if arm == 'A' else 'test-first'}) — order {order} ===")
        runner = run_arm_a if arm == "A" else run_arm_b
        outcomes[arm] = runner(inst, cfg, common, records)
    unit = {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": build_client(cfg.model, num_ctx=cfg.num_ctx).name,
        "split": cfg.split,
        "mode": "sweep_unit",
        "flags": {**_budget_flags(cfg), "run_group": cfg.run_group, "seed": cfg.seed,
                  **common},
        "arm_a": outcomes.get("A"),
        "arm_b": outcomes.get("B"),
        "calls": calls_in(records),
        "wall_seconds": round(time.time() - t0, 1),
    }
    return unit, records


# ── the sweep ─────────────────────────────────────────────────────────────────

@dataclass
class Summary:
    status: int = EXIT_OK
    done: list[dict] = field(default_factory=list)
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    failed_calls: int = 0          # calls spent by units that produced no record
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (id, env state)
    stop_reason: str | None = None


def run_sweep(cfg: Config, instances, *, save=save_run, results_path: Path = RESULTS_FILE,
              usage_path: Path | None = None) -> Summary:
    usage = Usage(usage_path or cfg.out_dir / cfg.run_group / "usage.json", cfg.daily_budget)
    model_name = build_client(cfg.model, num_ctx=cfg.num_ctx).name
    done = completed_units(results_path, model_name, cfg.run_group, cfg.arms)
    s = Summary()
    consecutive = 0
    for inst in instances:
        if inst.instance_id in done:
            s.skipped += 1
            continue
        if usage.remaining() < cfg.min_remaining:
            s.status, s.stop_reason = EXIT_QUOTA, (
                f"daily budget: {usage.used_today()}/{cfg.daily_budget} calls used today")
            break
        before = calls_served_total()
        try:
            unit, records = run_unit(inst, cfg)
        except Exception as exc:  # noqa: BLE001 - one unit must not end the sweep
            # A failed unit still SPENT its calls. Count them before deciding
            # anything: the provider's daily quota does not refund a unit that
            # died, and usage.json used to miss this entirely because it was only
            # written on the success path.
            spent = calls_served_total() - before
            if spent:
                usage.add(spent)
            if is_quota_error(exc):
                s.status, s.stop_reason = EXIT_QUOTA, f"quota error from the provider: {exc}"
                break
            reason = f"{type(exc).__name__}: {exc}"
            print(f"\n[unit failed] {inst.instance_id}: {reason} — nothing recorded; "
                  f"spent {spent} call(s), today={usage.used_today()}/{cfg.daily_budget}; "
                  "it will be retried on the next run")
            if not isinstance(exc, UnitFailed):
                traceback.print_exc()
            s.failed.append((inst.instance_id, reason))
            s.failed_calls += spent
            consecutive += 1
            if consecutive >= cfg.max_consecutive_failures:
                s.status, s.stop_reason = EXIT_FAILURES, (
                    f"{consecutive} consecutive unit failures — last: {reason}")
                break
            continue
        consecutive = 0
        for r in records:  # the pair is complete: now, and only now, persist it
            save(r)
        save(unit)
        usage.add(unit["calls"])
        s.done.append(unit)
        a, b = unit["arm_a"], unit["arm_b"]
        print(f"\n[unit done] {inst.instance_id}  order={unit['flags']['order']}  "
              f"A={'—' if a is None else a['resolved']}  "
              f"B={'—' if b is None else b['resolved']}  calls={unit['calls']}  "
              f"today={usage.used_today()}/{cfg.daily_budget}")
    return s


def print_summary(cfg: Config, s: Summary) -> None:
    print(f"\n{'=' * 64}\nSweep {cfg.run_group!r} — {cfg.model} — arms {cfg.arms}")
    print(f"  units done this run: {len(s.done)}   already done (skipped): {s.skipped}   "
          f"failed: {len(s.failed)}")
    for iid, why in s.failed:
        print(f"    failed {iid}: {why}")
    if s.excluded:
        print(f"  excluded (environment not valid): {len(s.excluded)}")
        for iid, state in s.excluded:
            print(f"    excluded {iid}: {state}")
    if s.done:
        if cfg.arms == "AB":
            a_only = sum(1 for u in s.done if u["arm_a"]["resolved"] and not u["arm_b"]["resolved"])
            b_only = sum(1 for u in s.done if u["arm_b"]["resolved"] and not u["arm_a"]["resolved"])
            both = sum(1 for u in s.done if u["arm_a"]["resolved"] and u["arm_b"]["resolved"])
            print(f"  this run: both resolved {both}, reactive only {a_only}, "
                  f"test-first only {b_only}, neither {len(s.done) - both - a_only - b_only}")
        for key, label in (("arm_a", "reactive"), ("arm_b", "test-first")):
            got = [u[key] for u in s.done if u[key] is not None]
            if got:
                print(f"  {label:10s} resolved {sum(g['resolved'] for g in got)}/{len(got)}")
        print(f"  model calls this run: {sum(u['calls'] for u in s.done)}"
              + (f" (+{s.failed_calls} on failed units)" if s.failed_calls else ""))
    elif s.failed_calls:
        print(f"  model calls this run: 0 ({s.failed_calls} spent on failed units)")
    if s.stop_reason:
        print(f"  STOPPED: {s.stop_reason}")
    print("  Re-run the same command to resume.")


def _load_ids(args) -> list[str]:
    ids: list[str] = list(args.ids or [])
    if args.sample:
        data = json.loads(Path(args.sample).read_text())
        ids += data["instances"] if isinstance(data, dict) else data
    if not ids:
        raise SystemExit("give --ids and/or --sample")
    return ids[: args.limit] if args.limit else ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="e.g. openrouter:google/gemma-4-31b-it:free")
    ap.add_argument("--arms", choices=["AB", "A", "B"], default="AB",
                    help="AB = paired RQ1 unit; B alone = the capability probe")
    ap.add_argument("--run-group", required=True, help="tags every record; resume key")
    ap.add_argument("--ids", nargs="+", help="instance ids")
    ap.add_argument("--sample", help="JSON file: a list of ids, or {'instances': [...]}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--seed", type=int, default=0, help="seeds the per-instance arm order")
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--max-tokens-per-turn", type=int, default=None,
                    help="default: 16000 for free OpenRouter models, else 4096")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="total per phase; default: 1000000 for free OpenRouter models, else 300000")
    ap.add_argument("--num-ctx", type=int, default=None, help="Ollama only")
    ap.add_argument("--no-score-brt", action="store_true",
                    help="skip the offline L0-L4 label (it costs Docker time, no model calls)")
    ap.add_argument("--daily-budget", type=int, default=950,
                    help="soft cap on model calls per UTC day (free tier: 1,000)")
    ap.add_argument("--min-remaining", type=int, default=60,
                    help="do not start a unit with fewer calls than this left today")
    ap.add_argument("--env-record", default="sample_lite_env.json",
                    help="environment record; instances not marked 'valid' are excluded "
                         "before any arm runs (frozen-sample rule)")
    ap.add_argument("--no-env-gate", action="store_true",
                    help="run instances even if the environment record does not mark them "
                         "valid — their verdicts are NOT trustworthy; say so if you report them")
    ap.add_argument("--log", default=None,
                    help="tee all output here (default: runs/sweep/<run-group>/sweep.log)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    per_turn, total = default_budgets(args.model)
    cfg = Config(
        model=args.model, arms=args.arms, run_group=args.run_group, split=args.split,
        seed=args.seed, max_iterations=args.max_iterations,
        max_tokens_per_turn=args.max_tokens_per_turn or per_turn,
        max_total_tokens=args.max_tokens or total, num_ctx=args.num_ctx,
        score_brt=not args.no_score_brt, verbose=args.verbose,
        daily_budget=args.daily_budget, min_remaining=args.min_remaining,
    )
    log_path = Path(args.log) if args.log else cfg.out_dir / cfg.run_group / "sweep.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(real_out, handle), Tee(real_err, handle)
    print(f"\n{'=' * 64}\n# sweep start {datetime.now(timezone.utc).isoformat()}  "
          f"log -> {log_path}")
    try:
        ids = _load_ids(args)
        excluded: list[tuple[str, str]] = []
        if not args.no_env_gate:
            ids, excluded = drop_invalid_envs(ids, ROOT / args.env_record)
            for iid, state in excluded:
                print(f"  [excluded] {iid}: environment state {state!r} — not scored")
        store = InstanceStore(cfg.split)
        instances = [store.get(i) for i in ids]
        print(f"Sweep {cfg.run_group!r}: {len(instances)} instance(s), model {cfg.model}, "
              f"arms {cfg.arms}, budgets {cfg.max_tokens_per_turn}/turn "
              f"{cfg.max_total_tokens} total, {cfg.max_iterations} iterations"
              + (f", {len(excluded)} excluded" if excluded else ""))
        summary = run_sweep(cfg, instances)
        summary.excluded = excluded
        print_summary(cfg, summary)
        return summary.status
    finally:
        # Put the real streams back BEFORE closing the log. Leaving sys.stdout as
        # a Tee over a closed handle makes CPython's shutdown flush raise, and the
        # interpreter then exits 120 — masking the real status (0 done / 3 quota /
        # 4 repeated failures) that a supervisor needs to tell those apart.
        sys.stdout.flush()
        sys.stdout, sys.stderr = real_out, real_err
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
