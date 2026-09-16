"""Deployable agentic TDBF pipeline: BRT agent -> GOLD-FREE gate -> fix agent.

This is the HONEST, deployable counterpart to scripts/06_agent_pipeline.py. The
difference is the gate and what the BRT phase is allowed to look at:

  1. Phase A (BRT): a real agent writes a Bug Reproduction Test. It is scored on
     the BUGGY tree ONLY (gold-free) — the gold patch is never touched and there
     is NO L0-L4 / L4 label produced here. (L4 measurement is a separate offline
     analysis; this pipeline mirrors what a real deployment can actually observe.)
  2. GATE (gold-free): hand the test to the fix phase iff it reproduces a clean
     failure on the buggy code (assertion fail = candidate, L3 OR L4 — we cannot
     and do not distinguish them without a fix). A non-reproducing (L1) or
     crashing (L2) or absent (L0) test is rejected.
  3. Phase B (fix): a real agent edits the buggy source until that BRT passes.

The fix is the goal. After the fix agent finishes, `resolved` is measured offline
via the held-out test_patch (the research scoreboard, RQ2) — it is never fed back
into any decision. So the gold patch / held-out tests stay strictly OUT of the
loop and out of the gate; they only score the finished fix.

Contrast: 06 gates on the gold L4 label (NOT deployable; the analysis "ceiling").
This script (07) gates gold-free and is the real system.

Usage:
  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907   # build image
  python3 scripts/07_deployable_pipeline.py --id astropy__astropy-12907 \\
      --model anthropic:claude-sonnet-4-6 -v
  python3 scripts/07_deployable_pipeline.py --limit 5 --model anthropic:claude-sonnet-4-6
  python3 scripts/07_deployable_pipeline.py --id astropy__astropy-12907 --brt-only  # stop after Phase A
  python3 scripts/07_deployable_pipeline.py --id astropy__astropy-14182 --breaker   # + post-fix Breaker arm

Two-arm comparison: run WITHOUT --breaker (single-agent baseline) and WITH --breaker
(adds the post-fix adversarial verifier) over the same instances, then compare
resolved-rate. Records are tagged flags.breaker / flags.breaker_verdicts so the arms
are separable in results.jsonl and the webapp.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent.pipeline import (
    brt_transcript_dict,
    take_served,
    fix_run_record,
    fix_transcript_dict,
    run_brt_phase,
    run_fix_phase,
    step_summary,
)
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import save_run


@dataclass
class DeployOutcome:
    """One instance through the deployable pipeline."""

    instance_id: str
    reproduces_on_buggy: bool  # gold-free: did the BRT fail cleanly on buggy code
    gate_accepted: bool        # did the gold-free gate hand the BRT to the fix phase
    resolved: bool | None      # None = fix phase skipped (gate rejected, or --brt-only)
    cost_usd: float
    brt_level: int | None = None  # offline label (only with --score-brt); info only
    breaker_verdicts: list | None = None  # post-fix Breaker verdicts (only with --breaker)

    @property
    def summary(self) -> str:
        gate = "accept" if self.gate_accepted else "REJECT"
        lvl = "" if self.brt_level is None else f" L{self.brt_level}"
        brk = f"  breaker={'/'.join(self.breaker_verdicts)}" if self.breaker_verdicts else ""
        if self.resolved is None:
            fix = "fix=skipped"
        else:
            fix = f"fix={'RESOLVED' if self.resolved else 'not-resolved'}{brk}"
        return (f"{self.instance_id}: reproduces={self.reproduces_on_buggy}{lvl}  "
                f"gate={gate}  {fix}  (${self.cost_usd:.2f})")


def brt_deployable_record(
    inst: SWEBenchInstance, model_name: str, *, split: str, flags: dict, phase,
    gate_accepted: bool, scored: bool = False
) -> dict:
    """BRT run record for the deployable pipeline. The GATE is always gold-free
    (`reproduces_on_buggy`). When `scored` is True (--score-brt), we ALSO attach the
    offline L0-L4 label + gold-fixed outcome for INFORMATION only — it never touched
    the gate. When False, no gold was consulted and best_level stays None."""
    r = phase.agent
    attempt = {
        "attempt": 1,
        "mode": "agent_deployable",
        "stop_cause": r.stop_cause,
        "iterations": r.iterations,
        "truncated_turns": r.truncated_turns,
        "num_tool_calls": sum(len(s.tool_calls) for s in r.steps),
        "outcome_buggy": phase.outcome_buggy,
        "failure_reason_buggy": phase.failure_reason_buggy,
        "test_code": phase.test_code,
        "input_tokens": r.total_input_tokens,
        "output_tokens": r.total_output_tokens,
        "cache_write_tokens": phase.total_cache_write_tokens,
        "cache_read_tokens": phase.total_cache_read_tokens,
        "cost_usd": phase.cost_usd,
        "duration_seconds": sum((s.duration_seconds or 0) for s in r.steps),
        "step_summary": step_summary(r.steps),
    }
    if scored:  # offline label, informational — NOT used by the gate
        attempt["level"] = phase.level
        attempt["outcome_fixed"] = phase.outcome_fixed
    return {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": model_name,
        "split": split,
        "mode": "agent_deployable",
        "flags": flags,
        "best_level": (phase.level if scored else None),
        "reproduces_on_buggy": phase.reproduces_on_buggy,
        "gate_accepted": gate_accepted,
        "outcome_buggy": phase.outcome_buggy,
        "failure_reason_buggy": phase.failure_reason_buggy,
        "best_attempt": 1,
        "attempts": [attempt],
    }


def run_one(
    inst: SWEBenchInstance,
    model_spec: str,
    *,
    split: str,
    max_iterations: int,
    max_total_tokens: int,
    think: bool,
    prompt_harden: bool,
    brt_only: bool,
    keep_container: bool,
    score_brt: bool = False,
    breaker: bool = False,
    breaker_rounds: int = 1,
    run_group: str | None = None,
    verbose: bool,
    num_ctx: int | None,
    max_tokens_per_turn: int = 4096,
    timeout: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    brt_dir: Path,
    fix_dir: Path,
) -> DeployOutcome | None:
    print(f"\n{'#'*60}\n# Instance:  {inst.instance_id}\n# Repo:      {inst.repo}\n{'#'*60}")

    client = build_client(model_spec, num_ctx=num_ctx, timeout=timeout,
                          temperature=temperature, seed=seed)
    if isinstance(client, OllamaClient):
        if think:
            client._think = True
        print(f"Model:     {client.name}  (num_ctx={client._num_ctx} "
              f"max_tokens/turn={max_tokens_per_turn} "
              f"prompt_budget={client._num_ctx - max_tokens_per_turn} "
              f"temp={client._temperature} seed={client._seed} "
              f"timeout={client._timeout}s)")
    else:
        print(f"Model:     {client.name}")

    # ===================== PHASE A — BRT agent ===================== #
    # score_brt=True adds an OFFLINE L0-L4 label (needs the gold-fixed tree) purely
    # for inspection; it never feeds the gate. Default (False) is pure gold-free.
    print(f"\n--- Phase A: BRT agent "
          f"({'buggy-only gate + offline label' if score_brt else 'buggy-only, no gold'}) ---")
    brt = run_brt_phase(
        inst, client,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        max_tokens_per_turn=max_tokens_per_turn,
        prompt_harden=prompt_harden,
        keep_container=keep_container,
        verbose=verbose,
        score_fixed=score_brt,  # only affects the offline label, never the gate
    )
    if brt is None:  # container could not start
        return None

    # ===================== GATE — gold-free (always) ===================== #
    # Accept iff the test reproduces a clean failure on the BUGGY tree (L3 or L4 —
    # indistinguishable without a fix, and that is the point). No gold consulted,
    # even when --score-brt computed a label above.
    gate_accepted = brt.reproduces_on_buggy
    brt_lvl = brt.level if score_brt else None
    if score_brt:
        print(f"[offline label] BRT = L{brt.level}  "
              f"(buggy={brt.outcome_buggy} / fixed={brt.outcome_fixed}) — info only, NOT gated")

    brt_dir.mkdir(parents=True, exist_ok=True)
    (brt_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(brt_transcript_dict(inst, client.name, brt), indent=2)
    )
    brt_flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
                 "prompt_harden": prompt_harden,
                 "num_ctx": (client._num_ctx if isinstance(client, OllamaClient) else None),
                 "max_tokens_per_turn": max_tokens_per_turn,
                 "temperature": (client._temperature if isinstance(client, OllamaClient) else None),
                 "seed": (client._seed if isinstance(client, OllamaClient) else None),
                 "submit_nudge_limit": brt.submit_nudge_limit,
                 "pipeline": "deployable", "arm": "deployable", "run_group": run_group}
    served = take_served(client)
    if served is not None:  # OpenRouter: record what actually answered Phase A
        brt_flags["served_by"] = served
    save_run(brt_deployable_record(
        inst, client.name, split=split, flags=brt_flags, phase=brt,
        gate_accepted=gate_accepted, scored=score_brt))

    cost = brt.cost_usd or 0.0

    if not gate_accepted:
        print(f"\n[gate:gold-free] BRT does not reproduce a clean failure on the buggy code "
              f"(outcome_on_buggy={brt.outcome_buggy}, "
              f"failure_reason={brt.failure_reason_buggy}) — rejected, skipping the fix phase.")
        return DeployOutcome(inst.instance_id, brt.reproduces_on_buggy, False, None, cost, brt_lvl)
    if brt_only:
        print(f"\n[gate:gold-free] --brt-only: gate accepted; stopping before the fix phase.")
        return DeployOutcome(inst.instance_id, brt.reproduces_on_buggy, True, None, cost, brt_lvl)

    # ===================== PHASE B — fix agent (+ optional post-fix Breaker) ===== #
    # The Breaker (same model self-reviews) runs AFTER the fix, gold-free, and tries
    # to defeat it; on 'broken' the fix agent re-fixes (bounded). --breaker enables
    # it; without the flag this is the single-agent baseline arm.
    print(f"\n--- Phase B: fix agent{' + post-fix Breaker' if breaker else ''} "
          f"(driving the gold-free-accepted BRT) ---")
    fix = run_fix_phase(
        inst, brt.test_code, client,
        split=split,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        max_tokens_per_turn=max_tokens_per_turn,
        keep_container=keep_container,
        verbose=verbose,
        run_dir=fix_dir,
        breaker_client=client if breaker else None,  # self-review: same model
        breaker_rounds=breaker_rounds if breaker else 0,
    )
    if fix is None:  # container could not start
        return DeployOutcome(inst.instance_id, brt.reproduces_on_buggy, True, None, cost, brt_lvl)

    fix_dir.mkdir(parents=True, exist_ok=True)
    (fix_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(fix_transcript_dict(inst, client.name, fix), indent=2)
    )
    fix_flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
                 "max_tokens_per_turn": max_tokens_per_turn,
                 "pipeline": "deployable", "arm": "deployable", "run_group": run_group,
                 "breaker": breaker,
                 "breaker_rounds": breaker_rounds if breaker else 0,
                 "breaker_verdicts": [b.verdict for b in (fix.breaks or [])]}
    served = take_served(client)
    if served is not None:  # OpenRouter: record what actually answered Phase B
        fix_flags["served_by"] = served
    save_run(fix_run_record(inst, client.name, split=split, flags=fix_flags, phase=fix))

    cost += fix.cost_usd or 0.0
    return DeployOutcome(inst.instance_id, brt.reproduces_on_buggy, True, fix.resolved, cost, brt_lvl,
                         breaker_verdicts=[b.verdict for b in (fix.breaks or [])] or None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, nargs="+",
                    help="one or more instance_ids (default: first in split)")
    ap.add_argument("--run-group", default=None,
                    help="Tag every record from this run with flags.run_group. Metrics dedup "
                         "per (arm, instance), so re-running instances that are already in "
                         "results.jsonl without a distinct group reports the best of old and "
                         "new and hides the change being measured.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-sonnet-4-6",
                    help="Model spec used for BOTH phases: anthropic:<model>, "
                         "ollama:<model>@<url>, openai:<model>")
    ap.add_argument("--max-iterations", type=int, default=30,
                    help="Iteration cap per phase (BRT and fix each get this). On top of "
                         "this, an agent that exhausts the cap without submitting gets a "
                         "bounded forced-submit grace so a test/fix is still produced.")
    ap.add_argument("--max-tokens", type=int, default=300_000,
                    help="Cumulative token budget per phase (hard cap).")
    ap.add_argument("--brt-only", action="store_true",
                    help="Stop after Phase A (write the BRT, skip the fix).")
    ap.add_argument("--score-brt", action="store_true",
                    help="Also compute the offline L0-L4 BRT label (runs on the gold-fixed tree) "
                         "for inspection. The gate stays gold-free — the label never gates.")
    ap.add_argument("--breaker", action="store_true",
                    help="Add a post-fix adversarial Breaker (same model self-reviews) that tries "
                         "to defeat the fix gold-free; on 'broken' the fix agent re-fixes. Without "
                         "this flag the run is the single-agent baseline arm.")
    ap.add_argument("--breaker-rounds", type=int, default=1,
                    help="Max re-fix rounds a 'broken' verdict triggers (0 = record the verdict as a "
                         "confidence signal only, no re-fix). Only used with --breaker.")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (Ollama).")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Ollama context window (prompt + generation). Default 16384. "
                         "Effective prompt budget is num_ctx - max-tokens-per-turn.")
    ap.add_argument("--max-tokens-per-turn", type=int, default=4096,
                    help="Cap on ONE model turn (default 4096). Use ~1024 for local "
                         "models: an agent turn is a tool call, and the remainder of "
                         "num_ctx is what the transcript gets.")
    ap.add_argument("--timeout", type=float, default=None,
                    help="Per-request timeout in seconds (Ollama; default 900). Raise "
                         "for a large local model on a slow card.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature (Ollama). Pin it (~0.2) for agent runs; "
                         "unset means the server default, typically 0.7+.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed (Ollama). Pin for reproducible local runs.")
    ap.add_argument("--prompt-harden", action="store_true",
                    help="Append the tool-use protocol block to the BRT prompt (weak-arm ladder).")
    ap.add_argument("--keep-container", action="store_true",
                    help="Do not remove agent containers after the run (debugging).")
    ap.add_argument("-v", "--verbose", action="store_true", help="Stream each agent's tool trace.")
    args = ap.parse_args()

    store = InstanceStore(args.split)
    if args.id:
        instances = [store.get(i) for i in args.id]
    elif args.limit:
        instances = list(store.iter_instances(limit=args.limit))
    else:
        instances = [store.first()]

    brt_dir = Path("runs") / "agent_brt"
    fix_dir = Path("runs") / "agent_fix"
    outcomes: list[DeployOutcome] = []
    for inst in instances:
        o = run_one(
            inst, args.model,
            split=args.split,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            prompt_harden=args.prompt_harden,
            brt_only=args.brt_only,
            score_brt=args.score_brt,
            breaker=args.breaker,
            breaker_rounds=args.breaker_rounds,
            run_group=args.run_group,
            keep_container=args.keep_container,
            verbose=args.verbose,
            num_ctx=args.num_ctx,
            max_tokens_per_turn=args.max_tokens_per_turn,
            timeout=args.timeout,
            temperature=args.temperature,
            seed=args.seed,
            brt_dir=brt_dir,
            fix_dir=fix_dir,
        )
        if o is not None:
            outcomes.append(o)

    # ===================== Aggregate report ===================== #
    print(f"\n{'='*60}\nDeployable pipeline summary ({len(outcomes)} instance(s)):")
    for o in outcomes:
        print(f"  {o.summary}")

    accepted = [o for o in outcomes if o.gate_accepted]
    if outcomes:
        print(f"\nGold-free gate accepted:  {len(accepted)}/{len(outcomes)}")
    scored_lvls = [o.brt_level for o in outcomes if o.brt_level is not None]
    if scored_lvls:  # --score-brt: show the offline BRT quality (info only)
        hist = {lvl: scored_lvls.count(lvl) for lvl in range(5)}
        print("BRT offline label:        "
              + "  ".join(f"L{lvl}={hist[lvl]}" for lvl in range(5))
              + f"   (L4/valid = {hist[4]}/{len(scored_lvls)})")
    broke = [o for o in outcomes if o.breaker_verdicts]
    if broke:  # --breaker: how often the adversary defeated the first fix
        n_broken = sum(1 for o in broke if "broken" in o.breaker_verdicts)
        print(f"Breaker fired:            {len(broke)}   (defeated first fix on {n_broken})")
    scored = [o for o in outcomes if o.resolved is not None]
    if scored:
        n_res = sum(1 for o in scored if o.resolved)
        print(f"Resolved (of fixes):      {n_res}/{len(scored)} = {n_res/len(scored):.2f}")
    total_cost = sum(o.cost_usd for o in outcomes)
    if total_cost:
        print(f"Total cost:               ${total_cost:.2f}")

    # exit 0 if anything resolved, else 0 if any BRT cleared the gate (brt-only), else 1
    if any(o.resolved for o in outcomes):
        return 0
    if args.brt_only and any(o.gate_accepted for o in outcomes):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
