"""End-to-end agentic TDBF pipeline: BRT agent -> (if L4) fix agent, in one run.

This is the single command that goes from a GitHub issue to an attempted fix:

  1. Phase A (BRT): a real agent writes a Bug Reproduction Test, scored L0-L4
     behind the gold barrier (scripts/04_agent_brt.py is the same thing, alone).
  2. GATE: only a validated L4 BRT (fail-on-buggy -> pass-on-fixed) is handed on.
     A non-reproducing / wrong test cannot correctly drive a fix.
  3. Phase B (fix): a real agent edits the buggy source until that BRT passes,
     scored resolved/not behind the gold barrier (scripts/05_agent_fix.py alone).

The BRT travels in memory from Phase A to Phase B — no results.jsonl round-trip.
Both phases reuse `tdbf.agent.pipeline`, so the records written here are identical
to running 04 then 05 separately (the webapp shows both).

GOLD-FREE: neither agent ever runs on the gold-fixed tree or sees the held-out
test_patch. The gold patch scores the BRT offline (L0-L4); the held-out test_patch
scores the fix offline (resolved). Both happen after the respective agent finishes.

Usage:
  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907   # build image
  python3 scripts/06_agent_pipeline.py --id astropy__astropy-12907 \\
      --model anthropic:claude-sonnet-4-6 -v
  python3 scripts/06_agent_pipeline.py --limit 5 --model anthropic:claude-sonnet-4-6
  python3 scripts/06_agent_pipeline.py --id astropy__astropy-12907 --brt-only   # stop after Phase A
  python3 scripts/06_agent_pipeline.py --id astropy__astropy-12907 --multi      # + Phase B Reviewer

`--multi` adds multi-agent review to the FIX phase only (a Reviewer that judges the
candidate fix; the same model self-reviews). Phase A (BRT) is always single-agent —
the BRT is only a means to drive the fix, and a Critic there was a near-no-op on a
capable model. The single-agent run (no flag) is the baseline. Roles are gold-free.
See `tdbf.agent.roles`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent.pipeline import (
    brt_run_record,
    brt_transcript_dict,
    fix_run_record,
    fix_transcript_dict,
    run_brt_phase,
    run_fix_phase,
)
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import save_run


@dataclass
class PipelineOutcome:
    """One instance through the whole pipeline."""

    instance_id: str
    brt_level: int
    is_valid_brt: bool
    resolved: bool | None  # None = fix phase skipped (BRT not L4, or --brt-only)
    cost_usd: float

    @property
    def summary(self) -> str:
        brt = f"BRT L{self.brt_level}"
        if self.resolved is None:
            fix = "fix=skipped"
        else:
            fix = f"fix={'RESOLVED' if self.resolved else 'not-resolved'}"
        return f"{self.instance_id}: {brt}  {fix}  (${self.cost_usd:.2f})"


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
    verbose: bool,
    num_ctx: int | None,
    brt_dir: Path,
    fix_dir: Path,
    multi: bool = False,
    review_rounds: int = 1,
) -> PipelineOutcome | None:
    print(f"\n{'#'*60}\n# Instance:  {inst.instance_id}\n# Repo:      {inst.repo}\n{'#'*60}")

    client = build_client(model_spec)
    if isinstance(client, OllamaClient):
        if think:
            client._think = True
        if num_ctx is not None:
            client._num_ctx = num_ctx
        print(f"Model:     {client.name}  (num_ctx={client._num_ctx})")
    else:
        print(f"Model:     {client.name}")

    # ===================== PHASE A — BRT agent (always single-agent) ======== #
    # The BRT is only a means to drive the fix; multi-agent effort lives in
    # Phase B. (A Critic here was a near-no-op on a capable model: revise-rate
    # 1/14 on Sonnet — so Phase A stays single-agent.)
    print("\n--- Phase A: BRT agent ---")
    brt = run_brt_phase(
        inst, client,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        prompt_harden=prompt_harden,
        keep_container=keep_container,
        verbose=verbose,
    )
    if brt is None:  # container could not start
        return None

    brt_dir.mkdir(parents=True, exist_ok=True)
    (brt_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(brt_transcript_dict(inst, client.name, brt), indent=2)
    )
    brt_flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
                 "prompt_harden": prompt_harden,
                 "num_ctx": (client._num_ctx if isinstance(client, OllamaClient) else None),
                 "submit_nudge_limit": brt.submit_nudge_limit,
                 "multi": False,  # Phase A is always single-agent now
                 "pipeline": True}
    save_run(brt_run_record(inst, client.name, split=split, flags=brt_flags, phase=brt))

    cost = brt.cost_usd or 0.0

    # ===================== GATE — L4 only ===================== #
    if not brt.is_valid_brt:
        print(f"\n[gate] BRT is L{brt.level} (not a validated L4) — skipping the fix phase. "
              "Only a fail-on-buggy/pass-on-fixed test can correctly drive a fix.")
        return PipelineOutcome(inst.instance_id, brt.level, False, None, cost)
    if brt_only:
        print(f"\n[gate] --brt-only: BRT is L4; stopping before the fix phase.")
        return PipelineOutcome(inst.instance_id, brt.level, True, None, cost)

    # ===================== PHASE B — fix agent (+ optional Reviewer) ======== #
    print(f"\n--- Phase B: fix agent{' + Reviewer' if multi else ''} (driving the L4 BRT) ---")
    fix = run_fix_phase(
        inst, brt.test_code, client,
        split=split,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        keep_container=keep_container,
        verbose=verbose,
        run_dir=fix_dir,
        reviewer_client=client if multi else None,  # self-review
        reviewer_rounds=review_rounds if multi else 0,
    )
    if fix is None:  # container could not start
        return PipelineOutcome(inst.instance_id, brt.level, True, None, cost)

    fix_dir.mkdir(parents=True, exist_ok=True)
    (fix_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(fix_transcript_dict(inst, client.name, fix), indent=2)
    )
    fix_flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
                 "multi": multi, "review_rounds": review_rounds if multi else 0,
                 "pipeline": True}
    save_run(fix_run_record(inst, client.name, split=split, flags=fix_flags, phase=fix))

    cost += fix.cost_usd or 0.0
    return PipelineOutcome(inst.instance_id, brt.level, True, fix.resolved, cost)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-sonnet-4-6",
                    help="Model spec used for BOTH phases: anthropic:<model>, "
                         "ollama:<model>@<url>, openai:<model>")
    ap.add_argument("--max-iterations", type=int, default=20,
                    help="Iteration cap per phase (BRT and fix each get this).")
    ap.add_argument("--max-tokens", type=int, default=300_000,
                    help="Cumulative token budget per phase (hard cap).")
    ap.add_argument("--brt-only", action="store_true",
                    help="Stop after Phase A (write the BRT, skip the fix).")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (Ollama).")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Override Ollama KV-cache window (num_ctx); see 04_agent_brt.py.")
    ap.add_argument("--prompt-harden", action="store_true",
                    help="Append the tool-use protocol block to the BRT prompt (weak-arm ladder).")
    ap.add_argument("--multi", action="store_true",
                    help="Add multi-agent review to the FIX phase (Phase B Reviewer), same "
                         "model self-reviewing. Phase A (BRT) is always single-agent. "
                         "Single-agent (no flag) is the baseline.")
    ap.add_argument("--review-rounds", type=int, default=1,
                    help="Max Reviewer revise rounds in Phase B when --multi is set.")
    ap.add_argument("--keep-container", action="store_true",
                    help="Do not remove agent containers after the run (debugging).")
    ap.add_argument("-v", "--verbose", action="store_true", help="Stream each agent's tool trace.")
    args = ap.parse_args()

    store = InstanceStore(args.split)
    if args.id:
        instances = [store.get(args.id)]
    elif args.limit:
        instances = list(store.iter_instances(limit=args.limit))
    else:
        instances = [store.first()]

    brt_dir = Path("runs") / "agent_brt"
    fix_dir = Path("runs") / "agent_fix"
    outcomes: list[PipelineOutcome] = []
    for inst in instances:
        o = run_one(
            inst, args.model,
            split=args.split,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            prompt_harden=args.prompt_harden,
            brt_only=args.brt_only,
            keep_container=args.keep_container,
            verbose=args.verbose,
            num_ctx=args.num_ctx,
            brt_dir=brt_dir,
            fix_dir=fix_dir,
            multi=args.multi,
            review_rounds=args.review_rounds,
        )
        if o is not None:
            outcomes.append(o)

    # ===================== Aggregate report ===================== #
    print(f"\n{'='*60}\nPipeline summary ({len(outcomes)} instance(s)):")
    for o in outcomes:
        print(f"  {o.summary}")

    l4 = [o for o in outcomes if o.is_valid_brt]
    l3 = [o for o in outcomes if o.brt_level == 3]
    c2p_denom = len(l4) + len(l3)
    if c2p_denom:
        print(f"\nC2P (L4 / (L3+L4)):   {len(l4)}/{c2p_denom} = {len(l4)/c2p_denom:.2f}")
    scored = [o for o in outcomes if o.resolved is not None]
    if scored:
        n_res = sum(1 for o in scored if o.resolved)
        print(f"Resolved (of fixes):  {n_res}/{len(scored)} = {n_res/len(scored):.2f}")
    total_cost = sum(o.cost_usd for o in outcomes)
    if total_cost:
        print(f"Total cost:           ${total_cost:.2f}")

    # exit 0 if anything resolved, else 0 if any valid BRT (brt-only), else 1
    if any(o.resolved for o in outcomes):
        return 0
    if args.brt_only and any(o.is_valid_brt for o in outcomes):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
