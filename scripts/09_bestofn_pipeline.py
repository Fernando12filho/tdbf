"""Best-of-N agentic TDBF pipeline: generate many BRTs -> select -> iterated fix.

Built to squeeze the most out of WEAK (local) models by sampling + verifying:

  1. Phase A (generate): run the BRT agent N times (default 5) -> N candidate tests.
  2. Score each candidate on the BUGGY tree only (gold-free) and PRE-FILTER to the
     ones that fail with a clean ASSERTION (a real reproduction).
  3. Selector agent: reads the issue + the surviving candidates and picks the best
     one (specificity) -> the gold-free gate is "a viable candidate was chosen".
  4. Phase B (fix, retry): the fix agent edits the source until the chosen BRT
     passes; if an attempt fails the gold-free check (BRT still failing), the tree
     is reverted and it retries with a hint — up to K attempts (default 3).

GOLD-FREE: every decision in the loop — candidate scoring, the viability filter,
the selection, and the fix-retry accept check — uses the buggy tree only. No agent
tool ever touches the gold patch. The gold patch / held-out test are used ONLY
afterwards, behind the gold barrier, to SCORE the result (chosen BRT -> L0-L4 with
--score-fixed; fix -> resolved). Use --gold-free-only for a run that never touches
gold at all (no L4 label, still measures resolved via the held-out test — pass
--brt-only too for zero gold usage whatsoever).

  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907   # build image
  python3 scripts/09_bestofn_pipeline.py --id astropy__astropy-12907 \\
      --model "ollama:qwen2.5-coder:14b-instruct-q3_K_M@http://192.168.1.192:11434" \\
      --num-candidates 5 --fix-attempts 3 --num-ctx 32768 -v
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
    run_brt_best_of_n,
    run_fix_phase,
)
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import save_run


@dataclass
class BestOfNOutcome:
    instance_id: str
    n_candidates: int
    n_viable: int
    chosen_index: int | None
    brt_level: int | None       # None when --gold-free-only (no L4 scoreboard)
    resolved: bool | None       # None = fix skipped (no viable BRT / --brt-only)
    cost_usd: float

    @property
    def summary(self) -> str:
        gate = f"viable {self.n_viable}/{self.n_candidates}"
        lvl = "" if self.brt_level is None else f" L{self.brt_level}"
        if self.chosen_index is None:
            fix = "gate=reject (no viable BRT)"
        elif self.resolved is None:
            fix = "fix=skipped"
        else:
            fix = f"fix={'RESOLVED' if self.resolved else 'not-resolved'}"
        return f"{self.instance_id}: {gate}{lvl}  {fix}  (${self.cost_usd:.2f})"


def run_one(
    inst: SWEBenchInstance,
    model_spec: str,
    *,
    split: str,
    n_candidates: int,
    fix_attempts: int,
    max_iterations: int,
    max_total_tokens: int,
    think: bool,
    prompt_harden: bool,
    brt_only: bool,
    score_fixed: bool,
    keep_container: bool,
    verbose: bool,
    num_ctx: int | None,
    brt_dir: Path,
    fix_dir: Path,
) -> BestOfNOutcome | None:
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

    # ===== PHASE A — best-of-N generate + select (same model self-selects) ===== #
    print(f"\n--- Phase A: best-of-{n_candidates} BRT + Selector ---")
    bon = run_brt_best_of_n(
        inst, client,
        n_candidates=n_candidates,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        selector_client=client,
        prompt_harden=prompt_harden,
        keep_container=keep_container,
        verbose=verbose,
        score_fixed=score_fixed,
    )
    if bon is None:  # container could not start
        return None

    cost = bon.cost_usd or 0.0
    brt_level = bon.brt.level if (bon.brt and score_fixed) else None

    # ===== GATE (gold-free): a viable candidate must have been chosen ===== #
    if bon.brt is None or bon.chosen_index is None:
        print(f"\n[gate] No viable BRT among {n_candidates} candidates "
              f"({bon.n_viable} clean-assertion-fails) — skipping the fix phase.")
        return BestOfNOutcome(inst.instance_id, n_candidates, bon.n_viable, None, brt_level, None, cost)

    brt_dir.mkdir(parents=True, exist_ok=True)
    (brt_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(brt_transcript_dict(inst, client.name, bon.brt), indent=2)
    )
    brt_flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
                 "prompt_harden": prompt_harden,
                 "num_ctx": (client._num_ctx if isinstance(client, OllamaClient) else None),
                 "submit_nudge_limit": bon.brt.submit_nudge_limit,
                 "bestofn": True, "n_candidates": n_candidates, "n_viable": bon.n_viable,
                 "chosen_index": bon.chosen_index,
                 "selector_reason": (bon.selection.reason if bon.selection else None),
                 "pipeline": True}
    save_run(brt_run_record(inst, client.name, split=split, flags=brt_flags, phase=bon.brt))

    if brt_only:
        print("\n[gate] --brt-only: viable BRT selected; stopping before the fix phase.")
        return BestOfNOutcome(inst.instance_id, n_candidates, bon.n_viable, bon.chosen_index,
                              brt_level, None, cost)

    # ===== PHASE B — fix with retry (gold-free accept check) ===== #
    print(f"\n--- Phase B: fix agent (up to {fix_attempts} attempts, retry on gold-free fail) ---")
    fix = run_fix_phase(
        inst, bon.brt.test_code, client,
        split=split,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        keep_container=keep_container,
        verbose=verbose,
        run_dir=fix_dir,
        max_attempts=fix_attempts,
    )
    if fix is None:
        return BestOfNOutcome(inst.instance_id, n_candidates, bon.n_viable, bon.chosen_index,
                              brt_level, None, cost)

    fix_dir.mkdir(parents=True, exist_ok=True)
    (fix_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(fix_transcript_dict(inst, client.name, fix), indent=2)
    )
    fix_flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
                 "bestofn": True, "fix_attempts": fix_attempts, "pipeline": True}
    save_run(fix_run_record(inst, client.name, split=split, flags=fix_flags, phase=fix))

    cost += fix.cost_usd or 0.0
    return BestOfNOutcome(inst.instance_id, n_candidates, bon.n_viable, bon.chosen_index,
                          brt_level, fix.resolved, cost)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-sonnet-4-6",
                    help="Model spec for BOTH generation and selection (self-select).")
    ap.add_argument("--num-candidates", type=int, default=5,
                    help="How many candidate BRTs to generate (best-of-N).")
    ap.add_argument("--fix-attempts", type=int, default=3,
                    help="Max fix attempts (gold-free retry on a still-failing BRT).")
    ap.add_argument("--max-iterations", type=int, default=20,
                    help="Iteration cap per agent run.")
    ap.add_argument("--max-tokens", type=int, default=300_000,
                    help="Cumulative token budget per agent run (hard cap).")
    ap.add_argument("--brt-only", action="store_true",
                    help="Stop after selecting a BRT (skip the fix phase).")
    ap.add_argument("--gold-free-only", action="store_true",
                    help="Do NOT score the chosen BRT on the gold-fixed tree (no L0-L4 label). "
                         "The whole run stays gold-free except the fix's resolved scoreboard.")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (Ollama).")
    ap.add_argument("--num-ctx", type=int, default=None, help="Override Ollama num_ctx.")
    ap.add_argument("--prompt-harden", action="store_true",
                    help="Append the tool-use protocol block to the BRT prompt (RQ1 weak-arm).")
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

    brt_dir = Path("runs") / "agent_bestofn"
    fix_dir = Path("runs") / "agent_fix"
    outcomes: list[BestOfNOutcome] = []
    for inst in instances:
        o = run_one(
            inst, args.model,
            split=args.split,
            n_candidates=args.num_candidates,
            fix_attempts=args.fix_attempts,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            prompt_harden=args.prompt_harden,
            brt_only=args.brt_only,
            score_fixed=not args.gold_free_only,
            keep_container=args.keep_container,
            verbose=args.verbose,
            num_ctx=args.num_ctx,
            brt_dir=brt_dir,
            fix_dir=fix_dir,
        )
        if o is not None:
            outcomes.append(o)

    print(f"\n{'='*60}\nBest-of-N pipeline summary ({len(outcomes)} instance(s)):")
    for o in outcomes:
        print(f"  {o.summary}")
    gated = [o for o in outcomes if o.chosen_index is not None]
    print(f"\nGate accepted (viable BRT):  {len(gated)}/{len(outcomes)}")
    scored = [o for o in outcomes if o.resolved is not None]
    if scored:
        n_res = sum(1 for o in scored if o.resolved)
        print(f"Resolved (of fixes):         {n_res}/{len(scored)} = {n_res/len(scored):.2f}")
    total_cost = sum(o.cost_usd for o in outcomes)
    if total_cost:
        print(f"Total cost:                  ${total_cost:.2f}")

    if any(o.resolved for o in outcomes):
        return 0
    if args.brt_only and gated:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
