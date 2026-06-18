"""Phase B: a single autonomous fix agent.

Takes a *validated* Bug Reproduction Test (a Phase A L4 BRT) and a real agent
(model-driven control) edits the buggy source until the BRT passes — without
breaking existing tests. The model is given the issue + the BRT + gold-free tools
(read_file, grep, glob, edit_file, run_brt, run_tests, revert, submit_fix) and
decides the trajectory itself: locate the bug -> edit -> run_brt -> iterate ->
run existing tests -> submit. Termination is the model's call (terminal submit_fix).

This is Phase B of docs/agent_plan.md, downstream of scripts/04_agent_brt.py.

GOLD-FREE: the agent's tools run only on the buggy checkout. Its success signal is
the BRT (its own reproduction) plus existing repo tests — never the held-out
FAIL_TO_PASS / test_patch. The fix patch (git diff of source edits) is captured
after the agent finishes and scored offline through the SWE-bench harness
(applies the held-out test_patch, runs FAIL_TO_PASS/PASS_TO_PASS -> resolved).
The agent never observes that resolved outcome.

Usage:
  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  # needs a validated BRT in results.jsonl (run scripts/04_agent_brt.py first)
  python3 scripts/05_agent_fix.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6 -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent.pipeline import fix_run_record, fix_transcript_dict, run_fix_phase
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import load_best_brt, save_run


def run_one(
    inst: SWEBenchInstance,
    model_spec: str,
    *,
    split: str,
    max_iterations: int,
    max_total_tokens: int,
    think: bool,
    brt_override: str | None,
    keep_container: bool,
    verbose: bool,
    transcript_dir: Path,
) -> bool | None:
    print(f"\n{'='*60}\nInstance:  {inst.instance_id}\nRepo:      {inst.repo}")

    # --- Obtain the validated BRT the fix must make pass. ---
    if brt_override is not None:
        brt_code = Path(brt_override).read_text()
        brt_src = f"file:{brt_override}"
    else:
        brt = load_best_brt(inst.instance_id, min_level=4)
        if brt is None:
            print("  [skip] No validated (L4) BRT for this instance in results.jsonl. "
                  "Run scripts/04_agent_brt.py first or pass --brt-file.")
            return None
        brt_code = brt["test_code"]
        brt_src = f"results.jsonl(L{brt['level']}, {brt['model']})"
    print(f"BRT:       {brt_src}")

    client = build_client(model_spec)
    if think and isinstance(client, OllamaClient):
        client._think = True
    print(f"Model:     {client.name}")

    # --- Run the gold-free fix agent + offline gold-barrier (harness) scoring. ---
    phase = run_fix_phase(
        inst, brt_code, client,
        split=split,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        keep_container=keep_container,
        verbose=verbose,
        run_dir=transcript_dir,
    )
    if phase is None:  # container could not start
        return None
    resolved = phase.resolved

    # --- Persist transcript + summary record (shared builders). ---
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(fix_transcript_dict(inst, client.name, phase), indent=2)
    )
    flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens}
    save_run(fix_run_record(inst, client.name, split=split, flags=flags, phase=phase))
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-sonnet-4-6",
                    help="Model spec: anthropic:<model>, ollama:<model>@<url>, openai:<model>")
    ap.add_argument("--brt-file", default=None,
                    help="Path to a BRT test file to use instead of looking one up in results.jsonl.")
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=300_000,
                    help="Cumulative token budget for one agent run (hard cap).")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (Ollama).")
    ap.add_argument("--keep-container", action="store_true",
                    help="Do not remove the agent's container after the run (debugging).")
    ap.add_argument("-v", "--verbose", action="store_true", help="Stream the agent's tool trace.")
    args = ap.parse_args()

    store = InstanceStore(args.split)
    if args.id:
        instances = [store.get(args.id)]
    elif args.limit:
        instances = list(store.iter_instances(limit=args.limit))
    else:
        instances = [store.first()]

    transcript_dir = Path("runs") / "agent_fix"
    results: list[bool] = []
    for inst in instances:
        r = run_one(
            inst, args.model,
            split=args.split,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            brt_override=args.brt_file,
            keep_container=args.keep_container,
            verbose=args.verbose,
            transcript_dir=transcript_dir,
        )
        if r is not None:
            results.append(r)

    if len(results) > 1:
        n_resolved = sum(results)
        print(f"\n{'='*60}\nFix agent on {len(results)} instance(s): "
              f"resolved {n_resolved}/{len(results)} = {n_resolved/len(results):.2f}")

    return 0 if any(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
