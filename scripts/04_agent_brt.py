"""Phase A: a single autonomous BRT agent.

A real agent (model-driven control) writes a Bug Reproduction Test for a
SWE-bench instance. The model is given the issue + gold-free tools (read_file,
grep, glob, run_python, run_test-on-buggy, submit_test) and decides the
trajectory itself: explore -> observe buggy behaviour -> write test -> run it ->
iterate -> submit. Termination is the model's call (the terminal submit_test
tool), not a fixed loop.

This is the agentic counterpart to scripts/02_generate_brt.py (the workflow
baseline). The same Agent loop runs on Claude or qwen by changing --model (the
weak-arm capability comparison — a scope condition, not an RQ).

GOLD-FREE: the agent's tools run only on the buggy checkout. The gold patch is
applied ONLY in the offline scoring section below, after the agent has finished,
to assign an L0-L4 level. The agent never observes the fixed-tree outcome.

Usage:
  export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
  python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907   # build image
  python3 scripts/04_agent_brt.py --id astropy__astropy-12907 --model anthropic:claude-opus-4-8 -v
  python3 scripts/04_agent_brt.py --id astropy__astropy-12907 \\
      --model ollama:qwen2.5-coder@http://192.168.1.192:11434       # weak baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent.pipeline import brt_run_record, brt_transcript_dict, run_brt_phase
from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import save_run
from tdbf.validation.brt_levels import BRTClassification, c2p_rate


def run_one(
    inst: SWEBenchInstance,
    model_spec: str,
    *,
    split: str,
    max_iterations: int,
    max_total_tokens: int,
    think: bool,
    prompt_harden: bool,
    keep_container: bool,
    verbose: bool,
    transcript_dir: Path,
    num_ctx: int | None = None,
) -> BRTClassification | None:
    print(f"\n{'='*60}\nInstance:  {inst.instance_id}\nRepo:      {inst.repo}")

    client = build_client(model_spec)
    if isinstance(client, OllamaClient):
        if think:
            client._think = True
        if num_ctx is not None:
            client._num_ctx = num_ctx
        print(f"Model:     {client.name}  (num_ctx={client._num_ctx})")
    else:
        print(f"Model:     {client.name}")

    # --- Run the gold-free BRT agent + offline gold-barrier scoring. ---
    phase = run_brt_phase(
        inst, client,
        max_iterations=max_iterations,
        max_total_tokens=max_total_tokens,
        prompt_harden=prompt_harden,
        keep_container=keep_container,
        verbose=verbose,
    )
    if phase is None:  # container could not start
        return None

    # --- Persist full transcript + summary record (shared builders). ---
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{inst.instance_id}.json").write_text(
        json.dumps(brt_transcript_dict(inst, client.name, phase), indent=2)
    )
    flags = {"max_iterations": max_iterations, "max_total_tokens": max_total_tokens,
             "prompt_harden": prompt_harden,
             "num_ctx": (client._num_ctx if isinstance(client, OllamaClient) else None),
             "submit_nudge_limit": phase.submit_nudge_limit}
    save_run(brt_run_record(inst, client.name, split=split, flags=flags, phase=phase))
    return phase.classification


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run the first N instances of the split (ignored if --id given).")
    ap.add_argument("--model", default="anthropic:claude-opus-4-8",
                    help="Model spec: anthropic:<model>, ollama:<model>@<url>, openai:<model>")
    ap.add_argument("--max-iterations", type=int, default=20,
                    help="Iteration cap. 15 was too low for harder instances "
                         "(astropy-6938 timed out mid-exploration → spurious L0; "
                         "submitted L4 at iter 13 with cap 25).")
    ap.add_argument("--max-tokens", type=int, default=200_000,
                    help="Cumulative token budget for one agent run (hard cap).")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (Ollama).")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Override Ollama KV-cache window (num_ctx). Default keeps the "
                         "client's 8192. Set to the model's max (e.g. 32768 for qwen2.5-coder) "
                         "to stop file reads from scrolling out of context (RQ1 failure mode #2). "
                         "Higher windows cost VRAM/decode speed.")
    ap.add_argument("--prompt-harden", action="store_true",
                    help="Append the tool-use protocol block (RQ1 scaffolding ladder). "
                         "Targets weak-model failures: tool-call-as-text, hallucinated output, "
                         "addressing the user. Logged in results flags.")
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

    transcript_dir = Path("runs") / "agent_brt"
    classifications: list[BRTClassification] = []
    for inst in instances:
        c = run_one(
            inst, args.model,
            split=args.split,
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_tokens,
            think=args.think,
            prompt_harden=args.prompt_harden,
            keep_container=args.keep_container,
            verbose=args.verbose,
            transcript_dir=transcript_dir,
            num_ctx=args.num_ctx,
        )
        if c is not None:
            classifications.append(c)

    # --- Aggregate report ---
    if len(instances) > 1:
        print(f"\n{'='*60}\nBatch of {len(instances)} instance(s):")
        dist = {}
        for c in classifications:
            dist[c.level.name] = dist.get(c.level.name, 0) + 1
        for name in sorted(dist):
            print(f"  {name}: {dist[name]}")
        rate = c2p_rate(classifications)
        print(f"C2P rate (L4 / (L3+L4)): {rate if rate is not None else 'n/a (no candidates)'}")

    any_valid = any(c.is_valid_brt for c in classifications)
    return 0 if any_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
