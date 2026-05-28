"""Phase 1 pipeline: generate a BRT and classify it.

Generates a Bug Reproduction Test for one SWE-bench instance using a
local or cloud LLM, then evaluates it on both the buggy and gold-fixed
trees to produce a BRT level (L0–L4). Only L4 counts as a valid BRT.

Requires:
  - Docker image already built (run 01_validate_gold.py first)
  - For Ollama: DOCKER_HOST set and Ollama reachable at base_url
  - For Anthropic/OpenAI: API key env var set

Usage:
  python scripts/02_generate_brt.py \\
      --model ollama:qwen3.5:9b@http://192.168.1.192:11434 \\
      --think --context --attempts 3

  python scripts/02_generate_brt.py --model anthropic:claude-opus-4-7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.data.instances import InstanceStore
from tdbf.generation.agent import TestGenerationAgent
from tdbf.harness.runner import (
    brt_test_path,
    fetch_relevant_code,
    inject_setup_preamble,
    make_test_patch,
    repo_test_hint,
    run_snippet_in_container,
    run_test_in_container,
)
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import save_run
from tdbf.validation.brt_levels import classify


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument("--id", default=None, help="instance_id (default: first in split)")
    ap.add_argument(
        "--model",
        default="ollama:qwen3.5:9b@http://192.168.1.192:11434",
        help="Model spec: ollama:<model>@<url>, anthropic:<model>, openai:<model>",
    )
    ap.add_argument(
        "--think", action="store_true",
        help="Enable Qwen3 thinking mode (slower, higher quality)",
    )
    ap.add_argument(
        "--context", action="store_true",
        help="Include buggy source files in the prompt",
    )
    ap.add_argument(
        "--attempts", type=int, default=1, metavar="N",
        help="Max generation attempts; stops early on first L4 (default: 1)",
    )
    ap.add_argument(
        "--observe", action="store_true",
        help="Two-phase: run probe in container to observe buggy output, then generate test",
    )
    args = ap.parse_args()

    # --- Load instance ---
    store = InstanceStore(args.split)
    inst = store.get(args.id) if args.id else store.first()
    print(f"Instance:  {inst.instance_id}")
    print(f"Repo:      {inst.repo}")

    # --- Build client ---
    client = build_client(args.model)
    if args.think and isinstance(client, OllamaClient):
        client._think = True
        print(f"Model:     {client.name}  [thinking=on]")
    else:
        print(f"Model:     {client.name}")

    # --- Fetch code context (once, reused across attempts) ---
    code_context: dict[str, str] | None = None
    if args.context:
        print("Fetching source context from Docker image ...")
        code_context = fetch_relevant_code(inst)
        if code_context:
            for path, content in code_context.items():
                print(f"  {path}  ({len(content)} chars)")
        else:
            print("  (no context found — proceeding without)")

    test_path = brt_test_path(inst.repo)
    env_hint = repo_test_hint(inst.repo)
    if test_path != "test_brt.py":
        print(f"Test path: {test_path}")

    agent = TestGenerationAgent(client)

    # --- Attempt loop ---
    best_result = None
    best_generated = None
    best_outcomes = (None, None)
    best_attempt_number = None
    attempts_log: list[dict] = []

    for attempt in range(1, args.attempts + 1):
        if args.attempts > 1:
            print(f"\n--- Attempt {attempt}/{args.attempts} ---")

        probe_code = observed_output = ""
        print(f"\nGenerating BRT with {client.name} ...")
        if args.observe:
            generated, probe_code, observed_output = agent.generate_with_observation(
                inst,
                run_snippet_fn=run_snippet_in_container,
                code_context=code_context,
                env_hint=env_hint,
            )
            if probe_code:
                print(f"Probe ({len(probe_code)} chars), observed output ({len(observed_output)} chars)")
        else:
            generated = agent.generate(inst, code_context=code_context, env_hint=env_hint)

        if not generated.extracted:
            print("  No code block extracted — skipping attempt.")
            attempts_log.append({
                "attempt": attempt, "level": None,
                "outcome_buggy": None, "outcome_fixed": None,
                "test_code": None, "probe_code": probe_code,
                "observed_output": observed_output,
                "raw_response": generated.raw_response,
                "prompt_sent": generated.prompt_sent,
                "probe_prompt": generated.probe_prompt,
                "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens,
                "duration_seconds": generated.duration_seconds,
            })
            continue

        # Inject framework setup preamble (e.g. django.setup()) so the model
        # doesn't have to write it and can't get it wrong.
        test_code = inject_setup_preamble(inst.repo, generated.test_code)

        print(f"Test extracted ({len(test_code)} chars)")
        print("--- generated test ---")
        print(test_code)
        print("--- end ---\n")

        test_patch = make_test_patch(test_path, test_code)

        print("Running on BUGGY code ...")
        outcome_buggy = run_test_in_container(inst, test_patch, test_path=test_path)
        print(f"  outcome_on_buggy:  {outcome_buggy}")

        print("Running on FIXED code ...")
        outcome_fixed = run_test_in_container(
            inst, test_patch, gold_patch=inst.gold_patch, test_path=test_path
        )
        print(f"  outcome_on_fixed:  {outcome_fixed}")

        result = classify(outcome_buggy, outcome_fixed)
        print(f"  → {result.level.name}")

        attempts_log.append({
            "attempt": attempt,
            "level": result.level.value,
            "outcome_buggy": outcome_buggy,
            "outcome_fixed": outcome_fixed,
            "test_code": test_code,
            "probe_code": probe_code,
            "observed_output": observed_output,
            "raw_response": generated.raw_response,
            "prompt_sent": generated.prompt_sent,
            "probe_prompt": generated.probe_prompt,
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "duration_seconds": generated.duration_seconds,
        })

        if best_result is None or result.level > best_result.level:
            best_result = result
            best_generated = generated
            best_outcomes = (outcome_buggy, outcome_fixed)
            best_attempt_number = attempt

        if result.is_valid_brt:
            print("  L4 reached — stopping early.")
            break

    # --- Final report ---
    if best_result is None:
        print("\nNo test was successfully extracted in any attempt.")
        return 1

    print(f"\n{'='*50}")
    if args.attempts > 1:
        print(f"Best result across {args.attempts} attempt(s):")
    print(f"BRT Level:  {best_result.level.name}  (L{best_result.level.value})")
    print(f"Valid BRT:  {best_result.is_valid_brt}")
    print(f"Outcomes:   buggy={best_outcomes[0]}  fixed={best_outcomes[1]}")
    print(f"Rationale:  {best_result.rationale}")
    print(f"{'='*50}")

    save_run({
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": client.name,
        "split": args.split,
        "flags": {"think": args.think, "context": args.context, "observe": args.observe},
        "best_level": best_result.level.value,
        "best_attempt": best_attempt_number,
        "attempts": attempts_log,
    })

    return 0 if best_result.is_valid_brt else 1


if __name__ == "__main__":
    raise SystemExit(main())
