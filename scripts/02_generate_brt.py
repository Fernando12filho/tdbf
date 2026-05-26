"""Phase 1 pipeline: generate a BRT and classify it.

Generates a Bug Reproduction Test for one SWE-bench instance using a
local or cloud LLM, then evaluates it on both the buggy and gold-fixed
trees to produce a BRT level (L0–L4). Only L4 counts as a valid BRT.

Requires:
  - Docker image already built (run 01_validate_gold.py first)
  - For Ollama: DOCKER_HOST set and Ollama reachable at base_url
  - For Anthropic/OpenAI: API key env var set

Usage:
  # Local Qwen via Ollama on another machine
  python scripts/02_generate_brt.py \\
      --model ollama:qwen3.5:9b@http://192.168.1.192:11434

  # Specific instance
  python scripts/02_generate_brt.py \\
      --model ollama:qwen3.5:9b@http://192.168.1.192:11434 \\
      --id astropy__astropy-12907

  # Cloud model (needs API key)
  python scripts/02_generate_brt.py --model anthropic:claude-opus-4-7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.data.instances import InstanceStore
from tdbf.generation.agent import TestGenerationAgent
from tdbf.harness.runner import BRT_TEST_PATH, make_test_patch, run_test_in_container
from tdbf.models.llm import build_client
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
    args = ap.parse_args()

    # --- Load instance ---
    store = InstanceStore(args.split)
    inst = store.get(args.id) if args.id else store.first()
    print(f"Instance:  {inst.instance_id}")
    print(f"Repo:      {inst.repo}")

    # --- Generate test ---
    client = build_client(args.model)
    agent = TestGenerationAgent(client)
    print(f"\nGenerating BRT with {client.name} ...")
    generated = agent.generate(inst)

    if not generated.extracted:
        print("FAILED: model returned no Python code block.")
        print("--- raw response ---")
        print(generated.raw_response[:1000])
        return 1

    print(f"Test extracted ({len(generated.test_code)} chars)")
    print("--- generated test ---")
    print(generated.test_code)
    print("--- end ---\n")

    # --- Build patch ---
    test_patch = make_test_patch(BRT_TEST_PATH, generated.test_code)

    # --- Evaluate on buggy tree ---
    print("Running on BUGGY code (no patch applied) ...")
    outcome_buggy = run_test_in_container(inst, test_patch)
    print(f"  outcome_on_buggy:  {outcome_buggy}")

    # --- Evaluate on fixed tree ---
    print("Running on FIXED code (gold patch applied) ...")
    outcome_fixed = run_test_in_container(
        inst, test_patch, gold_patch=inst.gold_patch
    )
    print(f"  outcome_on_fixed:  {outcome_fixed}")

    # --- Classify ---
    result = classify(outcome_buggy, outcome_fixed)
    print(f"\n{'='*50}")
    print(f"BRT Level:  {result.level.name}  (L{result.level.value})")
    print(f"Valid BRT:  {result.is_valid_brt}")
    print(f"Rationale:  {result.rationale}")
    print(f"{'='*50}")

    return 0 if result.is_valid_brt else 1


if __name__ == "__main__":
    raise SystemExit(main())
