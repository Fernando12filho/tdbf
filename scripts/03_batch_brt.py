"""Batch BRT generation over N SWE-bench instances.

Iterates instances, generates BRTs, writes incremental results.jsonl,
and prints a running C2P rate after each instance.

Resume-safe: instances already present in results.jsonl are skipped.
Pass --build to automatically build missing Docker images via the gold
patch harness (same as running 01_validate_gold.py per instance).

Usage:
  python scripts/03_batch_brt.py --limit 10 --build \\
      --model "ollama:qwen3.5:9b@http://192.168.1.192:11434" \\
      --think --observe --context --attempts 3

  # specific instances only
  python scripts/03_batch_brt.py --build \\
      --ids astropy__astropy-12907,astropy__astropy-14182 \\
      --model anthropic:claude-opus-4-7 --observe --context
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.data.instances import InstanceStore, SWEBenchInstance
from tdbf.generation.agent import TestGenerationAgent
from tdbf.harness.runner import (
    PredictionRecord,
    brt_test_path,
    fetch_python_version,
    fetch_relevant_code,
    inject_setup_preamble,
    make_test_patch,
    repo_test_hint,
    run_harness,
    run_snippet_in_container,
    run_test_in_container,
    write_predictions,
)
from tdbf.models.llm import OllamaClient, build_client
from tdbf.results import RESULTS_FILE, save_run
from tdbf.validation.brt_levels import BRTLevel, classify


def _image_exists(instance_id: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", f"sweb.eval.x86_64.{instance_id}:latest"],
        capture_output=True,
    )
    return proc.returncode == 0


def _build_image(inst: SWEBenchInstance, split: str) -> bool:
    """Build the Docker image by running the harness with the gold patch.

    Mirrors 01_validate_gold.py. cache_level=instance keeps the image so
    run_test_in_container can reuse it.
    """
    run_id = f"build_{inst.instance_id}"
    preds_path = Path("runs") / run_id / "predictions.jsonl"
    write_predictions(
        [PredictionRecord(inst.instance_id, "GOLD", inst.gold_patch)],
        preds_path,
    )
    rc = run_harness(
        predictions_path=preds_path,
        dataset_split=split,
        run_id=run_id,
        instance_ids=[inst.instance_id],
        max_workers=1,
        build_locally=True,
        cache_level="instance",
    )
    return rc == 0


def _load_done_ids(results_file: Path) -> set[str]:
    """Return instance_ids already recorded in results.jsonl."""
    if not results_file.exists():
        return set()
    done: set[str] = set()
    with results_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["instance_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _c2p_label(l3: int, l4: int) -> str:
    denom = l3 + l4
    if denom == 0:
        return "C2P: n/a"
    return f"C2P: {l4}/{denom} = {l4/denom:.1%}"


def process_instance(
    inst: SWEBenchInstance,
    agent: TestGenerationAgent,
    args: argparse.Namespace,
) -> dict:
    """Run the full BRT pipeline for one instance. Returns the run record."""
    print(f"\n{'─'*60}")
    print(f"Instance:  {inst.instance_id}")
    print(f"Repo:      {inst.repo}")

    code_context: dict[str, str] | None = None
    if args.context:
        code_context = fetch_relevant_code(inst)
        if code_context:
            for path, content in code_context.items():
                print(f"  context: {path}  ({len(content)} chars)")
        else:
            print("  context: (none found)")

    test_path = brt_test_path(inst.repo)
    python_version = fetch_python_version(inst)
    if python_version:
        print(f"  python:  {python_version}")
    env_hint = repo_test_hint(inst.repo, python_version=python_version)

    best_result = None
    best_outcomes = (None, None)
    best_attempt_number = None
    attempts_log: list[dict] = []

    for attempt in range(1, args.attempts + 1):
        if args.attempts > 1:
            print(f"  attempt {attempt}/{args.attempts}")

        probe_code = observed_output = ""
        if args.observe:
            generated, probe_code, observed_output = agent.generate_with_observation(
                inst,
                run_snippet_fn=run_snippet_in_container,
                code_context=code_context,
                env_hint=env_hint,
            )
        else:
            generated = agent.generate(inst, code_context=code_context, env_hint=env_hint)

        if not generated.extracted:
            print("  no code block extracted — skipping attempt")
            attempts_log.append({
                "attempt": attempt, "level": None,
                "outcome_buggy": None, "outcome_fixed": None,
                "test_code": None, "probe_code": probe_code,
                "observed_output": observed_output,
                "consistency": None,
                "raw_response": generated.raw_response,
                "prompt_sent": generated.prompt_sent,
                "probe_prompt": generated.probe_prompt,
                "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens,
                "duration_seconds": generated.duration_seconds,
            })
            continue

        test_code = inject_setup_preamble(inst.repo, generated.test_code)
        test_patch = make_test_patch(test_path, test_code)

        if generated.consistency:
            print(f"  consistency: {generated.consistency}")

        outcome_buggy = run_test_in_container(inst, test_patch, test_path=test_path)
        outcome_fixed = run_test_in_container(
            inst, test_patch, gold_patch=inst.gold_patch, test_path=test_path
        )
        result = classify(outcome_buggy, outcome_fixed)
        print(f"  buggy={outcome_buggy}  fixed={outcome_fixed}  → {result.level_label}")

        attempts_log.append({
            "attempt": attempt,
            "level": result.level.value,
            "outcome_buggy": str(outcome_buggy),
            "outcome_fixed": str(outcome_fixed),
            "test_code": test_code,
            "probe_code": probe_code,
            "observed_output": observed_output,
            "consistency": generated.consistency,
            "l3_subtype": result.l3_subtype,
            "failure_reason_buggy": outcome_buggy.failure_reason,
            "failure_reason_fixed": outcome_fixed.failure_reason,
            "raw_response": generated.raw_response,
            "prompt_sent": generated.prompt_sent,
            "probe_prompt": generated.probe_prompt,
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "duration_seconds": generated.duration_seconds,
        })

        if best_result is None or result.level > best_result.level:
            best_result = result
            best_outcomes = (outcome_buggy, outcome_fixed)
            best_attempt_number = attempt

        if result.is_valid_brt:
            print("  L4 — stopping early")
            break

    best_level = best_result.level.value if best_result else None
    return {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": agent.client.name,  # type: ignore[attr-defined]
        "split": args.split,
        "flags": {
            "think": getattr(args, "think", False),
            "context": args.context,
            "observe": args.observe,
        },
        "best_level": best_level,
        "best_attempt": best_attempt_number,
        "attempts": attempts_log,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    ap.add_argument(
        "--ids", default=None,
        help="Comma-separated instance_ids to run (overrides --limit)",
    )
    ap.add_argument(
        "--limit", type=int, default=10, metavar="N",
        help="Max instances to process (default: 10)",
    )
    ap.add_argument(
        "--model",
        default="ollama:qwen3.5:9b@http://192.168.1.192:11434",
        help="Model spec: ollama:<model>@<url>, anthropic:<model>, openai:<model>",
    )
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode")
    ap.add_argument("--context", action="store_true", help="Include buggy source files in prompt")
    ap.add_argument("--observe", action="store_true", help="Two-phase generation with probe")
    ap.add_argument(
        "--attempts", type=int, default=1, metavar="N",
        help="Max attempts per instance (default: 1)",
    )
    ap.add_argument(
        "--build", action="store_true",
        help="Build missing Docker images via the gold patch harness before processing",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run instances already in results.jsonl (useful for comparing models)",
    )
    args = ap.parse_args()

    client = build_client(args.model)
    if args.think and isinstance(client, OllamaClient):
        client._think = True
        print(f"Model: {client.name}  [thinking=on]")
    else:
        print(f"Model: {client.name}")

    store = InstanceStore(args.split)

    # Select instances
    if args.ids:
        instance_ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        instances = [store.get(iid) for iid in instance_ids]
    else:
        instances = list(store.iter_instances(limit=args.limit))

    print(f"Instances to process: {len(instances)}  (split={args.split})")

    done_ids = _load_done_ids(RESULTS_FILE)
    if done_ids:
        print(f"Resuming — {len(done_ids)} already in {RESULTS_FILE}")

    agent = TestGenerationAgent(client)

    total = skipped_done = skipped_no_image = processed = 0
    l3_count = l4_count = 0

    # Seed C2P counts from previously completed runs in results.jsonl
    if done_ids and RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    lvl = rec.get("best_level")
                    if lvl == BRTLevel.L3_CANDIDATE.value:
                        l3_count += 1
                    elif lvl == BRTLevel.L4_VALIDATED.value:
                        l4_count += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    for inst in instances:
        total += 1

        if inst.instance_id in done_ids and not args.force:
            print(f"  skip (done):    {inst.instance_id}")
            skipped_done += 1
            continue

        if not _image_exists(inst.instance_id):
            if args.build:
                print(f"  building image: {inst.instance_id} ...")
                if not _build_image(inst, args.split):
                    print(f"  build failed — skipping {inst.instance_id}")
                    skipped_no_image += 1
                    continue
            else:
                print(f"  skip (no image): {inst.instance_id}  (pass --build to auto-build)")
                skipped_no_image += 1
                continue

        record = process_instance(inst, agent, args)
        save_run(record)
        processed += 1

        lvl = record["best_level"]
        if lvl == BRTLevel.L3_CANDIDATE.value:
            l3_count += 1
        elif lvl == BRTLevel.L4_VALIDATED.value:
            l4_count += 1

        level_name = f"L{lvl}" if lvl is not None else "none"
        print(f"  saved  {inst.instance_id}  best={level_name}  {_c2p_label(l3_count, l4_count)}")

    print(f"\n{'='*60}")
    print(f"Done.  total={total}  processed={processed}  "
          f"skipped_done={skipped_done}  skipped_no_image={skipped_no_image}")
    print(f"Final  {_c2p_label(l3_count, l4_count)}  (L3={l3_count}  L4={l4_count})")
    print(f"Results written to {RESULTS_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
