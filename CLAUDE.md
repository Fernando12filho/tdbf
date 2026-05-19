# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TDBF (Test-Driven Bug Fixing) is a research project implementing Phase 1 of a thesis methodology: given a SWE-bench issue, generate a Bug Reproduction Test (BRT) and classify it into levels L0–L4 based on how it behaves on the buggy code vs. the gold-patched code. Only **L4** (fail on buggy → pass on gold-fixed) counts as a validated BRT.

## Commands

Install (editable, no extras needed for the offline smoke test):
```
pip install -e .
```

Install with live dependencies (HuggingFace datasets, LLM SDKs, SWE-bench harness):
```
pip install -e ".[dev]"   # or: pip install swebench datasets anthropic openai
```

Run the offline smoke test (no API keys, no Docker required):
```
python scripts/00_smoke_test.py
```

Validate the harness against the gold patch for a known instance (requires Docker + `pip install swebench datasets`):
```
python scripts/01_validate_gold.py --split lite
python scripts/01_validate_gold.py --split lite --id <instance_id>
```

## Architecture

The pipeline flows: **instance loading → test generation → harness execution → BRT classification**.

### Data layer (`src/tdbf/data/instances.py`)
`InstanceStore` lazily loads from HuggingFace (`princeton-nlp/SWE-bench_Lite` or `princeton-nlp/SWE-bench_Verified`). `SWEBenchInstance.context_safe_dict()` is the **only** method that should be passed to the generation agent — it deliberately excludes `gold_patch` and `test_patch` to preserve experimental validity.

### Model layer (`src/tdbf/models/llm.py`)
`LLMClient` is a `Protocol`. `AnthropicClient` and `OpenAIClient` are the two adapters. Both import their SDKs lazily so the offline smoke test can import without any SDKs installed. Use `build_client("anthropic:claude-opus-4-7")` or `build_client("openai:gpt-4-turbo")` to instantiate. The vendor-neutral protocol is intentional — RQ1 requires the same architecture with different models.

### Generation (`src/tdbf/generation/agent.py`)
`TestGenerationAgent.generate()` takes a `SWEBenchInstance`, builds a prompt from `context_safe_dict()`, and returns a `GeneratedTest` with the extracted Python code block. The agent must never see `gold_patch` or `test_patch`.

### Harness (`src/tdbf/harness/runner.py`)
This is a thin wrapper around `swebench.harness.run_evaluation`. A generated test is delivered as a unified diff that **adds** a new test file (not a code fix). `make_test_patch()` builds this diff. `run_harness()` writes a `predictions.jsonl` and invokes the official harness via subprocess. **`read_report()` is not yet implemented** — it must be filled in against the real harness output tree, which `scripts/01_validate_gold.py` surfaces on first run.

### Validation (`src/tdbf/validation/brt_levels.py`)
`classify(outcome_on_buggy, outcome_on_fixed)` maps the two observed outcomes to a `BRTLevel`. Outcomes are `"pass"`, `"fail"`, `"error"`, or `"absent"`. `c2p_rate()` computes the Candidate-to-Plausible rate = L4 / (L3 + L4), returning `None` when there are no candidates.

## Key invariants

- **Never pass `gold_patch` or `test_patch` to the generation agent.** This is the fairness control the whole experiment depends on.
- `read_report()` in `harness/runner.py` is intentionally unimplemented — complete it only after running `scripts/01_validate_gold.py` and reading the real harness output tree.
- The harness invocation uses `--namespace ''` (`build_locally=True`) to build Docker images locally rather than pulling prebuilt registry images.
- Run outputs go to `runs/<run_id>/` and harness logs to `logs/run_evaluation/<run_id>/`.
