# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TDBF (Test-Driven Bug Fixing) is a research project implementing Phase 1 of a thesis methodology: given a SWE-bench issue, generate a Bug Reproduction Test (BRT) and classify it into levels L0–L4 based on how it behaves on the buggy code vs. the gold-patched code. Only **L4** (fail on buggy → pass on gold-fixed) counts as a validated BRT. The metric tracked across instances is the **C2P rate** = L4 / (L3 + L4).

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

Build Docker image and validate gold patch for an instance (required before generating BRTs):
```
python scripts/01_validate_gold.py --split lite
python scripts/01_validate_gold.py --split lite --id <instance_id>
```

Generate a BRT for one instance (full pipeline):
```
# One-shot generation
python scripts/02_generate_brt.py --model ollama:qwen3.5:9b@http://192.168.1.192:11434 --think --context --attempts 3

# Two-phase generation (observe buggy output first, then write test) — preferred
python scripts/02_generate_brt.py --model ollama:qwen3.5:9b@http://192.168.1.192:11434 --think --context --observe --attempts 3

# Specify instance
python scripts/02_generate_brt.py --id astropy__astropy-14182 --model anthropic:claude-opus-4-7 --observe --attempts 3
```

## Architecture

The pipeline flows: **instance loading → test generation → harness execution → BRT classification**.

### Data layer (`src/tdbf/data/instances.py`)
`InstanceStore` lazily loads from HuggingFace (`princeton-nlp/SWE-bench_Lite` or `princeton-nlp/SWE-bench_Verified`). `SWEBenchInstance.context_safe_dict()` is the **only** method that should be passed to the generation agent — it deliberately excludes `gold_patch` and `test_patch` to preserve experimental validity.

`store.iter_instances(limit=N)` iterates the split for batch runs.

### Model layer (`src/tdbf/models/llm.py`)
`LLMClient` is a `Protocol`. Three adapters: `AnthropicClient`, `OpenAIClient`, `OllamaClient`. All import SDKs lazily so the offline smoke test works with no SDKs.

`build_client(spec)` parses:
- `"ollama:qwen3.5:9b@http://192.168.1.192:11434"` → `OllamaClient`
- `"anthropic:claude-opus-4-7"` → `AnthropicClient`
- `"openai:gpt-4-turbo"` → `OpenAIClient`

`OllamaClient` streams responses (avoids ReadTimeout during Qwen3's silent thinking phase). Pass `think=True` or prepend `/no_think` to suppress Qwen3's extended reasoning. The `--think` flag in `02_generate_brt.py` sets `client._think = True` on `OllamaClient`.

### Generation (`src/tdbf/generation/agent.py`)

Two generation modes:

**One-shot** (`generate()`): builds prompt from `context_safe_dict()` with chain-of-thought structure ("state buggy output / expected output / rationale before writing code"), returns `GeneratedTest`.

**Two-phase** (`generate_with_observation()`): 
1. Phase 1 (probe): extract runnable code from issue REPL blocks, or ask model to generate a probe script. Run it inside the Docker container to observe the actual buggy output. Each expression is echoed before its result (`>>> expr / result`) so Phase 2 can distinguish reference vs buggy calls.
2. Phase 2 (test): feed observed output into a second prompt. The model knows the exact buggy value and must assert the correct expected value. Falls back to one-shot if probe produces no output.

`extract_probe_from_issue()` parses `>>> ` REPL lines from fenced code blocks in the issue text, converts them to `print(repr(...))` calls with labeled echoes. Returns `None` if no example code found (model generates probe instead).

`env_hint` parameter threads through all prompts — used to inject repo-specific notes (e.g. "Django is already configured, don't call settings.configure()").

### Harness (`src/tdbf/harness/runner.py`)

Thin wrapper around `swebench.harness.run_evaluation`. Key functions:

- `brt_test_path(repo)`: returns the repo-relative path for the generated test file. Django repos → `tests/test_brt.py`; everything else → `test_brt.py` (repo root). Uses `_REPO_TEST_SUBDIRS` mapping.
- `inject_setup_preamble(repo, test_code)`: prepends framework bootstrap code (e.g. `django.setup()`) so the model doesn't have to and can't get it wrong. Uses `_REPO_PREAMBLES` mapping.
- `repo_test_hint(repo)`: returns a one-line string for the generation prompt about the environment (e.g. "Django is pre-configured").
- `make_test_patch(test_path, test_code)`: builds a unified diff that adds the test file at `test_path`.
- `run_test_in_container(instance, test_patch, *, gold_patch, test_path, ...)`: applies patch and runs `python -m pytest {test_path}` inside the Docker container. Returns `"pass"/"fail"/"error"/"absent"`. Uses `python -m pytest` (not bare `pytest`) for cross-env compatibility.
- `run_snippet_in_container(instance, snippet)`: runs a Python snippet (the probe) in the container and returns stdout. Returns traceback on crash (captures exception-manifesting bugs).
- `fetch_relevant_code(instance)`: reads source files changed by the gold patch from the Docker image (buggy version, test files excluded).
- `read_report(run_id, model_name, instance)`: reads the harness JSON report after an evaluation run.

**Docker/Colima**: On macOS with Colima, always set `DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock`. Temp dirs must be under `~/` (not `/tmp`) because Colima only mounts `~/.`. The runner uses `tempfile.TemporaryDirectory(dir=str(Path.home()))` for this reason.

**Image naming**: `sweb.eval.x86_64.{instance_id}:latest`. Images are built by `01_validate_gold.py`. Run with `cache_level=instance` to keep images after evaluation (default `env` deletes them).

### Validation (`src/tdbf/validation/brt_levels.py`)
`classify(outcome_on_buggy, outcome_on_fixed)` → `BRTClassification` with `.level` (L0–L4) and `.is_valid_brt`. `c2p_rate(classifications)` = L4 / (L3 + L4), returns `None` when denominator is 0.

BRT level semantics:
- L0: malformed / no code block
- L1: passes on buggy code (false positive — test doesn't reproduce the bug)
- L2: errors on both (crashes, import errors, wrong APIs)
- L3: fails on buggy AND fails on fixed (reproduces some failure but not the right one — wrong assertion)
- L4: fails on buggy, passes on fixed — **the only valid BRT**

## Key invariants

- **Never pass `gold_patch` or `test_patch` to the generation agent.** This is the fairness control the whole experiment depends on. `context_safe_dict()` is safe; direct field access is not.
- The harness invocation uses `--namespace ''` (`build_locally=True`) to build Docker images locally rather than pulling prebuilt registry images.
- Run outputs go to `runs/<run_id>/` and harness logs to `logs/run_evaluation/<run_id>/`.

## Current status and validated results

Phase 1 pipeline is fully implemented and working end-to-end.

**Validated results** (Qwen3.5:9b, thinking=on, --context --observe --attempts 3):
- `astropy__astropy-12907`: **L4** (attempt 2) — separability matrix bug, REPL code in issue
- `astropy__astropy-14182`: **L4** (attempt 1) — RST header_rows feature, model-generated probe
- `django__django-11001`: **L3** (3 attempts) — multiline RawSQL ordering bug, no example code in issue

The Django L3 is a **model knowledge gap** (Qwen3.5:9b hallucinates Django ORM internals like `RawSQL` import path, `SQLCompiler` constructor, `as_sql()` return signature). The infrastructure is correct. Switching to `anthropic:claude-opus-4-7` or `openai:gpt-4-turbo` is expected to resolve this.

## Environment setup (macOS + Colima)

```bash
# Start Docker (Colima)
colima start
export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock

# Ollama server is on Windows desktop at 192.168.1.192:11434
# Requires OLLAMA_HOST=0.0.0.0 on the Windows machine (set as system env var)
# and Windows Firewall inbound rule for TCP 11434
```

## Next steps — pivot to a real multi-agent system

**Read `docs/agent_plan.md` first.** The project is moving from the current
code-driven *workflow* (Phase 1 BRT generation with `--attempts`/`--repair`) to
**real agents**: LLMs that autonomously choose tools and decide when they're done.
The paper is about *using agents* to do test-driven bug fixing
(reproduce → write BRT → fix until BRT passes).

The batch runner (`scripts/03_batch_brt.py`) and the `--repair` loop already exist
and serve as the **non-agentic baseline**. The existing harness functions
(`run_snippet_in_container`, `run_test_in_container`, `fetch_relevant_code`,
`classify`) become the **tools** the agents call.

Foundational task: an agent runtime (vendor-agnostic tool-use loop over the
`LLMClient` protocol, so Claude vs qwen stays a clean RQ1 comparison). Then a
single BRT agent (Phase A), a fix agent (Phase B), multi-agent (Phase C). Keep the
**gold-free invariant**: no agent gets a tool that runs on the gold-fixed tree;
`gold_patch`/`test_patch` are offline scoring only. Budget is ~$500 — see
`docs/cost_estimate.md`. Full detail and phase checklists in `docs/agent_plan.md`.

## Adding support for a new repo type

To add framework setup for a new repo (e.g. Flask, Sympy):
1. Add entry to `_REPO_TEST_SUBDIRS` in `runner.py` if tests live in a subdir
2. Add entry to `_REPO_PREAMBLES` in `runner.py` if the framework needs bootstrapping
3. `repo_test_hint()` will auto-generate a hint for the prompt

The pattern is: runner injects the boilerplate, model writes only the test logic.
