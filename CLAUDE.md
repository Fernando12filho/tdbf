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

Run the full agentic pipeline end-to-end (BRT agent → gate on L4 → fix agent) for
one instance (build the image first via `01_validate_gold.py`):
```
python scripts/06_agent_pipeline.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6 -v
python scripts/06_agent_pipeline.py --id astropy__astropy-12907 --brt-only   # stop after the BRT
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

The project has moved past the Phase 1 *workflow* into **real agents** (see
`docs/agent_plan.md` for the authoritative, detailed status). Both the BRT agent
and the fix agent are implemented and live-validated end-to-end.

**Agent system (`src/tdbf/agent/`):**
- `runtime.py` — vendor-agnostic tool-use loop (`Agent`); termination = terminal
  `submit_*` tool or no-tool turn; iteration/token caps; real token capture.
- `session.py` — persistent per-instance container (`docker run -d sleep
  infinity` + `exec`) so an agent's many tool calls pay startup once.
- `tools.py` — gold-free tools wrapping `runner.py`. BRT set (Phase A):
  {read_file, grep, glob, run_python, run_test, submit_test}. Fix set (Phase B):
  {read_file, grep, glob, edit_file, run_brt, run_tests, revert, submit_fix}.
  `assert_gold_free` / `assert_fix_gold_free` guard the registries.
- `pipeline.py` — the shared phase logic both scripts and the end-to-end runner
  build on: `run_brt_phase` / `run_fix_phase` (gold-free agent run + offline
  gold-barrier scoring), the cost/serialise/step helpers, the system prompts, and
  pure transcript/results-record builders. Single source of truth — 04/05/06 do
  not re-implement the plumbing.
- Scripts: `04_agent_brt.py` (Phase A), `05_agent_fix.py` (Phase B),
  `06_agent_pipeline.py` (end-to-end BRT → gate-on-L4 → fix, in one command;
  the BRT travels in memory, no results.jsonl round-trip). Offline smoke tests:
  `scripts/agent_*_smoke.py` incl. `agent_pipeline_smoke.py` (no Docker/API needed).

**Validated results (Claude Sonnet `claude-sonnet-4-6`, astropy Lite):**
- **Phase A (BRT):** C2P = 5/6 = **0.83**, ~$0.24/instance.
- **Phase B (fix):** resolved 3/5 = **0.60**, ~$0.22/instance. The 2 non-resolved
  passed their own BRT but failed the held-out test (BRT necessary-not-sufficient).
- **RQ1 weak arm:** qwen2.5/3.5 can call tools but fail the autonomous loop
  (Phase A C2P 0.00; Phase B empty patch). A prompt-harden + bigger-model ladder
  did **not** close the gap → capability, not scaffolding.

**Non-agentic baseline (kept for comparison):** the `--attempts`/`--repair`
workflow in `02_generate_brt.py` / `03_batch_brt.py`.

## Environment setup (macOS + Colima)

```bash
# Start Docker (Colima)
colima start
export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock

# Ollama server is on Windows desktop at 192.168.1.192:11434
# Requires OLLAMA_HOST=0.0.0.0 on the Windows machine (set as system env var)
# and Windows Firewall inbound rule for TCP 11434
```

## Next steps

**Read `docs/agent_plan.md` first** — it holds the authoritative phase checklists
and findings. Phases 0 (runtime), A (BRT agent) and B (fix agent) are done and
live-validated. Remaining:
- **Broaden Phase B** beyond the 5 astropy instances (more repos/instances) for a
  credible resolved-rate; add the qwen weak arm.
- **Phase C** — multi-agent: split into reproduction / test-writer / (critic)
  agents + an orchestrator; compare single-capable-agent vs multi-agent (RQ4).

Hold the **gold-free invariant** in all new tools: no agent gets a tool that runs
on the gold-fixed tree; `gold_patch`/`test_patch` are offline scoring only
(`assert_gold_free` / `assert_fix_gold_free` enforce this). Budget ~$500 — Sonnet
sweeps run ~$0.22–0.24/instance, qwen is the free weak arm.

## Adding support for a new repo type

To add framework setup for a new repo (e.g. Flask, Sympy):
1. Add entry to `_REPO_TEST_SUBDIRS` in `runner.py` if tests live in a subdir
2. Add entry to `_REPO_PREAMBLES` in `runner.py` if the framework needs bootstrapping
3. `repo_test_hint()` will auto-generate a hint for the prompt

The pattern is: runner injects the boilerplate, model writes only the test logic.
