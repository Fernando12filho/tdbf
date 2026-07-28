# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TDBF (Test-Driven Bug Fixing) is a thesis research project on an **agentic** approach
to test-driven bug fixing on SWE-bench. Given a real GitHub issue, a real LLM agent
(1) reproduces the bug, (2) writes a **Bug Reproduction Test (BRT)** that fails on the
buggy code and passes once fixed, then (3) edits the source until that BRT passes. The
contribution is the *agentic* framing (the model drives its own tool-use trajectory,
not a fixed code loop) plus a way to validate BRTs **without the gold patch**.

Two anchors run through everything:
- **BRT levels (L0–L4)** classify a generated test by buggy-vs-fixed behavior; only
  **L4** (fail on buggy → pass on gold-fixed) is a valid BRT. **C2P = L4 / (L3 + L4)**
  is the headline BRT metric; **resolved-rate** (held-out FAIL_TO_PASS pass, no
  PASS_TO_PASS regression) is the headline fix metric.
- **Gold-free invariant** — no agent ever gets a tool that runs on the gold-fixed tree;
  `gold_patch`/`test_patch` are offline scoring instruments only. See "Key invariants".

**Authoritative status lives in `docs/agent_plan.md` and `docs/project_summary.md`**
(detailed phase checklists, findings, glossary). This file is the working orientation;
those two are the full record. Note: `docs/` is currently gitignored, so those files
are not version-controlled — keep that in mind before relying on their git history.

## Commands

Install (editable, no extras needed for the offline smoke test):
```
pip install -e .
```

Install with live dependencies (HuggingFace datasets, LLM SDKs, SWE-bench harness):
```
pip install -e ".[dev]"   # or: pip install swebench datasets anthropic openai
```

`.env` **auto-loads on `import tdbf`** (`load_env()` in `src/tdbf/__init__.py`) — no
`source .env` needed. Put `ANTHROPIC_API_KEY` there (the Claude Code login is *not* the
SDK key).

Run the offline smoke tests (no API keys, no Docker required):
```
python scripts/00_smoke_test.py
python scripts/agent_pipeline_smoke.py   # end-to-end agent plumbing, fake client/session
# all agent smokes: scripts/agent_*_smoke.py (runtime, tools, brt, fix, breaker,
# multiagent, bestofn, deployable, pipeline)
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

### Agentic pipelines (build the image first via `01_validate_gold.py`)

The **deployable, gold-free pipeline (`07`) is the primary experimental arm.** `06`
is kept only as the *L4-gated ceiling* (its gate consults the gold tree — an oracle a
real deployment can't have). Both accept `--brt-only` to stop after Phase A.

```
# Primary: honest gold-free gate (accept a BRT that cleanly fails on buggy = L3∪L4) → fix
python scripts/07_deployable_pipeline.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6 -v
python scripts/07_deployable_pipeline.py --id astropy__astropy-12907 --score-brt   # also emit offline L0–L4 label (info only, never gates)
python scripts/07_deployable_pipeline.py --id astropy__astropy-12907 --breaker      # + post-fix adversarial Breaker (RQ3 ablation; parked negative)

# Ceiling reference only (L4 gate = oracle): BRT → gate-on-L4 → fix
python scripts/06_agent_pipeline.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6 -v

# Single phases (share the same pipeline.py plumbing)
python scripts/04_agent_brt.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6   # Phase A only
python scripts/05_agent_fix.py --id astropy__astropy-12907 --model anthropic:claude-sonnet-4-6   # Phase B only (loads a validated BRT)

# Multi-agent (Phase C, depth-light): same model self-reviews (Critic on A, Reviewer on B)
python scripts/06_agent_pipeline.py --id astropy__astropy-12907 --multi --review-rounds 1
```

Metrics + reporting (read `results.jsonl`, no new runs):
```
python scripts/08_metrics_report.py            # C2P, Valid-BRT, Resolve@BRT, pipeline funnel, $/inst
cd webapp && python3 app.py                    # http://localhost:5000 — /metrics /research /docs /commands
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

## Agent system (`src/tdbf/agent/`)

The project inverted control from *workflow* (our code steps a generate→run→repair
loop) to **real agents** (the model drives its own tool-use trajectory). The old
harness functions became the tools the agents call.

- `runtime.py` — vendor-agnostic tool-use loop (`Agent`); termination = terminal
  `submit_*` tool or no-tool turn; iteration/token caps; real token capture.
  **Forced-submit grace:** if an agent exhausts `max_iterations` without ever
  submitting, it gets up to `submit_grace` (default 3) extra closing turns that
  force it to call its terminal tool with its best result — so a stuck agent still
  produces a test/fix instead of an empty L0 (`stop_cause="submit_forced"`). Those
  closing turns don't count toward the reported `iterations`.
- `session.py` — persistent per-instance container (`docker run -d sleep
  infinity` + `exec`) so an agent's many tool calls pay startup once.
- `tools.py` — gold-free tools wrapping `runner.py`, one registry + an
  `assert_*_gold_free` guard per role (see the role list below).
- `roles.py` + `pipeline.py` — the agent roles and the orchestration that wires them.
- `types.py` — shared agent dataclasses (`AgentResult`, step/usage records).

**The agents — 2 main + 4 auxiliary.** All are real gold-free `Agent`s with a terminal
tool; only the two main ones form the pipeline. The four auxiliary roles are optional
add-ons, and per the locked 2026-07-21 plan they are **RQ3 ablation material, not the
main experimental line** (the sweep runs the two main agents only).

*Main pipeline agents:*
- **BRT agent** (Phase A) — `brt_agent_tools`, `assert_gold_free`. Tools {read_file,
  grep, glob, run_python, run_test (buggy only), submit_test}. Writes the reproduction
  test. **Gate-retry:** if the submitted test lands L0/L1/L2 (does not fail cleanly on
  the buggy tree = doesn't reproduce), `run_brt_phase` re-runs the agent from a fresh
  conversation with gold-free feedback on what went wrong, up to `brt_retries` times
  (default **2**, on by default, no flag). The decision reads only the buggy-side
  outcome, so it stays gold-free; a capable model reproduces on attempt 0 and the loop
  is a no-op.
- **Fix agent** (Phase B) — `fix_agent_tools`, `assert_fix_gold_free`. Tools {read_file,
  grep, glob, edit_file, run_brt (validated BRT baked in), run_tests, revert,
  submit_fix}. Edits source until the BRT passes.

*Auxiliary roles (`roles.py`, each with a terminal verdict, fail-open to accept):*
- **Critic** (`run_critic`) — reviews BRT specificity before accept (Phase A).
  *Near-no-op on the strong arm* (on a capable model the first BRT is already specific).
- **Reviewer** (`run_reviewer`) — judges fix minimality/correctness (Phase B).
- **Breaker** (`run_breaker`) — post-fix adversarial verifier; probes the agent's *own*
  fix for a case it still gets wrong. *Parked as an RQ3 negative result* (no oracle →
  false negatives on the hardest verification-gap cases).
- **Selector** (`run_selector`) — best-of-N: picks among several candidate BRTs; used by
  `09_bestofn_pipeline.py`.

- `pipeline.py` — the single source of truth for all phases: `run_brt_phase` /
  `run_fix_phase` (gold-free agent run + offline gold-barrier scoring), optional
  `critic_client`/`reviewer_client`/`breaker_client` orchestration loops, a `score_fixed=`
  flag (True = 04/06 L0–L4 label; False = `07` buggy-only gold-free gate), plus the
  cost/serialise/step helpers, system prompts, and transcript/record builders. Scripts
  04–09 do not re-implement plumbing; `breaker_client=None` / single-agent paths stay
  byte-identical (asserted by the smokes).

**`src/tdbf/metrics.py`** — single source of truth for the metric suite (C2P,
Valid-BRT = L4/N, Resolve@BRT, whole-pipeline funnel = GateRate × Resolve@Gated,
$/inst, $/resolved, repo breakdown). `build_report` + `format_text`; consumed by both
`scripts/08_metrics_report.py` and the webapp `/metrics` page so they can't drift.

**Scripts (agentic):** `04_agent_brt.py` (Phase A), `05_agent_fix.py` (Phase B),
`06_agent_pipeline.py` (L4-gated ceiling, end-to-end in memory), **`07_deployable_pipeline.py`
(primary — gold-free gate; `--breaker`, `--score-brt`)**, `08_metrics_report.py`
(reporting), `09_bestofn_pipeline.py` (best-of-N BRT + Selector + iterated fix). Offline
smokes: `scripts/agent_*_smoke.py` (runtime, tools, brt, fix, breaker, multiagent,
bestofn, deployable, pipeline) — no Docker/API needed.

**`webapp/`** — Flask app (`app.py`, `db.py`, `research_data.py`, `seed_docs.py`)
presenting results to the advisor: `/metrics` (live from `results.jsonl` via
`metrics.py`), `/research` (locked plan + decisions + related-work), `/docs`,
`/commands`. SQLite `tdbf.db` is regenerated from `results.jsonl`.

## Status & findings (context, not official results)

Phases 0 (runtime), A (BRT agent), B (fix agent) are built and live-validated on two
repos (astropy, Django) with Claude Sonnet; the deployable `07` pipeline and the four
auxiliary roles are built and offline-green. **No official results yet** — the numbers
in `results.jsonl` / the webapp `/metrics` page are exploratory, from the analysis DB,
not a frozen sweep. Don't quote them as final.

Qualitative findings that steer the work (the *why*, no numbers):
- **RQ1 weak arm** — local 7B models (qwen2.5-coder:7b, qwen3.5:9b) can call tools
  correctly but can't sustain the autonomous loop to a terminal submit. Prompt-harden +
  a bigger local model did *not* close it → the gap is **capability, not scaffolding**.
  qwen stays as the free weak baseline; stop hacking on it.
- **Verification gap (RQ3)** — the dominant fix failure is "valid BRT, fix passes it,
  still not resolved": a BRT-passing fix is *necessary-not-sufficient*. This, not
  localization, is what the auxiliary roles were meant to attack.
- **Breaker parked** — with only ad-hoc `run_python` it has no oracle, so a
  wrong-but-plausible value looks fine and it fails open to `robust`. Kept as an honest
  RQ3 negative; no more Breaker tuning.

**Non-agentic baseline (kept for comparison):** the `--attempts`/`--repair` workflow in
`02_generate_brt.py` / `03_batch_brt.py`.

## Environment setup (macOS + Colima)

```bash
# Start Docker (Colima)
colima start
export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock

# Ollama server is on Windows desktop at 192.168.1.192:11434
# Requires OLLAMA_HOST=0.0.0.0 on the Windows machine (set as system env var)
# and Windows Firewall inbound rule for TCP 11434
```

## Research plan (LOCKED 2026-07-21) — what we're actually running

Read `docs/agent_plan.md` for the full checklists. The direction is locked:

- **Dataset: SWE-bench Lite** (the scoring dataset — *not* Verified). BugsInPy/GitBugs
  descoped unless time allows.
- **Two arms (the main comparison):**
  - **Arm A — reactive baseline:** same model / tools / budgets, but **drop `run_brt`**
    from the fix agent (keep `run_tests`, no Phase A BRT). The `submit_fix` description
    must lose its `run_brt` reference (`tools.py`). *Not built yet — this is the next
    build.*
  - **Arm B — test-first:** the deployable pipeline (`07`).
  - A test-last (TLD) third arm is **open**, revisit after the pilot.
- **Sequence:** build Arm A → freeze a random ~20 pilot → frozen stratified ~150 sample
  → paired sweep (McNemar) → RQ2 offline artifact analysis (AST properties, determinism
  3×, gold-patch-line coverage — gold is legal offline only).
- **Auxiliary roles (Critic / Reviewer / Breaker / Selector / best-of-N) are RQ3
  ablation material now**, including the negatives. No more Breaker tuning.

**Binding constraint is Docker image builds, not budget** (spend to date is a small
fraction of the ~$500; Sonnet runs are cheap, qwen is free). Getting instances built
via `01_validate_gold.py` is the bottleneck for growing the sample.

**Hold the gold-free invariant in all new tools** — see "Key invariants" below;
`assert_gold_free` / `assert_fix_gold_free` / `assert_role_gold_free` /
`assert_breaker_gold_free` enforce it per role.

## Repo layout beyond the code

- `docs/` — `agent_plan.md` (authoritative checklists), `project_summary.md`,
  `cost_estimate.md`. **Gitignored** (not version-controlled) — heads-up.
- `thesis/` — LaTeX thesis (report class, 7 chapters, `references.bib`). Source of the
  written deliverable; `thesisproposal/` holds the original proposal source.
- `figures/` — LaTeX/TikZ diagrams (`tdbf_*.tex` → PNG/PDF). **Gitignored.**
- `results.jsonl` (repo root) — the append-only run store; webapp `tdbf.db` and all
  metrics derive from it.
- `goldjson/` — per-instance gold-validation outputs from `01_validate_gold.py`.

## Adding support for a new repo type

To add framework setup for a new repo (e.g. Flask, Sympy):
1. Add entry to `_REPO_TEST_SUBDIRS` in `runner.py` if tests live in a subdir
2. Add entry to `_REPO_PREAMBLES` in `runner.py` if the framework needs bootstrapping
3. `repo_test_hint()` will auto-generate a hint for the prompt

The pattern is: runner injects the boilerplate, model writes only the test logic.
