# TDBF: Test-Driven Bug Fixing

Research code for a paper on test-first bug fixing with LLM agents, evaluated on
[SWE-bench Lite](https://www.swebench.com/).

Given a real GitHub issue, an agent first writes a **Bug Reproduction Test (BRT)**:
a test that fails on the buggy code and should pass once the bug is fixed. Only
after that test has been seen to fail does a second agent edit the source until it
passes. Both agents decide their own tool calls (read, grep, run code, run tests,
edit, submit). Our code supplies the tools and budgets and doesn't script the steps.

The question is whether this ordering helps. We compare it against a reactive
agent that has the same model, tools and budget but writes no test first.

## Research questions

- **RQ1 (ordering).** Does test-first resolve more issues than a matched reactive
  arm? Paired comparison on the same instances.
- **RQ2 (orchestration).** Which pipeline choices improve fix quality under a fixed
  budget: the accept rule for the test, retry budgets, a Critic/Reviewer,
  an adversarial Breaker, best-of-N test selection. Negative results count.
- **RQ3 (open-weight models).** Can open-weight models sustain the pipeline, and
  where does a trajectory break down: never submitting, a test that doesn't
  reproduce the bug, or a fix that doesn't resolve it?

## Current result (RQ1)

Model: `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter. Base sample plus
the extension sample run so far. Produced by `scripts/14_paired_analysis.py`.

|                          | test-first resolved | test-first not |
|--------------------------|--------------------:|---------------:|
| **reactive resolved**    | 51                  | 6              |
| **reactive not**         | 11                  | 55             |

- Paired instances: **123**
- Resolved: test-first **62/123 (0.50)**, reactive **57/123 (0.46)**
- McNemar exact two-sided p = **0.33**. Difference +0.041, 95% bootstrap CI
  [-0.024, +0.106]

The difference so far isn't statistically significant. The extension sample is still
running, so treat these numbers as interim.

"Resolved" uses the SWE-bench definition: the held-out `FAIL_TO_PASS` tests pass and
no `PASS_TO_PASS` test regresses.

## How a generated test is judged

Each BRT gets a level from how it behaves on the buggy tree and on the gold-fixed
tree (`src/tdbf/validation/brt_levels.py`):

| Level | On buggy code | On fixed code | Meaning |
|-------|---------------|---------------|---------|
| L0    | —             | —             | malformed, no usable test |
| L1    | passes        | —             | doesn't reproduce the bug |
| L2    | errors        | errors        | crashes on both (bad import, wrong API) |
| L3    | fails         | fails         | reproduces *a* failure, but the assertion is wrong |
| L4    | fails         | passes        | a valid BRT |

**C2P = L4 / (L3 + L4)** is the headline test metric.

### Gold-free gate

The gold patch is never shown to an agent and never used to make a decision. The
deployable pipeline accepts a test when it reaches its assertion and fails there on
the buggy tree (L3 or L4). It can't tell those two apart without the fix, and a
real deployment wouldn't have the fix. The L0–L4 label is computed offline for
scoring only. `scripts/06_agent_pipeline.py` gates on L4 and is kept only as a
ceiling reference.

## Layout

```
src/tdbf/
  agent/       tool-use runtime, container session, gold-free tools, roles, pipeline
  models/      LLM clients: Anthropic, OpenAI, OpenRouter, Ollama
  harness/     wrapper around the SWE-bench evaluation harness
  data/        SWE-bench instance loading
  validation/  BRT level classification
  metrics.py   metric suite (C2P, resolve rates, funnel, cost)
  pricing.py   per-model pricing table
scripts/       numbered entry points (see below) and offline *_smoke.py tests
webapp/        Flask app that presents results.jsonl
thesis/        the paper (ACM acmart)
results.jsonl  append-only store of every run
```

## Setup

Requires Python ≥ 3.10 and Docker. On macOS with Colima:

```bash
colima start
export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
pip install -e ".[dev]"
```

Put API keys in `.env` at the repo root (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
...). It loads automatically on `import tdbf`.

Use the prebuilt SWE-bench images where possible
(`docker.io/swebench/sweb.eval.x86_64.<id>`, `--platform linux/amd64`). Building
images locally pulls today's versions of unpinned dependencies, and that breaks some
environments.

## Running

Offline smoke tests (no Docker, no keys):

```bash
python3 scripts/00_smoke_test.py
for f in scripts/agent_*_smoke.py; do python3 "$f"; done
```

One instance:

```bash
python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907       # gold check
python3 scripts/07_deployable_pipeline.py --id astropy__astropy-12907 \
    --model openrouter:nvidia/nemotron-3-super-120b-a12b:free -v                    # test-first (Arm B)
python3 scripts/07_deployable_pipeline.py --id astropy__astropy-12907 --score-brt  # + offline L0–L4 label
```

Paired sweep and analysis:

```bash
python3 scripts/11_paired_sweep.py --model openrouter:<model>:free \
    --sample sample_lite_frozen.json --run-group <name>                            # both arms per instance
python3 scripts/16_nightly_batch.py ...     # pull image → gold check → paired unit → delete image
python3 scripts/14_paired_analysis.py       # McNemar + bootstrap CI
python3 scripts/08_metrics_report.py        # full metric suite
python3 scripts/15_status.py                # where every Lite instance stands
```

Results site:

```bash
cd webapp && python3 app.py   # http://localhost:5000: /results /development /status /metrics
```

| Script | Purpose |
|--------|---------|
| `04_agent_brt.py` / `05_agent_fix.py` | BRT phase / fix phase alone |
| `06_agent_pipeline.py` | L4-gated ceiling (uses gold, reference only) |
| `07_deployable_pipeline.py` | gold-free test-first pipeline, the main arm |
| `09_bestofn_pipeline.py` | best-of-N tests + Selector (RQ2) |
| `10_gate_ab.py` | paired A/B between two run groups |
| `11_paired_sweep.py` | RQ1 sweep, reactive vs test-first |
| `12_freeze_sample.py` / `13_build_sample.py` | freeze and build the sample |
| `02_generate_brt.py` / `03_batch_brt.py` | older non-agentic baseline |

## Notes on validity

- Every fix is scored under a fresh harness run id, and the report is checked to
  match the submitted patch. The harness silently reuses old reports when a run id
  repeats.
- Comparisons group runs by arm and `run_group`, never by model alone.
- Fix outcomes aren't fully deterministic. Repeat runs of the same instance
  occasionally disagree.
