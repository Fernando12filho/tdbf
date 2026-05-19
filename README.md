# TDBF — Test-Driven Bug Fixing (Phase 1: Test Generation)

Implementation companion to the thesis proposal
*"Test-Driven Bug Fixing: How Pre-Fix Test Generation Improves Agentic AI
Success Rates."*

## What this is

This repo implements **Phase 1** of the proposed methodology: the
test-generation and failure-reproduction stage. It produces candidate Bug
Reproduction Tests (BRTs) from a SWE-bench issue and classifies each one
through a multi-level validation framework.

## Core design decision

Phase 1's definition of success is *executable*: a generated test only
"counts" if it compiles, runs, and **fails on the buggy revision** (and,
for a fully validated BRT, **passes once the gold patch is applied**).
That property cannot be checked without an execution environment.

Therefore this implementation does **not** reimplement container
provisioning or F2P/P2P scoring. It drives the **official SWE-bench
evaluation harness** (`swebench.harness.run_evaluation`), which already
builds isolated per-instance Docker images, applies a patch, and reports
which `FAIL_TO_PASS` / `PASS_TO_PASS` tests pass. Our job is to produce
the correct prediction records and interpret the harness's report.

This keeps the novel surface (test generation + BRT classification) small
and correct, and avoids the multi-week tar pit of reimplementing
per-repo dependency setup.

## BRT validation levels

A generated test is classified by running it under two conditions — on
the buggy `base_commit`, and on the gold-patched tree — and observing
the transition:

| Level | Name              | On buggy code | On gold-fixed code | Meaning |
|-------|-------------------|---------------|--------------------|---------|
| L0    | Malformed         | does not run  | —                  | syntax/import/collection error |
| L1    | Non-reproducing   | passes        | passes             | false positive — never caught the bug |
| L2    | Crashes           | errors        | (any)              | runs but raises instead of asserting (weak) |
| L3    | Candidate BRT     | fails (assert)| fails / errors     | reproduces *a* failure, not the right one |
| L4    | Validated BRT     | fails (assert)| passes             | true fail→pass: a real BRT (the C2P case) |

Only **L4** is a scientifically valid BRT. L3 → L4 is the
candidate-to-plausible (C2P) transition the proposal centers on.

## Layout

```
src/tdbf/
  data/        SWE-bench instance loading (HF datasets)
  models/      Multi-model LLM interface (Claude + GPT-4)
  generation/  Phase 1 test-generation agent + prompts
  harness/     Thin wrapper around swebench.harness.run_evaluation
  validation/  BRT level classification from harness reports
scripts/       CLI entrypoints (smoke test, single-instance Phase 1)
configs/       Run configuration
```

## Status

Phase 1 scaffold. Start with `scripts/00_smoke_test.py` (no API keys, no
Docker) to verify data loading and prompt construction, then
`scripts/01_validate_gold.py` to confirm the harness scores a known-good
patch correctly before trusting any generated test.
