# Development round 1: strict procedure agent (2026-11-12)

Agent v0 (commit 75a24b7), gpt-5-mini, 50 turns, round-1 bugs (`data/dev/round1.json`,
development pool, not the 114). Strict only (decision 2026-11-12: baseline and soft run once
later on all development bugs). Records: run group `proc-dev-r1-strict` in results.jsonl;
transcripts in `runs/procedure/proc-dev-r1-strict/`. Total cost $0.54, 55 minutes.

## Results

| bug | label | turns | $ | outcome / flags |
|---|---|---|---|---|
| scikit-learn-13497 | L4 | 39 | 0.099 | forced test; truncated turn, write degraded |
| django-14608 | L4 | 40 | 0.041 | test; unreproduced |
| django-14855 | L0 | 50 | 0.058 | no test; budget exhausted |
| sympy-19254 | L3 (wrong assertion) | 26 | 0.045 | test; feature request, expected values not in the issue |
| requests-3362 | L4 | 50 | 0.054 | test; unreproduced |
| django-17051 | L2 | 48 | 0.054 | forced test; crashed |
| django-15695 | L0 | 50 | 0.051 | no test; budget exhausted |
| django-15498 | L4 | 33 | 0.028 | test |
| sympy-20212 | L4 | 15 | 0.026 | test |
| django-12589 | L3 (wrong assertion) | 50 | 0.086 | test; unreproduced |

**5 valid (L4), 2 wrong reason (L3), 1 crash (L2), 2 no test (L0).** $0.108 per valid test,
40 turns on average.

## Where the turns went (failure patterns)

1. **Reproduce mostly read code instead of running it.** 7 of 10 runs did not reproduce; in 6
   of them the model spent its 12 turns on grep/read_file and called run_python once or never,
   then finished with an empty run id. On Django one script called `settings.configure()`
   inside run_python (which already runs under Django's test runner).
2. **Write did the same.** In 3 Django runs (14855, 15695, 17051) Write used its whole budget on
   grep/read_file and never called run_test; two ended with no test, one with a crashing test.
3. **Pin ran on every bug and cost ~10 turns each.** Understand listed 3–5 "unknowns" per bug,
   most of them test-design questions ("equality or identity?", "which API?") rather than
   missing expected values. The design skips Pin when every expected claim is verified from the
   issue; v0 ran it whenever any unknown existed (a deviation from the design, fixed below).
4. **Understand needed a second visit in 7 of 10 runs**: the model quoted preflight's output
   (inline-expression results, probe output) as if it were issue text.
5. **`unmapped_assertion` in 5 runs**: the model left assertions out of `assertion_map`; each
   cost a Write revisit.
6. One reasoning turn hit the 4,096-token output limit (scikit-learn-13497).

Reproduce + Pin used about 22 turns before Write started, and the two L0 runs ran out of turns
in Write. The L3 tests are a feature request (sympy-19254: the issue asks for a better bound;
no expected values exist in it) and an unreproduced Django bug.

## Proposed changes for round 2 (agent v1), not yet applied

| # | Change | Targets |
|---|---|---|
| C1 | Understand: preflight output shown as a separate, labelled block; observed claims may quote it (source `preflight_run`, checked against that output); unknowns only for missing expected values, at most 3 | patterns 3, 4 |
| C2 | Pin skip rule as in the design: skip when no expected claim is unverified and at least one is verified; test-design questions never trigger Pin | pattern 3 |
| C3 | Read budget in Reproduce and Write: after 4 read_file/grep/glob calls without running code, those tools answer "run your script/test now" (deterministic) | patterns 1, 2 |
| C4 | Write tools as in the design: read_file and run_test (drop grep/glob) | pattern 2 |
| C5 | finish_write checks the assertion map itself and asks for the missing lines in the same phase (no send-back) | pattern 5 |
| C6 | Reproduce prompt: Django hint (run_python already runs under the test runner; no settings.configure(); use a test package's models via `module`) | pattern 1 |
| C7 | Preflight summary omits inline expressions that only raised NameError | pattern 4 |
| C8 | `max_tokens_per_turn` 8192 for every agent (baseline included, not yet run in Phase 1) | pattern 6 |
