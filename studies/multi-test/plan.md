# Multi-test reproduction tests: plan (2026-11-25, before any run)

User decision (2026-11-25), raised while reading the evaluation's tests: every agent writes **one**
test function per bug (single agent 50/50 files, reactive blackboard 47/49, ReProAgent 256/267). That
is our design (the BRT prompt asks for "one pytest test"), not the model's choice. More tests trade a
higher chance of reproducing the bug and more coverage against a higher chance that one wrong
assertion fails after the fix. For TDBF the question that matters is the **fix phase**: the main RQ2
finding is the verification gap (a fix passes the one test but does not resolve the bug). Several
tests covering related cases could make such a fix harder to submit.

## Question (revised before any run, user decision 2026-11-25: no fix runs)

Does a multi-test BRT (a main reproduction test plus as many related-case tests as the agent judges
the bug needs) write better reproduction tests than the one-test BRT: more valid tests, more coverage
of the fix, and how often does an extra test break strict success? The fix phase is **not** run.

## Arms (the 50 development bugs, gpt-5-mini)

| arm | tests |
|---|---|
| **single** | the stored one-test tests of the unchanged agent (`proc-dev-baseline` 30, `v2-base` 20); no new run |
| **multi** | new run of the same agent with the multi-test instruction (run group `mt-brt`) |

- **Multi-test instruction** (the only change): write a test module with (1) the main reproduction
  test, the issue's own scenario, and (2) **as many further tests as the agent judges the bug needs**
  for closely related cases the issue implies a correct fix must handle (other inputs it mentions,
  edge cases, the inverse or round-trip operation, another call path), each asserting only what the
  issue supports and reaching its assertions. No fixed number: a narrow bug may need only the main
  test. The main test must fail on the buggy code; the others should fail now too where they exercise
  the bug, and may pass if they guard behaviour that must stay correct.
- **Budget:** 50 turns per attempt, 2 gate retries, 1M tokens, 8192 per turn (as the stored tests).
  Gate unchanged (the module fails on the buggy code at an assertion).
- **Scoring:** both arms through the final evaluation's scorer (`scripts/20_final_eval.py`, official
  SWT-bench harness, the same image and gold run per bug), plus our offline L0–L4 label
  (module-level: L4 needs every test to pass on the fixed code).

## Metrics and rule (fixed now)

- **Primary: SWT-bench strict success S** (at least one test fails before the fix and passes after,
  none fails after it), multi vs single, paired on the 50 (exact McNemar).
- Secondary: **ΔC** (coverage of the fix's lines, where more tests should help), L4 and L3 counts,
  F→P (loose), tests per module, cost per bug and per valid test.
- **Adopted** only if multi has more S than single, no more L3, and cost per valid test at most
  1.5× single. ΔC is reported either way.
- 50 development bugs are a first look.

## Cost

Phase A multi ≈ $0.03 per bug → about **$1.50** for 50 bugs, capped at $3. Scoring is free.
