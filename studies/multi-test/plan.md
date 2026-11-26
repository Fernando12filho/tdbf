# Multi-test reproduction tests: plan (2026-11-25, before any run)

User decision (2026-11-25), raised while reading the evaluation's tests: every agent writes **one**
test function per bug (single agent 50/50 files, reactive blackboard 47/49, ReProAgent 256/267). That
is our design (the BRT prompt asks for "one pytest test"), not the model's choice. More tests trade a
higher chance of reproducing the bug and more coverage against a higher chance that one wrong
assertion fails after the fix. For TDBF the question that matters is the **fix phase**: the main RQ2
finding is the verification gap (a fix passes the one test but does not resolve the bug). Several
tests covering related cases could make such a fix harder to submit.

## Question

Does a multi-test BRT (a main reproduction test plus a few related cases) make the fix agent resolve
more bugs than the one-test BRT, and what does it cost in valid tests and money?

## Arms (the 50 development bugs, gpt-5-mini for every agent)

| arm | Phase A (test) | Phase B (fix) |
|---|---|---|
| **single** | the stored one-test tests of the unchanged agent (`proc-dev-baseline` 30, `v2-base` 20); no new run | new fix run with that test |
| **multi** | new run of the same agent with the multi-test instruction | new fix run with that test module |

- **Multi-test instruction** (the only Phase A change): write a test module with (1) the main
  reproduction test, the issue's own scenario, and (2) 2–4 more tests of closely related cases the
  issue implies the fix must handle (other inputs it mentions, edge cases, the inverse or round-trip
  operation, another call path), each asserting only what the issue supports and reaching its
  assertions. The main test must fail on the buggy code; the others should fail now too where they
  exercise the bug, and may pass if they guard behaviour that must stay correct.
- **Phase A budget:** 50 turns per attempt, 2 gate retries, 1M tokens, 8192 per turn (as the stored).
  Gate unchanged (the module fails on the buggy code at an assertion).
- **Phase B:** the unchanged fix agent (`run_fix_phase`, run_brt runs the whole module), 50 turns,
  1M tokens, 8192 per turn, one attempt; only for tests that passed the gate (as in the deployable
  pipeline); a bug whose test did not pass the gate counts as unresolved.
- **Scoring:** SWE-bench harness (held-out FAIL_TO_PASS / PASS_TO_PASS), a fresh run id per fix, the
  verified-verdict rule; offline L0–L4 label for every test (module-level: L4 needs every test to
  pass on the fixed code).

## Metrics and rule (fixed now)

- **Primary: resolved bugs**, multi vs single, paired on the 50 (exact McNemar).
- Secondary: L4 tests, gate acceptance, tests per module, Phase A + B cost, cost per resolved bug,
  and among resolved-by-one-arm bugs whether the other arm's fix passed its own tests (verification
  gap).
- **Adopted** only if multi resolves more bugs and its cost per resolved bug is at most 1.5× single.
- 50 development bugs are a first look; a positive result would be confirmed on clean bugs.

## Cost

Phase A multi ≈ $0.03 per bug; Phase B ≈ $0.03–0.05 per fix × 2 arms → about **$5** for 50 bugs,
capped at $8. The final evaluation's long run waits until this study is done.
