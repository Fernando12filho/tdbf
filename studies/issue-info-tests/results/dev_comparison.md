# Development comparison: baseline vs strict vs soft on 30 development bugs (2026-11-16)

The stopping-rule comparison of design §10c. All three agents use gpt-5-mini and 50 turns per
attempt, on the 30 development bugs of rounds 1–3 (SWT-bench Lite outside the 114).

- **baseline**: the unchanged BRT agent (`pipeline.run_brt_phase`: one free-form agent, gold-free
  gate, up to 2 retries). Run group `proc-dev-baseline`, $0.74, 72 minutes.
- **strict**: the procedure agent with phases and gates. Its 30 results come from the three
  development rounds, so from three versions (v0, v1, v2), not the frozen `procedure-v1`
  (re-running it was skipped by decision). Run groups `proc-dev-r{1,2,3}-strict`.
- **soft**: the frozen `procedure-v1` soft mode (the same steps as one prompt, all tools, no
  gates). Run group `proc-dev-soft`, $1.57, 2.8 hours.

Each agent ran once per bug. Labels L0–L4 come from the offline gold label (L4 = fails on the
buggy code, passes on the fixed code: a valid test; L3 = fails on both).

## Totals

| agent | valid (L4) | L3 | L0 / L1 / L2 | total $ | $ per valid test | mean turns |
|---|---|---|---|---|---|---|
| baseline | **21/30** | 9 | 0 / 0 / 0 | 0.74 | **0.035** | 22.7 |
| strict | 15/30 | 11 | 2 / 1 / 1 | 1.39 | 0.093 | 31.0 |
| soft | 20/30 | 6 | 0 / 3 / 1 | 1.57 | 0.079 | 30.0 |

By round (strict version in brackets):

| round | baseline L4 | strict L4 | soft L4 |
|---|---|---|---|
| 1 (v0) | 7 | 5 | 5 |
| 2 (v1) | 6 | 5 | 8 |
| 3 (v2) | 8 | 5 | 7 |

Paired, per bug (L4 or not):

| pair | both valid | only the first | only the second |
|---|---|---|---|
| strict vs baseline | 12 | 3 | 9 |
| soft vs baseline | 16 | 4 | 5 |
| strict vs soft | 12 | 3 | 8 |

By repository: baseline 12/16 Django, 9/14 other; strict 8/16 and 7/14; soft 9/16 and 11/14.

## Reading

- **Strict is worse than the baseline on every criterion of the stopping rule**: fewer valid tests
  (15 vs 21), more L3 (11 vs 9), and 2.7× the cost per valid test (the rule allows 1.5×). It is
  behind in each of the three rounds, whatever its version. Paired, 9 bugs only the baseline got
  right against 3 only strict did (exact McNemar p ≈ 0.15 on 30 bugs: a consistent direction,
  not a significant one).
- **Soft is level with the baseline** (20 vs 21, 4 vs 5 discordant) at 2.3× the cost per valid
  test. Giving the model the procedure as instructions neither helps nor hurts much.
- So the loss comes from **enforcing** the procedure (phases with separate contexts, turn budgets
  per phase, gates and send-backs), not from the steps themselves. The round reviews show where:
  turns spent in Understand/Pin/Locate before any test exists, Write starting cold from a handoff
  record, forced submissions when Write's budget runs out (5 of round 3's failures).
- The baseline used its gate retry once in 30 bugs: with gpt-5-mini it usually gets a test that
  fails on the buggy code at the first attempt.

## Decision for the stopping rule

The rule (design §10c) adopts the procedure agent only if it shows more valid tests, no more L3,
and cost per valid test ≤ 1.5× the baseline. Strict meets none; soft meets none of the "more"
criteria. The procedure agent is **not adopted** as the improved agent. Its development is kept
as a reported negative result (enforced procedure < free agent; same steps as instructions ≈ free
agent).

## Per bug

| bug | baseline L / turns / $ / notes | strict L / turns / $ / notes | soft L / turns / $ / notes |
|---|---|---|---|
| scikit-learn__scikit-learn-13497 | L3(wrong_assertion) / 15 / 0.022 /  | L4 / 39 / 0.099 / forced_test [truncated_turns,degraded:write,forced_submit,unknown_expected] | L4 / 32 / 0.066 / test |
| django__django-14608 | L4 / 33 / 0.023 /  | L4 / 40 / 0.041 / test [unreproduced,unknown_expected] | L4 / 26 / 0.037 / test |
| django__django-14855 | L4 / 24 / 0.034 /  | L0 / 50 / 0.058 / no_test [unreproduced,budget_exhausted] | L3(wrong_exception) / 50 / 0.076 / forced_test [forced_submit] |
| sympy__sympy-19254 | L3(wrong_assertion) / 33 / 0.056 /  | L3(wrong_assertion) / 26 / 0.045 / test [symptom_unchecked,unknown_expected] | L4 / 13 / 0.030 / test |
| psf__requests-3362 | L4 / 16 / 0.010 /  | L4 / 50 / 0.054 / test [unreproduced,unknown_expected] | L3(wrong_assertion) / 28 / 0.037 / test |
| django__django-17051 | L4 / 30 / 0.023 /  | L2 / 48 / 0.054 / forced_test [unreproduced,no_turns_left:write,forced_submit,unknown_expected] | L4 / 24 / 0.030 / test |
| django__django-15695 | L4 / 37 / 0.063 /  | L0 / 50 / 0.051 / no_test [unreproduced,budget_exhausted] | L1 / 33 / 0.042 / test |
| django__django-15498 | L4 / 18 / 0.011 /  | L4 / 33 / 0.028 / test [symptom_unchecked] | L2 / 12 / 0.013 / test |
| sympy__sympy-20212 | L4 / 5 / 0.003 /  | L4 / 15 / 0.026 / test [symptom_not_checked,unknown_expected] | L4 / 32 / 0.038 / test |
| django__django-12589 | L3(wrong_assertion) / 24 / 0.037 /  | L3(wrong_assertion) / 50 / 0.086 / test [unreproduced,unknown_expected] | L3(wrong_assertion) / 22 / 0.038 / test |
| sympy__sympy-13971 | L4 / 28 / 0.024 /  | L3(wrong_assertion) / 18 / 0.044 / test [symptom_from_preflight_claim,symptom_not_checked] | L4 / 42 / 0.131 / test |
| sympy__sympy-18087 | L4 / 7 / 0.006 /  | L4 / 11 / 0.030 / test [symptom_unchecked] | L4 / 20 / 0.044 / test |
| django__django-11630 | L3(wrong_assertion) / 33 / 0.045 /  | L3(wrong_assertion) / 42 / 0.069 / test [unreproduced] | L3(wrong_assertion) / 48 / 0.118 / test |
| sympy__sympy-18199 | L3(wrong_assertion) / 22 / 0.032 /  | L4 / 19 / 0.044 / test [symptom_unchecked] | L4 / 30 / 0.084 / test |
| django__django-10924 | L4 / 36 / 0.040 /  | L1 / 42 / 0.055 / forced_test [unreproduced,degraded:write,forced_submit] | L4 / 39 / 0.075 / test |
| django__django-14155 | L3(wrong_assertion) / 23 / 0.025 /  | L4 / 13 / 0.021 / test [symptom_unchecked] | L3(wrong_assertion) / 14 / 0.020 / test |
| sympy__sympy-23191 | L3(wrong_assertion) / 19 / 0.017 /  | L3(runtime) / 23 / 0.046 / forced_test [symptom_unchecked,degraded:write,forced_submit,unknown_expected] | L4 / 26 / 0.036 / test |
| django__django-14016 | L4 / 8 / 0.005 /  | L4 / 3 / 0.011 / test [symptom_from_preflight_claim] | L4 / 15 / 0.031 / test |
| sympy__sympy-13773 | L4 / 14 / 0.010 /  | L3(wrong_assertion) / 18 / 0.046 / test [symptom_unchecked] | L4 / 37 / 0.080 / test |
| django__django-14411 | L4 / 42 / 0.034 /  | L4 / 26 / 0.021 / test [unreproduced] | L4 / 43 / 0.044 / test |
| sympy__sympy-24066 | L4 / 10 / 0.015 /  | L4 / 13 / 0.029 / test [symptom_from_preflight_claim] | L4 / 23 / 0.050 / test |
| sympy__sympy-15346 | L4 / 3 / 0.004 /  | L4 / 12 / 0.021 / test [symptom_unchecked] | L4 / 29 / 0.051 / test |
| sphinx-doc__sphinx-8474 | L4 / 73 / 0.081 / 1 retry | L3(wrong_assertion) / 48 / 0.036 / forced_test [no_turns_left:write,forced_submit,unknown_expected] | L1 / 28 / 0.038 / test |
| sympy__sympy-19007 | L4 / 20 / 0.016 /  | L3(runtime) / 29 / 0.035 / forced_test [symptom_unchecked,degraded:write,forced_submit] | L4 / 41 / 0.066 / test [default_location] |
| django__django-11039 | L4 / 21 / 0.024 /  | L4 / 42 / 0.061 / test [unreproduced,unknown_expected] | L4 / 37 / 0.061 / test |
| django__django-13551 | L4 / 7 / 0.005 /  | L4 / 14 / 0.011 / test [symptom_unchecked] | L4 / 14 / 0.020 / test |
| django__django-16910 | L4 / 24 / 0.034 /  | L3(wrong_exception) / 48 / 0.074 / forced_test [symptom_unchecked,no_turns_left:write,forced_submit] | L4 / 31 / 0.039 / test |
| sympy__sympy-17022 | L3(wrong_assertion) / 11 / 0.012 /  | L3(runtime) / 34 / 0.077 / forced_test [symptom_partial,degraded:write,forced_submit] | L1 / 42 / 0.064 / test |
| django__django-11049 | L4 / 11 / 0.006 /  | L4 / 26 / 0.046 / test | L4 / 25 / 0.031 / test |
| django__django-11905 | L3(wrong_assertion) / 33 / 0.028 /  | L3(wrong_assertion) / 48 / 0.073 / forced_test [symptom_unchecked,no_turns_left:write,forced_submit] | L3(wrong_assertion) / 45 / 0.080 / test |

| agent | bugs | valid (L4) | wrong reason (L3) | total $ | $ per valid test | mean turns |
|---|---|---|---|---|---|---|
| baseline | 30 | 21 | 9 | 0.744 | 0.035 | 22.7 |
| strict | 30 | 15 | 11 | 1.392 | 0.093 | 31.0 |
| soft | 30 | 20 | 6 | 1.572 | 0.079 | 30.0 |
