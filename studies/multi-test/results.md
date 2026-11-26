# Multi-test reproduction tests: results (2026-11-26)

The comparison of `plan.md` (rule fixed before the run; fix phase dropped before the run). 50
development bugs, gpt-5-mini; the multi-test agent decides how many tests a bug needs. Both arms
scored with the final evaluation's scorer (official SWT-bench harness, one image and one gold run
per bug, `scripts/20_final_eval.py --set dev --systems single rbb reproagent multi`). All 50 gold runs
pass. 50/50 bugs written, no crashes, $1.56.

## Tests the agent chose to write

| tests in the module | bugs | multi S | single S on the same bugs |
|---|---|---|---|
| 1 | 24 | 13 | 15 |
| 2 | 21 | 9 | 12 |
| 3 | 5 | 2 | 2 |

Mean 1.6 tests per module; the agent kept one test for about half the bugs.

## Results

| arm | strict S | F→P (loose) | ΔC (coverage of the fix, mean) | L4 / L3 (our label) |
|---|---|---|---|---|
| single (stored one-test tests) | **29/50** | **29** | 0.56 | 38 / 12 |
| **multi** | 24/50 | 25 | 0.57 | 36 / 14 |

- Paired on S: multi only 3 (django-14997, sklearn-13497, sympy-13971), single only 8
  (django-14855, django-15695, requests-3362, pytest-7168, sympy-12481, sympy-13437, sympy-18087,
  sympy-20212); exact McNemar p = 0.23; difference in S rate −0.10, 95% bootstrap CI [−0.22, +0.02].
- Coverage: +0.011 on average over 49 bugs (multi higher on 6, lower on 5): no real gain.
- Rule (more S than single, no more L3, cost per valid test ≤ 1.5×): fewer S and more L3 →
  **not adopted**.

## Reading

- More tests did **not** cover more of the fix or catch more bugs: F→P fell (25 vs 29) and coverage
  stayed level. The extra tests mostly exercise the same code path as the main test.
- They cost valid tests: a module with one wrong related-case test fails SWT-bench's strict success
  (no test may fail after the fix), and the agent's main tests were also slightly weaker (L4 36 vs 38).
- The one clear win, sklearn-13497 (both input forms the issue names), shows the idea can work when
  the issue itself lists the related cases; across 50 bugs it is outweighed.

## Also from this scoring pass (the final evaluation's development view and scorer check)

| system | S / 50 | F→P | ΔC |
|---|---|---|---|
| ReProAgent | 33 | 33 | 0.67 |
| reactive blackboard | 30 | 32 | 0.62 |
| single agent | 29 | 29 | 0.56 |

- **Scorer check passed:** our re-scoring agrees with ReProAgent's published verdicts on 48/48 bugs
  where both exist.
- Paired: single vs ReProAgent 8 vs 12 (p = 0.50, CI [−0.26, +0.10]); reactive vs ReProAgent 8 vs 11
  (p = 0.65); reactive vs single 8 vs 7 (p = 1.0). ReProAgent's extra wins are mostly sympy (10 of its
  12 over the single agent); ours are mostly Django.
- These are development bugs; the comparison that counts is the 226 clean bugs.
