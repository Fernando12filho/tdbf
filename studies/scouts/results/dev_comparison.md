# Scouts architecture: development comparison on 30 bugs (2026-11-24)

The comparison of `plan.md` (rule fixed before the runs). All arms use gpt-5-mini for every role, the
same 30 development bugs (SWT-bench Lite outside the 114), and the same writer budget (50 turns per
attempt, 2 gate retries). Labels are the offline gold L0–L4.

- **baseline:** the unchanged BRT agent (`proc-dev-baseline`, agent/procedure@c797b2f).
- **match:** the unchanged BRT agent + gate 2 (judge: does the failure match the issue?), up to 3
  match retries (`scouts-dev-match`).
- **full:** three scouts in parallel (100 turns each, amendment 1) + organizer + BRT agent with the
  brief + gate 2 (`scouts-dev-full-s100`).

## Totals

| arm | L4 (valid) | L3 | other | cost | $ per valid test | writer turns (mean) | time per bug |
|---|---|---|---|---|---|---|---|
| baseline | **21** | 9 | 0 | $0.74 | **$0.035** | 21.0 | n/a |
| match | **21** | 8 | 1 | $0.84 | $0.040 (1.1×) | 23.0 | 3 min |
| full | 19 | 11 | 0 | $3.01 | $0.158 (4.5×) | 12.1 | 7 min |

Paired (L4 or not):

| pair | only the first valid | only the second valid | exact McNemar p |
|---|---|---|---|
| match vs baseline | 3 (sklearn-13497, sympy-18199, sympy-23191) | 3 (django-15695, sympy-13773, sphinx-8474) | 1.0 |
| full vs baseline | 1 (sklearn-13497) | 3 (django-15695, sympy-18087, sphinx-8474) | 0.63 |
| full vs match | 1 (sympy-13773) | 3 (sympy-18087, sympy-18199, sympy-23191) | 0.63 |

## Decision (rule of plan.md)

Adopt only with more L4, no more L3, and cost per valid test ≤ 1.5× the baseline.

- **match:** L4 equal (21 = 21) → not adopted. L3 8 ≤ 9 and cost 1.1× would pass; it adds nothing.
- **full:** fewer L4 (19), more L3 (11), 4.5× the cost per valid test → fails all three; not adopted.

Neither arm is adopted. The differences are within run-to-run variation (every McNemar p ≥ 0.6).

## Gate 2 (the match judge)

| arm | judge calls | 'mismatch' | final tests: label / last verdict |
|---|---|---|---|
| match | 30 | 1 (sympy-13773; the retry was still L3) | L4/match 21, L3/match 8 |
| full | 32 | 3 (sympy-20212 → L4, django-11039 → L4, sphinx-8474 → retries failed gate 1, kept L3) | L4/match 19, L3/match 10, L3/mismatch 1 |

The judge accepted 18 of 19 final L3 tests. `gate2_l3_match_arm.md` explains why: on the match
arm, 4 of 8 L3 tests follow the issue and fail only because the benchmark's fix differs from what the
issue asks (no gold-free check can see that); the other 4 are invented or over-specified assertions a
stricter text-only judge might catch.

## Scouts

| role | turns mean / max | stop | cost |
|---|---|---|---|
| Template Generator | 13 / 42 | submit 30/30 | $0.28 |
| Edit Localizer | 35 / 64 | submit 26, 1M-token ceiling 4 | $1.23 |
| Issue Reproducer | 16 / 46 | submit 30/30 | $0.68 |
| Organizer | 1 call | brief 30/30 | $0.14 |

Scouts and organizer cost $2.34 of the full arm's $3.01 (78%); the Localizer alone 41%. The brief
halves the writer's turns (12 vs 21–23): the writer mostly transcribes the brief's proposed assertion,
so a wrong brief is copied, not corrected.

## Reading

- Decomposing Phase A into parallel scouts + organizer did **not** produce more valid tests, at 4.5×
  the cost. This repeats the procedure study (strict 15, soft 20 vs baseline 21): more structure in
  Phase A, same or fewer valid tests.
- A text-only "failure matches the issue" judge rarely fires with gpt-5-mini (4 mismatches in 62
  calls), and half of the L3 tests it could not catch even in principle.
- Development results on 30 bugs, reported as such.
