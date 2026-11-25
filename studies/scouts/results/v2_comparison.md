# Scouts v2: comparison on 20 fresh development bugs (2026-11-25)

The comparison of `plan_v2.md` (rule fixed before the runs). 20 fresh bugs (rounds 4–5 of the
procedure study's order); all 20 environments pass the gold check (official harness, fresh run ids).
Writer and judge gpt-5-mini; scouts and organizer gpt-5-nano. Labels: offline gold L0–L4.

- `v2-base`: the old BRT agent, unchanged.
- `v2-loop`: the BRT agent with judge v2 inside its loop (check_test + guarded submit_test).
- `v2-full-1m`: gpt-5-nano scouts (50 turns, 1M tokens) + organizer, then the same writer loop.

## Totals

| arm | L4 (valid) | L3 | cost | $ per valid test | writer turns | time per bug |
|---|---|---|---|---|---|---|
| base | **17** | 3 | $0.44 | **$0.026** | 18.5 | 2.0 min |
| loop | 16 | 4 | $0.57 | $0.035 (1.4×) | 15.3 | 2.7 min |
| full | 16 | 4 | $1.06 | $0.066 (2.5×) | 14.2 | 7.0 min |

Paired (L4 or not), exact McNemar:

| pair | only the first valid | only the second valid | p |
|---|---|---|---|
| loop vs base | 0 | 1 (sympy-17630) | 1.0 |
| full vs base | 2 (django-14997, sphinx-10451) | 3 (astropy-14182, pytest-7168, pylint-5859) | 1.0 |
| full vs loop | 3 (sympy-17630, django-14997, sphinx-10451) | 3 (astropy-14182, pytest-7168, pylint-5859) | 1.0 |

## Decision (rule of plan_v2.md)

More L4, no more L3, cost per valid test ≤ 1.5× `v2-base`:

- **loop:** 16 < 17 L4, 4 > 3 L3 (cost 1.4× passes) → not adopted.
- **full:** 16 < 17 L4, 4 > 3 L3, 2.5× → not adopted.

The arms differ from the base by one bug net, each way within run-to-run variation (every p = 1.0).

## The in-loop judge

| arm | judge checks | mismatches | bugs with a mismatch | of those, ended 'match' | final L3 judged 'match' |
|---|---|---|---|---|---|
| loop | 29 | 9 | 8 | 8 | 4 of 4 |
| full | 30 | 5 | 5 | 5 | 4 of 4 |

- Judge v2 inside the loop **fires** (9 mismatches in 29 checks, against 1 in 30 for the v1 outer
  judge) and the writer **always** answers it: every bug with a mismatch ended with a test judged
  'match'. Unverified submissions: 0 (loop), 1 (full).
- It does **not** change the label. In the loop arm, the 8 bugs where the judge objected ended 6 L4
  and 2 L3, and the base had the same level on all 8: the objections were to tests that were
  already valid, or to bugs no arm solved.
- Every final L3 test was judged 'match' (4/4 in both arms).

## Scouts on gpt-5-nano

| role | turns mean / max | stops | cost |
|---|---|---|---|
| Template Generator | 20 / 37 | submit 20 | $0.16 |
| Edit Localizer | 37 / 50 | submit 16, forced 3, no report 1 | $0.27 |
| Issue Reproducer | 11 / 33 | submit 20 | $0.13 |
| Organizer | 1 call | brief 20 | $0.04 |

Scouts + organizer: $0.03 per bug with nano, against $0.08 per bug with gpt-5-mini in v1: the cheaper
helpers cut the full arm's cost from 4.5× to 2.5× the base per valid test, with the same L4 as the
loop arm.

## Reading

- On these 20 bugs the old agent already writes a valid test for 17 (85%); only 3 L3 were left to
  win, so 20 bugs cannot show a small gain. None was won by either arm.
- A judge inside the writer's loop works mechanically (it objects and the writer revises), but the
  revisions did not turn any L3 into L4. Together with v1's L3 analysis (half the L3 tests follow the
  issue and fail only against the benchmark's fix), a text-only judge has little left to fix.
- Third development study in a row (procedure agent, scouts v1, scouts v2) where adding structure to
  Phase A does not beat the single BRT agent on the same model. Development results.
