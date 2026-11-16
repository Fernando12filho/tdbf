# Development round 2: strict procedure agent v1 (2026-11-15)

Agent v1 (commit a537185, changes C1–C8 from round 1), gpt-5-mini, 50 turns, round-2 bugs
(`data/dev/round2.json`, positions 10–19 of the development order, not the 114). Strict only.
Records: run group `proc-dev-r2-strict` in results.jsonl; transcripts in
`runs/procedure/proc-dev-r2-strict/`. Preflight live check before the run: all 10 environments
valid with the official command. Total cost $0.39, 44 minutes.

## Results

| bug | label | turns | $ | outcome / flags |
|---|---|---|---|---|
| sympy-13971 | L3 (wrong assertion) | 18 | 0.044 | test; symptom_from_preflight_claim, symptom_not_checked |
| sympy-18087 | L4 | 11 | 0.030 | test; symptom_unchecked |
| django-11630 | L3 (wrong assertion) | 42 | 0.069 | test; unreproduced |
| sympy-18199 | L4 | 19 | 0.044 | test; symptom_unchecked |
| django-10924 | L1 (passes on buggy) | 42 | 0.055 | forced test; unreproduced, degraded:write |
| django-14155 | L4 | 13 | 0.021 | test; symptom_unchecked |
| sympy-23191 | L3 (crashes on both) | 23 | 0.046 | forced test; degraded:write, unknown_expected |
| django-14016 | L4 | 3 | 0.011 | test; symptom_from_preflight_claim |
| sympy-13773 | L3 (wrong assertion) | 18 | 0.046 | test; symptom_unchecked |
| django-14411 | L4 | 26 | 0.021 | test; unreproduced |

**5 valid (L4), 4 wrong reason (L3), 1 passes on buggy (L1), 0 no test (L0).** $0.077 per valid
test, 21.5 turns on average.

## Round 1 (v0) vs round 2 (v1)

Different bugs in each round, so this shows where the turns went, not an effect size.

| | round 1, v0 | round 2, v1 |
|---|---|---|
| valid (L4) | 5/10 | 5/10 |
| wrong reason (L3) | 2 | 4 |
| no test (L0) / crash (L2) / L1 | 2 / 1 / 0 | 0 / 0 / 1 |
| mean turns | 40 | 21.5 |
| $ per valid test | 0.108 | 0.077 |
| Pin ran | 10/10 | 2/10 (C2) |
| runs that ran out of turns | 2 | 0 |
| read refusals (C3) | — | 6 bugs: reproduce 2 (5, 3 refusals), write 4 (1–4) |
| Understand needed a 2nd visit | 7/10 | 8/10 (6 failed the 2nd as well) |
| assertion-map send-backs | 5 | 0 (C5 refusals inside the visit) |

What C1–C8 did: the turns moved from reading to running (no run exhausted its budget; mean turns
halved), Pin stopped running on test-design questions, and assertion-map problems no longer cost
a visit. The valid count did not move, and the failures changed kind (below).

## Failure patterns

1. **Understand still fails its quote check, now because of how the preflight block is shown.**
   Of 15 rejected quotes: 9 are the block's own status line "no code from the issue could be
   run as is." quoted as an observation (4 bugs, both visits); 3 copy the block's headers
   (`--- code block:` / `--- printed:`) into the quote; 2 are a real misquote (13971 dropped
   `\right` from a LaTeX string; the check was right); 1 is a long warning line. Each costs one
   turn and the claim, not the run.
2. **Preflight noise taken as the bug (C1 side effect).** In sympy-13971 the preflight output
   carried the environment's `DeprecationWarning: Using or importing the ABCs from 'collections'`;
   the model made it an observed claim, Reproduce was skipped on it (flag
   `symptom_from_preflight_claim`), and the test checks the warning, not the LaTeX bug. In
   django-10924 the observed claim is preflight's `NameError: name 'models' is not defined` (the
   issue snippet lacks imports): an authoring error of the snippet, not behaviour of the code.
   That test passes on buggy (L1).
3. **Expected values that are wrong at the source.** sympy-13773: the issue says scalar `@`
   should raise ValueError "as in NumPy"; the fix raises TypeError, so a test faithful to the
   issue fails on the fixed code. django-11630: Pin verified "expected" values from existing tests
   that describe today's (buggy) check, and the test asserts the E028 error the issue reports.
   Both are issue-information cases (the study's subject), not something a gate can catch.
4. **sympy-23191** (pretty-printing of vectors): no expected output in the issue, three unknowns
   left, the test crashed in all three Write visits (L3 runtime, forced).
5. `symptom_unchecked` in 5 runs: observed values were descriptions ("misplaced i_C"), not
   tokens, so no symptom check applied.

## Proposed changes for round 3 (agent v2), not yet applied

| # | Change | Targets |
|---|---|---|
| C9 | No PREFLIGHT block when nothing ran (the Understand prompt says "no preflight output" instead); block headers marked so they read as labels, and a preflight quote is checked with those labels stripped | pattern 1 |
| C10 | A preflight-sourced observed claim that names an authoring error (NameError, ImportError, ModuleNotFoundError, SyntaxError) or only a warning is not accepted as an observation | pattern 2 |
| C11 | Preflight display drops `DeprecationWarning` lines emitted at import time (`/testbed/...: DeprecationWarning:` plus its source line); the record keeps them | pattern 2 |

Patterns 3–5 are left as they are: 3 is what the removal study measures, 4 is one bug, and 5 has
no deterministic fix that would not invent tokens.

Stopping rule (design §10c): 20 development bugs so far, the baseline not yet run on them; the
comparison needs ≥ 30 bugs and the baseline.
