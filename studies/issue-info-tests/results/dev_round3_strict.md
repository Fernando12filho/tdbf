# Development round 3: strict procedure agent v2 (2026-11-16)

Agent v2 (commit 4151727, changes C9–C11 from round 2), gpt-5-mini, 50 turns, round-3 bugs
(`data/dev/round3.json`, positions 20–29 of the development order, not the 114). Strict only.
Records: run group `proc-dev-r3-strict` in results.jsonl; transcripts in
`runs/procedure/proc-dev-r3-strict/`. Preflight live check before the run: all 10 environments
valid with the official command. Total cost $0.46, 50 minutes.

## Results

| bug | label | turns | $ | outcome / flags |
|---|---|---|---|---|
| sympy-24066 | L4 | 13 | 0.029 | test; symptom_from_preflight_claim |
| sympy-15346 | L4 | 12 | 0.021 | test; symptom_unchecked |
| sphinx-8474 | L3 (wrong assertion) | 48 | 0.036 | forced test; no_turns_left:write, unknown_expected |
| sympy-19007 | L3 (crashes on both) | 29 | 0.035 | forced test; degraded:write |
| django-11039 | L4 | 42 | 0.061 | test; unreproduced, unknown_expected |
| django-13551 | L4 | 14 | 0.011 | test; symptom_unchecked |
| django-16910 | L3 (wrong exception) | 48 | 0.074 | forced test; no_turns_left:write |
| sympy-17022 | L3 (crashes on both) | 34 | 0.078 | forced test; symptom_partial, degraded:write |
| django-11049 | L4 | 26 | 0.046 | test |
| django-11905 | L3 (wrong assertion) | 48 | 0.073 | forced test; no_turns_left:write |

**5 valid (L4), 5 wrong reason (L3), 0 L0/L1/L2.** $0.093 per valid test, 31.4 turns on average.
All five L3 tests are forced submissions (Write never produced a test that passed its gate).

## Rounds 1–3

| | round 1, v0 | round 2, v1 | round 3, v2 |
|---|---|---|---|
| valid (L4) | 5/10 | 5/10 | 5/10 |
| wrong reason (L3) | 2 | 4 | 5 |
| L0 / L1 / L2 | 2 / 0 / 1 | 0 / 1 / 0 | 0 / 0 / 0 |
| mean turns | 40 | 21.5 | 31.4 |
| $ per valid test | 0.108 | 0.077 | 0.093 |
| Understand needed a 2nd visit | 7/10 | 8/10 | 4/10 |
| forced submissions | 2 | 2 | 5 |

Different bugs each round, so the columns show where the turns went, not an effect size.
v2's C9–C11 did what they targeted: Understand revisits fell from 8 to 4, and of those 4,
2 are the real noise refusals C10 was built for (django-16910, sympy-17022: the issue's snippet
failed on its own NameError / missing numpy) and 2 are a quote-check bug (pattern 1).

## Failure patterns

1. **Zero-width spaces break the quote check (a bug in the check).** django-11039 and
   django-11905 contain U+200B (left by Trac links). The model echoes each as a control
   character (`\x00`, `\x0b`), so every quote around one fails: both visits of Understand in both
   bugs. In 11039 the three expected claims were lost and became unknowns, sending the run through
   Pin (10 turns). Removing zero-width and control characters on both sides makes all 8 quotes
   verify (checked on the stored answers).
2. **Write spends its turns reading.** In the three runs that ran out of Write turns
   (sphinx-8474, django-16910, django-11905), the first visit starts with 5 `read_file` calls (the
   5th refused by the read budget) and then alternates reads and runs; Write closes on a forced
   submission. The read budget (C3, 4 in a row) still leaves most of Write's 15 turns to reading.
3. **Django test home and the models it uses.** django-11905: Locate chose `tests/utils_tests/`
   (the default), the test imports `many_to_one_null.models`, and Django's runner refuses it
   ("Model class many_to_one_null.models.Reporter doesn't declare an explicit app_label and isn't
   in an application in INSTALLED_APPS"). The model then spent 9 test runs on settings and app
   labels. The fix is deterministic: the error names the package.
4. **Missing dependency.** sympy-17022 is about lambdify with NumPy; NumPy is not installed in
   the official image, so neither preflight nor the test can show the bug (L3, crashes on both).
   No change: the environment is the benchmark's.
5. **Crashing tests stay crashing.** sympy-19007 crashed in all three Write visits (as sympy-23191
   did in round 2).

## Proposed changes for agent v3, not yet applied

| # | Change | Targets |
|---|---|---|
| C12 | Quote check removes zero-width characters (U+200B–U+200D, U+FEFF) and control characters from both the source and the quote | pattern 1 (a bug fix) |
| C13 | Write's read budget is 2 in a row (Reproduce keeps 4); Locate's `style_notes` already carry the directory's style | pattern 2 |
| C14 | Django: when a test run reports "Model class `<pkg>.models.<X>` doesn't declare an explicit app_label", the controller moves the test home to `tests/<pkg>/` if that is a test package (Locate's own check) and tells the model, within the same Write visit | pattern 3 |

## Stopping rule (design §10c)

30 development bugs now: 15 valid (L4), 11 wrong reason (L3), $0.091 per valid test overall.
The rule compares against the baseline on the same bugs, which has not run yet. The 30 strict
results come from three agent versions (v0, v1, v2).
