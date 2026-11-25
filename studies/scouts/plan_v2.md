# Scouts architecture v2: plan (2026-11-25, before any v2 run)

v1 result (`results/dev_comparison.md`): on the 30 development bugs, the full arm got 19/30 valid tests
against the old agent's 21/30, at 4.5× the cost per valid test; the judge said *mismatch* 4 times in 62
calls. The L3 analysis (`results/gate2_l3_match_arm.md`) found half the L3 tests uncatchable without
gold and half caused by a lenient judge. v2 changes three things (user decisions 2026-11-25):

1. **Cheaper helpers.** The three scouts and the organizer run on **gpt-5-nano**
   (`openrouter:openai/gpt-5-nano@openai`, $0.05/$0.40 per 1M tokens, ~5× cheaper than gpt-5-mini).
   The test writer and the judge stay on **gpt-5-mini**, the old agent's model and ReProAgent's.
   Scouts are capped at 50 turns and 300K tokens each (v1: Localizer up to 64 turns, 4 runs hit 1M).
2. **The writer loops until its test matches the issue, inside its own conversation.** Its `run_test`
   becomes `check_test`: run the test on the buggy code and, when it fails at an assertion, ask the
   judge in the same call; the result carries the outcome and the verdict with its reason.
   `submit_test` is refused until a `check_test` of that exact code returned *match*; after 3
   mismatches the writer may submit anyway (flagged). Gate 1 (executed, buggy code) stays the final
   check. No outer judge loop.
3. **Judge v2, stricter**, from the L3 analysis: an assertion the issue does not support (exception
   type, exact message or format, an unstated number) is a mismatch; the expected value must come from
   the issue or from an observed run; the judge also sees the brief's expected-behaviour section when
   there is one. Still gold-free (issue, brief, test, buggy output) and fail-open to *match*.

## Validation

- **Bugs:** 20 fresh development bugs, rounds 4–5 of the procedure study's fixed order (positions
  30–49; seed 20261112; pool = SWT-bench Lite minus the 114): `data/dev_round4.json`,
  `data/dev_round5.json`. Not the 30 v1 bugs, because judge v2 was derived from their L3 cases.
- **Arms** (all with the same writer budget as v1: 50 turns per attempt, 2 gate retries, 1M tokens):
  - `v2-base`: the old agent, unchanged, gpt-5-mini: the reference on these bugs.
  - `v2-loop`: the old agent with the writer loop and judge v2, no scouts.
  - `v2-full`: gpt-5-nano scouts + organizer, then the writer loop and judge v2.
- **Label:** offline L0–L4 with the gold patch, after the agents.

## Comparison rule (fixed now, the same as v1)

An arm is adopted only if, against `v2-base` on the same 20 bugs, it has more L4, no more L3, and cost
per valid test at most 1.5× `v2-base`. Reported in every case: L4/L3/other, paired discordant bugs,
cost per valid test and per role, writer turns, judge verdicts per check, mismatches the writer fixed,
and judge false accepts (final L3 with *match*) and false rejects (an L4-labelled final test submitted
after 3 mismatches). 20 bugs is small; these are development results.

## Estimated cost

`v2-base` ≈ $0.50 (the old agent cost $0.025 per bug); `v2-loop` ≈ $0.04 per bug ≈ $0.80; `v2-full`
≈ $0.05–0.06 per bug ≈ $1.20. Total ≈ $2.50, with a `--max-usd` cap per arm. Images: 20 pulls
(prebuilt, `--platform linux/amd64`), 82 GB free.
