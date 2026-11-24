# Scouts architecture for Phase A: plan (2026-11-24)

Branch `agent/scouts`, from `main` (36ff973) plus three cherry-picks: in-container exec timeouts
(a2885a1), sympy BRTs not collecting sympy's own `test()` (e83ce10), and run provenance in every
record (8fca7a2). Study `scouts-dev` in `src/tdbf/studies.py`.

## Why

Two observations from the literature review and our own data:

- **Wrong expected values are the main test failure.** With gpt-4o-mini, 60 of the 74 tests our gate
  accepted were L3: they fail on the buggy code at an assertion, but the asserted value is wrong.
  Issue2Test, with the same model on the same 114 instances, got 35 valid tests to our 11. It refines
  a test until its failure *matches the issue*. Our gate accepts any assertion failure.
- **MASAI** (arXiv 2406.11638) splits issue resolution into sub-agents with separate contexts; its
  Test Template Generator, Issue Reproducer and Edit Localizer run before the fix. Its future work asks
  for different (even small) models per sub-agent.

## Architecture

```
issue ─┬─ Test Template Generator ─┐   parallel; each its own container (same image)
       ├─ Edit Localizer ──────────┤   and its own model client
       └─ Issue Reproducer ────────┘
                    │
                Organizer          no tools: merges the three reports into a test brief
                    │
                BRT agent + brief  pipeline.run_brt_phase, its tools unchanged
                    │
       gate 1: fails on buggy at its assertion             (unchanged)
       gate 2: the failure matches the issue               NEW
               judge sees issue + test + buggy output only (gold-free, no tools)
               mismatch -> retry with the judge's feedback, up to 3 times
                    │
                fix agent          unchanged (not run in this validation)
```

| role | tools | output |
|---|---|---|
| Test Template Generator | read_file, grep, glob, run_python, run_test | a bug-independent test that PASSES, plus conventions |
| Edit Localizer | read_file, grep, glob, run_python | code locations, entry point to call, existing tests |
| Issue Reproducer | read_file, grep, glob, run_python | script, observed output verbatim, expected behaviour quoted (or UNSTATED) |
| Organizer | none | test brief: how to run, what to call, observed, expected, assertion, conflicts |
| BRT agent (writer) | the existing BRT tools | the test |
| Match judge | none | match / mismatch + feedback |

- **Models per role.** Every role takes its own client (`--role-model ROLE=SPEC`); default: one model
  for all. A role's model must be registered in the study.
- **Gold-free.** New guards: `assert_scout_gold_free`, `assert_organizer_gold_free`,
  `assert_match_gold_free`. The judge fails open to *match*, so it can never discard a test that
  passed gate 1. If a match retry ends on a test that fails gate 1, the last gate-passing test is kept.
- **Baseline unchanged.** `brief` and `match_client` default to off; with both off, `run_brt_phase`
  and its records are byte-identical (asserted in `scripts/agent_scouts_smoke.py`; all 20 offline
  smoke suites pass).

## Validation (development, Phase A only)

- **Bugs:** the 30 development bugs of the procedure study (`data/dev_bugs.json`): SWT-bench Lite
  outside the 114, so outside RQ1 and the SWT studies. All 30 images are on disk.
- **Model:** gpt-5-mini for every role (`openrouter:openai/gpt-5-mini@openai`).
- **Budgets:** writer 50 turns per attempt, 1M-token ceiling, 8192 tokens per turn, 2 gate retries:
  the same as the reference baseline. Scouts 20 turns each. Up to 3 match retries.
- **Arms** (to separate the two changes):
  - `scouts-dev-match`: the unchanged BRT agent + gate 2.
  - `scouts-dev-full`: scouts + organizer + BRT agent + gate 2.
- **Reference:** the unchanged BRT agent on the same 30 bugs, same model and budgets
  (`proc-dev-baseline`, agent/procedure@c797b2f; copied to `data/baseline_proc_dev.jsonl`):
  **21/30 L4, 9 L3, $0.74**. Its agent code is identical to this branch's.
- **Label:** offline L0–L4 with the gold patch, after the agents, as always.

## Comparison rule (fixed before any run)

The same rule as the procedure study, so the two development results read the same way. An arm is
adopted only if, against the reference on the same 30 bugs, it has:

1. more valid tests (L4),
2. no more L3, and
3. cost per valid test at most 1.5× the reference.

Reported in every case: L4/L3/L0–L2 counts, paired discordant bugs per pair of arms, cost per valid
test, turns, and per-role cost. For gate 2: how often the judge said *mismatch* on tests the label
calls L4 (false rejections) and *match* on L3 (false acceptances), which measures the judge itself.

With 30 bugs, a difference of a few tests is not significant (the procedure study's 9-vs-3 discordant
split gave McNemar p ≈ 0.15). These are development results, reported as such.

## Estimated cost

The reference cost $0.025 per bug. Match arm ≈ 1.5–2× (judge calls, some retries) ≈ $1.5. Full arm
≈ 3–4× (three scouts and an organizer) ≈ $3. Each batch has a `--max-usd` cap.

## Commands

```
python3 scripts/scouts_brt.py --arm match --model openrouter:openai/gpt-5-mini@openai \
    --run-group scouts-dev-match --max-usd 3
python3 scripts/scouts_brt.py --arm full  --model openrouter:openai/gpt-5-mini@openai \
    --run-group scouts-dev-full --max-usd 5
```

## Amendment 1 (2026-11-24, during the run)

- **Scout turn budget 20 → 100** (user decision). With 20 turns the Localizer was forced to submit on
  4 of the first 6 bugs and the Reproducer on 3 of 6, so their reports could be incomplete. The full
  arm restarts on all 30 bugs as `scouts-dev-full-s100`; the 7 bugs finished at 20 turns stay under
  `scouts-dev-full` and are reported only as that configuration. 100 is a loop guard, not a target:
  a scout stops when it submits.
- **Hung requests:** a scout's OpenRouter request hung 25+ minutes (keep-alive bytes defeat the SDK's
  read timeout). Every request now has a 300 s wall-clock deadline with two re-sends (fbd0acb).
  Infrastructure only; agent behaviour is unchanged.
- The comparison rule is unchanged.
