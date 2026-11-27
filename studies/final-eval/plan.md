# Final evaluation: TDBF test writing vs ReProAgent on SWT-bench Lite (plan, 2026-11-25)

Written and frozen before any evaluation run (user decisions 2026-11-25). Branch `eval/swt-lite`.

## Questions

1. Does TDBF's **single test-writing agent** (gpt-5-mini) write as many valid reproduction tests as
   **ReProAgent** (four-stage pipeline, GPT-5-mini) on the same bugs, scored the same way?
2. Does the **reactive blackboard** (multi-agent, multi-model: gpt-5-nano scouts, gpt-5-mini writer
   and judge) write more valid tests than the single agent, with enough bugs to tell?
3. Same for reactive blackboard vs ReProAgent.

All comparisons two-sided. Development results so far (`studies/scouts/results/`) tie on 50 bugs.

## Bugs

`data/sample_swt_lite_final.json`: SWT-bench Lite, **276** bugs (SWE-bench Lite minus
swt-bench@eee3862 `dataset/filter_cases_lite.txt`).
- **Primary view: the 226 clean bugs**, never used to develop anything.
- Secondary view: all 276 (comparable with the leaderboard), including the 50 development bugs.

## Systems

| system | how it runs | on the 50 development bugs |
|---|---|---|
| **single** | `pipeline.run_brt_phase`, unchanged: gpt-5-mini, 50 turns per attempt, 2 gate retries, 1M tokens, 8192 tokens per turn | reuse `proc-dev-baseline` (30; agent/procedure@c797b2f) and `v2-base` (20): same agent, code and budgets |
| **reactive blackboard** | `blackboard.run_blackboard_brt_phase(reactive=True)`: nano scouts (50 turns), mini writer and judge v2, soft start 90 s | reuse `rbb-dev30` and `rbb-fresh` |
| **ReProAgent** | its published predictions, re-scored by us | its predictions; 9 SWT-Lite bugs without a prediction count as failures, as on the leaderboard |

One run per bug (pass@1), like ReProAgent. The code is tagged before the run starts.

## Scoring (the same for every system)

- Official SWT-bench harness (`~/swt-bench` @eee3862) through `scripts/18_swt_score.py`, on the
  prebuilt SWE-bench images (amd64), one bug at a time.
- Per bug: a **gold run** first (the environment must grade the gold tests as success; bugs where it
  does not are excluded for every system and listed), then the three predictions.
- **Primary metric: S** (strict success: at least one test fails before the fix and passes after, and
  none fails after it). Secondary: ΔC (coverage of the fix), our own L4 label, cost per bug and per
  valid test.
- **Scorer check:** our re-scoring of ReProAgent should match its published report (156 of 267) and
  the leaderboard (56.2% of 276) within a few bugs. A larger gap is investigated and reported before
  any comparison is.

## Analysis

- S rate per system on the 226 clean bugs (primary) and on all 276.
- Paired, per bug: exact McNemar on the discordant bugs, and a paired bootstrap 95% CI of the
  difference in S rate (10,000 resamples, seed 20261125), for single vs ReProAgent, reactive vs
  ReProAgent, and reactive vs single.
- By repository (descriptive only).
- The bugs the single agent fails become the development set for any later multi-agent work.

## Execution and cost

- Per bug: pull the prebuilt image → run single → run reactive blackboard → export both tests as
  SWT-bench predictions → score gold, single, reactive, ReProAgent → remove the image (kept: images of
  the development bugs). Resumable; unattended under `caffeinate`.
- About 12–15 minutes per bug → ~45–55 hours for 226 bugs, over several nights.
- Model cost: single ≈ $0.025 per bug, reactive ≈ $0.06 → **≈ $20** for 226 bugs; capped at $30.

## Amendment 1 (2026-11-27, during the run, user decision)

- **Our offline L0–L4 label is skipped for the clean bugs scored from the restart at bug 93 on**
  (`--no-label`): it ran each new test on the buggy and gold-fixed code in extra containers after the
  agents, ~2 minutes per bug, and it is a secondary metric. The agents' behaviour, the gate and the
  SWT-bench scoring (the primary metric S) are unchanged; such records carry `best_level: null` and
  `flags.offline_label: false`. The first 92 clean bugs keep their labels.
- **Disk:** the run's free-disk stop was lowered from 20 to 12 GB (restart at bug 79): Colima's Docker
  disk file grows to at most 100 GB and does not shrink, so 20 GB would have stopped the run early.
- **The two agents run at the same time** for each bug from the same restart on (user decision; each
  has its own containers; `--sequential-agents` restores the old order). Scoring stays one bug at a
  time (the harness shares one base image name). A busier machine can make an agent's test run hit
  its time limit slightly more often; this applies to both of our systems alike, not to ReProAgent.
- Measured pace before these changes: ~18 minutes per bug (single agent 4.1, reactive blackboard
  6.9, scoring 6.7, all sequential).
