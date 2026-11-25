# Blackboard Phase A: plan (2026-11-25, before any blackboard run)

Third design in the scouts line (user decision 2026-11-25), after v1 (parallel scouts → organizer →
writer, outer judge) and v2 (gpt-5-nano scouts, judge inside the writer's loop). Both v1 and v2 had
the scouts work **alone** and hand over a brief at the end. The literature has a third family,
parallel agents that **exchange findings while working** (blackboard systems: LbMAS, Salemi et al.
2510.01285, PatchBoard; see `docs/literature_search_2026-11-23.md` §8). This study tests that family.

## Architecture

```
            ┌──────────── BLACKBOARD (fixed sections, evidence required) ────────────┐
            │ entry_point │ locations │ observed_behaviour │ expected_behaviour │       │
            │ test_conventions │ test_status │ questions │ answers                     │
            └──────▲────────────▲────────────▲────────────▲───────────────────────────┘
       read/post   │            │            │            │
         Template ─┘  Localizer ┘ Reproducer ┘    Writer ─┘   all four start at the same time,
         (nano)       (nano)      (nano)         (mini)       each in its own container
```

- **Board tools** for every agent: `read_board` (new entries since its last read, or all),
  `post_finding(section, text, evidence)`: evidence (a file:line, an observed run's output, or a quote
  from the issue) is required, `ask_board(question)`, `answer(question_id, text, evidence)`. Scouts
  also get `wait_for_board(seconds)`, which blocks without a model call until something new appears.
- **Notices:** every non-terminal tool result of an agent ends with how many board entries it has not
  read yet, so findings reach agents without polling.
- **Scouts** (gpt-5-nano): their v2 investigation tools (read_file, grep, glob, run_python; the
  Template Generator also run_test), post findings as soon as they have them, answer questions in
  their area, and finish when they have nothing to add or when the writer has finished.
- **Writer** (gpt-5-mini): the v2 writer, check_test with judge v2 inside its loop and the guarded
  submit_test, plus the board tools. It starts immediately and reads the board when it needs facts.
  The judge's context is the board at the time of the check.
- **No organizer:** the board replaces it.
- Gate 1, gate retries, the offline label and every gold-free rule are unchanged; new allowlists and
  guards for the board-enabled scouts and writer. The board holds only agents' own text.

## Validation

- **Bugs:** the 20 fresh bugs (primary; all gold-valid) and the 30 development bugs (secondary).
- **Compared with stored results only (no reruns, user decision):**
  - 20 fresh: `v2-base` 17, `v2-loop` 16, `v2-full-1m` 16 (L4). Blackboard vs `v2-full-1m` isolates the
    communication structure (same models, same writer loop, same judge; parallel-then-merge vs
    exchange).
  - 30 dev: `proc-dev-baseline` 21, `scouts-dev-match` 21, `scouts-dev-full-s100` 19. Secondary:
    judge v2's rules were derived from these bugs' L3 cases, and v1 used gpt-5-mini scouts.
- **Budgets:** writer 50 turns per attempt, 2 gate retries, 3 judge mismatches before an unverified
  submit, 1M tokens; scouts 50 turns and 1M tokens each.
- **Rule (unchanged):** adopted only with more L4 than the old agent on the same bugs, no more L3,
  and cost per valid test ≤ 1.5× the old agent.
- **Also reported:** board use (entries per agent and section, questions asked and answered, whether
  the writer read the board before its first test), per-role cost, time per bug.

## Estimated cost

About $0.05–0.07 per bug (nano scouts, mini writer and judge) → ~$3 for 50 bugs, capped at $5.
