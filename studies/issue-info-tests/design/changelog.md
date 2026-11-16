# Procedure agent: changelog

Every change to the agent, with the reason and the commit (design §10c, step 4).
Development rounds add entries here; the version frozen as `procedure-v1` is the last one.

## v0 — first build, before any model run (2026-11-12)

Built from `procedure_agent.md` as written. Where the build had to decide something the design
leaves open, or where a check on development bugs showed the design would fail, the choice is
listed here.

**Files.** `src/tdbf/agent/preflight.py`, `procedure.py` (controller), `procedure_gates.py`,
`procedure_tools.py`, `procedure_prompts.py`, `procedure_score.py` (offline label, gold);
`src/tdbf/harness/official.py`; `scripts/procedure_brt.py` (runs strict / soft / baseline),
`procedure_smoke.py`, `procedure_preflight_smoke.py`, `procedure_live_check.py`,
`procedure_preflight_live.py`. Run groups `proc-dev-r1-{strict,soft,baseline}` in study
`procedure-dev` (gpt-5-mini). The baseline (`run_brt_phase`) is unchanged.

**Decisions taken in the build**

| # | What | Why |
|---|---|---|
| B1 | Every model turn counts against the 50, including a phase's closing "finish now" turn and nudges | The budget must bind; the baseline's closing turns are not counted in its `max_iterations`, so the procedure agent is if anything stricter |
| B2 | A phase that was skipped and is later re-entered by a send-back draws on the reserve | Skipping moves its turns to the reserve; without this a sent-back Reproduce had 0 turns |
| B3 | Reproduce is skipped (0 turns, not 1) when a preflight run matches the observed claim | Deterministic, costs nothing; the design allowed 1 turn |
| B4 | Locate is skipped when the top test candidate is a file with score ≥ 4 and at least twice the second's; the Locate gate (directory in HEAD, trivial probe runs there) still applies | Design: "one obvious candidate" |
| B5 | Write gets `read_file`, `grep`, `glob`, `run_test`; not `run_python` | Design §3.6 lists read_file and run_test; grep/glob are read-only |
| B6 | Every assertion statement must be in `assertion_map` and point to verified evidence; observed claims (O-ids) count as evidence as well as expected (E-ids) | Enforces D2 deterministically; an assertion against an observed buggy value is weaker but is not an unverified expected value |
| B7 | **Symptom check skipped when the failure output shows no values** (bare `AssertionError`, as sympy's `bin/test` and plain unittest asserts print); flag `symptom_not_checked` | Live check on sympy-20212: the check rejected a test the offline label calls L4, because the runner prints no values. A gate must not reject what it cannot see |
| B8 | Short value tokens (≤ 3 chars, e.g. `0`) match only as a printed or compared value (after `->`, `=`, `:`, `assert`, or a whole line; before a comparison or end of line) | `0` also matched inside `0**-oo` |
| B9 | A send-back that names the previous test includes it in the next phase's prompt | The handoff record alone did not say what the rejected test was |
| B10 | No outer retry: one procedure run of 50 turns; send-backs replace the baseline's cold retries | Design §0. The baseline keeps its own two gate retries (it is unchanged), so it can use up to 3 × 50 turns |
| B11 | Unverified expected quotes from Understand become unknowns with the proposed value kept; at the end each remaining unknown is a `# UNVERIFIED:` comment in the test header and the test is flagged `unknown_expected` | D2 |
| B12 | Soft mode: one agent, all tools, same instructions in one prompt, `run_test`/`submit_test` take the test directory, no gates; same 50 turns | Design §1, D1 |

**Checks.** Offline: `procedure_smoke.py` (gates, routing, skips, send-backs, D2, budget,
soft) and `procedure_preflight_smoke.py`, 0 failures. Live, scripted client (no model):
sympy-20212 strict and soft, django-15498 strict, all L4 under the offline label through the
official runner.

## v1 — after development round 1 (2026-11-15)

Round 1 (strict v0, 10 bugs, 5 L4; review in `results/dev_round1_strict.md`) showed the turns
going to reading instead of running code, Pin running on test-design questions, and preflight
output quoted as issue text. Changes C1–C8 from that review, all applied:

| # | Change | Where | Targets (round-1 pattern) |
|---|---|---|---|
| C1 | Understand sees the preflight output as a separate block labelled "not part of the issue". An observed claim names its `source` (`issue` or `preflight_run`) and its quote is checked against that text; expected claims must still quote the issue. At most 3 unknowns, only missing expected values (`finish_understand` refuses more) | `procedure_prompts.UNDERSTAND`, `render_preflight`; `procedure_tools.finish_understand`; `Procedure.p_understand` | 3, 4 |
| C2 | Pin skip rule as in design §3.4: skipped when ≥ 1 expected claim is verified and none is unverified. Unknowns the model listed besides are kept as `open_questions` in the record, not as `# UNVERIFIED:` values. An assertion without evidence still reaches Pin through the Write gate | `Procedure._after_reproduce`; `Unknown.origin` | 3 |
| C3 | Read budget in Reproduce and Write: after 4 `read_file`/`grep`/`glob` calls in a row without running code, the read tools refuse ("run your script/test now") until `run_python`/`run_test` runs. Refusals counted per phase in `read_refusals` | `procedure_tools.ReadBudget`, `budgeted` | 1, 2 |
| C4 | Write tools: `read_file`, `run_test`, `finish_write` (grep/glob dropped). Supersedes B5 | `Procedure.p_write` | 2 |
| C5 | `finish_write` checks the assertion map itself (every assertion statement mapped, every id exists) and refuses with the missing lines, so the model fixes it in the same visit; the Write gate's own check stays as a backstop | `procedure_tools.assertion_map_problem`, `finish_write(ids)` | 5 |
| C6 | Django hint in the Reproduce prompt: run_python already runs under the test runner; no `settings.configure()`/`django.setup()`; models via `module` | `procedure_prompts.DJANGO_RUN_HINT` | 1 |
| C7 | The preflight summary leaves out inline expressions whose only output was a NameError (prose such as `f(x)`); an inline run with nothing left is dropped. Fenced blocks unchanged; preflight's own record unchanged | `procedure_prompts.shown_snippets` | 4 |
| C8 | `max_tokens_per_turn` 8192 for every agent: `Procedure` default and the `procedure_brt.py` default, which the baseline run also takes (baseline code unchanged; it is a run parameter) | `procedure.py`, `scripts/procedure_brt.py` | 6 |

**Decisions taken while applying them**

| # | What | Why |
|---|---|---|
| B13 | The Django hint also goes into Pin's prompt | Pin uses run_python the same way; same mistake possible |
| B14 | Soft mode's single prompt gets the same wording (preflight quotes, ≤ 3 unknowns, run code early, Django hint), but no read budget and no `finish_*` checks | B12: soft = same instructions, nothing enforced |
| B15 | The read budget is not applied in Pin or Locate | Round 1 showed no over-reading there; Locate's job is reading |
| B16 | An observed quote is accepted from whichever text contains it, recording that one as its source | Both checks are deterministic; rejecting a correct quote for a wrong label would only cost a visit |
| B17 | Reproduce's preflight skip matches issue-sourced observed claims first; if only a preflight-sourced claim matches, the skip is taken and flagged `symptom_from_preflight_claim` | A claim quoted from preflight output matches that output by construction; the flag keeps those cases countable |

**Checks.** `procedure_smoke.py` gains `test_v1_*` (C1–C8; the send-back test now expects C5's
in-phase refusal instead of an `unmapped_assertion` send-back); all offline smoke suites pass.
No model run yet with v1.
