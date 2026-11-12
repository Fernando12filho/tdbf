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
