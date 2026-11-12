# Handoff: where this study stands and how to continue (2026-11-12)

Read this first in a new session.

## Documents
- `plan.md`: the study protocol (draft). Amendment 2 has the plan: Phase 1 develop the improved
  agent on development bugs, Phase 2 main runs on the 114 bugs, Phase 3 comparisons.
- `design/procedure_agent.md`: the design of the improved agent (the "procedure agent"), with the
  decisions taken on 2026-11-12 in §10. **This is what to build.**
- `results/agreement_v3.txt`: the labelling prompt v3 agrees with human labels at human level.

## Branches
| Branch | What it is | Use it for |
|---|---|---|
| `main` | Older stable code; behind | Nothing new |
| `openrouter-paid` | Complete stored results up to 2026-11-05 | Reference only |
| `study/issue-info-tests` | Everything above plus run provenance, labels for all 114 bugs, protocol, design | Study documents and analyses |
| `agent/procedure` | Starts identical to `study/issue-info-tests` | **Build the procedure agent here.** Merge back into the study branch when frozen (tag `procedure-v1`) |

Rules for the build:
- The existing agent (`run_brt_phase` in `src/tdbf/agent/pipeline.py`) must stay runnable and
  unchanged: it is the baseline. The procedure agent is new code next to it
  (`src/tdbf/agent/procedure.py`, `preflight.py`), selected by a flag.
- Register every new run group in `src/tdbf/studies.py` before running it.
- Development bugs come from SWE-bench Lite **outside** `sample_tests_swt114_frozen.json`.
- Downloads from the internet are done by the user; ask for a drop folder under `external/`.
- Ask before any paid run. Commit messages: lowercase `area:` prefix, no co-author trailer.

## Model
`openai/gpt-5-mini` for every agent (decision D7 in the design). Checked 2026-11-12 with one
two-turn tool call ($0.0002): the client works unchanged. Spec string
`openrouter:openai/gpt-5-mini@openai` (paid ids need a pinned provider); no temperature is sent;
reasoning tokens are inside `completion_tokens` and `billed_usd` records the real cost.
`max_tokens=4096` includes reasoning tokens: log `finish_reason=length` in the new agent.

## Done (2026-11-12)
- Preflight built, no model call: `src/tdbf/agent/preflight.py` and
  `src/tdbf/harness/official.py` (official SWE-bench command per repo/version, with
  `eval_commands`; astropy `--tb=no` -> `--tb=short`; pytest fallback). Offline tests:
  `scripts/procedure_preflight_smoke.py`. Live runner: `scripts/procedure_preflight_live.py`.
  Baseline files untouched; baseline smoke tests pass.
- Development round 1 drawn (`scripts/draw_dev_round.py`, seed 20261112, pool = SWT-bench
  Lite minus the 114 = 162 bugs) -> `data/dev/round1.json`; images pulled (amd64).
- Preflight live on round 1 plus a sphinx and an astropy runner check: all 12 valid with the
  official command (`results/preflight_dev_round1.md`). A live mechanics check also ran on
  3 of the 114 (no model, no agent decisions; results not kept in the study).
- Images: `docker pull --platform linux/amd64 swebench/sweb.eval.x86_64.<id with __ -> _1776_>`
  then tag as `sweb.eval.x86_64.<id>:latest` (what ContainerSession looks for).

- Controller and phases built (strict and soft): `src/tdbf/agent/procedure*.py`; build
  decisions B1–B12 in `design/changelog.md`. Batch runner `scripts/procedure_brt.py
  --agent strict|soft|baseline` (cost cap `--max-usd`, resume by run group, offline label).
  Run groups `proc-dev-r1-{strict,soft,baseline}` registered (study `procedure-dev`).
  Offline smoke tests pass; live scripted check (no model) gives L4 on sympy-20212 and
  django-15498. **No model run yet.**

## Next steps
1. ~~Preflight~~ (done, see above).
2. ~~Controller and phases, strict and soft~~ (done, see above).
3. Development round 1: 10 random development bugs, procedure agent and baseline, review.
4. Repeat rounds until the stopping rule in design §10c.
