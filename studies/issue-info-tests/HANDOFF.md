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

## Next steps
1. Implement the preflight (environment, official SWE-bench test command, issue snippet runs,
   location candidates) with offline smoke tests.
2. Implement the controller and phases (strict), then the soft variant.
3. Development round 1: 10 random development bugs, procedure agent and baseline, review.
4. Repeat rounds until the stopping rule in design §10c.
