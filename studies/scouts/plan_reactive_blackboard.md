# Reactive blackboard: plan (2026-11-25, before any reactive run)

The blackboard run (`plan_blackboard.md`) showed light board use: about 5 entries per bug, the writer
reading the board about twice per bug, 1 question in 12 bugs, and every exchange left to the model's
choice. User decision (2026-11-25): make the board central. The changes make the information flow
**automatic** rather than adding rules that block the writer, because enforced steps with gates hurt
before (strict procedure agent 15 vs 21 L4) and MAST warns that forced communication adds overhead.

## Changes from the blackboard design

1. **Push content, not counts.** New entries by other agents are inserted into each agent's next
   non-terminal tool result (the entries themselves, capped in length), and marked read. The writer
   cannot skip a finding.
2. **Auto-posted progress.** Every writer `check_test` posts its outcome, the failure line and the
   judge's verdict with its reason to `test_status`, written by the system, not left to the model.
3. **Problems routed to scouts.** When check_test ends in a judge *mismatch*, the system posts a
   question to the Issue Reproducer (re-run the issue's example and confirm the correct expected
   value, quoting the issue). When the test crashes before asserting, it posts a question to the
   Template Generator and the Edit Localizer (the right setup and the right call). Waiting scouts wake
   up on it.
4. **Soft start.** The writer's first check_test waits up to 90 s until the Reproducer has posted
   both observed and expected behaviour; after that it proceeds regardless. No later waits.

Unchanged: models (gpt-5-nano scouts, gpt-5-mini writer and judge), budgets, the in-loop judge v2,
guarded submit, gate 1, retries, the offline label, all gold-free rules. Not added: sourced-assertion
gate on submit_test (a hard rule; kept for a separate ablation if wanted).

## Validation

- **Bugs:** the same 50 as the blackboard run: 20 fresh (primary; old agent 17/20) and 30
  development (secondary; judge v2 rules were derived from their L3 cases).
- **Compared with stored results only:** the old agent, the blackboard (`bb-fresh`, `bb-dev30`) and
  every earlier arm on the same bugs. Reactive vs blackboard isolates the four changes.
- **Rule (unchanged):** adopted only with more L4 than the old agent on the same bugs, no more L3,
  and cost per valid test ≤ 1.5× the old agent.
- **Also reported:** entries per bug and by author, routed questions and how many scouts answered,
  writer reads, how long the soft start waited, judge checks and mismatches, cost per role.
- **Cost:** about the blackboard's (~$0.04–0.05 per bug), plus scouts answering routed questions;
  ~$3 for 50 bugs, capped at $5.
