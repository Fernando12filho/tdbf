# Reactive blackboard: comparison on 50 bugs (2026-11-25)

The comparison of `plan_reactive_blackboard.md` (rule fixed before the run). The blackboard with an
automatic information flow: new entries pushed into every agent's tool results, the writer's
check_test results auto-posted, judge rejections routed to the Issue Reproducer and crashes to the
Template Generator and Edit Localizer, a 90 s soft start. gpt-5-nano scouts, gpt-5-mini writer and
judge. 50/50 bugs, no crashes after the run was resumed (OpenRouter credits ran out after 13 fresh
bugs; a new key was set and the same configuration resumed; no record was lost or repeated). $2.99.

## Results

| arm | fresh L4 / L3 | dev L4 / L3 | all 50 L4 | $ per valid (fresh / dev) | vs old agent (fresh; dev) |
|---|---|---|---|---|---|
| old agent | 17 / 3 | 21 / 9 | **38** | $0.026 / $0.035 | |
| writer loop + judge v2 | 16 / 4 | — | — | 1.4× / — | −1 |
| scouts v2 | 16 / 4 | — | — | 2.6× / — | +2/−3 |
| scouts v1 | — | 19 / 11 | — | — / 4.5× | +1/−3 |
| blackboard | 15 / 5 | 23 / 6 | **38** | 2.4× / 2.2× | +1/−3; +5/−3 |
| **reactive blackboard** | **18 / 2** | 20 / 9 | **38** | 2.2× / 2.8× | +2/−1 (p=1.0); +3/−4 (p=1.0) |

- Fresh: gains django-14997, sphinx-10451; loss sklearn-25570. More L4 and fewer L3 than the old agent
  (the first design to do both on the clean set), but 2.2× per valid test → **not adopted**.
- Dev: gains sklearn-13497, sympy-18199, django-11905; losses django-14855, django-15695, sympy-18087,
  sphinx-8474. Fewer L4, 2.8× → **not adopted**.
- Over all 50 bugs the old agent, the blackboard and the reactive blackboard each get **38** valid tests.

## The reactive flow

| | fresh | dev |
|---|---|---|
| board entries per bug | 9.2 | 9.5 |
| times the writer receives board content, per bug | 18.1 | 24.9 |
| questions routed to scouts (answered) | 26 (10) | 31 (5) |
| judge checks / mismatches / bugs with a mismatch (ended 'match') | 24 / 2 / 2 (2) | 49 / 18 / 13 (11) |

Soft start: mean 65 s; it waited the full 90 s on 35 of 50 bugs. Scouts cost $1.52 of $2.99 (51%);
4.3 minutes per bug. Scouts answered only 15 of 57 routed questions (26%).

## Why it helped on the fresh bugs and hurt on the development bugs

The development bugs are harder (old agent 70% vs 85%; 3 bugs no design ever solved vs 1). Reading
the boards of two losses:

- **django-14855:** the Template Generator posted root-cause claims outside its role, and the Issue
  Reproducer posted a wrong reproduction from a dummy setup ("returns a plain string `namespace(pk=7)`
  instead of a hyperlink"); pushed into the writer's results, they shaped its test.
- **sympy-18087:** the Reproducer's "expected behaviour" entry only restated the bug; the writer then
  asserted an exact form nobody stated (`res == cos(x) + Abs(sin(x))`), and judge v2 accepted it.

Pushing the board makes the writer depend on what the scouts post. With gpt-5-nano scouts that is
mostly right on the easier fresh bugs (18 vs 17) and often confidently wrong on the harder development
bugs (20 vs 21; the plain blackboard, whose writer mostly ignored the board, got 23). This is MAST's
inter-agent misalignment: a weak agent's error propagates to the strong one.

## Reading

- Fifth Phase-A design not adopted. On 50 bugs the old single agent, the blackboard and the reactive
  blackboard all write 38 valid tests; the multi-agent designs cost 2–3× per valid test.
- The reactive flow does what it was built for (the board becomes central), and its effect follows the
  scouts' quality. Two open questions for any next design: stronger scouts (gpt-5-mini, at cost), and
  pushing only findings backed by an observed run.
