# Blackboard Phase A: comparison on 50 bugs (2026-11-25)

The comparison of `plan_blackboard.md` (rule fixed before the run). gpt-5-nano scouts and the
gpt-5-mini in-loop-judge writer work concurrently around a structured board; no organizer. Compared
with stored results only. 50/50 bugs, no crashes, $2.74 in total, ~3.9 minutes per bug.

## 20 fresh bugs (primary: nothing was designed on them)

| arm | L4 | L3 | cost | $ per valid | vs old agent | McNemar p |
|---|---|---|---|---|---|---|
| old agent | **17** | 3 | $0.44 | **$0.026** | | |
| writer loop + judge v2 | 16 | 4 | $0.57 | $0.035 (1.4×) | +0 / −1 | 1.0 |
| scouts v2 | 16 | 4 | $1.06 | $0.066 (2.6×) | +2 / −3 | 1.0 |
| **blackboard** | 15 | 5 | $0.95 | $0.063 (2.4×) | +1 (django-14997) / −3 (sympy-17630, django-14752, pylint-5859) | 0.63 |

Rule: fewer L4, more L3, 2.4× → **not adopted**.

## 30 development bugs (secondary)

| arm | L4 | L3 | other | cost | $ per valid | vs old agent | McNemar p |
|---|---|---|---|---|---|---|---|
| old agent | 21 | 9 | 0 | $0.74 | $0.035 | | |
| old agent + judge v1 | 21 | 8 | 1 | $0.84 | $0.040 (1.1×) | +3 / −3 | 1.0 |
| scouts v1 | 19 | 11 | 0 | $3.01 | $0.158 (4.5×) | +1 / −3 | 0.63 |
| **blackboard** | **23** | 6 | 1 | $1.83 | $0.079 (2.2×) | +5 / −3 | 0.73 |

Gains: sklearn-13497, sympy-19254, django-11630, sympy-18199, django-11905. Losses: requests-3362,
sphinx-8474, django-11039. Rule: more L4 and no more L3 pass; 2.2× cost per valid test fails →
**not adopted**.

**Caveat:** three of the five gains (sympy-19254, django-11630, django-11905) are cases of the match
arm's L3 analysis (`gate2_l3_match_arm.md`) from which judge v2's rules were derived. The writer loop
with judge v2 was never run on these 30, so the gain cannot be split between the board and judge v2,
and on the clean fresh bugs the blackboard lost ground. Read it as: a likely effect of judge v2 on
the bugs it was derived from, not evidence for the board.

## Board use (as designed: findings pulled by the agents)

| | fresh | dev |
|---|---|---|
| entries per bug | 5.1 | 4.7 |
| by author (per bug) | localizer 2.1, reproducer 1.8, writer 0.8, template 0.5 | localizer 2.0, reproducer 1.6, template 0.8, writer 0.4 |
| writer reads per bug | 2.2 | 2.2 |
| questions (answered) | 2 (1) | 2 (2) |
| judge checks / mismatches / bugs with a mismatch (ended 'match') | 36 / 13 / 7 (7) | 47 / 12 / 9 (8) |

Light use: the writer asked 4 questions in 50 bugs and read the board about twice per bug. Scouts
cost $1.25 of $2.74 (46%); per bug the full blackboard costs $0.055, against $0.025–0.035 for the old
agent. Scout stops: 143 of 150 submitted, 4 forced, 2 at the turn cap, 1 still running at the join.

## Reading

- On clean bugs the blackboard did not beat the old agent (15 vs 17), like every earlier design.
- The board was under-used: the information flow depended on the writer choosing to read or ask.
  The reactive blackboard (`plan_reactive_blackboard.md`) makes that flow automatic.
