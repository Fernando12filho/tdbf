# Study protocol: what information does an agent need to write a correct reproduction test?

**Status: DRAFT, 2026-11-09. Not frozen.** Nothing in this plan has been run. The protocol
is frozen by committing the agreed version and tagging it `issue-info-tests-protocol`.
No new result may be produced before that tag. The open decisions are in Section 10.

## Amendment 2 (2026-11-11, end of day; supersedes the order of work below)

**Done.** Every one of the 114 bugs has information-type labels (prompt v3, GPT-5-mini).
On the 36 bugs with human labels from the Bug Reports paper, v3 agrees with the humans at
least as well as the humans agree with each other (expected behaviour kappa 0.76 vs 0.67,
location 0.78 vs 0.43; `results/agreement_v3.txt`). Still open: a person spot-checks about
15 of the 78 LLM-only labels.

**Framing.** We try to improve the test agent. The same runs measure what the agent
depends on (removal) and whether the improvement makes it more robust when information is
missing. The interaction is the link between the two paths.

**Phase 1. Develop the improved agent on development bugs, never on the 114.**
- Development bugs: about 30 SWE-bench Lite bugs outside the 114, drawn with a fixed seed.
  Our runner labels any test there (L0–L4), since the real fix is available offline.
- Candidates, from our own data and the papers read on 2026-11-10:
  1. Grounding step before the test: the agent investigates and states the expected value
     with its source (issue, documentation, existing test, observed), or marks it unknown.
     Targets wrong-reason tests (L3). DPIAgent found a pasted summary hurt, so the agent
     must build and check it itself.
  2. Retry feedback: replace "it passed, make it fail" with diagnostic feedback and carry
     the findings into the retry. In stored gpt-4o-mini runs, 17 of 20 tests that went from
     passing to failing after feedback ended L3.
  3. Step cap 30 → 60, optionally with a budget reminder. Runs that hit the cap rarely
     produced a valid test.
  4. A prompt rule: assert only expected values that have a source.
- One change at a time on the development bugs; keep what helps; freeze the improved agent
  with a git tag before Phase 2. Development results are reported as development.

**Phase 2. Main runs on the 114 bugs: nemotron-3-super (paid), 5 repeats per condition.**

| Condition | Agent | Issue text | Bugs |
|---|---|---|---|
| C1 | unchanged (baseline) | original | 114 |
| C2 | improved | original | 114 |
| C3 | unchanged | expected behaviour removed | bugs that contain it |
| C4 | improved | expected behaviour removed | bugs that contain it |
| C5 | unchanged | location removed | bugs that contain it |
| C6 | improved | location removed | bugs that contain it |

C1 is step C. C3–C6 need the advisor's approval. gpt-4o-mini is optional, later.
Before C1: the batch runner reuses each bug's Docker image across all repeats and
conditions instead of downloading it per pass.
Estimate (list prices, past token use): about $15 for C1, about $70 for C1–C6, and roughly
100 hours of machine time for C1–C6 at the last measured 2 minutes per bug-run.

**Phase 3. Comparisons, fixed before the runs.**
- Outcome per bug and condition: share of the 5 runs that produce a valid test (L4); also
  the L3 share, SWT-bench success, cost and steps.
- Noise (C1): mean and standard deviation of the valid-test rate across the 5 repeats.
- Improvement: C2 vs C1, paired by bug.
- Information: C3 vs C1 and C5 vs C1, paired by bug, on the bugs that contain that type.
- Robustness: does the improvement help more when information is missing
  ((C4 − C3) vs (C2 − C1))? Mixed logistic model
  `L4 ~ agent * removal + (1 | bug)`, effects in percentage points with 95% intervals
  from resampling bugs, Holm correction within each family, smallest effect of interest
  10 points.
- Repositories. Django is 53 of the 114 bugs; most other repositories have under 12.
  (a) every model includes the repository (fixed effect, with Django vs the rest as the
  pre-registered split); (b) each comparison is also reported for Django and for the rest
  separately; (c) per-repository numbers are descriptive only, with no tests, because the
  samples are too small; (d) patterns across repositories are described for the labels
  (which information each repository's issues tend to contain).
- Link to fixing (stored data, exploratory): wrong tests and the reactive fixer's failures.

## Amendment 1 (2026-11-11, before any new run)

The study is now framed as part of improving the test agent: the same runs serve as an
ablation of the agent's input (which information it depends on) and as the information
study. The steps replace the order of work in Section 13 as follows.

| Step | What | Needs | Kind |
|---|---|---|---|
| A | Human labels for the 37 of our 114 bugs that are in SWE-bench Verified, from the Bug Reports paper's replication package (Zenodo, CC-BY-4.0), downloaded by hand into `external/` | Nothing | Measurement material |
| B | An LLM splits the issue text of those 37 bugs into the nine information types without rewriting it; agreement with the human labels is reported (Cohen's kappa per type and character overlap), alongside the paper's own human–human agreement (kappa 0.57, 78.6% of characters) | A few cents | **Measurement validation, not a result about agents.** Decides whether LLM labels can be used for the other 77 bugs |
| C | RQ0: 5 repeats of the unchanged test agent on the 114 bugs, original issues | About $26; user approval | Confirmatory for RQ0 only |
| D | RQ2: removal conditions | Advisor approval | Confirmatory |
| E | RQ3: a grounding step before test writing (the candidate improved agent) | Advisor approval | Confirmatory |

Facts this amendment relies on: the 37 bugs have identical issue text and base commit in
SWE-bench Lite and Verified (checked 2026-11-11). It is not yet confirmed that all 37 are
among the 441 issues the Bug Reports paper kept as bug reports. Every run record now stores
the code commit and whether the checkout had uncommitted changes (`code` key, from
`tdbf.results.code_provenance`).

## 1. Purpose

Agentic test generators accept a test because it fails on the buggy code. That check
cannot tell a right test from a wrong one. Prior work has measured how often tests are
wrong and what that does to repair (CoHarden), and which issue information helps
*fixing* (Writing Bug Reports for Repair Agents, TrajSpec). It has not measured which
issue information an agent needs to write a *correct test*.

Our hypothesis is that tests and fixes need different information. A fixer can succeed
knowing where the bug is. A test has to state the correct value. We do not claim a
better agent: the agent is held fixed and used as the measuring instrument.

## 2. Terms

| Term | Meaning |
|---|---|
| Bug / instance | One SWE-bench Lite issue with its repository state and real fix |
| Run | One execution of the test agent on one bug under one condition |
| Condition | The version of the issue text given to the agent |
| Repeat | Another run with everything identical; LLM output varies between runs |
| L4 (valid) | Fails on the buggy code at an assertion and passes after the real fix |
| L3 (wrong reason) | Fails on the buggy code and does not pass after the real fix |
| L0, L1, L2 | Never ran; passed on the buggy code; crashed on the buggy code |
| L4 rate | L4 runs / all runs |
| Precision of the check (C2P) | L4 / (L3 + L4) |
| SD | Standard deviation, √(Σ(x − x̄)² / (n − 1)) |
| SESOI | Smallest effect of interest, fixed in advance: **10 percentage points** |

## 3. Questions and hypotheses

Labelled before any result. *Confirmatory* results may be claimed; *exploratory* results
are reported as such and only generate hypotheses.

**RQ0 (noise, descriptive).** How much does the L4 rate vary between repeats of the same
bugs? Reported as the SD across repeats, per model.

**RQ1 (confirmatory).** Which issue information predicts a valid test?
- H1: issues that explicitly state the expected behaviour have higher odds of L4 than
  issues that do not, controlling for the other information types.

**RQ2 (confirmatory).** What happens when that information is removed?
- H2a: removing expected behaviour lowers the L4 rate by at least the SESOI.
- H2b: removing expected behaviour lowers it more than removing location does. This is
  the contrast with the fixing result, where location mattered more.

**RQ3 (confirmatory).** Does an investigated report help, or does it turn guesses into facts?
- H3a: the investigated report raises the L4 rate by at least the SESOI over the original issue.
- H3b: it does not raise the L3 share by more than 5 points (non-inferiority margin).
- Exploratory: how often the report's expected behaviour is right, by the source it cites.

**RQ4 (exploratory, no new runs).** How does test validity relate to the fix agent
stopping by itself? We already looked at this data (valid: 67% vs 43%; wrong: 31% vs 39%),
so it cannot be confirmatory here.

## 4. Materials

**Bugs.** `sample_tests_swt114_frozen.json`: 114 bugs, `ids_sha256` 39c25380…bc33f,
the paired RQ1 bugs that are in SWT-bench Lite. Django is 53 of the 114.

**Development bugs, kept out of all analysis.**
- Model pilot and report-prompt development: the 14 bugs in `excluded_development_set`
  of `sample_lite_frozen.json`.
- Labelling guide development: 20 bugs drawn from SWE-bench Lite outside the 114 and
  outside those 14, `random.Random(20261109)` over the sorted ids.

**Agent.** The test agent (arm T) at the tagged commit: 30 iterations per attempt, up to
3 attempts, unchanged prompts and tools. Only the issue text changes, through
`Instance.problem_statement`. The agent never sees the real fix or the hidden tests.

**Models** (Decision D4).
- Main: `nvidia/nemotron-3-super-120b-a12b`, paid. It has room to go down (76/114 L4).
- `openai/gpt-4o-mini-2024-07-18`, to compare with stored runs and Issue2Test. Its L4
  rate is 14/114, so RQ2 has a floor effect on it and it is reported as secondary there.
- Temperature and seed are left at provider defaults, and recorded as defaults, along
  with the serving provider.

## 5. Procedure

**Step 1: labelling (RQ1).** Two people independently label each issue using the nine
types from Writing Bug Reports for Repair Agents, marked as character spans: summary,
observed behaviour, expected behaviour, reproduction steps, environment, reference,
affected area, suggested fix, workaround.
- Expected behaviour is also coded E0 (absent), E1 (implied, no value or rule), E2
  (explicit correct value or rule).
- Raters see only the issue text. They do not see test levels, models or outcomes.
- The guide is refined only on the 20 development bugs. Target before labelling the
  114: Cohen's kappa ≥ 0.70 per type. Report kappa and raw agreement, and resolve
  disagreements by discussion.

**Step 2: noise baseline (RQ0).** 5 repeats per model on the 114 bugs with the original
issue. These runs are also the "original issue" condition for RQ2 and RQ3.

**Step 3: checkpoint.** Run the RQ1 analysis on stored tests (nemotron 114, gpt-4o-mini
114). If the E2 vs E0/E1 difference is under 5 points and its interval includes 0, stop
and amend RQ2 and RQ3 in writing before spending on them.

**Step 4: removal experiment (RQ2).** 5 repeats per model under each condition. Each
removal runs on the bugs that contain that category.
- Remove expected behaviour (including assertions or example output that state it).
- Remove reproduction steps.
- Remove location (affected area).

Edited issues are built from the labelled spans. A second person checks that each one
reads coherently and no longer contains the removed information.

**Step 5: report experiment (RQ3).** 5 repeats per model under each condition.
- Plain rewrite: one LLM rewrite of the issue, with no repository access.
- Investigated report: an agent searches and runs code on the buggy revision only, then
  writes observed behaviour, expected behaviour and steps. Each expected-behaviour
  statement carries its source: quoted from the issue, documentation or existing test,
  observed by running code, or marked unknown. Report prompts are developed only on the
  14 development bugs.

Offline, two people with access to the real fix judge each expected-behaviour statement
as correct, incorrect or not determinable. No agent ever sees this.

**Step 6: scoring.** Every test gets its L0–L4 label from our runner and a SWT-bench verdict.

## 6. Outcomes

- **Primary:** L4 rate.
- **Secondary:** L3 share; precision (C2P); SWT-bench success; coverage delta; cost and
  iterations per bug. Disagreements between L4 and SWT-bench success are listed.

## 7. Analysis

Fixed before any run. The analysis lives in `analysis.py` in this folder, and every number
in the paper comes from it.

- **RQ0:** mean ± SD of the L4 rate across repeats, plus how many bugs change level.
- **RQ1:** mixed logistic regression, one row per stored test:
  `L4 ~ E2 + repro_steps + observed + location + suggested_fix + model + django + (1 | bug)`.
  Odds ratios with 95% intervals. Secondary outcome: L4 among L3+L4. Sensitivity: E as
  ordinal, and without Django.
- **RQ2, RQ3:** mixed logistic regression `L4 ~ condition * model + (1 | bug)` for the
  test, and the difference in L4 rate in points with a 95% bootstrap interval that
  resamples bugs (10,000 resamples, seed 20261109). If the model does not converge, use
  the per-bug majority across repeats with Cochran's Q and exact McNemar.
- **Claiming an effect:** the Holm-corrected p is below 0.05 *and* the interval is
  compatible with the SESOI. An interval entirely inside ±10 points is reported as no
  meaningful effect, which is a result.
- **Multiple tests:** Holm correction within each RQ. Family RQ2 = H2a, reproduction
  steps removal vs original, location removal vs original, H2b. Family RQ3 = H3a, H3b,
  plain rewrite vs original.
- **Results are reported per model.** Nothing is pooled without showing each model.
- **Other systems:** compared only on the same bugs, and differences are attributed to
  the agent only when the model matches.

The minimum detectable effect will be recomputed from the RQ0 discordance before Step 4.

## 8. Rules

- **Excluded run:** infrastructure failure only (provider error, Docker failure, scoring
  error). It is retried up to 3 times with the same configuration, and all exclusions
  are listed. A poor or empty test is an outcome, never an exclusion.
- **Stopping:** all planned runs are completed unless the budget cap is reached; no
  stopping because results look good or bad.
- **Deviations:** any change after the tag is logged in `deviations.md` with date, reason
  and whether it was made before or after seeing results.
- **Gold-free:** the real fix and hidden tests are used only for offline labels.

## 9. Threats to validity

- **Construct:** L4 depends on the real fix and hidden tests, and there can be other
  correct behaviours. SWT-bench success is reported beside it. Span labels are human
  judgments, hence two raters.
- **Internal:** removing text may also remove context; a coherence check limits this.
  Stored tests in RQ1 were generated before labelling, but raters are blind to them.
  Free and paid nemotron are different endpoints, so stored results and new repeats
  are compared and any drift is reported.
- **External:** Python only, Django-heavy, one main agent (D6), two models.
- **Data leakage:** the issues predate these models, which may have seen the fixes.
  Removal experiments partly address this; a leaked answer would make removal look harmless.
- **Reproducibility:** provider models can be withdrawn. Prompts, transcripts, commit
  hashes and results are archived.

## 10. Open decisions

| # | Decision | Proposal |
|---|---|---|
| D1 | Advisors accept test creation as the main direction | Needed before freezing |
| D2 | What the advisor meant by "standard deviation" | Assumed: SD across repeats (RQ0) |
| D3 | Who is the second labeller | A person, not an LLM |
| D4 | Final models | The two above; optionally one more open model, such as Qwen3 Coder 30B |
| D5 | Budget cap | $200 (Section 12) |
| D6 | Run RQ2 on a second agent (mini-SWE-agent) | Do it if budget and time allow; otherwise report as a limitation |
| D7 | Target venue | Affects length and artifact rules |

## 11. Reproducibility and repository

- Branch `study/issue-info-tests`; protocol tag before any run; one registered study in
  `src/tdbf/studies.py`, with run groups `iit-baseline`, `iit-removal`, `iit-report`.
- Required before Step 2: every run record stores the code commit and a prompt hash;
  dependencies are pinned; unit tests cover labels and analysis.
- Folder contents: `plan.md`, `deviations.md`, `labelling_guide.md`, `sample.json`,
  `config.json`, `analysis.py`, `results/`.

## 12. Budget estimate

Per run per bug: $0.026 nemotron (list price and stored token use) and $0.020 gpt-4o-mini
(measured). Together $0.046. One bug-condition with 5 repeats and both models is $0.23.

| Step | Conditions | Cost |
|---|---|---|
| Pilot on 14 development bugs | Original issue | about $3 |
| RQ0 baseline | 1 | $26 |
| RQ2, upper bound (all 114 eligible) | 3 | $79 |
| RQ3, including one report per bug and model | 2 | $58 |
| **Total** | | **about $170** |

Proposed cap: $200, about 20% above the estimate. RQ1 and RQ4 use stored runs and cost
nothing. These are estimates from list prices and past token use.

## 13. Order of work

1. Settle D1–D7 and freeze this protocol with the tag.
2. Infrastructure from Section 11.
3. Labelling guide on the 20 development bugs, then label the 114.
4. Pilot, then RQ0.
5. RQ1 analysis and the checkpoint.
6. RQ2, then RQ3, then RQ4.
7. Paper numbers from `analysis.py` only, and the archive.

## 14. References

Method and reporting
- Baltes, Wagner et al. Guidelines for Empirical Studies in SE involving LLMs. arXiv:2508.15503
- Sallou, Durieux, Panichella. Breaking the Silence. ICSE-NIER 2024. arXiv:2312.08055
- Arcuri, Briand. A Hitchhiker's Guide to Statistical Tests. STVR 24(3), 2014. doi:10.1002/stvr.1486
- Ralph et al. Empirical Standards for SE Research. arXiv:2010.03525
- Dietterich. Approximate Statistical Tests. Neural Computation 10(7), 1998
- Agarwal et al. Deep RL at the Edge of the Statistical Precipice. NeurIPS 2021. arXiv:2108.13264
- Wilson et al. Good Enough Practices in Scientific Computing. arXiv:1609.00037

Closest work
- CoHarden (Beyond Fail-to-Pass). arXiv:2607.19843
- Writing Bug Reports for Software Repair Agents. arXiv:2607.09553
- TrajSpec. arXiv:2607.07882
- Validation Evidence in LLM Repair Agents. arXiv:2607.28871
- SWE-Doctor. arXiv:2607.00990
- Echo. arXiv:2603.07326
- e-Otter++. arXiv:2508.06365
- AssertFlip. arXiv:2507.17542
- Issue2Test. arXiv:2503.16320
- SWT-Bench. arXiv:2406.12952
