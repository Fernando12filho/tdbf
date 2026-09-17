# TDBF — plan to submission (revised 2026-09-16 → 2026-09-27)

Working plan for finishing the thesis. Update the checkboxes as things land;
if a decision changes, change it *here* first so the chapters follow one source.

**Deadline: Sunday 27 September 2026.** Data freeze **Thursday 24 September**.

> **Status 2026-09-16:** Phase 1 (originally due Sat 12) has not started in code —
> about four days behind. `main` is pushed and in sync with GitHub. The plan below
> is re-cut around the **no-budget pivot**; the freeze date holds.

---

## The no-budget pivot (2026-09-16)

There is no more paid API budget. Every new run uses **free OpenRouter models** or
**local Ollama**. What that changes:

- **RQ1 now runs on a free model**, not Sonnet. Whether any free model is capable
  enough is unknown, so a capability probe gates the sweep (task 2.1). If both arms
  would score ~0, the comparison says nothing — rescope before sweeping.
- **RQ1 and RQ3 share runs.** Every free model is also an open-weight data point.
- **RQ2 is answered from stored data** (Critic, Breaker, best-of-N on Sonnet, plus the
  gate A/B). No new paid calls.
- **The Sonnet n=14 run is the frontier reference point**, not the headline:
  gate 14/14, resolved 10/14, C2P 13/14 (non-random development set). The July run of
  the same instances resolved 9/14. *(Corrected 09-16: the earlier "identical per
  instance" claim was a scoring artefact — see the 09-16 scoring fix.)*
- **Fix outcomes are not fully deterministic**: 1 of 15 repeat (instance, model) pairs
  disagrees. One run per arm per instance carries noise — another reason RQ1 is
  reported with a bootstrap CI.

## Decisions

| Date | Decision | Choice |
|---|---|---|
| 09-10 | Deadline | Sat 26 – Sun 27 Sep; no new runs after Thu 24 |
| 09-10 | RQ3 framing | **Open-weight models generally** — hosted (OpenRouter) for variety, local Ollama as the on-device configuration. "Code never leaves the machine" must come out of the RQ3 paragraph. |
| 09-10 | Thesis prose | Claude drafts ch3 and ch4; Fernando writes ch1, ch2, ch6, ch7 and owns the argument in ch5 |
| 09-10 | BRT-properties analysis | Reinstated as analysis (not an RQ) — free, over stored artifacts |
| 09-10 | Sample | Stratified across repos, frozen to a committed file before the sweep |
| 09-16 | Budget | **None.** Free OpenRouter models + local Ollama only |
| 09-16 | OpenRouter limits | Fernando adds a one-time **$10** credit → free models go from 50 to 1,000 requests/day. Free models do not draw the credit down. |
| 09-16 | Local host | The Ollama desktop (192.168.1.192) is available on request — it was simply switched off |

## Research questions (canonical — `thesis/chapters/1_introduction.tex` is the source)

- **RQ1 — ordering.** Does test-first (repair withheld until a test is *observed*
  to fail on the buggy revision) resolve more than a matched reactive arm?
- **RQ2 — orchestration.** Which pipeline characteristics improve fix quality and
  reliability under a fixed budget? Ablations, negatives included.
- **RQ3 — open-weight models.** Can they sustain the pipeline, and at which stage
  does the trajectory break down (submits? reproduces? resolves?)

> The older numbering (RQ1 = weak arm as a scope condition, RQ2 = BRT properties,
> RQ3 = orchestration) is **superseded**. `docs/agent_plan.md` and chapters 4–6
> still carry it in places — fixing that is task 1.1.

---

## Measured constraints (do not re-derive; re-measure only if something changes)

**OpenRouter (checked 2026-09-16)**
- Key is **free tier**. 444 models, 24 free, **20 free with tool calling**. Larger
  candidates: `nvidia/nemotron-3-super-120b-a12b:free`,
  `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-31b-it:free`,
  `cohere/north-mini-code:free`. Exclude `stealth/union-alpha` and the
  `openrouter/free` auto-router — neither can be named or reproduced.
- Free-model limits (OpenRouter docs): **20 requests/min**; **50/day** under $10 of
  purchased credit, **1,000/day** at $10+ (granted from $9).
- **Requests per instance** (Sonnet, gate-fix run): test phase median 9 turns (max 31),
  fix phase median 11 (max 26) → ~20 for test-first, ~11 for reactive, **~30–40 per
  paired instance**; weaker models run closer to the 30-turn cap.
  At 1,000/day that is **~15–30 paired instances per day**.
- **Measured 09-16 on a free model:** one pair = **62 calls, 29 min** (nemotron-3-super,
  an easy instance; the test-first fix phase used all 30 turns). So **~16 pairs/day**,
  quota-bound, ~8 h of running. Fri 18 – Wed 23 fits **~60–80 pairs** shared with RQ3 —
  hence n = 60.

**Infrastructure**
- **Image build ≈ 3.4 min** including gold validation (sympy-11400: 203 s, resolved).
- **Disk is not the binding constraint.** Real image store ~29 GB despite ~153 GB of
  nominal sizes (shared layers); a new repo family costs ~1.8 GB. 63 GB free.
- **SWE-bench Lite:** django 114, sympy 77, matplotlib 23, sklearn 23, pytest 17,
  sphinx 16, astropy 6, requests 6, pylint 6, xarray 5, seaborn 4, flask 3.
  The 37 images built so far are astropy + django + one sympy.

---

## Phase 1 — Infrastructure (Wed 16 Sep)

- [x] **1.0** $10 OpenRouter credit added — `is_free_tier: False`, `total_credits: 10`, usage 0 (checked 09-16)
- [ ] **1.1** Fix RQ numbering in `4_experiments.tex`, `5_results.tex`,
      `6_discussion.tex`; rewrite the RQ3 paragraph in `1_introduction.tex`
- [x] **1.2** OpenRouter client (`openrouter:<model>` in `build_client`), with:
      - retry with exponential back-off on 429 / 5xx (free endpoints throttle often)
      - `:free` models priced at an **explicit $0.00**, never `None`
      - the served model id and provider recorded on every run (free models change
        or vanish; this is a reproducibility record)
      - an offline smoke; one live call to confirm tool calling
      - **Live check 09-16** (one `add` tool call each): nemotron-3-super-120b ✅ 3.5 s,
        gemma-4-31b-it ✅ 1.4 s, north-mini-code ✅ 1.2 s — all *structured* tool calls,
        none text-recovered. **nemotron-3-ultra-550b timed out** (60 s) → dropped from
        the probe for now.
- [x] **1.2b** Scoring fix (09-16): fresh verified run id per scoring + `report_dir`
      for slashed model names. 20 stale stored verdicts re-scored; 1 changed.
- [x] **1.2c** Free reasoning models: truncated turns named and counted; malformed
      `read_file` line args rejected. Budgets: **16k tokens/turn, 1M total**, both arms.
- [x] **1.3** Arm A, the reactive baseline: drop `run_brt` from `_FIX_TOOL_NAMES`,
      strip it from `submit_fix`'s description, no Phase A. Offline smoke asserting the
      arms differ *only* in that. Live check 09-16 on nemotron-3-super: plumbing works
      (20 calls, $0, provenance recorded) — it exposed the budget issues fixed in 1.2c.
- [x] **1.4** Paired sweep driver (`11_paired_sweep.py`): per instance, both arms back to
      back in a seeded random order; a unit is saved only when both arms finished;
      resumable; stops cleanly at the daily cap; waits out upstream shared-pool throttles.
      **Live check 09-16** (astropy-12907, nemotron-3-super free): both arms resolved,
      test L4, verified scoring, 0 throttles, 0 truncated turns — **62 calls, 29 min**.
      The first attempt on gemma-4-31b died on a Google AI Studio shared-pool 429.
- [~] **1.5** Building + gold-checking the frozen sample (`13_build_sample.py`, started
      09-16 evening): per instance, gold patch through the verified scorer + the gate's
      BRT path with a trivial fail/pass test. pytest (a never-built repo) validated first.
- [ ] **1.6** Bring the Ollama desktop up

## Phase 2 — Capability probe and freeze (Thu 17 Sep)

- [ ] **2.1** **Capability probe:** 3 free models (nemotron-3-super-120b, gemma-4-31b-it,
      north-mini-code) × 5 dev-set instances, `11_paired_sweep.py --arms B`. Record C2P,
      gate rate, resolved, calls and turns per phase, **and throttle events** — a model
      whose shared pool keeps throttling cannot carry a multi-day sweep.
- [ ] **2.2** **Pick the RQ1 model** (best resolved, then C2P). If no free model
      resolves anything, stop and rescope RQ1 before sweeping.
- [ ] **2.3** Verify each candidate is actually **open-weight** (licence + published
      weights) before the thesis calls it that
- [x] **2.4** **Sample frozen 09-16** (`c86c899`, pushed): 60 instances, seed 20260916,
      stratified by repo, dev set excluded — django 22, sympy 16, matplotlib 5, sklearn 5,
      pytest 4, sphinx 3, one each of requests/pylint/xarray/seaborn/flask. **No astropy**
      (all six Lite astropy instances are dev set) — disclose. Pre-registered rules are in
      the file. Frozen early is fine: the probe uses the dev set only.
- [ ] **2.5** Claude drafts ch3 Methodology; Fernando reviews

## Phase 3 — Sweeps (Fri 18 – Wed 23 Sep)

- [ ] **3.1** Paired RQ1 sweep on the chosen free model, running continuously
- [ ] **3.2** RQ3: the other free models + local qwen, staged reporting
      (terminal submit? reproduces on buggy? resolves?)
- [ ] **3.3** RQ2 from stored data (Critic, Breaker, best-of-N, gate A/B)
- [ ] **3.4** BRT-properties analysis over stored artifacts
- [ ] **3.5** Figures: pipeline diagram, funnel, L0–L4 distribution, per-model breakdown
- [ ] **3.6** Claude drafts ch4 Experimental Design
- [ ] **3.7** Fernando writes ch2 Related Work, then ch5 Results as numbers land

## Phase 4 — Freeze (Thu 24 Sep)

- [ ] **4.1** **No new runs after tonight.** Snapshot `results.jsonl`; record the
      commit hash, frozen sample file, and every model id + provider used.
- [ ] **4.2** Final metrics, tables, figures regenerated from the frozen data

## Phase 5 — Finish (Fri 25 – Sun 27 Sep)

- [ ] **5.1** ch6 Discussion, ch7 Conclusion, abstract
- [ ] **5.2** ch1 Contributions + Thesis Organization (deleted in the trim — must come
      back); rewrite the Thesis Statement section
- [ ] **5.3** Threats to validity — non-random development set, model churn on free
      endpoints, the text-recovery asymmetry below, two-repo history
- [ ] **5.4** Bibliography, consistency pass, full read-through, compile

---

## Standing rules

1. **Verify every citation against the PDF.** A WebFetch summary of Cheng FSE'26 was
   previously wrong on dataset, model and numbers.
2. **The gold-free invariant is not negotiable.** No agent tool may observe the gold
   patch, the gold-fixed tree, or the held-out tests.
3. **Freeze the sample before the sweep.** Report n honestly.
4. **Negatives are results** — the parked Breaker, the near-no-op Critic, a null RQ1.
5. **Validate on one instance before launching a batch.**
6. **Never use a model you cannot name.** Pin exact model ids; no auto-routers.

## Open risks

| Risk | Mitigation |
|---|---|
| **No free model is capable enough** — both arms score ~0 and RQ1 is uninformative | Capability probe (2.1) before the sweep; rescope RQ1 if it fails |
| **McNemar power depends on discordant pairs, not n** | Report the paired difference with a bootstrap 95% CI; a null result is reportable |
| **Free models change or disappear mid-sweep** | Pin model ids, record the served model + provider per run, run each model's instances in one contiguous block |
| **429 throttling on free endpoints** | Back-off in the client; the driver resumes where it stopped |
| **"Open-weight" claim unverified** for some free models | Task 2.3; exclude anything without published weights from RQ3 |
| **Text-recovery asymmetry:** the OpenAI-format path turns a tool call written as prose into a real call; the Anthropic path does not | Conservative (it helps the weaker models); disclose in the methodology |
| Repos never built (matplotlib, sklearn, pytest, sphinx…) | sympy proven; build one of each family early; disclose any that fail |
| Writing is the real critical path | Claude drafts ch3/ch4 so Fernando's time goes to ch2/ch5/ch6 |
| `ContainerSession.close()` leaks containers | Reap before the freeze |
