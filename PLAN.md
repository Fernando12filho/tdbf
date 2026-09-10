# TDBF — plan to submission (2026-09-10 → 2026-09-27)

Working plan for finishing the thesis. Update the checkboxes as things land;
if a decision changes, change it *here* first so the chapters follow one source.

**Deadline: Sunday 27 September 2026.** Data freeze **Thursday 24 September**.

---

## Decisions taken 2026-09-10

| Decision | Choice |
|---|---|
| Deadline | Sat 26 – Sun 27 Sep; no new runs after Thu 24 |
| RQ3 framing | **Open-weight models generally** — hosted (OpenRouter) for model variety, local Ollama as the on-device configuration. The "code never leaves the machine" wording must come out of the RQ3 paragraph. |
| Thesis prose | Claude drafts ch3 (Methodology) and ch4 (Experimental Design); Fernando writes ch1, ch2, ch6, ch7 and owns the argument in ch5 |
| BRT-properties analysis | Reinstated as analysis (not an RQ) — it is free, over stored artifacts, and gold is legal offline |
| Sample | Stratified across repos, frozen to a committed file before the sweep |

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

- **Image build ≈ 3.4 min** including gold validation (sympy-11400, first ever
  build of that repo family: 203 s, `resolved=True`, F2P 2/0, P2P 0 fail).
- **Disk is not the binding constraint.** Nominal image sizes sum to ~153 GB but
  the real store is ~29 GB — SWE-bench images share base and per-repo env layers.
  A whole new repo family costs ~1.8 GB actual. **63 GB free** on the Colima disk
  (98 GB, `/dev/vdb1`) supports the full stratified sample.
- **Pipeline run ≈ 2.5 min/instance** per arm (14 instances in ~35 min), **≈$0.20**
  with prompt caching on (90% cache hit rate; $2.73 for 14 instances end-to-end).
- **SWE-bench Lite composition:** django 114, sympy 77, matplotlib 23, sklearn 23,
  pytest 17, sphinx 16, astropy 6, requests 6, pylint 6, xarray 5, seaborn 4,
  flask 3. The 36 images built so far are **astropy + django only**.
- **Baseline result to beat/reproduce:** Arm B, Sonnet, n=14 — gate 14/14,
  resolved 10/14, C2P 13/14.

---

## Phase 1 — Infrastructure (Thu 10 – Sat 12 Sep)

- [ ] **1.1** Fix RQ numbering in `4_experiments.tex`, `5_results.tex`,
      `6_discussion.tex`; rewrite the RQ3 paragraph in `1_introduction.tex` to drop
      "local hardware / never leaves the machine"
- [ ] **1.2** OpenRouter client (`openrouter:<model>` in `build_client`).
      `_OpenAIToolMixin` already speaks the format, so this is a base_url + key +
      pricing lookup. `OPENROUTER_API_KEY` is already set.
- [ ] **1.3** Arm A, the reactive baseline: drop `run_brt` from `_FIX_TOOL_NAMES`,
      strip its mention from `submit_fix`'s description, no Phase A. Same model,
      tools, budgets, container, harness. Add an offline smoke asserting the arms
      differ *only* in that.
- [ ] **1.4** Paired sweep driver (`11_paired_sweep.py`): per instance, build →
      Arm A → Arm B → score → optionally reclaim the image. Resumable; tags every
      record with `arm` and `run_group`.
- [ ] **1.5** Start the background builder on the stratified list; let it run.

## Phase 2 — Pilot and freeze (Sun 13 – Wed 16 Sep)

- [ ] **2.1** Pilot: ~20 random Lite instances through both arms → Δ estimate,
      per-instance cost, wall-clock per instance
- [ ] **2.2** **Freeze the sample**: stratified random ~150 instances written to
      `sample_lite_frozen.json` and committed *before* the sweep
- [ ] **2.3** Claude drafts ch3 Methodology; Fernando reviews
- [ ] **2.4** Fernando begins ch2 reading

## Phase 3 — Main sweep (Thu 17 – Sun 20 Sep)

- [ ] **3.1** Paired sweep runs continuously over the frozen sample
- [ ] **3.2** Claude drafts ch4 Experimental Design
- [ ] **3.3** Fernando writes ch2 Related Work
- [ ] **3.4** Bring Ollama up (192.168.1.192:11434) before Phase 4

## Phase 4 — Ablations and analysis (Mon 21 – Wed 23 Sep)

- [ ] **4.1** RQ2 ablations — prefer *stored* data (breaker, critic, best-of-N all
      already run); only add runs where the stored data cannot answer it
- [ ] **4.2** RQ3 open-weight runs: OpenRouter model set + local qwen, staged
      reporting (terminal submit? reproduces on buggy? resolves?)
- [ ] **4.3** BRT-properties analysis over stored artifacts (AST assertion density,
      exact-value vs raises-only, determinism 3×, gold-patch-line coverage)
- [ ] **4.4** Figures: pipeline diagram, funnel, L0–L4 distribution, cost/resolve
- [ ] **4.5** Fernando writes ch5 Results as numbers land

## Phase 5 — Freeze (Thu 24 Sep)

- [ ] **5.1** **No new runs after tonight.** Snapshot `results.jsonl`, record the
      commit hash and the frozen sample file in the thesis.
- [ ] **5.2** Final metrics, tables, figures regenerated from the frozen data

## Phase 6 — Finish (Fri 25 – Sun 27 Sep)

- [ ] **6.1** ch6 Discussion, ch7 Conclusion, abstract
- [ ] **6.2** ch1 Contributions + Thesis Organization (deleted in the trim, must
      come back), uncomment/rewrite the Thesis Statement section
- [ ] **6.3** Threats to validity — including the non-random 14-instance
      development set, single primary model, two-repo history
- [ ] **6.4** Bibliography, consistency pass, full read-through, compile

---

## Standing rules

1. **Verify every citation against the PDF.** A WebFetch summary of Cheng FSE'26
   was previously wrong on dataset, model and numbers. If a claim cannot be
   confirmed from the source, flag it rather than smoothing it over.
2. **The gold-free invariant is not negotiable.** No agent tool may observe the
   gold patch, the gold-fixed tree, or the held-out tests. Gold is an offline
   measurement instrument.
3. **Freeze the sample before the sweep**, not after. Report n honestly.
4. **Negatives are results.** The parked Breaker, the near-no-op Critic and any
   null RQ1 effect are findings, not failures.
5. **Validate on one instance before launching a batch.** This caught a parser
   regression on 2026-09-09 that would have corrupted a 14-instance run.

## Open risks

| Risk | Mitigation |
|---|---|
| **McNemar power depends on discordant pairs, not n.** Both arms share a model, so they will agree often; n=150 may still give few discordant pairs. | Pre-commit to reporting the paired difference with a bootstrap 95% CI. A null result is reportable. |
| Repos never built (matplotlib, sklearn, pytest, sphinx…) may have environment problems | sympy already proven. Build one instance of each family early in Phase 1; if a family fails, disclose and stratify over what builds. |
| Ollama host currently unreachable | Needed only by Phase 4.2; bring it up during Phase 3. |
| Writing is the real critical path | Claude drafts ch3/ch4 in Phases 2–3 so Fernando's time goes to ch2/ch5/ch6. |
| `ContainerSession.close()` leaks containers (confirmed) | Cosmetic during runs; reap before the freeze so disk figures are honest. |
