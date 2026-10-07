# TDBF — plan to submission (revised 2026-09-16; re-cut 2026-10-07 → ~2026-10-20)

Working plan for finishing the thesis. Update the checkboxes as things land;
if a decision changes, change it *here* first so the paper sections follow one source.

**Deadline: ~Tuesday 20 October 2026** (moved; the 27 September date below is history).
Current work: **Phase 6 — Django-native BRT execution** (end of file).

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
| 09-10 | Paper prose | Claude drafts Sec. 3 (Approach) and Sec. 4 (Setup); Fernando writes Sec. 1, 2, 6, 7 and owns the argument in Sec. 5 |
| 09-10 | BRT-properties analysis | Reinstated as analysis (not an RQ) — free, over stored artifacts |
| 09-10 | Sample | Stratified across repos, frozen to a committed file before the sweep |
| 09-16 | Budget | **None.** Free OpenRouter models + local Ollama only |
| 09-16 | OpenRouter limits | Fernando adds a one-time **$10** credit → free models go from 50 to 1,000 requests/day. Free models do not draw the credit down. |
| 09-16 | Local host | The Ollama desktop (192.168.1.192) is available on request — it was simply switched off |
| 09-21 | **Document format** | Advisor read: the write-up must be a **research paper**, not a thesis. `thesis/` is now an **ACM `acmart` paper**, single-column manuscript style — the layout of ReProAgent (arXiv:2607.09123), which Fernando picked as the format model. `main.tex` + `sections/`. Register, page budgets and layout mechanics: `thesis/WRITING_GUIDE.md` §0 |
| 10-04 | RQ1 extension | n=90, seed 20261004, run group `rq1-ext`, tag `extension-freeze`; pooled with the base 55 |
| 10-07 | **Django BRT execution** | The BRT tools run Django tests under bare pytest with an empty `INSTALLED_APPS` (22/41 Django BRT trajectories hit it). Fix = run the BRT the repo's own way (`runtests.py`, inside an existing test module). Test-first-only change; the reactive arm is untouched |
| 10-07 | **Django re-run + substitution** | If Phase 6 validates, re-run the test-first arm on **every** paired Django instance (not just failures) under `rq1-djv2`; those records **replace** the old Django test-first results. Non-Django results stay. Report both the frozen and the corrected RQ1 numbers, with the departure described |
| 10-07 | Quota while Phase 6 is built | The extension runs its 56 non-Django instances on the frozen code (valid: the change is Django-only); the 15 pending Django instances wait for Phase 6 |

## Research questions (canonical — `thesis/sections/1_introduction.tex` is the source)

- **RQ1 — ordering.** Does test-first (repair withheld until a test is *observed*
  to fail on the buggy revision) resolve more than a matched reactive arm?
- **RQ2 — orchestration.** Which pipeline characteristics improve fix quality and
  reliability under a fixed budget? Ablations, negatives included.
- **RQ3 — open-weight models.** Can they sustain the pipeline, and at which stage
  does the trajectory break down (submits? reproduces? resolves?)

> The older numbering (RQ1 = weak arm as a scope condition, RQ2 = BRT properties,
> RQ3 = orchestration) is **superseded**. `docs/agent_plan.md` and paper Sections IV–VI
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
- **Disk IS a binding constraint** (corrected 09-17 — the 09-10 estimate was wrong). This
  Docker uses containerd's image store, which keeps compressed layers *and* the unpacked
  copy; new repo families cost 4–10 GB each and prebuilt matplotlib images ~7–8 GB each.
  Colima disk 98 GB: 7.6 GB free after cleanup (09-17). Deleting images frees far less
  than `docker system df -v` "unique size" suggests, because layers are shared. Growing the
  Colima disk (Mac has ~73 GB free) is the way to add the 4 deferred instances.
- **SWE-bench Lite:** django 114, sympy 77, matplotlib 23, sklearn 23, pytest 17,
  sphinx 16, astropy 6, requests 6, pylint 6, xarray 5, seaborn 4, flask 3.
  The 37 images built so far are astropy + django + one sympy.

---

## Phase 1 — Infrastructure (Wed 16 Sep)

- [x] **1.0** $10 OpenRouter credit added — `is_free_tier: False`, `total_credits: 10`, usage 0 (checked 09-16)
- [ ] **1.1** Fix RQ numbering in `4_setup.tex`, `5_results.tex`,
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
- [x] **1.5** Sample environments built and checked 09-17 (`13_build_sample.py`; record in
      **`sample_lite_env.json`**, committed). **Limit for this run: 50 valid, 6 pending
      re-check, 4 deferred to a later run** (decided 09-17).
      - **Dependency drift** broke 7 local builds (newer pandas/setuptools_scm/pip, a removed
        sympy `1.7` branch). Fix: SWE-bench's **prebuilt images** (`docker.io/swebench/…`,
        amd64, digests recorded in the env record). pylint-7114 re-checked valid on it; the
        other 6 (matplotlib 23913/24334/24970, sklearn 25500/25638, sympy 20590) have their
        prebuilt image but are **not re-checked yet** — no disk needed to do so.
      - **Deferred (4):** matplotlib 25332/25498 (Docker disk full, image not pulled);
        sphinx 7686/8801 (my mid-build image deletion crashed their checks — operator error;
        they need a local rebuild, which needs disk). This departs from the frozen rules
        (they are not environment-invalid) and must be reported as such.
      - Memory: conda environment solves (matplotlib) get Claude Code's background jobs
        killed on this 16 GB Mac — run heavy builds from a separate terminal.
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

- [ ] **5.1** Sec. 6 Discussion, Sec. 7 Conclusion, abstract (the abstract in
      `main.tex` is scaffolding with `\ph{}` placeholders — rewrite it last)
- [x] **5.2** Sec. 1 Contributions + roadmap, and the problem statement rewritten as
      an architectural question (done 09-21 in the paper conversion)
- [ ] **5.3** Threats to validity (Sec. 6.4, one paragraph) — non-random development
      set, model churn on free endpoints, the text-recovery asymmetry below,
      two-repo history
- [ ] **5.4** Bibliography, consistency pass, full read-through, compile
- [ ] **5.5** Page-budget pass: the target is 8–10 two-column pages. Grep for `\ph{`
      — none may remain. Drop `\nocite{*}` from `main.tex` so the reference list
      holds only what the paper cites.

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

---

## Phase 6 — Django-native BRT execution (from Wed 7 Oct)

**Why (analysis 2026-10-07, 74 pairs).** Reactive 37 vs test-first 34. By BRT
outcome: L4 accepted 48 → 30 vs 29; L3 accepted 18 → 4 vs 5; gate rejected 8 →
**0 vs 3** — the whole gap is gate rejections, 7 of 8 on Django. Root cause: the
Django preamble (`runner.py` `_REPO_PREAMBLES`) loads `test_sqlite`, whose
`INSTALLED_APPS` is empty — `runtests.py` normally adds the apps and creates the DB —
so any BRT or probe touching models / auth / admin / contenttypes crashes, while the
hint says "Django is already configured" and forbids `TestCase`. 22 of 41 Django BRT
trajectories hit it, including all 5 Django no-submit runs. The fix agent's
`run_tests` already uses `runtests.py`; only the BRT side was left behind.

**Scope.** Test-first arm only (BRT tools `run_test` / `run_python` / `submit_test`,
the gate check, the offline label, the fix agent's `run_brt`). Non-Django repos keep
the identical path (smokes assert it). Later candidates, decided separately because
some touch code both arms share: gate fallback (fix anyway with an *unverified*
test), closing turns that offer only the terminal tool + keep the last
assertion-failing test, retries that carry the last test and its error, `read_file`
windowing / output trimming.

**Measured 2026-10-07 (gold used offline, infrastructure only).** All 40 paired
Django instances: the SWE-bench FAIL_TO_PASS tests, run with `runtests.py`, fail on
buggy and pass on gold-fixed in **40/40** (0–4 s each, Django 3.0–5.0). On buggy
code 27 fail only at an assertion (`FAIL`), **12 only with an exception (`ERROR`)**,
1 both — so the assertion-only gate would reject the official test on 30% of Django
instances. The 12 raise ValueError / TypeError / KeyError / IntegrityError /
OperationalError / NotSupportedError / binascii.Error / Http404; none raise
ImportError / NameError / SyntaxError.

**Decision 10-07 — Django gate rule:** accept a BRT that `FAIL`s *or* `ERROR`s on
buggy, **except** test-authoring errors: ImportError / ModuleNotFoundError /
NameError / SyntaxError / IndentationError, or a test module that fails to load.
Django only (it lives in the runtests.py output parser); other repos keep today's
rule. Reported with the environment change as one departure.

- [x] **6.1 Feasibility (no model calls).** Done 10-07 on django-13447: existing
  module + own `brt_app` both work (models, auth, test DB), ~1 s, files untracked;
  `runtests.py` output separates assertion failure (`FAIL:`) from exception (`ERROR:`).
- [x] **6.2 Implemented 10-07** — `runner.brt_run_spec` / `parse_runtests_result`
  (+ setup errors and "no such table" counted as a broken test, found in the live
  check), `tools.run_brt_in_session` shared by run_test / run_brt / gate / fix
  loop, `run_brt_in_container` for the label, `DJANGO_BRT_SYSTEM_PROMPT`.
  Smokes 18/18 (new `agent_django_brt_smoke.py`). Live, no model: django-13447
  through the real tools → gate accept → label **L4**; Django 3.0/py3.6 OK.
  Original plan text:
- [ ] ~~6.2 Implement~~ (branch `worktree-django-native-brt`). `runner.py`: Django BRT
  path chosen by module, run with `runtests.py`, outcome + failure reason parsed from
  unittest output; hint rewritten (TestCase/SimpleTestCase in an existing module).
  `tools.py`: `module` argument on `run_test` / `submit_test`, `run_brt` uses it;
  `run_python` gets real Django settings (standard apps + in-memory DB). `pipeline.py`:
  gate check, offline label and `run_brt` share the one path.
- [ ] **6.3 Offline validation.** All `scripts/agent_*_smoke.py`, `batch_smoke.py`,
  `00_smoke_test.py` before and after. Runner check against ground truth: for ~8
  Django instances (incl. 13447, 12453) the SWE-bench FAIL_TO_PASS test, run through
  the new path, must fail on buggy and pass on gold-fixed (L4). Gold used offline
  for infrastructure validation only — never shown to an agent.
- [ ] **6.4 Live pilot** (`--arms B`, run group `v2-pilot`, ~300 calls, not counted):
  ~5 Django instances — the 3 rejected (12453, 13447, 13757) + 2 that worked.
- [ ] **6.5 Re-run** (`--arms B`, run group `rq1-djv2`): all 37 paired Django
  instances (~2,000 calls ≈ 2 days of quota). Then the 15 pending Django extension
  instances run both arms with the new code.
- [ ] **6.6 Results + paper.** Metrics take Django test-first from `rq1-djv2`
  (`metrics.EXPERIMENT_ARMS`); report the frozen pre-registered result and the
  corrected one; the departure and its evidence go in Setup / Threats.

**Quota math (~950 calls/day):** non-Django pending ≈ 5.5 days, Django re-run ≈ 2,
pending Django ≈ 2 → ~9.5 days of quota by ~20 Oct. Run in the daytime too, not only
overnight (the counter resets at 00:00 UTC).
