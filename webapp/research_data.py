"""Research plan, decisions, and related-work comparison for the /research page.

Git-versioned Python data (not DB rows) on purpose: the plan and the
related-work table are research artifacts — changes to them should show up in
diffs and be reviewable, exactly like code. The /research route renders these
structures directly.

Status vocabulary: "done" | "active" | "pending" (plan items and phases),
"open" | "decided" (decisions).
"""

from __future__ import annotations

# ── The plan to finish the research ──────────────────────────────────────────
# One phase per card; items are the concrete, checkable steps.

RESEARCH_PLAN: list[dict] = [
    {
        "name": "Built & validated",
        "status": "done",
        "items": [
            {"text": "Agent runtime (vendor-agnostic tool loop, token caps)", "status": "done",
             "note": "src/tdbf/agent/runtime.py"},
            {"text": "Phase A — BRT agent", "status": "done",
             "note": "C2P 5/6 = 0.83 on astropy Lite (Sonnet)"},
            {"text": "Phase B — fix agent", "status": "done",
             "note": "resolved 3/5 = 0.60 on the L4-gated arm"},
            {"text": "Deployable test-first pipeline (script 07, gold-free gate)", "status": "done",
             "note": "THE proposed system: BRT → gate → fix, no gold in any decision"},
            {"text": "Metric suite + webapp metrics page", "status": "done",
             "note": "src/tdbf/metrics.py, /metrics"},
            {"text": "RQ3 ablation material (negative results included)", "status": "done",
             "note": "Reviewer revise-rate 1/14 (no-op on strong model); post-fix Breaker "
                     "n=2 both false-negative 'robust' (no oracle → fail-open); best-of-N BRT"},
        ],
    },
    {
        "name": "Current evidence (treat as pilot-quality)",
        "status": "done",
        "items": [
            {"text": "Test-first pipeline: 9/13 resolved = 0.69", "status": "done",
             "note": "NON-random sample (all 6 astropy Lite + 7 early django) — will drop "
                     "on a drawn sample; do not headline this number"},
            {"text": "Weak-arm finding: qwen 7–14B cannot sustain the autonomous loop", "status": "done",
             "note": "C2P 0.00 / empty patches; prompt-harden ladder did not close it — capability, "
                     "not scaffolding"},
        ],
    },
    {
        "name": "Next: reactive baseline (arm A)",
        "status": "active",
        "items": [
            {"text": "Build reactive fix agent — same model/tools/budgets, no BRT", "status": "active",
             "note": "tool-registry variant: drop run_brt, keep run_tests; per proposal §Experimental Setup"},
            {"text": "Offline smoke test (mirrors agent_*_smoke.py pattern)", "status": "pending"},
        ],
    },
    {
        "name": "Pilot",
        "status": "pending",
        "items": [
            {"text": "Draw + freeze a random sample of ~20 Lite instances (stratified by repo)",
             "status": "pending", "note": "sample file committed BEFORE any run"},
            {"text": "Run both arms on the pilot (~$25)", "status": "pending",
             "note": "real Δ + real per-instance cost decide the sweep size"},
        ],
    },
    {
        "name": "Sweep (RQ1)",
        "status": "pending",
        "items": [
            {"text": "Frozen random sample ~150 Lite × 2 arms (~$180)", "status": "pending",
             "note": "paired design: same instance through both arms; McNemar on discordant pairs"},
            {"text": "Docker images pre-built for the sample (01_validate_gold)", "status": "pending",
             "note": "wall-clock bottleneck, not API cost"},
        ],
    },
    {
        "name": "RQ2 — offline artifact analysis (no new agent runs)",
        "status": "pending",
        "items": [
            {"text": "Static test properties from stored test_code (ast): assertion density, "
                     "LOC, exact-value vs raises-only, imports", "status": "pending"},
            {"text": "Determinism: re-run each BRT 3× in container", "status": "pending"},
            {"text": "Gold-patch-line coverage: does the BRT execute the lines the gold patch "
                     "changes?", "status": "pending",
             "note": "gold is LEGAL here — offline measurement only, never in the loop"},
            {"text": "Correlate properties with resolved (needs sweep n first)", "status": "pending"},
        ],
    },
    {
        "name": "Analysis & writing",
        "status": "pending",
        "items": [
            {"text": "RQ1: Δ resolved-rate + bootstrap CI + McNemar", "status": "pending"},
            {"text": "RQ3: assemble ablations (attempt budget, reviewer, breaker, best-of-N)",
             "status": "pending"},
            {"text": "Thesis chapters from skeleton (thesis/)", "status": "pending"},
        ],
    },
]


# ── Decisions log ────────────────────────────────────────────────────────────

DECISIONS: list[dict] = [
    {"date": "2026-07-21", "status": "decided",
     "text": "SWE-bench Lite is the scoring dataset (not Verified).",
     "note": "Proposal §Dataset must be updated — it currently names Verified as primary."},
    {"date": "2026-07-21", "status": "decided",
     "text": "Two arms committed: A = reactive baseline (no BRT), B = test-first 07 pipeline.",
     "note": "RQ1 = B vs A under matched model/tools/budgets."},
    {"date": "2026-07-21", "status": "open",
     "text": "Third arm (test-last / TLD) to engage Cheng '26 cogeneration empirically.",
     "note": "Revisit after the pilot; +~50% sweep cost if added."},
    {"date": "2026-07-21", "status": "open",
     "text": "BugsInPy / GitBugs auxiliary datasets: descoped unless time allows.",
     "note": "Named in the proposal; Lite sweep is the priority."},
    {"date": "2026-07-15", "status": "decided",
     "text": "Multi-agent direction was post-fix Breaker, now parked as an RQ3 negative result.",
     "note": "n=2 false-negative 'robust': depth-1 self-review without an oracle fail-opens."},
]


# ── Related work — the comparison table ──────────────────────────────────────
# vs_tdbf = how our work differs / what we take from it. Keep each entry short
# enough to scan; the thesis Related Work chapter is the long form.

RELATED_WORK: list[dict] = [
    {
        "key": "Cheng2026Cogeneration",
        "title": "Dynamic Cogeneration of Bug Reproduction Test in Agentic Program Repair",
        "venue": "FSE Companion '26", "year": 2026, "url": "https://arxiv.org/abs/2601.19066",
        "dataset": "120 internal Google bugs (GITS), 6 languages",
        "model": "Gemini 2.5 Pro (Passerine, 20 traj × 25 steps)",
        "focus": "Cogenerate fix+BRT in ONE patch; compares TDD / TLD / Freeform orderings",
        "key_result": "All cogeneration ≥ fix-only; Freeform best on pass@20; TLD > TDD on "
                      "candidateBRT@k; agent defaults to test-LAST when unconstrained",
        "vs_tdbf": "THE key comparison. Their TDD is a prompt nudge inside one agent; ours is an "
                   "architectural gate (BRT must verifiably fail before the fix agent starts, and "
                   "the fix agent cannot edit the BRT — their #1 failure mode, deleted tests, is "
                   "impossible here). Private data vs our public Lite. Their failure #3 (fix "
                   "overfits its BRT) = our independently-found verification gap.",
    },
    {
        "key": "Cheng2025BRTAgent",
        "title": "Agentic Bug Reproduction for Effective Automated Program Repair at Google",
        "venue": "arXiv", "year": 2025, "url": "https://arxiv.org/abs/2502.01821",
        "dataset": "Internal Google bugs",
        "model": "Gemini",
        "focus": "Dedicated BRT-generation agent feeding an APR system",
        "key_result": "28% plausible BRTs (vs 10% LIBRO); +30% bugs reach plausible fixes when "
                      "BRTs are provided",
        "vs_tdbf": "Closest prior work; motivates our RQ1. They hand BRTs to a separate APR "
                   "system; we study the end-to-end two-agent pipeline with a gold-free gate "
                   "between phases, on a public benchmark.",
    },
    {
        "key": "Kang2023LIBRO",
        "title": "LLMs are Few-shot Testers: LLM-based General Bug Reproduction (LIBRO)",
        "venue": "ICSE", "year": 2023, "url": "https://arxiv.org/abs/2209.11515",
        "dataset": "Defects4J + GitHub",
        "model": "Codex",
        "focus": "Few-shot (non-agentic) BRT generation from bug reports",
        "key_result": "Reproduces ~30% of Defects4J bugs; post-processing ranks candidates",
        "vs_tdbf": "The non-agentic baseline idea — our scripts 02/03 (--attempts/--repair "
                   "workflow) play this role against the real agent.",
    },
    {
        "key": "Yang2024SWEagent",
        "title": "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering",
        "venue": "NeurIPS", "year": 2024, "url": "https://arxiv.org/abs/2405.15793",
        "dataset": "SWE-bench",
        "model": "GPT-4",
        "focus": "Agent-computer interface design for repo-level issue solving",
        "key_result": "12.5% on SWE-bench full; interface design materially changes agent ability",
        "vs_tdbf": "The canonical reactive (patch-first) agent — the ~12% our proposal cites. "
                   "Our arm A is this workflow under our own scaffold, so the comparison is "
                   "same-model same-tools, not cross-paper.",
    },
    {
        "key": "Jimenez2023SWEbench",
        "title": "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
        "venue": "ICLR", "year": 2024, "url": "https://arxiv.org/abs/2310.06770",
        "dataset": "2,294 GitHub issues (Lite = 300, Verified = 500)",
        "model": "—",
        "focus": "The benchmark: held-out FAIL_TO_PASS / PASS_TO_PASS scoring",
        "key_result": "Real-world issue resolution is hard (Claude 2: 1.96%)",
        "vs_tdbf": "Our substrate and scoreboard. We score offline via the held-out tests; "
                   "agents never see them (gold-free invariant).",
    },
    {
        "key": "Meng2024Empirical",
        "title": "An Empirical Study on LLM-based Agents for Automated Bug Fixing",
        "venue": "arXiv", "year": 2024, "url": "https://arxiv.org/abs/2411.10213",
        "dataset": "SWE-bench Verified",
        "model": "7 systems compared",
        "focus": "Cross-agent comparison; bottleneck analysis",
        "key_result": "Fault localization + bug reproduction are the key bottlenecks",
        "vs_tdbf": "Independent evidence that reproduction (our Phase A) is the binding "
                   "constraint — supports making the BRT a first-class gated artifact.",
    },
    {
        "key": "ReProAgent2026",
        "title": "ReProAgent: Tool-Augmented Multi-Stage Agentic Generation of BRTs",
        "venue": "arXiv", "year": 2026, "url": "https://arxiv.org/abs/2607.09123",
        "dataset": "SWT-bench-lite / SWT-bench-verified",
        "model": "—",
        "focus": "Multi-stage BRT generation from issue reports",
        "key_result": "58.4% / 70.3% reproduction on SWT-bench lite/verified",
        "vs_tdbf": "State of the art for Phase A alone. We do not compete on BRT rate; we study "
                   "what the BRT buys the DOWNSTREAM fix (end-to-end Δ), which they don't measure.",
    },
    {
        "key": "SWEDoctor2026",
        "title": "SWE-Doctor: Guiding SE Agents with Runtime Diagnosis from Multi-Faceted BRTs",
        "venue": "arXiv", "year": 2026, "url": "https://arxiv.org/abs/2607.00990",
        "dataset": "SWE-bench",
        "model": "—",
        "focus": "Multiple behavioral BRTs per issue as runtime diagnosis for the fixer",
        "key_result": "Multi-faceted BRT execution guides patch generation",
        "vs_tdbf": "A richer Phase A→B information channel (N tests, not 1). Natural future-work "
                   "citation; our best-of-N selector is a light version of this idea.",
    },
    {
        "key": "Traceability2026",
        "title": "Understanding APR Agents Through the Lens of Traceability",
        "venue": "arXiv", "year": 2026, "url": "https://arxiv.org/abs/2506.08311",
        "dataset": "Agent trajectories",
        "model": "—",
        "focus": "Empirical trajectory analysis of repair agents",
        "key_result": "Test generation / regression selection are major bottlenecks; argues for "
                      "shift-left (early, high-quality test generation)",
        "vs_tdbf": "'Shift-left' is exactly our architectural claim — cite as converging evidence "
                   "for test-first ordering.",
    },
    {
        "key": "Yuan2025TDFlow",
        "title": "TDFlow: Agentic Workflows for Test Driven Development",
        "venue": "arXiv", "year": 2025, "url": "https://arxiv.org/abs/2510.23761",
        "dataset": "—", "model": "—",
        "focus": "TDD-style agentic workflows (general SE, not APR-specific)",
        "key_result": "Workflow-level TDD framing for agents",
        "vs_tdbf": "Shares the TDD framing; does not run the controlled ordering experiment "
                   "against a matched reactive baseline — our RQ1 does.",
    },
]
