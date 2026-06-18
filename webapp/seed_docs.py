"""Seed content for the research-log / docs area of the webapp.

These documents capture the design decisions, results, and discussion that
shaped TDBF, so the knowledge lives next to the data instead of only in chat
history or local-only files. They are inserted once, on first run, when the
``docs`` table is empty (see ``db.seed_docs``). After that they are editable
in-browser like any other doc — edit freely, this file is only the initial
snapshot.

Each entry: (topic, title, created_at, body-markdown). ``topic`` groups the
docs in the sidebar; ``created_at`` lets you read them as a dated journal.
"""

from __future__ import annotations

# Date the bulk of this knowledge was discussed / recorded.
_D = "2026-06-16"

SEED_DOCS: list[dict] = [
    {
        "topic": "Overview",
        "title": "What TDBF is",
        "created_at": _D,
        "pinned": 1,
        "body": """\
**TDBF (Test-Driven Bug Fixing)** is a research project / thesis. Given a
SWE-bench issue, the system:

1. **Reproduces** the bug,
2. writes a **Bug Reproduction Test (BRT)** that fails on the buggy code, and
3. **fixes** the code until the BRT passes.

The contribution is the **agentic** approach to TDD bug-fixing *and* a way to
**validate BRTs without ever seeing the gold patch**.

### BRT levels (L0–L4)
A generated test is scored by how it behaves on the buggy tree vs. the
gold-patched tree:

| Level | Buggy | Fixed | Meaning |
|---|---|---|---|
| L0 | — | — | malformed / no code block |
| L1 | pass | — | false positive — doesn't reproduce the bug |
| L2 | error | error | crashes on both (import errors, wrong API) |
| L3 | fail | fail | reproduces *a* failure but not the right one |
| **L4** | **fail** | **pass** | **the only valid BRT** |

### The headline metric
**C2P rate = L4 / (L3 + L4)** — of the tests that reproduce *something*, what
fraction are genuinely valid. For the fix phase the metric is the
**resolved-rate** (held-out SWE-bench tests pass).
""",
    },
    {
        "topic": "Invariants",
        "title": "The gold-free invariant (non-negotiable)",
        "created_at": _D,
        "pinned": 1,
        "body": """\
**No agent ever gets a tool that runs on the gold-fixed tree.** Producing the
fix is the *output* of a deployed system, so nothing in the loop may depend on
seeing the answer.

- Agents see only **gold-free signals**: does the test run? does it fail
  *cleanly* on the buggy code? does it pass on buggy (= L1, doesn't reproduce)?
  does the assertion match the probe's observed output / the issue's stated
  value?
- `gold_patch` and `test_patch` are **offline measurement instruments only** —
  used to score L0–L4 and compute C2P *after* the agent finishes, never inside
  the loop.
- `context_safe_dict()` is the **only** view of an instance an agent ever gets.
  It deliberately excludes `gold_patch` and `test_patch`. Direct field access
  is a fairness violation.
- Enforced in code by `assert_gold_free()` (BRT tools) and
  `assert_fix_gold_free()` (fix tools) — allowlist guards over the tool
  registries. **Code-review the tool surface for this on every change.**

Leaking the fixed-run outcome into an agent's feedback = a system that cannot
deploy. This is the control the whole experiment's validity rests on.
""",
    },
    {
        "topic": "Architecture",
        "title": "Workflow → real agent (the pivot)",
        "created_at": _D,
        "body": """\
The key distinction that drives the whole project:

- **Workflow (what existed first):** *our code* decides the steps
  (generate → run → repair → retry). The LLM is a text generator in fixed
  slots. `--attempts` (stateless resampling) and `--repair` (narrow traceback
  feedback) are workflow patches. **This is not what the thesis is about** —
  but it's kept as a non-agentic baseline for comparison.
- **Real agent (the target):** the LLM gets a goal + tools and decides the
  trajectory itself — read this, run that, inspect, write, re-run, debug,
  declare done. **Termination is the model's call, not a loop counter.**

The pivot **inverts control from code → model.** The existing harness
functions (`run_snippet_in_container`, `run_test_in_container`,
`fetch_relevant_code`, `classify`) were *not* thrown away — they became the
**tools the agent calls**.

### Model-agnostic by design
Hold the architecture fixed, vary the model. The agent loop is **not** bound to
any vendor's SDK tool runner (that would make the Claude-vs-qwen RQ1 comparison
unclean). One vendor-agnostic loop runs the same system prompt, same tools,
same termination logic — only the model swaps.
""",
    },
    {
        "topic": "Architecture",
        "title": "The agent system (src/tdbf/agent/)",
        "created_at": _D,
        "body": """\
- **`runtime.py`** — vendor-agnostic tool-use loop (`Agent`). Termination =
  terminal `submit_*` tool (primary) or a no-tool turn (fallback);
  iteration/token caps; real per-turn token capture into `AgentResult`.
- **`session.py`** — persistent per-instance container (`docker run -d sleep
  infinity` + `docker exec`) so an agent's many tool calls pay container
  startup **once** instead of per call (~10–40s each otherwise). File streaming
  over stdin, conda-testbed activation, auto `DOCKER_HOST` for Colima,
  context-managed cleanup.
- **`tools.py`** — gold-free tools wrapping `runner.py`:
  - **BRT set (Phase A):** `read_file`, `grep`, `glob`, `run_python`,
    `run_test` (buggy only), `submit_test` (terminal).
  - **Fix set (Phase B):** `read_file`, `grep`, `glob`, `edit_file`, `run_brt`
    (validated BRT baked in — agent can't weaken its own success criterion),
    `run_tests` (regression check), `revert`, `submit_fix` (terminal).
  - `assert_gold_free` / `assert_fix_gold_free` guard the registries.
- **Scripts:** `04_agent_brt.py` (Phase A), `05_agent_fix.py` (Phase B).
  Offline smoke tests (`scripts/agent_*_smoke.py`) need no Docker/API.

### Tool layer detail
- `read_file` prefixes **line numbers** and accepts `start_line`/`end_line`
  (added after Sonnet burned ~8 iterations grepping for ranges — commit
  d13ea88).
- `run_brt` bakes the validated BRT in at construction so the fixer can't
  weaken the bar it has to clear.
- `run_tests` is capped at 180s and steered to a single targeted test file
  (see the 6938 self-sabotage fix).
""",
    },
    {
        "topic": "Results",
        "title": "Phase A — BRT agent results",
        "created_at": "2026-06-12",
        "pinned": 1,
        "body": """\
**Claude Sonnet (`claude-sonnet-4-6`), 6 astropy Lite instances (2026-06-12).**
The agent drives its own trajectory (glob/grep/read_file → run_python to
observe buggy output → run_test → terminal `submit_test`); a gold barrier
scores offline.

- Results: `12907` L4, `14182` L4, `14365` L3, `14995` L4, `7746` L4,
  `6938` L4 → **C2P = 5/6 = 0.83**.
- Cost **~$0.24/instance** ($0.10–$0.35); ~67k in / 2–6k out tokens per run.
  → ~$24 for a 100-instance Sonnet sweep, comfortably inside the $500 budget.

**Finding — L0-by-timeout is a step-budget artifact, not a capability gap.**
`6938` hit `max_iterations=15` mid-exploration → scored L0; the same instance
at `--max-iterations 25` submitted at iter 13 → **L4**. Default cap should be
≥20 for harder instances.

**Django broadening (2026-06-16):** Phase A on the 8 pre-built Django images →
**7/8 L4 (C2P 1.00)**, one L0 from max_iter. Sonnet handles Django's framework
bootstrap cleanly.
""",
    },
    {
        "topic": "Results",
        "title": "Phase B — Fix agent results",
        "created_at": "2026-06-16",
        "pinned": 1,
        "body": """\
The fix agent edits the buggy tree until the **validated L4 BRT** passes, then
is scored gold-free behind a barrier via the official SWE-bench harness
(applies the held-out `test_patch`, runs FAIL_TO_PASS / PASS_TO_PASS).

**Combined (Sonnet, 12 instances, 2 repos): resolved 9/12 = 0.75**, $2.95 total
(~$0.25/instance). astropy 3/5, django 6/7.

astropy detail (3/5, ~$1.11):

| inst | resolved | $ |
|---|---|---|
| 12907 | yes | 0.42 |
| 14995 | yes | 0.11 |
| 6938  | yes | 0.15 |
| 14182 | no  | 0.28 |
| 7746  | no  | 0.15 |

**RQ3 validity finding:** both astropy failures share a signature —
applied=True, `run_brt` PASSED (agent submitted), but the held-out FAIL_TO_PASS
still fails. **A BRT-passing fix is necessary but not sufficient**; on those two
the BRT under-specifies the correct fix.

**Failure modes seen:**
- *necessary-but-not-sufficient* (astropy 14182, 7746) — directly motivates the
  Phase C Critic role.
- *expensive flail* (django 11019) — hard 16-FAIL_TO_PASS instance, couldn't
  fix, hit max_iter, broke 3 PASS_TO_PASS, **$0.99**. The budget risk →
  motivates a Reviewer role / tighter early-abort.
- *self-sabotage* (6938, since fixed) — made the correct fix, then ran
  `run_tests` over a whole dir, hit unrelated pre-existing failures, and
  **reverted its own good fix**. Mitigated by steering run_tests to one file +
  forbidding reverts of a BRT-passing fix without an identified self-introduced
  regression. Re-run → resolved in 10 iters / $0.15.
""",
    },
    {
        "topic": "Results",
        "title": "RQ1 — Claude vs qwen as agents (the weak arm)",
        "created_at": "2026-06-12",
        "body": """\
Same loop, same tools, same instances — **only the model swaps**.

**qwen2.5-coder:7b** on the 6 astropy instances → **C2P = 0.00** (vs Claude
0.83). The tool layer is *clean* on qwen — it emits valid `grep`/`run_python`
calls and even recovers a wrong import path. But it **cannot sustain the
trajectory to completion**: 5/6 runs ended `natural` (a text turn, then quit,
never calling the terminal `submit_test`). Its one submission was a
wrong-assertion L3. **Zero L4.**

**Scaffolding ladder — does cheap help close the gap? No.**

| arm | L4/L3/L0 | C2P |
|---|---|---|
| qwen2.5-coder:7b (baseline) | 0/1/5 | 0.00 |
| qwen2.5-coder:7b + harden   | 0/0/6 | 0.00 |
| qwen3.5:9b + harden         | 0/0/6 | 0.00 |
| Claude Sonnet (reference)   | 5/1/0 | 0.83 |

`--prompt-harden` adds a tool-use protocol block (use the function-calling
channel, never write the call as text, never narrate/guess a tool's output,
args must be valid JSON, not done until run_test asserts + submit_test). All 12
ladder runs ended `natural` (zero `submit_test`). Hardening suppressed the
tool-call-as-text bug but exposed a deeper **reasoning** failure: qwen talks
itself out of the bug ("replacement works correctly, no assertion failure") and
quits — it even *lost* the baseline's one L3.

**The RQ1 result, in its strongest form:** the gap is **capability, not
tool-format or prompt scaffolding.** All free (local), so it costs nothing
against the budget. This is itself a finding, not a setback.
""",
    },
    {
        "topic": "Research questions",
        "title": "RQ1–RQ4",
        "created_at": _D,
        "body": """\
- **RQ1** — Claude vs qwen *as agents* (same loop, different model). Hypothesis:
  the 7B open model can't sustain an autonomous tool-use loop; this gap is
  itself a result. *(Confirmed so far — see the weak-arm doc.)*
- **RQ2** — does agentic TDD (BRT-in-the-loop) beat a no-test agent baseline on
  resolved-rate? The clean control is *our* agent with the BRT tool removed
  (comparing to a different paper's number is confounded by
  scaffold/model/harness).
- **RQ3** — does a gold-free BRT-confidence signal predict true validity (L4)?
  *(The necessary-but-not-sufficient Phase B failures bear directly on this.)*
- **RQ4** (optional) — single capable agent vs multi-agent decomposition.
""",
    },
    {
        "topic": "Constraints",
        "title": "Budget & reality checks",
        "created_at": _D,
        "body": """\
- **Budget: ~$500 total Claude spend.** Agentic loops are token-heavy (growing
  context × many tool calls). Mitigations: iteration/token caps, prompt
  caching, a small fixed eval subset (~100–150 instances, not all 300 Lite),
  **Sonnet** for the main sweeps + Opus on a slice, **qwen free local** as the
  weak arm. *Measure real token cost on ~5 instances before committing the
  rest of the budget* — done: Sonnet sweeps run ~$0.22–0.25/instance.
- **Model reality:** real autonomous tool use is hard for a 7B model. The
  headline agentic results are on **Claude**; qwen is the documented weak
  baseline.
- **Baseline validity (RQ2):** comparing our agent's resolved-rate to a
  *different paper's* no-test number is confounded. The clean control is *our*
  agent with the BRT tool removed.
- **Docker latency:** every fresh `docker run` is ~10–40s → the persistent
  container session is essential, not optional.
- **Budget risk:** the *expensive-flail* failure (django 11019, $0.99 on one
  hard instance) — watch the token cap / consider early-abort.
""",
    },
    {
        "topic": "Roadmap",
        "title": "Phase C (multi-agent) & next steps",
        "created_at": _D,
        "body": """\
Phases 0 (runtime), A (BRT agent) and B (fix agent) are **done and
live-validated**. Remaining:

### Phase C — go depth-LIGHT first (decided 2026-06-16)
Rather than the full reproduction/test-writer/critic + localizer/fixer/reviewer
split, start with **one collaborator role per phase**:
- **Phase A: a Critic** that reviews the BRT for *specificity* before it's
  accepted — directly targets the necessary-but-not-sufficient failure (astropy
  14182, 7746). Gold-free: reasons from the issue, never sees gold.
- **Phase B: a Reviewer** that judges fix minimality/correctness and which
  targeted tests to run — targets the expensive-flail failure (django 11019).

Topology (working assumption): **orchestrator-mediated code-controller**, all
roles share one `ContainerSession`, each role gold-free with its own
iteration/token cap. The single-agent pipeline is the RQ4 baseline. Escalate to
the full multi-role split only if the light version shows it's needed.

### Open items
- **Refocus on local LLMs first** — make the local arm viable (other
  tool-tuned models, better Ollama tool-calling path, tailored scaffolding) so
  RQ1 has a stronger local story before multi-agent.
- Add the **qwen weak arm** to the Phase B sweep; broaden further (more repos)
  for a paper-grade resolved-rate.
- Then: Phase A Critic + Phase B Reviewer + orchestrator wiring; compare
  single-capable-agent vs light multi-agent (RQ4).
""",
    },
    {
        "topic": "Environment",
        "title": "macOS + Colima + Ollama setup",
        "created_at": _D,
        "body": """\
```bash
# Start Docker (Colima) and point DOCKER_HOST at its socket
colima start
export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock

# Load API keys from .env (gitignored — never commit it)
set -a; source .env; set +a
```

- **Colima quirk:** temp dirs must live under `~/` (not `/tmp`) because Colima
  only mounts `~/`. The runner uses `tempfile.TemporaryDirectory(dir=Path.home())`.
- **Image naming:** `sweb.eval.x86_64.{instance_id}:latest`, built by
  `01_validate_gold.py`. Run with `cache_level=instance` to keep images.
- **Ollama** server is the Windows desktop at `192.168.1.192:11434` (needs
  `OLLAMA_HOST=0.0.0.0` + a firewall inbound rule for TCP 11434).
- The harness uses `--namespace ''` (`build_locally=True`) to build images
  locally rather than pulling prebuilt registry images.
""",
    },
]
