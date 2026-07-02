"""End-to-end agentic TDBF pipeline: BRT phase -> fix phase.

This is the single source of truth for the two agent phases, so the BRT script
(`04_agent_brt.py`), the fix script (`05_agent_fix.py`) and the end-to-end runner
(`06_agent_pipeline.py`) all share identical agent + gold-barrier logic.

Each `run_*_phase` function:
  1. runs a real agent (model-driven control) against the BUGGY tree only
     (gold-free — the agent never sees the gold patch or held-out test_patch);
  2. crosses an explicit GOLD BARRIER and scores the finished artifact offline
     (BRT -> L0-L4 classification; fix patch -> resolved/not via the harness).

The functions return structured results; persistence (`results.jsonl`, webapp
POST) and CLI presentation stay in the scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.instances import SWEBenchInstance
from ..harness.runner import (
    PredictionRecord,
    brt_test_path,
    inject_setup_preamble,
    make_test_patch,
    read_report,
    repo_test_hint,
    run_harness,
    run_test_in_container,
    write_predictions,
)
from ..models.llm import LLMClient
from ..validation.brt_levels import BRTClassification, classify
from .runtime import Agent, AgentResult
from .session import ContainerSession
from .tools import brt_agent_tools, fix_agent_tools

# $/1M tokens (input, output), keyed by substring of the model id. Local = $0.
_PRICING = {"opus": (5.0, 25.0), "sonnet": (3.0, 15.0), "haiku": (1.0, 5.0)}


def cost_usd(model: str, in_tok: int, out_tok: int) -> float | None:
    """USD cost for a run, or None for local/unknown (free) models."""
    for key, (pin, pout) in _PRICING.items():
        if key in model:
            return round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 4)
    return None


def serialise_messages(messages) -> list[dict]:
    """Flatten the agent message trace to JSON-serialisable dicts (transcripts)."""
    out = []
    for m in messages:
        d = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
            ]
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def step_summary(steps) -> list[dict]:
    """Per-iteration tool/token summary for the run record."""
    return [
        {
            "iteration": s.iteration,
            "tools": [tc.name for tc in s.tool_calls],
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "duration_seconds": s.duration_seconds,
        }
        for s in steps
    ]


# ── persistence record builders (shared by scripts 04/05/06) ──────────────────
# Pure dict builders so the BRT script, fix script and end-to-end runner emit
# byte-identical transcripts + results.jsonl records. The scripts own the I/O
# (write the transcript, call save_run) so save_run stays patchable in tests.

def brt_transcript_dict(inst: SWEBenchInstance, model_name: str, phase: "BrtPhaseResult") -> dict:
    r = phase.agent
    return {
        "instance_id": inst.instance_id,
        "model": model_name,
        "system": phase.system,
        "task": phase.task,
        "stop_cause": r.stop_cause,
        "messages": serialise_messages(r.messages),
        "submission": phase.submission,
    }


def brt_run_record(
    inst: SWEBenchInstance, model_name: str, *, split: str, flags: dict, phase: "BrtPhaseResult"
) -> dict:
    r = phase.agent
    attempt = {
        "attempt": 1,
        "mode": "agent",
        "level": phase.level,
        "l3_subtype": phase.classification.l3_subtype if phase.classification else None,
        "stop_cause": r.stop_cause,
        "iterations": r.iterations,
        "num_tool_calls": sum(len(s.tool_calls) for s in r.steps),
        "outcome_buggy": phase.outcome_buggy,
        "outcome_fixed": phase.outcome_fixed,
        "test_code": phase.test_code,
        "input_tokens": r.total_input_tokens,
        "output_tokens": r.total_output_tokens,
        "cost_usd": phase.cost_usd,
        "duration_seconds": sum((s.duration_seconds or 0) for s in r.steps),
        "step_summary": step_summary(r.steps),
    }
    return {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": model_name,
        "split": split,
        "mode": "agent",
        "flags": flags,
        "best_level": phase.level,
        "best_attempt": 1,
        "attempts": [attempt],
    }


def fix_transcript_dict(inst: SWEBenchInstance, model_name: str, phase: "FixPhaseResult") -> dict:
    r = phase.agent
    return {
        "instance_id": inst.instance_id,
        "model": model_name,
        "system": phase.system,
        "task": phase.task,
        "stop_cause": r.stop_cause,
        "messages": serialise_messages(r.messages),
        "fix_patch": phase.fix_patch,
        "resolved": phase.resolved,
    }


def fix_run_record(
    inst: SWEBenchInstance, model_name: str, *, split: str, flags: dict, phase: "FixPhaseResult"
) -> dict:
    r = phase.agent
    h = phase.harness
    attempt = {
        "attempt": 1,
        "mode": "fix_agent",
        "resolved": phase.resolved,
        "applied": h.applied if h else False,
        "stop_cause": r.stop_cause,
        "iterations": r.iterations,
        "num_tool_calls": sum(len(s.tool_calls) for s in r.steps),
        "fix_patch": phase.fix_patch,
        "fail_to_pass_failed": h.fail_to_pass_failed if h else None,
        "pass_to_pass_failed": h.pass_to_pass_failed if h else None,
        "input_tokens": r.total_input_tokens,
        "output_tokens": r.total_output_tokens,
        "cost_usd": phase.cost_usd,
        "duration_seconds": sum((s.duration_seconds or 0) for s in r.steps),
        "step_summary": step_summary(r.steps),
    }
    return {
        "instance_id": inst.instance_id,
        "repo": inst.repo,
        "model": model_name,
        "split": split,
        "mode": "fix_agent",
        "flags": flags,
        "best_level": 4 if phase.resolved else 0,  # reuse field: resolved -> sentinel
        "resolved": phase.resolved,
        "best_attempt": 1,
        "attempts": [attempt],
    }


# ── Phase A: BRT agent ────────────────────────────────────────────────────────

BRT_SYSTEM_PROMPT = """You are an autonomous software-testing agent. Your single goal is to write a \
Bug Reproduction Test (BRT): one pytest test that reproduces the bug described in a GitHub issue.

You are working inside a checkout of `{repo}` at the BUGGY commit `{base_commit}`. The bug has NOT \
been fixed. Your tools run against this buggy code.

A good BRT:
- FAILS on this buggy code with an ASSERTION failure that directly reflects the bug in the issue.
- Fails for the RIGHT reason: it must reach its assertion and fail there, NOT crash with an \
ImportError / NameError / setup error before exercising the behaviour.
- Would PASS once the bug is correctly fixed. You cannot run the fixed code, so reason from the \
issue about what the correct behaviour should be and assert exactly that.

Recommended workflow (you decide the specifics and when you are done):
1. Read the issue. Use glob/grep/read_file to locate the relevant code.
2. Use run_python to OBSERVE the actual buggy behaviour and confirm your understanding before \
asserting anything.
3. Write the test and use run_test to check it FAILS on the buggy code with failure_reason=assertion.
4. Iterate until run_test reports outcome=fail with failure_reason=assertion, then call submit_test.

Rules:
- Framework setup boilerplate is injected for you automatically. Write plain pytest functions; do \
NOT write framework bootstrap (no settings.configure(), no django.setup()).
- {env_hint}
- The test file is placed for you; do not worry about its path or filename.
- Only assert behaviour you are confident a correct fix would produce.
- Call submit_test EXACTLY ONCE, only after run_test has confirmed a clean assertion failure on the \
buggy code. Never submit a test that passes or merely errors on the buggy code.
- Be efficient — you have a limited number of steps and a token budget."""


# Optional scaffolding (`--prompt-harden`). Targets the diagnosed qwen failure
# modes: emitting a tool call as text instead of via the function-calling channel,
# narrating/guessing a tool's output instead of calling it, and addressing the user.
# Apply to BOTH models (Claude is unaffected) so the RQ1 comparison stays fair.
PROMPT_HARDEN = """

CRITICAL TOOL-USE PROTOCOL — follow exactly:
- You act ONLY by invoking tools through the function-calling channel. NEVER write a tool call as \
text, markdown, or a ```json``` / ```python``` block in your message. If you write the call as text \
instead of invoking it, NOTHING happens and your turn is wasted.
- NEVER state, guess, narrate, or assume what a tool will output. You do not know a tool's result \
until the tool actually returns it to you. Do not write "this will output ..." — call the tool and \
wait for the real result.
- You are an autonomous agent. NEVER address a human or ask anyone to run a command, save a file, or \
check a directory. There is no human to help; you do everything yourself via tools.
- Tool arguments MUST be a valid JSON object. Put code in a JSON STRING (a normal double-quoted, \
escaped string) — never a bare identifier and never a triple-quoted block.
- You are NOT done until run_test has returned outcome=fail with failure_reason=assertion AND you \
have then invoked submit_test. Producing test code in your message is not submitting it."""


@dataclass
class BrtPhaseResult:
    """Outcome of the BRT phase. `classification` is None if the agent never
    submitted a test (scored as L0 by the caller). `test_code` is the submission
    with the framework preamble injected (what was scored and what the fix phase
    should consume)."""

    agent: AgentResult
    system: str
    task: str
    submission: str | None
    test_code: str | None
    outcome_buggy: str | None
    outcome_fixed: str | None
    classification: BRTClassification | None
    cost_usd: float | None
    submit_nudge_limit: int

    @property
    def level(self) -> int:
        return self.classification.level.value if self.classification else 0

    @property
    def is_valid_brt(self) -> bool:
        """Gold L4 label (offline analysis ONLY — needs the gold-fixed tree)."""
        return bool(self.classification and self.classification.is_valid_brt)

    @property
    def reproduces_on_buggy(self) -> bool:
        """GOLD-FREE accept signal: the test reached an assertion and FAILED on
        the buggy tree (a candidate BRT, L3 or L4). This is the only usability
        signal observable without a fix, so it — not `is_valid_brt` — is what a
        deployable pipeline may gate on. (Cannot tell L3 from L4; that needs gold.)"""
        return self.outcome_buggy == "fail"


def run_brt_phase(
    inst: SWEBenchInstance,
    client: LLMClient,
    *,
    max_iterations: int,
    max_total_tokens: int,
    prompt_harden: bool = False,
    submit_nudge_limit: int = 2,
    keep_container: bool = False,
    verbose: bool = False,
    score_fixed: bool = True,
) -> BrtPhaseResult | None:
    """Run the BRT agent (gold-free) then score its test through the gold barrier.

    `score_fixed=True` (default; used by 04/06) runs the test on BOTH the buggy and
    the gold-fixed tree and assigns an L0-L4 label. `score_fixed=False` (the
    deployable pipeline, 07) runs the test on the BUGGY tree ONLY — the gold patch
    is never touched, so there is no L4 label, only the gold-free `outcome_buggy`
    (and `reproduces_on_buggy`). Use this when the gate must stay gold-free.

    Returns None only if the container could not be started (instance skipped).
    """
    test_path = brt_test_path(inst.repo)
    env_hint = repo_test_hint(inst.repo) or "(no special framework setup for this repo)"
    system = BRT_SYSTEM_PROMPT.format(
        repo=inst.repo, base_commit=inst.base_commit, env_hint=env_hint
    )
    if prompt_harden:
        system += PROMPT_HARDEN
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for instance `{ctx['instance_id']}` "
        f"({ctx['repo']} @ {ctx['base_commit']}):\n\n{ctx['problem_statement']}"
    )

    # --- AGENT (gold-free): the model drives the tools on the buggy tree. ---
    try:
        session = ContainerSession(inst, keep=keep_container).start()
    except RuntimeError as e:
        print(f"  [skip] {e}")
        return None
    try:
        tools = brt_agent_tools(session)  # gold-free set + assert_gold_free guard
        agent = Agent(
            client, tools, system,
            max_iterations=max_iterations,
            max_total_tokens=max_total_tokens,
            submit_nudge_limit=submit_nudge_limit,
            verbose=verbose,
        )
        print(f"Running BRT agent (max_iter={max_iterations}, max_tok={max_total_tokens}) ...")
        result = agent.run(task)
    finally:
        session.close()

    cost = cost_usd(client.name, result.total_input_tokens, result.total_output_tokens)
    print(
        f"BRT agent done: stop={result.stop_cause}  iters={result.iterations}  "
        f"tokens={result.total_input_tokens}/{result.total_output_tokens}"
        + (f"  cost=${cost}" if cost is not None else "  cost=$0 (local)")
    )

    # =================== GOLD BARRIER =================== #
    # Everything above is gold-free. Below, the gold patch is used PURELY to
    # score the finished test into L0-L4. The agent never saw any of this.
    submission = result.submission
    classification = None
    test_code = None
    outcome_buggy = outcome_fixed = None
    if not submission or not submission.strip():
        print(f"  Agent did not submit a test (stop_cause={result.stop_cause}). Scoring as L0.")
    else:
        test_code = inject_setup_preamble(inst.repo, submission)
        test_patch = make_test_patch(test_path, test_code)
        print("Scoring offline — BUGGY tree ...")
        outcome_buggy = run_test_in_container(inst, test_patch, test_path=test_path)
        print(f"  outcome_on_buggy:  {outcome_buggy}")
        if score_fixed:
            print("Scoring offline — gold-FIXED tree ...")
            outcome_fixed = run_test_in_container(
                inst, test_patch, gold_patch=inst.gold_patch, test_path=test_path
            )
            print(f"  outcome_on_fixed:  {outcome_fixed}")
            classification = classify(outcome_buggy, outcome_fixed)
            print(f"  -> {classification.level_label}  (valid BRT: {classification.is_valid_brt})")
        else:
            # Deployable pipeline: NO gold-fixed run, NO L4 label. The gate uses
            # only the gold-free buggy-side outcome (reproduces_on_buggy).
            reproduces = str(outcome_buggy) == "fail"
            print(f"  gold-free: reproduces failure on buggy = {reproduces} "
                  "(L3/L4 not distinguished — gold not consulted)")

    return BrtPhaseResult(
        agent=result,
        system=system,
        task=task,
        submission=submission,
        test_code=test_code,
        outcome_buggy=str(outcome_buggy) if outcome_buggy is not None else None,
        outcome_fixed=str(outcome_fixed) if outcome_fixed is not None else None,
        classification=classification,
        cost_usd=cost,
        submit_nudge_limit=agent.submit_nudge_limit,
    )


# ── Phase B: fix agent ────────────────────────────────────────────────────────

FIX_SYSTEM_PROMPT = """You are an autonomous bug-fixing agent. Your single goal is to FIX the bug \
described in a GitHub issue by editing the source code of `{repo}` (checked out at the buggy commit \
`{base_commit}`).

You are given a bug-reproduction test (BRT) that currently FAILS because of the bug. Your fix is \
correct when that test PASSES and you have not broken existing behaviour.

Recommended workflow (you decide the specifics and when you are done):
1. Read the issue and the BRT. Use glob/grep/read_file to locate the buggy code.
2. Call run_brt to see the test fail and understand the failure.
3. Use edit_file to apply a fix to the source.
4. Call run_brt again. Iterate edit -> run_brt until it reports outcome=pass.
5. Use run_tests on the SPECIFIC existing test file(s) closest to your change (e.g. the test module \
for the file you edited) to check you did not break related behaviour. Do NOT run whole test \
directories — they are slow and full of failures unrelated to your change.
6. Once run_brt passes and the targeted tests look consistent, call submit_fix.

Rules:
- Fix the ROOT CAUSE in the source. NEVER edit the test to make it pass — the test is fixed and you \
cannot change it (run_brt always uses the original BRT).
- Keep the change minimal and targeted; do not reformat unrelated code.
- IMPORTANT: existing tests in this repo may already fail for reasons unrelated to you (environment, \
network, pre-existing flakiness). A failure reported by run_tests does NOT by itself mean your fix is \
wrong. Only treat it as a regression if you can point to a specific test that your edit plausibly broke.
- Once run_brt reports outcome=pass you have a working fix. Do NOT call revert on a fix that makes the \
BRT pass unless you have clearly identified a regression YOUR change introduced. When unsure, keep the \
fix and submit. revert discards ALL your edits — use it only to escape a state that is clearly worse.
- Call submit_fix EXACTLY ONCE, only after run_brt reports outcome=pass.
- Be efficient — you have a limited number of steps and a token budget."""


@dataclass
class FixPhaseResult:
    """Outcome of the fix phase. `harness` is the SWE-bench report (None if no
    source edits were produced). `resolved` is the held-out verdict the agent
    never saw."""

    agent: AgentResult
    system: str
    task: str
    fix_patch: str
    resolved: bool
    harness: object | None
    cost_usd: float | None


def run_fix_phase(
    inst: SWEBenchInstance,
    brt_code: str,
    client: LLMClient,
    *,
    split: str,
    max_iterations: int,
    max_total_tokens: int,
    keep_container: bool = False,
    verbose: bool = False,
    run_dir: Path = Path("runs") / "agent_fix",
) -> FixPhaseResult | None:
    """Run the fix agent (gold-free, BRT baked into run_brt) then score the fix
    patch through the gold barrier (held-out test_patch via the harness).

    Returns None only if the container could not be started (instance skipped).
    """
    system = FIX_SYSTEM_PROMPT.format(repo=inst.repo, base_commit=inst.base_commit)
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']} @ {ctx['base_commit']}):\n\n"
        f"{ctx['problem_statement']}\n\n"
        f"--- Bug-reproduction test (currently FAILS; make it PASS by fixing the source) ---\n"
        f"{brt_code}"
    )

    # --- AGENT (gold-free): the model edits the buggy tree. ---
    try:
        session = ContainerSession(inst, keep=keep_container).start()
    except RuntimeError as e:
        print(f"  [skip] {e}")
        return None
    try:
        tools = fix_agent_tools(session, brt_code)  # gold-free; BRT baked into run_brt
        agent = Agent(
            client, tools, system,
            max_iterations=max_iterations,
            max_total_tokens=max_total_tokens,
            verbose=verbose,
        )
        print(f"Running fix agent (max_iter={max_iterations}, max_tok={max_total_tokens}) ...")
        result = agent.run(task)
        # Capture the fix patch = tracked source edits (the untracked BRT is excluded).
        fix_patch = session.exec("git diff").output
    finally:
        session.close()

    cost = cost_usd(client.name, result.total_input_tokens, result.total_output_tokens)
    print(
        f"Fix agent done: stop={result.stop_cause}  iters={result.iterations}  "
        f"tokens={result.total_input_tokens}/{result.total_output_tokens}"
        + (f"  cost=${cost}" if cost is not None else "  cost=$0 (local)")
        + f"  patch={len(fix_patch)} chars"
    )

    # =================== GOLD BARRIER =================== #
    # Below uses the held-out test_patch (via the SWE-bench harness) PURELY to
    # decide resolved/not. The agent never saw any of this.
    resolved = False
    harness = None
    if not fix_patch.strip():
        print("  Agent produced no source edits — not resolved.")
    else:
        run_id = f"agent_fix_{inst.instance_id}"
        pred = PredictionRecord(
            instance_id=inst.instance_id,
            model_name_or_path=client.name,
            model_patch=fix_patch,
        )
        pred_path = write_predictions([pred], run_dir / f"{run_id}.jsonl")
        print("Scoring offline via SWE-bench harness (held-out test_patch) ...")
        run_harness(pred_path, split, run_id, [inst.instance_id])
        harness = read_report(run_id, client.name, inst)
        resolved = harness.resolved
        print(f"  applied={harness.applied}  resolved={resolved}  "
              f"FAIL_TO_PASS pass/fail={len(harness.fail_to_pass_passed)}/"
              f"{len(harness.fail_to_pass_failed)}  PASS_TO_PASS fail={len(harness.pass_to_pass_failed)}")

    return FixPhaseResult(
        agent=result,
        system=system,
        task=task,
        fix_patch=fix_patch,
        resolved=resolved,
        harness=harness,
        cost_usd=cost,
    )
