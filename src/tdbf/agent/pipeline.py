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
    ContainerResult,
    PredictionRecord,
    _parse_failure_reason,
    _parse_pytest_outcome,
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
from .roles import Break, Review, Selection, run_breaker, run_critic, run_reviewer, run_selector
from .runtime import Agent, AgentResult
from .session import ContainerSession
from .tools import brt_agent_tools, fix_agent_tools

# Pricing lives in tdbf.pricing so this module and metrics.py cost runs
# identically (re-exported here: callers and smokes import it from pipeline).
from ..pricing import cost_usd  # noqa: E402,F401  (re-export)


def take_served(client) -> dict | None:
    """Which model + provider actually answered since the last call, then reset.

    Only clients that track provenance (OpenRouter) return anything; for the rest
    this is None and callers leave their records unchanged. Free endpoints can
    re-point or withdraw a model id mid-sweep, so this is the record of what a
    run was really made with.
    """
    snap = getattr(client, "served_snapshot", None)
    if not callable(snap):
        return None
    served = snap()
    client.reset_served()
    return served


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

def _reviews_dict(phase) -> list[dict]:
    return [{"verdict": rv.verdict, "feedback": rv.feedback} for rv in (phase.reviews or [])]


def _breaks_dict(phase) -> list[dict]:
    return [{"verdict": bk.verdict, "feedback": bk.feedback} for bk in (getattr(phase, "breaks", None) or [])]


def brt_transcript_dict(inst: SWEBenchInstance, model_name: str, phase: "BrtPhaseResult") -> dict:
    r = phase.agent
    d = {
        "instance_id": inst.instance_id,
        "model": model_name,
        "system": phase.system,
        "task": phase.task,
        "stop_cause": r.stop_cause,
        "messages": serialise_messages(r.messages),
        "submission": phase.submission,
    }
    if phase.reviews:
        d["reviews"] = _reviews_dict(phase)
    return d


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
        "failure_reason_buggy": phase.failure_reason_buggy,
        "outcome_fixed": phase.outcome_fixed,
        "test_code": phase.test_code,
        "input_tokens": phase.total_input_tokens if phase.total_input_tokens is not None else r.total_input_tokens,
        "output_tokens": phase.total_output_tokens if phase.total_output_tokens is not None else r.total_output_tokens,
        "cache_write_tokens": phase.total_cache_write_tokens,
        "cache_read_tokens": phase.total_cache_read_tokens,
        "cost_usd": phase.cost_usd,
        "duration_seconds": sum((s.duration_seconds or 0) for s in r.steps),
        "step_summary": step_summary(r.steps),
    }
    if phase.reviews:  # Phase C only — single-agent records stay byte-identical
        attempt["critic_rounds"] = len(phase.reviews)
        attempt["critic_verdicts"] = [rv.verdict for rv in phase.reviews]
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
    d = {
        "instance_id": inst.instance_id,
        "model": model_name,
        "system": phase.system,
        "task": phase.task,
        "stop_cause": r.stop_cause,
        "messages": serialise_messages(r.messages),
        "fix_patch": phase.fix_patch,
        "resolved": phase.resolved,
    }
    if phase.reviews:
        d["reviews"] = _reviews_dict(phase)
    if getattr(phase, "breaks", None):
        d["breaks"] = _breaks_dict(phase)
    return d


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
        "input_tokens": phase.total_input_tokens if phase.total_input_tokens is not None else r.total_input_tokens,
        "output_tokens": phase.total_output_tokens if phase.total_output_tokens is not None else r.total_output_tokens,
        "cache_write_tokens": phase.total_cache_write_tokens,
        "cache_read_tokens": phase.total_cache_read_tokens,
        "cost_usd": phase.cost_usd,
        "duration_seconds": sum((s.duration_seconds or 0) for s in r.steps),
        "step_summary": step_summary(r.steps),
    }
    if phase.reviews:  # Phase C only
        attempt["reviewer_rounds"] = len(phase.reviews)
        attempt["reviewer_verdicts"] = [rv.verdict for rv in phase.reviews]
    if getattr(phase, "breaks", None):  # Phase C — post-fix breaker only
        attempt["breaker_rounds"] = len(phase.breaks)
        attempt["breaker_verdicts"] = [bk.verdict for bk in phase.breaks]
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
    # Phase C (multi-agent). Empty/None in the single-agent path so records stay
    # byte-identical. `total_*_tokens` aggregate worker + critic runs when set.
    reviews: list = None  # type: ignore[assignment]
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    # How the buggy-tree run failed: "assertion" (reached and fired its assert),
    # "runtime" (crashed before asserting) or None (undetermined). Gold-free —
    # parsed from buggy-side pytest output only. Gates `reproduces_on_buggy`.
    failure_reason_buggy: str | None = None

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
        deployable pipeline may gate on. (Cannot tell L3 from L4; that needs gold.)

        A pytest exit code of 1 means "a test failed", which conflates two
        opposite things: an ASSERTION that fired (a real reproduction) and a test
        that CRASHED before reaching its assertion (invented API, bad import).
        A crash fails identically on the fixed tree, so it can never be L4 and is
        not a reproduction. `failure_reason` already distinguishes them, so the
        gate consults it. Note the criterion is `!= "runtime"`, NOT
        `== "assertion"`: a bare `assert x == y` with no message emits no
        `E AssertionError:` line, so its reason is None while still being a
        perfectly good reproduction."""
        return self.outcome_buggy == "fail" and self.failure_reason_buggy != "runtime"


def _brt_retry_feedback(outcome_buggy, submission: str | None, failure_reason: str | None = None) -> str:
    """Gold-free feedback appended to the task when a submitted BRT lands L0/L1/L2
    (it did not fail cleanly on the BUGGY tree). Reasons ONLY from the buggy-side
    outcome — never the gold-fixed result — so the retry stays deployable."""
    prev = f"\n\nYour previous test:\n{submission}\n" if submission and submission.strip() else "\n"
    if outcome_buggy == "fail" and failure_reason == "runtime":  # L2 — crashed, never asserted
        why = ("It FAILED on the buggy code, but it CRASHED before reaching its assertion "
               "(an ImportError, NameError, AttributeError or wrong API — not an "
               "AssertionError). A crashing test fails the same way on fixed code, so it "
               "does NOT reproduce the bug. Use run_python to confirm the API you call "
               "actually exists and behaves as you expect, then make the test REACH its "
               "assertion and fail there.")
    elif outcome_buggy == "pass":  # L1 — the test does not reproduce the bug
        why = ("It PASSED on the buggy code, so it does NOT reproduce the bug. A bug "
               "reproduction test must FAIL on the buggy code. Use run_python to observe the "
               "ACTUAL buggy behaviour, then assert the CORRECT expected value so the test "
               "fails now and would pass once the bug is fixed.")
    elif outcome_buggy == "error":  # L2 — broken test, not a clean assertion failure
        why = ("It ERRORED on the buggy code instead of failing cleanly (an ImportError, "
               "NameError, wrong API or setup error — not an assertion). Use run_test to see "
               "the error and fix the test so it REACHES its assertion and fails there.")
    else:  # L0 — absent / no usable test submitted
        why = ("You did not submit a usable test that fails on the buggy code. Write the test, "
               "use run_test to confirm it fails with failure_reason=assertion on the buggy "
               "code, and only then call submit_test.")
    return (f"\n\n--- Your previous attempt did NOT reproduce the bug ---{prev}"
            f"{why}\n\nTry again.")


def run_brt_phase(
    inst: SWEBenchInstance,
    client: LLMClient,
    *,
    max_iterations: int,
    max_total_tokens: int,
    max_tokens_per_turn: int = 4096,
    prompt_harden: bool = False,
    submit_nudge_limit: int = 2,
    brt_retries: int = 2,
    keep_container: bool = False,
    verbose: bool = False,
    score_fixed: bool = True,
    critic_client: LLMClient | None = None,
    critic_rounds: int = 0,
    critic_max_iterations: int = 8,
) -> BrtPhaseResult | None:
    """Run the BRT agent (gold-free) then score its test through the gold barrier.

    `score_fixed=True` (default; used by 04/06) runs the test on BOTH the buggy and
    the gold-fixed tree and assigns an L0-L4 label. `score_fixed=False` (the
    deployable pipeline, 07) runs the test on the BUGGY tree ONLY — the gold patch
    is never touched, so there is no L4 label, only the gold-free `outcome_buggy`
    (and `reproduces_on_buggy`). Use this when the gate must stay gold-free.

    `brt_retries` (default 2): if the submitted test does NOT fail cleanly on the
    buggy tree (L0 no test / L1 passes / L2 errors — it does not reproduce the bug),
    the BRT agent is re-run with gold-free feedback on what went wrong, up to this
    many extra times. The decision looks ONLY at the buggy-side outcome, so it stays
    deployable/gold-free. A capable model reproduces on the first attempt, so the
    loop breaks immediately and the run is byte-identical to `brt_retries=0`.

    Phase C (multi-agent): pass `critic_client` + `critic_rounds > 0` to add a
    Critic that reviews the candidate BRT for specificity; on a 'revise' verdict
    the worker gets another round with the critic's feedback. `critic_client=None`
    (default) is the single-agent ablation baseline (RQ3) — byte-identical to before.

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
    reviews: list[Review] = []
    tot_in = tot_out = 0
    tot_cw = tot_cr = 0
    submission = test_code = test_patch = None
    outcome_buggy = outcome_fixed = None
    failure_reason_buggy = None
    classification = None

    def _run_brt_agent(run_task: str):
        a = Agent(
            client, brt_agent_tools(session), system,  # gold-free set + assert_gold_free guard
            max_iterations=max_iterations,
            max_total_tokens=max_total_tokens,
            max_tokens_per_turn=max_tokens_per_turn,
            submit_nudge_limit=submit_nudge_limit,
            verbose=verbose,
        )
        return a, a.run(run_task)

    def _score_on_buggy(sub):
        # GOLD-FREE: run the submitted test on the BUGGY tree only. Returns
        # (test_code, test_patch, outcome_on_buggy); the outcome is a ContainerResult
        # (compares == "fail"/"pass"/"error"). (None, None, None) if nothing usable.
        #
        # ALWAYS the live session. The gate is a deployment decision, so it is
        # taken in the runtime the agent actually worked in, with the same
        # budget — see _score_on_buggy_in_session for why the cold path skewed
        # it. Crucially this does not depend on `score_fixed`: --score-brt is
        # documented as "info only, never gates", and an earlier version of this
        # branched the gate's substrate on it, so passing an informational flag
        # silently changed which BRTs were accepted. The offline L0-L4 label
        # runs its own buggy+fixed pair below, through one substrate, so that
        # comparison stays like-for-like without touching the gate.
        if not (sub and sub.strip()):
            return None, None, None
        tc = inject_setup_preamble(inst.repo, sub)
        tp = make_test_patch(test_path, tc)
        print("Verifying the submitted BRT — BUGGY tree, in-session ...")
        ob = _score_on_buggy_in_session(session, sub)
        print(f"  outcome_on_buggy:  {ob}"
              + (f" (failure_reason={ob.failure_reason})" if ob.failure_reason else ""))
        return tc, tp, ob

    try:
        # Gate-retry loop: a submitted BRT that does NOT fail cleanly on the buggy
        # tree is L0 (no test), L1 (passes) or L2 (errors) — it does not reproduce
        # the bug. Rather than give up, re-run the BRT agent (same tools/session, a
        # fresh conversation) with gold-free feedback on what went wrong, up to
        # `brt_retries` times. The decision uses ONLY the buggy-side outcome — never
        # the gold-fixed result — so it stays deployable/gold-free. A capable model
        # submits a clean-failing BRT on attempt 0, so the loop breaks immediately
        # (byte-identical to the pre-retry behaviour).
        run_task = task
        for attempt in range(brt_retries + 1):
            label = "BRT agent" if attempt == 0 else f"BRT agent retry {attempt}/{brt_retries}"
            print(f"Running {label} (max_iter={max_iterations}, max_tok={max_total_tokens}) ...")
            agent, result = _run_brt_agent(run_task)
            tot_in += result.total_input_tokens
            tot_out += result.total_output_tokens
            tot_cw += result.total_cache_write_tokens
            tot_cr += result.total_cache_read_tokens
            submission = result.submission
            test_code, test_patch, outcome_buggy = _score_on_buggy(submission)
            failure_reason_buggy = getattr(outcome_buggy, "failure_reason", None)
            if outcome_buggy == "fail" and failure_reason_buggy != "runtime":
                break  # reached its assertion and failed = candidate BRT (L3/L4)
            if attempt < brt_retries:
                bad = ("L1 (passes on buggy)" if outcome_buggy == "pass"
                       else "L2 (crashes before asserting)" if failure_reason_buggy == "runtime"
                       else "L2 (errors on buggy)" if outcome_buggy == "error"
                       else "L0 (no usable test)")
                print(f"  BRT does not reproduce the bug — {bad}. "
                      f"Retrying with feedback ({attempt + 1}/{brt_retries}).")
                run_task = task + _brt_retry_feedback(
                    outcome_buggy, submission, failure_reason_buggy)

        # Phase C: Critic reviews the candidate BRT for specificity; on 'revise'
        # the worker re-runs with the feedback (bounded by critic_rounds). The
        # critic shares this session and is gold-free (see roles.py).
        for rnd in range(critic_rounds if critic_client is not None else 0):
            if not (result.submission and result.submission.strip()):
                break  # nothing submitted to review
            review = run_critic(
                inst, session, critic_client,
                candidate_test=result.submission,
                max_iterations=critic_max_iterations,
                max_total_tokens=max_total_tokens,
                verbose=verbose,
            )
            reviews.append(review)
            tot_in += review.agent.total_input_tokens
            tot_out += review.agent.total_output_tokens
            print(f"  Critic round {rnd + 1}: verdict={review.verdict}")
            if review.accepted:
                break
            revise_task = (
                f"{task}\n\n--- A critic reviewed your previous test and asked for revisions ---\n"
                f"Your previous test:\n{result.submission}\n\n"
                f"Critic feedback: {review.feedback}\n\n"
                "Revise the test to address this feedback. Use run_test to confirm it still fails "
                "cleanly (assertion) on the buggy code, then call submit_test."
            )
            agent, result = _run_brt_agent(revise_task)
            tot_in += result.total_input_tokens
            tot_out += result.total_output_tokens
            tot_cw += result.total_cache_write_tokens
            tot_cr += result.total_cache_read_tokens
            # Re-score the revised submission on the buggy tree (gold-free).
            submission = result.submission
            test_code, test_patch, outcome_buggy = _score_on_buggy(submission)
            failure_reason_buggy = getattr(outcome_buggy, "failure_reason", None)
    finally:
        session.close()

    cost = cost_usd(client.name, tot_in, tot_out,
                    cache_write_tok=tot_cw, cache_read_tok=tot_cr)
    print(
        f"BRT agent done: stop={result.stop_cause}  iters={result.iterations}  "
        f"tokens={tot_in}/{tot_out}"
        + (f"  cost=${cost}" if cost is not None else "  cost=$0 (local)")
        + (f"  critic_rounds={len(reviews)}" if reviews else "")
    )

    # =================== GOLD BARRIER =================== #
    # Everything above is gold-free (the retry loop scored the BUGGY tree only).
    # Below, the gold patch is used PURELY to assign the offline L0-L4 label when
    # score_fixed=True. The agent never saw any of this.
    if not (submission and submission.strip()):
        print(f"  Agent did not submit a test (stop_cause={result.stop_cause}). Scoring as L0.")
    elif score_fixed:
        # The label observes BOTH sides through the patch-based container, so
        # L3-vs-L4 is a like-for-like comparison. Its buggy run is separate from
        # (and never feeds) the gate above, which is why the label can disagree
        # with the gate — that disagreement is itself worth seeing.
        print("Scoring offline — BUGGY tree (label) ...")
        outcome_buggy_label = run_test_in_container(inst, test_patch, test_path=test_path)
        print(f"  outcome_on_buggy(label):  {outcome_buggy_label}")
        print("Scoring offline — gold-FIXED tree ...")
        outcome_fixed = run_test_in_container(
            inst, test_patch, gold_patch=inst.gold_patch, test_path=test_path
        )
        print(f"  outcome_on_fixed:  {outcome_fixed}")
        classification = classify(outcome_buggy_label, outcome_fixed)
        print(f"  -> {classification.level_label}  (valid BRT: {classification.is_valid_brt})")
        if classification.is_valid_brt and not (
            outcome_buggy == "fail" and failure_reason_buggy != "runtime"
        ):
            print("  [warn] the gate REJECTED a test the offline label calls L4 — "
                  "a valid BRT was thrown away; investigate before trusting GateRate.")
    else:
        # Deployable pipeline: NO gold-fixed run, NO L4 label. The gate uses only
        # the gold-free buggy-side outcome (reproduces_on_buggy).
        print(f"  gold-free: reproduces failure on buggy = "
              f"{outcome_buggy == 'fail' and failure_reason_buggy != 'runtime'} "
              f"(outcome={outcome_buggy} reason={failure_reason_buggy}; "
              "L3/L4 not distinguished — gold not consulted)")

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
        failure_reason_buggy=failure_reason_buggy,
        reviews=reviews,
        total_input_tokens=tot_in,
        total_output_tokens=tot_out,
        total_cache_write_tokens=tot_cw,
        total_cache_read_tokens=tot_cr,
    )


# ── Phase A (best-of-N): generate many BRTs, select the best ──────────────────
# Gold-free throughout the loop: every candidate is scored on the BUGGY tree only,
# the viability filter + the Selector never see the fixed tree. Only the CHOSEN
# candidate is optionally scored through the gold barrier afterwards (scoreboard).


def _score_on_buggy_in_session(session: ContainerSession, submission: str) -> ContainerResult:
    """Gold-free: write a candidate BRT into the shared (buggy) session and run it.

    Buggy tree only — the fixed tree is never touched — so this is a valid gate /
    pre-filter signal. It writes the SUBMISSION, not whatever the agent happened
    to pass to its last `run_test` call, so the artifact that is actually kept is
    the one that gets verified; the agent cannot submit code it never ran.

    This is the substrate the deployable gate uses, in preference to a fresh
    `run_test_in_container`, because the cold path gave the same test a strictly
    worse deal: 120s total (covering container start, `pip install pytest` and
    `git apply`) against the 180s of warm test time the agent's own `run_test`
    got, plus a hand-built unified diff that can fail to apply and score L0. A
    BRT could therefore be rejected on time or patch mechanics rather than on
    whether it reproduces the bug.
    """
    rel = brt_test_path(session.repo)
    code = inject_setup_preamble(session.repo, submission)
    session.write_file(f"/testbed/{rel}", code)
    session.ensure_pytest()
    res = session.exec(f"python -m pytest {rel} -x --tb=short -q", timeout=180)
    outcome = _parse_pytest_outcome(res.returncode, res.output)
    reason = _parse_failure_reason(res.output) if outcome == "fail" else None
    return ContainerResult(outcome, reason, res.output[-3000:])


def _brt_passes_in_session(session: ContainerSession, brt_code: str) -> bool:
    """Gold-free success check for the fix loop: does the BRT PASS on the current
    (edited) tree? Buggy checkout + the agent's own edits only — no gold patch."""
    rel = brt_test_path(session.repo)
    code = inject_setup_preamble(session.repo, brt_code)
    session.write_file(f"/testbed/{rel}", code)
    session.ensure_pytest()
    res = session.exec(f"python -m pytest {rel} -x --tb=short -q", timeout=180)
    return _parse_pytest_outcome(res.returncode, res.output) == "pass"


@dataclass
class BrtCandidate:
    """One generated candidate BRT + its gold-free (buggy-tree) outcome."""

    index: int
    submission: str | None
    outcome_buggy: str | None
    failure_reason: str | None
    viable: bool  # clean assertion-fail on buggy (the gate criterion)
    agent: AgentResult


@dataclass
class BestOfNResult:
    """Outcome of best-of-N BRT generation + selection. `brt` is a normal
    BrtPhaseResult for the CHOSEN candidate (so the usual record/transcript
    builders apply); None if no candidate was viable."""

    brt: BrtPhaseResult | None
    candidates: list[BrtCandidate]
    chosen_index: int | None
    selection: Selection | None
    n_viable: int
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float | None


def run_brt_best_of_n(
    inst: SWEBenchInstance,
    client: LLMClient,
    *,
    n_candidates: int,
    max_iterations: int,
    max_total_tokens: int,
    max_tokens_per_turn: int = 4096,
    selector_client: LLMClient,
    prompt_harden: bool = False,
    submit_nudge_limit: int = 2,
    selector_max_iterations: int = 8,
    keep_container: bool = False,
    verbose: bool = False,
    score_fixed: bool = False,
) -> BestOfNResult | None:
    """Generate `n_candidates` BRTs (gold-free), keep only the clean-assertion-fail
    ones, let a Selector pick the best, then optionally gold-barrier-score the
    CHOSEN candidate for the scoreboard. Returns None if the container won't start.

    Fully gold-free in the loop: candidate scoring, the viability filter and the
    selection all use the buggy tree only. `score_fixed` governs ONLY the offline
    scoreboard of the chosen test (L0-L4); no decision ever sees the fixed tree.
    """
    test_path = brt_test_path(inst.repo)
    env_hint = repo_test_hint(inst.repo) or "(no special framework setup for this repo)"
    system = BRT_SYSTEM_PROMPT.format(repo=inst.repo, base_commit=inst.base_commit, env_hint=env_hint)
    if prompt_harden:
        system += PROMPT_HARDEN
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for instance `{ctx['instance_id']}` "
        f"({ctx['repo']} @ {ctx['base_commit']}):\n\n{ctx['problem_statement']}"
    )

    try:
        session = ContainerSession(inst, keep=keep_container).start()
    except RuntimeError as e:
        print(f"  [skip] {e}")
        return None

    candidates: list[BrtCandidate] = []
    tot_in = tot_out = 0
    selection: Selection | None = None
    chosen_index: int | None = None
    chosen_submission: str | None = None
    chosen_agent: AgentResult | None = None
    try:
        # --- GENERATE N candidates (gold-free), score each on the BUGGY tree ---
        for i in range(n_candidates):
            agent = Agent(
                client, brt_agent_tools(session), system,
                max_iterations=max_iterations, max_total_tokens=max_total_tokens,
                max_tokens_per_turn=max_tokens_per_turn,
                submit_nudge_limit=submit_nudge_limit, verbose=verbose,
            )
            print(f"Generating candidate {i + 1}/{n_candidates} ...")
            result = agent.run(task)
            tot_in += result.total_input_tokens
            tot_out += result.total_output_tokens
            sub = result.submission
            res = _score_on_buggy_in_session(session, sub) if sub and sub.strip() else None
            outcome, reason = (
                (res.outcome, res.failure_reason) if res is not None else (None, None)
            )
            # `!= "runtime"`, not `== "assertion"`: a bare `assert x == y` with no
            # message emits no `E AssertionError:` line, so reason is None while the
            # test is still a valid reproduction. Matches `reproduces_on_buggy`.
            viable = outcome == "fail" and reason != "runtime"
            print(f"  candidate {i}: submit={'yes' if sub else 'no'}  "
                  f"buggy={outcome} reason={reason}  viable={viable}")
            candidates.append(BrtCandidate(i, sub, outcome, reason, viable, result))

        viable = [c for c in candidates if c.viable]
        print(f"Viable (clean assertion-fail on buggy): {len(viable)}/{n_candidates}")

        # --- SELECT among viable candidates (gold-free) ---
        if len(viable) == 1:
            chosen_index = viable[0].index
            print(f"  only 1 viable candidate -> index {chosen_index} (selector skipped)")
        elif len(viable) >= 2:
            selection = run_selector(
                inst, session, selector_client,
                candidates=[c.submission for c in viable],
                max_iterations=selector_max_iterations, max_total_tokens=max_total_tokens,
                verbose=verbose,
            )
            tot_in += selection.agent.total_input_tokens
            tot_out += selection.agent.total_output_tokens
            chosen_index = viable[selection.index].index  # viable-local -> global index
            print(f"  selector chose viable#{selection.index} -> candidate index {chosen_index}")
        if chosen_index is not None:
            chosen_submission = candidates[chosen_index].submission
            chosen_agent = candidates[chosen_index].agent
    finally:
        session.close()

    cost = cost_usd(client.name, tot_in, tot_out)

    # =================== GOLD BARRIER (scoreboard only) =================== #
    brt: BrtPhaseResult | None = None
    if chosen_submission:
        test_code = inject_setup_preamble(inst.repo, chosen_submission)
        test_patch = make_test_patch(test_path, test_code)
        print("Scoring CHOSEN candidate offline — BUGGY tree ...")
        outcome_buggy = run_test_in_container(inst, test_patch, test_path=test_path)
        outcome_fixed = None
        classification = None
        if score_fixed:
            print("Scoring CHOSEN candidate offline — gold-FIXED tree ...")
            outcome_fixed = run_test_in_container(
                inst, test_patch, gold_patch=inst.gold_patch, test_path=test_path
            )
            classification = classify(outcome_buggy, outcome_fixed)
            print(f"  -> {classification.level_label}  (valid BRT: {classification.is_valid_brt})")
        brt = BrtPhaseResult(
            agent=chosen_agent, system=system, task=task,
            submission=chosen_submission, test_code=test_code,
            outcome_buggy=str(outcome_buggy) if outcome_buggy is not None else None,
            outcome_fixed=str(outcome_fixed) if outcome_fixed is not None else None,
            classification=classification, cost_usd=cost, submit_nudge_limit=submit_nudge_limit,
            failure_reason_buggy=getattr(outcome_buggy, "failure_reason", None),
            reviews=[], total_input_tokens=tot_in, total_output_tokens=tot_out,
        )
    else:
        print("No viable candidate produced — nothing to hand to the fix phase.")

    return BestOfNResult(
        brt=brt, candidates=candidates, chosen_index=chosen_index, selection=selection,
        n_viable=len(viable), total_input_tokens=tot_in, total_output_tokens=tot_out, cost_usd=cost,
    )


# ── Phase B: fix agent ────────────────────────────────────────────────────────

FIX_SYSTEM_PROMPT = """You are an autonomous bug-fixing agent. Your single goal is to FIX the bug \
described in a GitHub issue by editing the source code of `{repo}` (checked out at the buggy commit \
`{base_commit}`).

You are given a bug-reproduction test (BRT) to HELP you: it currently fails because of the bug, so it \
is a useful signal of progress. But the BRT is a GUIDE, not the definition of success — it may be \
imperfect (a wrong assertion, or it may error for reasons unrelated to the bug). Your real target is \
the ROOT CAUSE described in the issue, not merely making the test pass.

Recommended workflow (you decide the specifics and when you are done):
1. Read the issue and the BRT. Use glob/grep/read_file to locate the buggy code.
2. Call run_brt to see the test fail and understand the failure.
3. Use edit_file to apply a root-cause fix to the source.
4. Call run_brt again. Iterate edit -> run_brt; normally you are on track once it reports outcome=pass.
5. Use run_tests on the SPECIFIC existing test file(s) closest to your change (e.g. the test module \
for the file you edited) to check you did not break related behaviour. Do NOT run whole test \
directories — they are slow and full of failures unrelated to your change.
6. Submit once your fix addresses the issue (see the submit rules below).

Rules:
- Fix the ROOT CAUSE in the source. NEVER edit the test to make it pass — the test is fixed and you \
cannot change it (run_brt always uses the original BRT).
- Keep the change minimal and targeted; do not reformat unrelated code.
- DEFAULT: call submit_fix once run_brt reports outcome=pass and your change addresses the issue. \
Making the BRT pass is your PRIMARY success signal — aim for it.
- BAIL-OUT (use sparingly): if, after genuine effort, you have concrete evidence the BRT itself is \
flawed (it asserts a wrong value, or errors for reasons unrelated to the bug) and so cannot pass even \
with a correct fix, then fix the actual root cause from the issue and submit your best fix anyway, \
briefly stating why the BRT could not pass. Do NOT bail out just because the fix is hard — only when \
the BRT is demonstrably wrong.
- IMPORTANT: existing tests in this repo may already fail for reasons unrelated to you (environment, \
network, pre-existing flakiness). A failure reported by run_tests does NOT by itself mean your fix is \
wrong. Only treat it as a regression if you can point to a specific test that your edit plausibly broke.
- Do NOT call revert on a fix that makes the BRT pass unless you have clearly identified a regression \
YOUR change introduced. When unsure, keep the fix and submit. revert discards ALL your edits.
- Call submit_fix EXACTLY ONCE.
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
    # Phase C (multi-agent). Empty/None in the single-agent path.
    reviews: list = None  # type: ignore[assignment]
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    breaks: list = None  # type: ignore[assignment]  # post-fix breaker verdicts


def run_fix_phase(
    inst: SWEBenchInstance,
    brt_code: str,
    client: LLMClient,
    *,
    split: str,
    max_iterations: int,
    max_total_tokens: int,
    max_tokens_per_turn: int = 4096,
    keep_container: bool = False,
    verbose: bool = False,
    run_dir: Path = Path("runs") / "agent_fix",
    reviewer_client: LLMClient | None = None,
    reviewer_rounds: int = 0,
    reviewer_max_iterations: int = 8,
    breaker_client: LLMClient | None = None,
    breaker_rounds: int = 1,
    breaker_max_iterations: int = 8,
    max_attempts: int = 1,
) -> FixPhaseResult | None:
    """Run the fix agent (gold-free, BRT baked into run_brt) then score the fix
    patch through the gold barrier (held-out test_patch via the harness).

    `max_attempts > 1` adds a gold-free RETRY loop: after each attempt, if the BRT
    does not pass on the (edited) tree, the tree is reverted and the agent tries
    again with a hint about what the previous attempt did — up to `max_attempts`
    tries. The retry decision uses only the gold-free run_brt-pass signal; the
    held-out test is never consulted inside the loop.

    Phase C (multi-agent): pass `reviewer_client` + `reviewer_rounds > 0` to add a
    Reviewer that judges the fix's minimality/root-cause before it is accepted.
    `reviewer_client=None` (default) is the single-agent ablation baseline (RQ3).

    Phase C (post-fix Breaker): pass `breaker_client` to add an adversarial verifier
    that runs AFTER the fix (and any Reviewer) and tries to defeat it gold-free —
    passing the BRT is necessary-not-sufficient. On a 'broken' verdict the fix agent
    re-runs with the breaker's failing case (bounded by `breaker_rounds`); the
    verdict is always recorded (a gold-free confidence signal). `breaker_client=None`
    (default) skips the breaker entirely, keeping the single-agent path identical.

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
    reviews: list[Review] = []
    breaks: list[Break] = []
    tot_in = tot_out = 0
    tot_cw = tot_cr = 0
    try:
        # Gold-free RETRY loop: try up to max_attempts, reverting between failures.
        result = None
        fix_patch = ""
        attempt_hint = ""
        for attempt in range(max_attempts):
            agent = Agent(
                client, fix_agent_tools(session, brt_code), system,  # gold-free; BRT baked into run_brt
                max_iterations=max_iterations,
                max_total_tokens=max_total_tokens,
                max_tokens_per_turn=max_tokens_per_turn,
                verbose=verbose,
            )
            print(f"Running fix agent (attempt {attempt + 1}/{max_attempts}, "
                  f"max_iter={max_iterations}, max_tok={max_total_tokens}) ...")
            result = agent.run(task + attempt_hint)
            # Capture the fix patch = tracked source edits (the untracked BRT is excluded).
            fix_patch = session.exec("git diff").output
            tot_in += result.total_input_tokens
            tot_out += result.total_output_tokens
            tot_cw += result.total_cache_write_tokens
            tot_cr += result.total_cache_read_tokens
            if attempt == max_attempts - 1:
                break  # last (or only) attempt — keep whatever we have
            # Gold-free accept check: BRT passes on the edited tree AND a patch exists.
            if fix_patch.strip() and _brt_passes_in_session(session, brt_code):
                print(f"  attempt {attempt + 1}: BRT passes (gold-free) — accepting this fix.")
                break
            attempt_hint = (
                f"\n\n--- Your previous attempt (#{attempt + 1}) did NOT make the BRT pass ---\n"
                + (f"It changed:\n{fix_patch}\n" if fix_patch.strip() else "It produced no source edits.\n")
                + "Take a DIFFERENT approach this time; the same edit will not work."
            )
            session.reset_tree()  # revert to clean buggy tree before retrying
            print(f"  attempt {attempt + 1}: BRT not passing — reverting and retrying.")

        # Phase C: Reviewer judges the candidate fix; on 'revise' the worker
        # re-runs with the feedback (bounded by reviewer_rounds). Gold-free.
        for rnd in range(reviewer_rounds if reviewer_client is not None else 0):
            if not fix_patch.strip():
                break  # no fix to review
            review = run_reviewer(
                inst, session, reviewer_client,
                brt_code=brt_code, fix_diff=fix_patch,
                max_iterations=reviewer_max_iterations,
                max_total_tokens=max_total_tokens,
                verbose=verbose,
            )
            reviews.append(review)
            tot_in += review.agent.total_input_tokens
            tot_out += review.agent.total_output_tokens
            print(f"  Reviewer round {rnd + 1}: verdict={review.verdict}")
            if review.accepted:
                break
            revise_task = (
                f"{task}\n\n--- A reviewer examined your fix and asked for revisions ---\n"
                f"Reviewer feedback: {review.feedback}\n\n"
                "Address this feedback by editing the source. Keep run_brt passing, then submit_fix."
            )
            agent = Agent(
                client, fix_agent_tools(session, brt_code), system,
                max_iterations=max_iterations,
                max_total_tokens=max_total_tokens,
                max_tokens_per_turn=max_tokens_per_turn,
                verbose=verbose,
            )
            result = agent.run(revise_task)
            fix_patch = session.exec("git diff").output
            tot_in += result.total_input_tokens
            tot_out += result.total_output_tokens
            tot_cw += result.total_cache_write_tokens
            tot_cr += result.total_cache_read_tokens

        # Phase C (post-fix Breaker): an adversarial verifier attacks the fix even
        # though the BRT passes (necessary-not-sufficient). It always runs at least
        # once when enabled — the verdict is a gold-free confidence signal — and a
        # 'broken' verdict triggers up to `breaker_rounds` re-fix rounds (0 =
        # confidence-only, no re-fix). Gold-free: the breaker probes the agent's own
        # fixed tree, never the gold patch or held-out tests.
        refixes = 0
        while breaker_client is not None and fix_patch.strip():
            brk = run_breaker(
                inst, session, breaker_client,
                brt_code=brt_code, fix_diff=fix_patch,
                max_iterations=breaker_max_iterations,
                max_total_tokens=max_total_tokens,
                verbose=verbose,
            )
            breaks.append(brk)
            tot_in += brk.agent.total_input_tokens
            tot_out += brk.agent.total_output_tokens
            print(f"  Breaker round {len(breaks)}: verdict={brk.verdict}")
            if not brk.broken or refixes >= breaker_rounds:
                break  # robust, or out of re-fix budget — accept the current fix
            break_task = (
                f"{task}\n\n--- An adversarial verifier defeated your fix ---\n"
                f"It found a case your fix still gets wrong: {brk.feedback}\n\n"
                "Address this case by editing the source (keep run_brt passing), then submit_fix."
            )
            agent = Agent(
                client, fix_agent_tools(session, brt_code), system,
                max_iterations=max_iterations,
                max_total_tokens=max_total_tokens,
                max_tokens_per_turn=max_tokens_per_turn,
                verbose=verbose,
            )
            result = agent.run(break_task)
            fix_patch = session.exec("git diff").output
            tot_in += result.total_input_tokens
            tot_out += result.total_output_tokens
            tot_cw += result.total_cache_write_tokens
            tot_cr += result.total_cache_read_tokens
            refixes += 1
    finally:
        session.close()

    cost = cost_usd(client.name, tot_in, tot_out,
                    cache_write_tok=tot_cw, cache_read_tok=tot_cr)
    print(
        f"Fix agent done: stop={result.stop_cause}  iters={result.iterations}  "
        f"tokens={tot_in}/{tot_out}"
        + (f"  cost=${cost}" if cost is not None else "  cost=$0 (local)")
        + f"  patch={len(fix_patch)} chars"
        + (f"  reviewer_rounds={len(reviews)}" if reviews else "")
        + (f"  breaker={'/'.join(b.verdict for b in breaks)}" if breaks else "")
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
        reviews=reviews,
        total_input_tokens=tot_in,
        total_output_tokens=tot_out,
        total_cache_write_tokens=tot_cw,
        total_cache_read_tokens=tot_cr,
        breaks=breaks,
    )
