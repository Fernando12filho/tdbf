"""Multi-agent collaborator roles (Phase C, depth-light).

Two advisory roles, each a *real* Agent (model-driven, gold-free tools) with a
terminal `submit_review` verdict:

  - **Critic** (Phase A): reviews a candidate BRT for *specificity* before it is
    accepted. Targets the measured failure where an L4 BRT still under-specifies
    the bug (astropy 14182, 7746 resolve in Phase B only sometimes), and the weak
    model's habit of asserting the observed buggy output as "correct".
  - **Reviewer** (Phase B): reviews a candidate fix for *minimality / root-cause*
    before it is submitted, and suggests a targeted regression test. Targets the
    expensive-flail failure (django 11019).

A code orchestrator (in `pipeline.py`) sequences worker -> role -> revise, sharing
the worker's ContainerSession. Every role is gold-free: it reasons from the issue
and the buggy tree, never the gold patch or held-out tests.

Fail-open: a role that does not return a clean verdict defaults to ACCEPT, so a
flaky critic can never destroy a good artifact — it can only *add* revise signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..data.instances import SWEBenchInstance
from ..models.llm import LLMClient
from .runtime import Agent, AgentResult
from .session import ContainerSession
from .tools import (
    breaker_agent_tools,
    critic_agent_tools,
    match_agent_tools,
    reviewer_agent_tools,
    selector_agent_tools,
)

MATCH_SYSTEM_PROMPT = """You are a judge working on `{repo}`. Another agent wrote a Bug Reproduction \
Test (BRT) for a GitHub issue, and the test already FAILS on the buggy code. Your job is to decide \
whether it fails FOR THE REASON THE ISSUE DESCRIBES. You do not write or run anything: you read the \
issue, the test and the test's output on the buggy code.

Answer 'match' only if ALL three hold:
1. SCENARIO: the test exercises the behaviour the issue is about (the same function, feature and kind \
of input), not a neighbouring one.
2. SYMPTOM: the failure in the output shows the wrong behaviour the issue reports (the value, message or \
exception the issue says happens now), not an unrelated error.
3. EXPECTATION: what the test asserts as correct is what the issue says should happen, stated in the \
issue or directly implied by it. A value the test's author guessed or invented is a mismatch, even if \
it looks plausible.

If the issue reports an exception (a crash), a test that fails because the code under test raises that \
exception, or asserts it is not raised, satisfies SYMPTOM. If the issue states no exact expected value, \
EXPECTATION holds when the test asserts a property the issue says the correct behaviour must have.

Finish by calling submit_match:
- verdict='match' when all three hold;
- verdict='mismatch' otherwise, with feedback naming the condition that failed and what the test \
should assert instead, quoting the issue. Be strict about invented expected values, but do not ask for \
changes the issue gives no basis for."""


MATCH_SYSTEM_PROMPT_V2 = """You are a strict judge working on `{repo}`. Another agent is writing a Bug \
Reproduction Test (BRT) for a GitHub issue, and its current test FAILS on the buggy code. Decide whether \
it fails FOR THE REASON THE ISSUE DESCRIBES and asserts ONLY what the issue supports. You do not write or \
run anything: you read the issue, the test, its output on the buggy code and, when given, a brief from \
investigators who ran the issue's code (the brief may contain mistakes; the issue is the authority).

Answer 'match' only if ALL hold:
1. SCENARIO: the test exercises the behaviour the issue is about (same function, feature, kind of input).
2. SYMPTOM: the failure shows the wrong behaviour the issue reports, not an unrelated error.
3. SOURCED EXPECTATION: every expected value the test asserts comes from the issue (stated, or directly \
implied) or from an observed run of correct code described in the issue. A number, string or result the \
test's author worked out or guessed is NOT sourced.
4. NO OVER-SPECIFICATION: the test asserts nothing the issue does not require. In particular:
   - if the issue says something "should fail/raise/error" without naming the exception, the test must \
accept any exception (or the type the issue names), not pick one;
   - if the issue does not give an exact message, repr, format or SQL text, the test must not compare \
against an exact string; check the property the issue states instead (e.g. "contains the argument", \
"is a str", "does not raise");
   - if the issue only asks for a problem to go away (an error, a warning), the test must check that \
problem is gone, not that nothing else at all is reported.

If the issue reports a crash, a failure caused by that crash (or an assertion that it is not raised) \
satisfies SYMPTOM.

Finish by calling submit_match:
- verdict='match' when all four hold;
- verdict='mismatch' otherwise, with feedback naming the condition that failed and the weaker or \
corrected assertion the issue does support, quoting the issue."""


@dataclass
class MatchCheck:
    """The match judge's verdict on one gate-passing test, plus its run."""

    verdict: str  # "match" | "mismatch"
    feedback: str
    agent: AgentResult

    @property
    def matched(self) -> bool:
        return self.verdict == "match"


def _parse_match(result: AgentResult) -> tuple[str, str]:
    """(verdict, feedback) from submit_match. Fail-open to 'match': a judge that
    returns no clean verdict never discards a test that passed the gate."""
    if result.submission:
        try:
            data = json.loads(result.submission)
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in ("match", "mismatch"):
                return verdict, str(data.get("feedback", "")).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return "match", ""


def run_match_judge(
    inst: SWEBenchInstance,
    client: LLMClient,
    *,
    test_code: str,
    buggy_output: str,
    max_iterations: int = 3,
    max_total_tokens: int = 200_000,
    max_tokens_per_turn: int = 4096,
    verbose: bool = False,
    version: int = 1,
    context: str | None = None,
) -> MatchCheck:
    """Judge whether a gate-passing test fails for the issue's reason (gold-free:
    issue, test, BUGGY-tree output and, for v2, the investigators' brief; no
    environment tool). version=1 is the v1 judge, unchanged."""
    system = (MATCH_SYSTEM_PROMPT_V2 if version == 2 else MATCH_SYSTEM_PROMPT).format(repo=inst.repo)
    ctx = inst.context_safe_dict()
    out = buggy_output.strip()
    if len(out) > 6000:
        out = "...[earlier output clipped]\n" + out[-6000:]
    brief = ""
    if version == 2 and context and context.strip():
        c = context.strip()
        if len(c) > 6000:
            c = c[:6000] + "\n...[brief clipped]"
        brief = f"--- Investigators' brief (may contain mistakes) ---\n{c}\n\n"
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']}):\n\n{ctx['problem_statement']}\n\n"
        f"{brief}--- The test ---\n{test_code}\n\n"
        f"--- Its output on the BUGGY code (it fails) ---\n{out or '(no output captured)'}\n\n"
        "Decide whether it fails for the reason the issue describes, then call submit_match."
    )
    agent = Agent(client, match_agent_tools(), system, max_iterations=max_iterations,
                  max_total_tokens=max_total_tokens, max_tokens_per_turn=max_tokens_per_turn,
                  verbose=verbose)
    result = agent.run(task)
    verdict, feedback = _parse_match(result)
    return MatchCheck(verdict, feedback, result)

CRITIC_SYSTEM_PROMPT = """You are a meticulous test reviewer (a Critic) working on `{repo}` at the \
buggy commit. Another agent wrote a candidate Bug Reproduction Test (BRT) that already FAILS on the \
buggy code. Your job is to judge whether that test is SPECIFIC enough — not to write a test yourself.

A specific BRT:
- asserts the value a CORRECT fix would produce, reasoned from the issue — NOT merely "not the buggy \
output" and NOT the observed buggy value copied in as if it were correct;
- exercises the exact behaviour the issue describes, so a genuine fix makes it pass;
- could NOT be satisfied by a trivial or wrong change that doesn't actually fix the bug.

You have read-only tools (read_file, grep, glob) and run_python to check what the correct value should \
be. You are gold-free: you cannot see the fixed code; reason from the issue.

Finish by calling submit_review:
- verdict='accept' if the test specifically pins the bug;
- verdict='revise' with concrete, actionable feedback if it is under-specified, asserts the buggy value \
as correct, or could pass without a real fix. Be strict but fair — only ask for revision when a concrete \
weakness exists. Be efficient; you have a limited step budget."""

REVIEWER_SYSTEM_PROMPT = """You are a careful code reviewer working on `{repo}` at the buggy commit. \
Another agent produced a candidate FIX (shown as a diff) that makes the bug-reproduction test PASS. \
The edited source is already applied in the checkout, so read_file shows the fixed code. Your job is to \
judge the fix — not to edit code yourself.

A good fix:
- addresses the ROOT CAUSE described in the issue, not a superficial hack that only satisfies the test;
- is MINIMAL and targeted — no unrelated edits, no dead code, no reformatting;
- does not plausibly break related behaviour.

You have read-only tools (read_file, grep, glob) and run_tests to run ONE targeted existing test file as \
a regression check. You are gold-free: you cannot see the held-out tests; reason from the issue and the \
diff. Remember a run_tests failure may be pre-existing/environmental, not caused by this fix.

Finish by calling submit_review:
- verdict='accept' if the fix is a minimal, root-cause change;
- verdict='revise' with concrete feedback if it is a hack, non-minimal, or plausibly regressive. Be \
efficient; you have a limited step budget."""


BREAKER_SYSTEM_PROMPT = """You are an adversarial verifier (a Breaker) working on `{repo}` at the \
buggy commit. Another agent produced a candidate FIX (shown as a diff) that already makes the \
bug-reproduction test (BRT) PASS. But passing the BRT is NECESSARY, NOT SUFFICIENT: a fix can satisfy \
the test yet still be incomplete or wrong. Your job is to TRY TO DEFEAT this fix — not to edit code.

Reason from the ISSUE about what a genuinely correct fix must do, then attack. The BRT's exact \
scenario ALREADY passes — do NOT just re-run it. A fix is under-specified precisely on the cases the \
BRT does NOT cover, so deliberately probe those:
- the INVERSE / round-trip operation: if the fix adds or changes a writer, exercise the READER (and \
write->read->write round-trips); if it adds a getter, exercise the setter — any symmetry the issue \
implies must hold in BOTH directions, even when the BRT only checked one;
- VARY every parameter beyond the BRT's exact value: try more, fewer, zero, one, and other valid \
values (e.g. if the BRT passes a 2-element option, also try 3 elements, 1, an empty one, an unusual one);
- other inputs, edge cases, and related code paths the issue implies but the BRT never touches.
Use run_python to actually EXECUTE these on the ALREADY-APPLIED fix and observe what it does — never \
guess. Look for any input or behaviour where the fixed code still violates what the issue says is correct.

The fix is already applied in the checkout, so read_file shows the fixed source and run_python runs the \
fixed code. You are gold-free: you CANNOT see the held-out tests or the reference fix — reason only from \
the issue and the code.

Finish by calling submit_break:
- verdict='broken' if you found a concrete case where the fixed code is still wrong — put that exact \
failing case in `feedback` so the author can address it;
- verdict='robust' if, after genuine effort, you could not defeat the fix. Be a strict but honest \
adversary: only report 'broken' with a real, demonstrated failing case, never a vague doubt. Be \
efficient; you have a limited step budget."""


SELECTOR_SYSTEM_PROMPT = """You are a test selector working on `{repo}` at the buggy commit. Several \
candidate Bug Reproduction Tests (BRTs) were generated for the same GitHub issue; each already FAILS \
with a clean assertion on the buggy code. Your job is to pick the SINGLE best one — the test that will \
most reliably become a passing test once the bug is correctly fixed.

Prefer the candidate that:
- asserts the specific value a CORRECT fix would produce, reasoned from the issue (not a vague or \
overly broad assertion, and never the buggy value dressed up as correct);
- targets exactly the behaviour the issue describes, so a genuine fix makes it pass and a wrong/trivial \
change would not;
- is clear and self-contained.

You have read-only tools (read_file, grep, glob) and run_python to check what the correct value should \
be. You are gold-free: you cannot see the fixed code; reason from the issue. Finish by calling \
submit_selection with the 0-based index of your chosen candidate and a short reason. Be efficient."""


@dataclass
class Review:
    """A role's verdict on a worker artifact, plus the role agent's run (telemetry)."""

    verdict: str  # "accept" | "revise"
    feedback: str
    agent: AgentResult

    @property
    def accepted(self) -> bool:
        return self.verdict == "accept"


@dataclass
class Selection:
    """The Selector's choice among candidate BRTs, plus its agent run (telemetry)."""

    index: int
    reason: str
    agent: AgentResult


@dataclass
class Break:
    """The Breaker's verdict on a candidate fix, plus its agent run (telemetry)."""

    verdict: str  # "robust" | "broken"
    feedback: str
    agent: AgentResult

    @property
    def broken(self) -> bool:
        return self.verdict == "broken"


def _parse_verdict(result: AgentResult) -> tuple[str, str]:
    """Extract (verdict, feedback) from the submit_review artifact. Fail-open to
    accept if the role never returned a clean verdict (never block the worker)."""
    if result.submission:
        try:
            data = json.loads(result.submission)
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in ("accept", "revise"):
                return verdict, str(data.get("feedback", "")).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return "accept", ""


def _parse_break(result: AgentResult) -> tuple[str, str]:
    """Extract (verdict, feedback) from the submit_break artifact. Fail-open to
    'robust' if the breaker never returned a clean verdict — a flaky adversary must
    never discard a good fix; it can only ADD a 'broken' signal."""
    if result.submission:
        try:
            data = json.loads(result.submission)
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in ("robust", "broken"):
                return verdict, str(data.get("feedback", "")).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return "robust", ""


def run_critic(
    inst: SWEBenchInstance,
    session: ContainerSession,
    client: LLMClient,
    *,
    candidate_test: str,
    max_iterations: int,
    max_total_tokens: int,
    verbose: bool = False,
) -> Review:
    """Run the Critic over a candidate BRT (gold-free); return its verdict."""
    system = CRITIC_SYSTEM_PROMPT.format(repo=inst.repo)
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']}):\n\n{ctx['problem_statement']}\n\n"
        f"--- Candidate Bug Reproduction Test (fails on the buggy code) ---\n{candidate_test}\n\n"
        "Judge whether this test SPECIFICALLY pins the bug, then call submit_review."
    )
    agent = Agent(
        client, critic_agent_tools(session), system,
        max_iterations=max_iterations, max_total_tokens=max_total_tokens, verbose=verbose,
    )
    result = agent.run(task)
    verdict, feedback = _parse_verdict(result)
    return Review(verdict, feedback, result)


def run_reviewer(
    inst: SWEBenchInstance,
    session: ContainerSession,
    client: LLMClient,
    *,
    brt_code: str,
    fix_diff: str,
    max_iterations: int,
    max_total_tokens: int,
    verbose: bool = False,
) -> Review:
    """Run the Reviewer over a candidate fix diff (gold-free); return its verdict."""
    system = REVIEWER_SYSTEM_PROMPT.format(repo=inst.repo)
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']}):\n\n{ctx['problem_statement']}\n\n"
        f"--- Bug-reproduction test the fix must keep passing ---\n{brt_code}\n\n"
        f"--- Candidate fix (already applied in the checkout) ---\n{fix_diff}\n\n"
        "Judge whether this fix is a minimal, root-cause change, then call submit_review."
    )
    agent = Agent(
        client, reviewer_agent_tools(session), system,
        max_iterations=max_iterations, max_total_tokens=max_total_tokens, verbose=verbose,
    )
    result = agent.run(task)
    verdict, feedback = _parse_verdict(result)
    return Review(verdict, feedback, result)


def run_selector(
    inst: SWEBenchInstance,
    session: ContainerSession,
    client: LLMClient,
    *,
    candidates: list[str],
    max_iterations: int,
    max_total_tokens: int,
    verbose: bool = False,
) -> Selection:
    """Run the Selector over `candidates` (each a viable BRT that fails cleanly on
    the buggy tree); return its 0-based choice. Fails OPEN to index 0 (all inputs
    are already viable, so the first is a safe default) if the choice is invalid."""
    system = SELECTOR_SYSTEM_PROMPT.format(repo=inst.repo)
    ctx = inst.context_safe_dict()
    listing = "\n\n".join(
        f"=== Candidate index {i} ===\n{code}" for i, code in enumerate(candidates)
    )
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']}):\n\n{ctx['problem_statement']}\n\n"
        f"There are {len(candidates)} candidate BRTs, each failing cleanly on the buggy code:\n\n"
        f"{listing}\n\n"
        "Pick the single best candidate, then call submit_selection with its 0-based index."
    )
    agent = Agent(
        client, selector_agent_tools(session), system,
        max_iterations=max_iterations, max_total_tokens=max_total_tokens, verbose=verbose,
    )
    result = agent.run(task)
    index, reason = 0, ""
    if result.submission:
        try:
            data = json.loads(result.submission)
            idx = int(data.get("index"))
            if 0 <= idx < len(candidates):
                index, reason = idx, str(data.get("reason", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
    return Selection(index, reason, result)


def run_breaker(
    inst: SWEBenchInstance,
    session: ContainerSession,
    client: LLMClient,
    *,
    brt_code: str,
    fix_diff: str,
    max_iterations: int,
    max_total_tokens: int,
    verbose: bool = False,
) -> Break:
    """Run the post-fix Breaker over a candidate fix (gold-free); return its verdict.

    The fix is already applied in `session`, so the breaker's run_python probes the
    agent's own fixed tree. It never sees the gold patch or the held-out tests.
    """
    system = BREAKER_SYSTEM_PROMPT.format(repo=inst.repo)
    ctx = inst.context_safe_dict()
    task = (
        f"GitHub issue for `{ctx['instance_id']}` ({ctx['repo']}):\n\n{ctx['problem_statement']}\n\n"
        f"--- Bug-reproduction test the fix already makes pass ---\n{brt_code}\n\n"
        f"--- Candidate fix (already applied in the checkout) ---\n{fix_diff}\n\n"
        "Try to defeat this fix by finding a case it still gets wrong, then call submit_break."
    )
    agent = Agent(
        client, breaker_agent_tools(session), system,
        max_iterations=max_iterations, max_total_tokens=max_total_tokens, verbose=verbose,
    )
    result = agent.run(task)
    verdict, feedback = _parse_break(result)
    return Break(verdict, feedback, result)
