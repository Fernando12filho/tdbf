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
from .tools import critic_agent_tools, reviewer_agent_tools, selector_agent_tools

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
