"""Scouts architecture for Phase A: three investigators in parallel, an organizer,
then the BRT agent with a second gate (the failure must match the issue).

    issue ─┬─ Template Generator ─┐   (parallel; each its own container of the
           ├─ Edit Localizer ─────┼─  same image and its own model client)
           └─ Issue Reproducer ───┘
                     │
                 Organizer          (no tools: merges the three reports into a brief)
                     │
                 BRT agent + brief  (pipeline.run_brt_phase, unchanged tools)
                     │
           gate 1: fails on buggy at its assertion        (as before)
           gate 2: the failure matches the issue (judge)  (retries with feedback)

Each role takes its own LLMClient, so roles can run on different models; the
default is the same model for every role. One client OBJECT per role even when
the spec is shared: OpenRouter bills per client, so each role's cost stays its own.

GOLD-FREE: scouts get the BRT agent's buggy-tree tools (assert_scout_gold_free),
the organizer and judge get no environment tool at all. Nothing here sees the gold
patch or the held-out tests; the offline label stays behind run_brt_phase's barrier.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..data.instances import SWEBenchInstance
from ..harness.runner import brt_env_hint
from ..models.llm import LLMClient
from .pipeline import billed_mark, phase_cost, run_brt_phase
from .runtime import Agent, AgentResult
from .session import ContainerSession
from .tools import organizer_agent_tools, scout_agent_tools

ROLES = ("template", "localizer", "reproducer", "organizer", "writer", "judge")
SCOUT_ROLES = ("template", "localizer", "reproducer")

_COMMON = """
You are autonomous: never address a human or ask anyone to do something for you; use your tools. \
Tool arguments must be a valid JSON object with code in a JSON string. You are working on the BUGGY \
code at commit `{base_commit}`; the bug is NOT fixed and you cannot see a fixed version. You do not \
fix the bug and you do not write the final test: other agents do. Be efficient; you have a limited \
number of steps. Finish by calling your submit tool."""

TEMPLATE_SYSTEM = """You are the Test Template Generator for `{repo}`. Your job: find out how tests are \
written and run in this repository for the code the issue is about, and produce a minimal test module \
that PASSES on the current code. It must be independent of the bug: it only shows how to import, set up \
and call that code in a test (for example, construct the object the issue uses and assert something \
trivially true about it).

How:
1. Find the existing tests closest to the code in the issue (glob/grep), read one or two.
2. Write a minimal template in the same style (imports, fixtures, base classes, helpers).
3. Run it with run_test until it reports outcome=pass. run_test's description talks about failing \
tests; ignore that, your template must PASS.

Environment: {env_hint}

Submit with submit_template: the template module (as it passed), and short notes on the conventions a \
test writer must follow here and any setup pitfall you hit.""" + _COMMON

LOCALIZER_SYSTEM = """You are the Edit Localizer for `{repo}`. Your job: find the code responsible for \
the behaviour the issue describes.

Report:
- the code locations (path, line range, function or class) where the wrong behaviour is produced, with \
one line each on why;
- the public entry point a test should call to trigger it (an exact, runnable call);
- the existing test files that cover this code.

Use glob/grep/read_file; use run_python to confirm the call path when it is not obvious. Do not guess \
locations you have not read.

Submit with submit_locations.""" + _COMMON

REPRODUCER_SYSTEM = """You are the Issue Reproducer for `{repo}`. Your job: run the issue's example on \
this buggy code and report exactly what happens.

How:
1. Take the code from the issue (or the smallest script that does what the issue describes) and run \
it with run_python. Fix only what is needed to run it here (imports, setup); keep its behaviour.
2. Compare what you observe with what the issue reports.

Report:
- the script you ran;
- the observed output, copied verbatim (values, messages, traceback);
- whether it shows the bug the issue describes (yes / no / partly, and why);
- the EXPECTED correct behaviour as the issue states it, quoted from the issue. If the issue does not \
state it, write "UNSTATED" and quote what the issue does say. Never invent an expected value.

Environment: {env_hint}

Submit with submit_reproduction.""" + _COMMON

ORGANIZER_SYSTEM = """You are the Organizer. Three investigators studied a GitHub issue in `{repo}` (at \
the buggy commit) and wrote reports: a Test Template Generator (how tests are written and run here), an \
Edit Localizer (where the bug lives) and an Issue Reproducer (what the buggy code does now). You have no \
tools; you only read. Merge their findings into one TEST BRIEF for the agent who will write the bug \
reproduction test.

Write the brief with these sections:
1. How to write and run the test here (from the template: imports, setup, conventions, pitfalls).
2. What to call (the entry point, and the code locations involved).
3. Observed buggy behaviour (verbatim from the reproducer's output).
4. Expected correct behaviour (quote the issue; write UNSTATED if the issue does not say).
5. The assertion to make: what the test should check, so that it fails now and passes once fixed.
6. Conflicts and uncertainties between the reports, or gaps none of them covered.

Use only what the issue and the reports say. Do not invent values, names or APIs. Keep it short: the \
test writer will read the issue too.

Submit with submit_brief."""


@dataclass
class RoleRun:
    """One role's agent run (scout or organizer)."""

    role: str
    model: str
    report: str | None
    agent: AgentResult | None
    cost_usd: float | None
    seconds: float
    error: str | None = None

    def record(self) -> dict:
        a = self.agent
        return {
            "role": self.role, "model": self.model, "submitted": bool(self.report),
            "stop_cause": a.stop_cause if a else None,
            "iterations": a.iterations if a else 0,
            "input_tokens": a.total_input_tokens if a else 0,
            "output_tokens": a.total_output_tokens if a else 0,
            "cache_write_tokens": a.total_cache_write_tokens if a else 0,
            "cache_read_tokens": a.total_cache_read_tokens if a else 0,
            "cost_usd": self.cost_usd, "seconds": round(self.seconds, 1),
            "report_chars": len(self.report or ""), "error": self.error,
        }


@dataclass
class ScoutsResult:
    """The three scout runs and the organizer's brief."""

    scouts: dict[str, RoleRun]
    organizer: RoleRun
    seconds: float
    brief: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float | None:
        costs = [r.cost_usd for r in (*self.scouts.values(), self.organizer)]
        known = [c for c in costs if c is not None]
        return round(sum(known), 6) if known else None


def _system(role: str, inst: SWEBenchInstance) -> str:
    env_hint = brt_env_hint(inst.repo) or "(no special framework setup for this repo)"
    tpl = {"template": TEMPLATE_SYSTEM, "localizer": LOCALIZER_SYSTEM,
           "reproducer": REPRODUCER_SYSTEM, "organizer": ORGANIZER_SYSTEM}[role]
    return tpl.format(repo=inst.repo, base_commit=inst.base_commit, env_hint=env_hint)


def _issue_task(inst: SWEBenchInstance) -> str:
    ctx = inst.context_safe_dict()
    return (f"GitHub issue for instance `{ctx['instance_id']}` "
            f"({ctx['repo']} @ {ctx['base_commit']}):\n\n{ctx['problem_statement']}")


def _cost(client: LLMClient, mark, result: AgentResult | None) -> float | None:
    """Same billing helper as every other phase, so costs are comparable."""
    if result is None:
        return None
    return phase_cost(client, mark, result.total_input_tokens, result.total_output_tokens,
                      cache_write_tok=result.total_cache_write_tokens,
                      cache_read_tok=result.total_cache_read_tokens)


def run_scout(inst: SWEBenchInstance, role: str, client: LLMClient, *,
              max_iterations: int, max_total_tokens: int, max_tokens_per_turn: int,
              verbose: bool = False) -> RoleRun:
    """One scout in its OWN container (same image), so parallel scouts never write
    over each other's probe or test files. Never raises: a scout that cannot start
    or crashes returns an empty report and the organizer works without it."""
    t0 = time.monotonic()
    mark = billed_mark(client)
    try:
        session = ContainerSession(inst).start()
    except RuntimeError as e:
        return RoleRun(role, client.name, None, None, None, time.monotonic() - t0,
                       error=f"container: {e}")
    try:
        agent = Agent(client, scout_agent_tools(session, role), _system(role, inst),
                      max_iterations=max_iterations, max_total_tokens=max_total_tokens,
                      max_tokens_per_turn=max_tokens_per_turn, verbose=verbose)
        result = agent.run(_issue_task(inst))
    except Exception as e:  # noqa: BLE001 - one scout must not sink the others
        return RoleRun(role, client.name, None, None, None, time.monotonic() - t0,
                       error=f"{type(e).__name__}: {e}")
    finally:
        session.close()
    report = result.submission.strip() if result.submission and result.submission.strip() else None
    return RoleRun(role, client.name, report, result, _cost(client, mark, result),
                   time.monotonic() - t0)


def run_organizer(inst: SWEBenchInstance, client: LLMClient, scouts: dict[str, RoleRun], *,
                  max_iterations: int = 3, max_total_tokens: int = 300_000,
                  max_tokens_per_turn: int = 8192, verbose: bool = False) -> RoleRun:
    t0 = time.monotonic()
    mark = billed_mark(client)
    titles = {"template": "Test Template Generator", "localizer": "Edit Localizer",
              "reproducer": "Issue Reproducer"}
    parts = [_issue_task(inst)]
    for role in SCOUT_ROLES:
        r = scouts.get(role)
        body = (r.report if r and r.report else
                "(no report: this investigator did not finish"
                + (f" — {r.error}" if r and r.error else "") + ")")
        parts.append(f"--- Report from the {titles[role]} ---\n{body}")
    parts.append("Write the test brief, then call submit_brief.")
    agent = Agent(client, organizer_agent_tools(), _system("organizer", inst),
                  max_iterations=max_iterations, max_total_tokens=max_total_tokens,
                  max_tokens_per_turn=max_tokens_per_turn, verbose=verbose)
    result = agent.run("\n\n".join(parts))
    report = result.submission.strip() if result.submission and result.submission.strip() else None
    return RoleRun("organizer", client.name, report, result, _cost(client, mark, result),
                   time.monotonic() - t0)


def run_scouts(inst: SWEBenchInstance, clients: dict[str, LLMClient], *,
               scout_max_iterations: int = 20, max_total_tokens: int = 1_000_000,
               max_tokens_per_turn: int = 8192, verbose: bool = False) -> ScoutsResult:
    """Run the three scouts in parallel, then the organizer. If the organizer
    produces no brief, the scouts' raw reports are passed on instead, so the test
    writer is never left with less than the scouts found."""
    t0 = time.monotonic()
    print(f"Scouts: {', '.join(SCOUT_ROLES)} in parallel (max_iter={scout_max_iterations}) ...")
    with ThreadPoolExecutor(max_workers=len(SCOUT_ROLES)) as pool:
        futs = {role: pool.submit(run_scout, inst, role, clients[role],
                                  max_iterations=scout_max_iterations,
                                  max_total_tokens=max_total_tokens,
                                  max_tokens_per_turn=max_tokens_per_turn, verbose=verbose)
                for role in SCOUT_ROLES}
        scouts = {role: f.result() for role, f in futs.items()}
    for role, r in scouts.items():
        print(f"  {role}: {'report' if r.report else 'NO report'}"
              f" ({r.agent.stop_cause if r.agent else r.error}, {r.seconds:.0f}s, ${r.cost_usd})")
    print("Organizer: merging the reports into a test brief ...")
    org = run_organizer(inst, clients["organizer"], scouts,
                        max_total_tokens=max_total_tokens,
                        max_tokens_per_turn=max_tokens_per_turn, verbose=verbose)
    res = ScoutsResult(scouts=scouts, organizer=org, seconds=time.monotonic() - t0)
    if org.report:
        res.brief = org.report
    else:
        res.notes.append("organizer gave no brief; passing the raw scout reports")
        raw = [f"[{role}]\n{r.report}" for role, r in scouts.items() if r.report]
        res.brief = "\n\n".join(raw) if raw else None
    print(f"  brief: {len(res.brief or '')} chars ({org.agent.stop_cause if org.agent else '-'})")
    return res


def run_scouts_brt_phase(inst: SWEBenchInstance, clients: dict[str, LLMClient], *,
                         max_iterations: int, max_total_tokens: int, max_tokens_per_turn: int,
                         scout_max_iterations: int = 20, match_retries: int = 3,
                         brt_retries: int = 2, use_scouts: bool = True,
                         verbose: bool = False):
    """The new Phase A. `use_scouts=False` is the match-only arm: the unchanged BRT
    agent plus gate 2, no scouts and no brief (separates the two changes).

    Returns (ScoutsResult | None, BrtPhaseResult | None)."""
    scouts = run_scouts(inst, clients, scout_max_iterations=scout_max_iterations,
                        max_total_tokens=max_total_tokens,
                        max_tokens_per_turn=max_tokens_per_turn,
                        verbose=verbose) if use_scouts else None
    phase = run_brt_phase(
        inst, clients["writer"], max_iterations=max_iterations,
        max_total_tokens=max_total_tokens, max_tokens_per_turn=max_tokens_per_turn,
        brt_retries=brt_retries, verbose=verbose,
        brief=scouts.brief if scouts else None,
        match_client=clients["judge"], match_retries=match_retries)
    return scouts, phase
