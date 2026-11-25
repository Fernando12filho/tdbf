"""Blackboard Phase A: scouts and the test writer work AT THE SAME TIME and share
findings on a structured board while they work (studies/scouts/plan_blackboard.md).

    Template ─┐
    Localizer ┼── read / post / ask / answer ──► BLACKBOARD ◄── read / post / ask ── Writer
    Reproducer┘                                  (fixed sections,               (check_test +
     (each its own container)                     evidence required)             judge v2 loop)

Differences from scouts v1/v2 (parallel, then an organizer merges the reports):
no organizer, the writer starts immediately, and findings flow both ways while
everyone works. Every non-terminal tool result tells an agent how many board
entries it has not read, so findings reach it without polling.

GOLD-FREE: the board holds only text the agents wrote from the issue and the
BUGGY tree; scouts keep their buggy-only tools, the writer keeps the v2 tools.
New allowlists in tools.py guard both registries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..data.instances import SWEBenchInstance
from ..harness.runner import brt_env_hint
from ..models.llm import LLMClient
from .pipeline import billed_mark, phase_cost, run_brt_phase
from .runtime import Agent, AgentResult
from .session import ContainerSession
from .tools import (
    GlobTool,
    GrepTool,
    ReadFileTool,
    RunPythonTool,
    RunTestTool,
    SubmitReportTool,
    Tool,
    ToolResult,
    _BrtModuleState,
    assert_bb_scout_gold_free,
)

SECTIONS = ("entry_point", "locations", "observed_behaviour", "expected_behaviour",
            "test_conventions", "test_status", "questions", "answers")
SCOUT_ROLES = ("template", "localizer", "reproducer")
_MAX_ENTRY = 2000


@dataclass
class Entry:
    id: int
    author: str
    section: str
    text: str
    evidence: str
    t: float                      # seconds since the board opened
    answers: int | None = None    # for 'answers': the question id
    to: str | None = None         # reactive: the role a routed question is for

    def render(self, limit: int | None = None) -> str:
        head = f"#{self.id} [{self.section}] by {self.author} (t={self.t:.0f}s)"
        if self.to:
            head += f", FOR {self.to}"
        if self.answers is not None:
            head += f", answering #{self.answers}"
        text, ev = self.text, self.evidence
        if limit and len(text) + len(ev) > limit:
            text = text[:limit] + ("…" if len(text) > limit else "")
            ev = ev[:max(0, limit - len(text))] + ("…" if ev else "")
        ev = f"\n  evidence: {ev}" if ev else ""
        return f"{head}\n  {text}{ev}"


class Board:
    """Thread-safe shared board. Each reader has its own read cursor."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self.entries: list[Entry] = []
        self._seen: dict[str, int] = {}
        self._t0 = time.monotonic()
        self.closed = False
        self.reads: dict[str, int] = {}
        self.soft_start_wait: float | None = None  # reactive: seconds the writer waited

    def post(self, author: str, section: str, text: str, evidence: str = "",
             answers: int | None = None, to: str | None = None) -> Entry:
        with self._cv:
            e = Entry(len(self.entries) + 1, author, section, text[:_MAX_ENTRY],
                      evidence[:_MAX_ENTRY], time.monotonic() - self._t0, answers, to)
            self.entries.append(e)
            self._cv.notify_all()
            return e

    def unseen(self, reader: str) -> int:
        with self._cv:
            return sum(1 for e in self.entries[self._seen.get(reader, 0):] if e.author != reader)

    def read(self, reader: str, *, everything: bool = False) -> list[Entry]:
        with self._cv:
            start = 0 if everything else self._seen.get(reader, 0)
            self._seen[reader] = len(self.entries)
            self.reads[reader] = self.reads.get(reader, 0) + 1
            return [e for e in self.entries[start:] if everything or e.author != reader]

    def wait(self, reader: str, timeout: float) -> None:
        """Block until an entry by someone else appears, the board closes, or timeout."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while not self.closed and self.unseen_locked(reader) == 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return
                self._cv.wait(left)

    def wait_until(self, predicate, timeout: float) -> bool:
        """Block until predicate(entries) holds or timeout; True if it held."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while not predicate(self.entries):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
            return True

    def unseen_locked(self, reader: str) -> int:
        return sum(1 for e in self.entries[self._seen.get(reader, 0):] if e.author != reader)

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()

    def render_all(self) -> str:
        with self._cv:
            return "\n\n".join(e.render() for e in self.entries) or "(the board is empty)"

    def open_questions(self) -> list[Entry]:
        with self._cv:
            answered = {e.answers for e in self.entries if e.answers is not None}
            return [e for e in self.entries if e.section == "questions" and e.id not in answered]

    def summary(self) -> dict:
        with self._cv:
            by_author: dict[str, int] = {}
            by_section: dict[str, int] = {}
            for e in self.entries:
                by_author[e.author] = by_author.get(e.author, 0) + 1
                by_section[e.section] = by_section.get(e.section, 0) + 1
            qs = [e for e in self.entries if e.section == "questions"]
            answered = {e.answers for e in self.entries if e.answers is not None}
            routed = [q for q in qs if q.to]
            return {"entries": len(self.entries), "by_author": by_author, "by_section": by_section,
                    "questions": len(qs), "answered": sum(1 for q in qs if q.id in answered),
                    "routed": len(routed),
                    "routed_answered": sum(1 for q in routed if q.id in answered),
                    "soft_start_wait": (round(self.soft_start_wait, 1)
                                        if self.soft_start_wait is not None else None),
                    "reads": dict(self.reads)}

    def log(self) -> list[dict]:
        with self._cv:
            return [e.__dict__.copy() for e in self.entries]


# ── board tools ──────────────────────────────────────────────────────────────

class _BoardTool(Tool):
    def __init__(self, board: Board, author: str) -> None:
        self.board = board
        self.author = author


class ReadBoardTool(_BoardTool):
    name = "read_board"
    description = ("Read the shared board: findings, questions and answers posted by the other agents "
                   "working on this issue right now. Returns the entries you have not read yet; "
                   "pass everything=true for the whole board.")
    input_schema = {"type": "object", "properties": {
        "everything": {"type": "boolean", "description": "Return the whole board, not only new entries."}}}

    def execute(self, arguments: dict) -> ToolResult:
        es = self.board.read(self.author, everything=bool(arguments.get("everything")))
        if not es:
            return ToolResult("(nothing new on the board)")
        return ToolResult("\n\n".join(e.render() for e in es))


class PostFindingTool(_BoardTool):
    name = "post_finding"
    description = ("Post a finding to the shared board so the other agents can use it at once. "
                   "`section` is one of: entry_point, locations, observed_behaviour, "
                   "expected_behaviour, test_conventions, test_status. `evidence` is REQUIRED: the "
                   "file:line you read, the output of a run you made, or a quote from the issue. "
                   "Post facts, not guesses; keep each finding short and specific.")
    input_schema = {"type": "object", "properties": {
        "section": {"type": "string", "enum": [s for s in SECTIONS if s not in ("questions", "answers")]},
        "text": {"type": "string", "description": "The finding."},
        "evidence": {"type": "string", "description": "file:line, run output, or issue quote."}},
        "required": ["section", "text", "evidence"]}

    def execute(self, arguments: dict) -> ToolResult:
        section = str(arguments.get("section", "")).strip()
        text = str(arguments.get("text", "")).strip()
        evidence = str(arguments.get("evidence", "")).strip()
        if section not in SECTIONS or section in ("questions", "answers"):
            return ToolResult(f"post_finding: section must be one of {SECTIONS[:6]}.", is_error=True)
        if not text or not evidence:
            return ToolResult("post_finding requires non-empty 'text' and 'evidence'.", is_error=True)
        e = self.board.post(self.author, section, text, evidence)
        return ToolResult(f"posted as #{e.id}")


class AskBoardTool(_BoardTool):
    name = "ask_board"
    description = ("Post a question for the other agents (e.g. 'what does X return for an empty "
                   "input?'). They see it and may answer; read_board shows the answers.")
    input_schema = {"type": "object", "properties": {
        "question": {"type": "string"}}, "required": ["question"]}

    def execute(self, arguments: dict) -> ToolResult:
        q = str(arguments.get("question", "")).strip()
        if not q:
            return ToolResult("ask_board requires a non-empty 'question'.", is_error=True)
        e = self.board.post(self.author, "questions", q)
        return ToolResult(f"question posted as #{e.id}")


class AnswerTool(_BoardTool):
    name = "answer"
    description = ("Answer a question on the board. `evidence` is REQUIRED (file:line, run output "
                   "or issue quote).")
    input_schema = {"type": "object", "properties": {
        "question_id": {"type": "integer"}, "text": {"type": "string"},
        "evidence": {"type": "string"}}, "required": ["question_id", "text", "evidence"]}

    def execute(self, arguments: dict) -> ToolResult:
        try:
            qid = int(arguments.get("question_id"))
        except (TypeError, ValueError):
            return ToolResult("answer requires an integer 'question_id'.", is_error=True)
        if not any(e.id == qid and e.section == "questions" for e in self.board.entries):
            return ToolResult(f"#{qid} is not a question on the board.", is_error=True)
        text = str(arguments.get("text", "")).strip()
        evidence = str(arguments.get("evidence", "")).strip()
        if not text or not evidence:
            return ToolResult("answer requires non-empty 'text' and 'evidence'.", is_error=True)
        e = self.board.post(self.author, "answers", text, evidence, answers=qid)
        return ToolResult(f"answer posted as #{e.id}")


class WaitForBoardTool(_BoardTool):
    name = "wait_for_board"
    description = ("Wait (up to `seconds`, max 60) until another agent posts something new, without "
                   "spending effort. Use it when you have nothing to do but may be asked a question.")
    input_schema = {"type": "object", "properties": {"seconds": {"type": "integer"}}}

    def execute(self, arguments: dict) -> ToolResult:
        try:
            secs = max(1, min(60, int(arguments.get("seconds") or 30)))
        except (TypeError, ValueError):
            secs = 30
        self.board.wait(self.author, secs)
        n = self.board.unseen(self.author)
        return ToolResult(f"{n} new entr{'y' if n == 1 else 'ies'} on the board" if n
                          else "nothing new yet")


class NoticeTool(Tool):
    """Wraps a non-terminal tool: appends the board notice (unread entries, open
    questions, or that the writer has finished) to its result."""

    def __init__(self, inner: Tool, board: Board, reader: str, *, scout: bool,
                 push: bool = False) -> None:
        self.inner = inner
        self.board = board
        self.reader = reader
        self.scout = scout
        self.push = push
        self.name = inner.name
        self.description = inner.description
        self.input_schema = inner.input_schema
        self.terminal = inner.terminal

    def execute(self, arguments: dict) -> ToolResult:
        res = self.inner.execute(arguments)
        notes = []
        if self.scout and self.board.closed:
            notes.append("The test writer has FINISHED. Stop now: call your finish tool.")
        elif self.push and self.inner.name != "read_board":
            # Reactive: the new entries themselves, not a count; marked read.
            new = [e for e in self.board.read(self.reader)
                   if self.scout or not (e.author == "system" and e.section == "test_status")]
            if new:
                shown, budget = [], 3000
                for e in new:
                    r = e.render(limit=500)
                    if budget - len(r) < 0:
                        shown.append(f"... {len(new) - len(shown)} more: call read_board(everything=true)")
                        break
                    shown.append(r)
                    budget -= len(r)
                notes.append("NEW ON THE BOARD:\n" + "\n\n".join(shown))
            if self.scout:
                mine = [e for e in self.board.open_questions() if e.to == self.reader]
                if mine:
                    notes.append("Questions FOR YOU (answer with `answer`): "
                                 + ", ".join(f"#{e.id}" for e in mine) + ".")
        else:
            n = self.board.unseen(self.reader)
            if n:
                notes.append(f"{n} unread board entr{'y' if n == 1 else 'ies'}: call read_board.")
            if self.scout:
                q = self.board.open_questions()
                if q:
                    notes.append("Open questions: " + ", ".join(f"#{e.id}" for e in q[:5]) + ".")
        if not notes:
            return res
        return ToolResult(res.content + "\n\n[board] " + " ".join(notes), is_error=res.is_error)


def board_tools(board: Board, author: str, *, scout: bool) -> list[Tool]:
    tools: list[Tool] = [ReadBoardTool(board, author), PostFindingTool(board, author),
                         AskBoardTool(board, author)]
    if scout:
        tools += [AnswerTool(board, author), WaitForBoardTool(board, author)]
    return tools


def with_notices(tools: list[Tool], board: Board, reader: str, *, scout: bool,
                 push: bool = False) -> list[Tool]:
    return [t if t.terminal else NoticeTool(t, board, reader, scout=scout, push=push)
            for t in tools]


# ── agents ───────────────────────────────────────────────────────────────────

_SCOUT_COMMON = """
How you work here:
- Four agents work on this issue AT THE SAME TIME and share a board: you, two other investigators, \
and the test writer, who is already writing the reproduction test. Post each finding with \
post_finding THE MOMENT you have it (short, specific, with evidence); do not save them for the end.
- Read the board when a tool result says there are unread entries; build on what others posted \
instead of repeating it. Answer open questions in your area with `answer`.
- When you have nothing to add, call wait_for_board instead of exploring further; when the board says \
the test writer has finished, or you are sure you have nothing more to contribute, call finish.
- You do not fix the bug and you do not write the final test. You work on the BUGGY code at \
`{base_commit}`; you cannot see a fixed version. Never address a human. Tool arguments must be valid \
JSON. Be efficient."""

_SCOUT_JOBS = {
    "template": """You are the Test Template Generator for `{repo}`. Find how tests are written and run \
here for the code the issue is about: imports, setup, fixtures, base classes. Check a minimal \
bug-independent test PASSES with run_test (its description talks about failing tests; a template must \
pass). Post what the writer needs under test_conventions (with the template as evidence).
Environment: {env_hint}""",
    "localizer": """You are the Edit Localizer for `{repo}`. Find the code that produces the behaviour \
in the issue. Post under entry_point the exact public call a test should make, and under locations the \
path:line and function of the code involved (one finding each). Do not post locations you have not \
read.""",
    "reproducer": """You are the Issue Reproducer for `{repo}`. Run the issue's example (or the \
smallest script doing what it describes) with run_python. Post the observed output VERBATIM under \
observed_behaviour (the script as evidence), and under expected_behaviour what the issue says should \
happen, quoting it as evidence; if the issue states no expected value, post that it is UNSTATED. Never \
invent an expected value.
Environment: {env_hint}""",
}

WRITER_BOARD_ADDENDUM = """

Shared board (this run):
- Three investigators started at the same time as you and post findings to a shared board: the \
entry point to call, code locations, the observed buggy behaviour, the expected behaviour (quoted from \
the issue), and how tests are written here. read_board shows what they posted; tool results tell you \
when there are unread entries.
- Start working right away; read the board when you need a fact or when there are new entries. If you \
need something specific, ask_board; post your test's current outcome under test_status so they can \
help. Verify what you rely on: findings can be wrong."""


REACTIVE_SCOUT_ADDENDUM = """
Reactive board (this run):
- New board entries appear in your tool results automatically.
- The writer's test results are posted to the board automatically. When its test is rejected or \
crashes, the system posts a question FOR a specific investigator. After posting your findings, do NOT \
finish: call wait_for_board and answer every question FOR YOU with `answer` (evidence required), \
running code to check when needed. Finish only when the board says the test writer has finished."""

REACTIVE_WRITER_ADDENDUM = """
- Reactive board: new entries appear in your tool results automatically. Every check_test result is \
posted to the board for you, and when the reviewer rejects your test or it crashes, the investigators \
are asked to help; their answers appear in your next tool results. Your first check_test may wait \
briefly for the investigators' observed and expected behaviour."""


@dataclass
class ScoutRun:
    role: str
    model: str
    agent: AgentResult | None
    cost_usd: float | None
    seconds: float
    error: str | None = None

    def record(self) -> dict:
        a = self.agent
        return {"role": self.role, "model": self.model,
                "stop_cause": a.stop_cause if a else None, "iterations": a.iterations if a else 0,
                "input_tokens": a.total_input_tokens if a else 0,
                "output_tokens": a.total_output_tokens if a else 0,
                "cost_usd": self.cost_usd, "seconds": round(self.seconds, 1), "error": self.error}


@dataclass
class BlackboardResult:
    board: Board
    scouts: dict[str, ScoutRun] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def scouts_cost_usd(self) -> float | None:
        known = [r.cost_usd for r in self.scouts.values() if r.cost_usd is not None]
        return round(sum(known), 6) if known else None


def scout_bb_tools(session: ContainerSession, role: str, board: Board,
                   reactive: bool = False) -> list[Tool]:
    tools: list[Tool] = [ReadFileTool(session), GrepTool(session), GlobTool(session),
                         RunPythonTool(session)]
    if role == "template":
        tools.append(RunTestTool(session, _BrtModuleState()))
    tools += board_tools(board, role, scout=True)
    tools.append(SubmitReportTool("finish", "Finish your work. Give a one-paragraph summary of "
                                            "what you posted. Call once.", field="summary"))
    tools = with_notices(tools, board, role, scout=True, push=reactive)
    assert_bb_scout_gold_free(tools)
    return tools


def _issue_task(inst: SWEBenchInstance) -> str:
    ctx = inst.context_safe_dict()
    return (f"GitHub issue for instance `{ctx['instance_id']}` "
            f"({ctx['repo']} @ {ctx['base_commit']}):\n\n{ctx['problem_statement']}")


def _run_bb_scout(inst, role, client, board, *, max_iterations, max_total_tokens,
                  max_tokens_per_turn, verbose, reactive: bool = False) -> ScoutRun:
    t0 = time.monotonic()
    mark = billed_mark(client)
    try:
        session = ContainerSession(inst).start()
    except RuntimeError as e:
        return ScoutRun(role, client.name, None, None, time.monotonic() - t0, f"container: {e}")
    try:
        env_hint = brt_env_hint(inst.repo) or "(no special framework setup for this repo)"
        system = (_SCOUT_JOBS[role] + _SCOUT_COMMON).format(
            repo=inst.repo, base_commit=inst.base_commit, env_hint=env_hint)
        if reactive:
            system += REACTIVE_SCOUT_ADDENDUM
        agent = Agent(client, scout_bb_tools(session, role, board, reactive), system,
                      max_iterations=max_iterations, max_total_tokens=max_total_tokens,
                      max_tokens_per_turn=max_tokens_per_turn, verbose=verbose)
        result = agent.run(_issue_task(inst))
    except Exception as e:  # noqa: BLE001 - one scout must not sink the others
        return ScoutRun(role, client.name, None, None, time.monotonic() - t0,
                        f"{type(e).__name__}: {e}")
    finally:
        session.close()
    cost = phase_cost(client, mark, result.total_input_tokens, result.total_output_tokens,
                      cache_write_tok=result.total_cache_write_tokens,
                      cache_read_tok=result.total_cache_read_tokens)
    return ScoutRun(role, client.name, result, cost, time.monotonic() - t0)


_ROUTE_CAP = 3  # routed questions per role per bug


def reactive_hooks(board: Board, *, soft_start: float = 90.0):
    """(before_check, on_check) for the writer's check_test (plan_reactive_blackboard.md):
    a one-time soft start, the result auto-posted, and problems routed to scouts."""
    state = {"started": False, "routed": {}}

    def have_repro(entries) -> bool:
        secs = {e.section for e in entries if e.author == "reproducer"}
        return {"observed_behaviour", "expected_behaviour"} <= secs

    def before_check() -> None:
        if state["started"]:
            return
        state["started"] = True
        t = time.monotonic()
        board.wait_until(have_repro, soft_start)
        board.soft_start_wait = time.monotonic() - t

    def route(role: str, text: str, evidence: str) -> None:
        if state["routed"].get(role, 0) >= _ROUTE_CAP:
            return
        state["routed"][role] = state["routed"].get(role, 0) + 1
        board.post("system", "questions", text, evidence, to=role)

    def on_check(info: dict) -> None:
        tail = (info.get("output") or "")[-600:]
        status = f"writer's check_test: outcome={info['outcome']}" + (
            f" failure_reason={info['failure_reason']}" if info.get("failure_reason") else "")
        if info.get("verdict"):
            status += f"; reviewer verdict={info['verdict']}"
        ev = tail + (f"\nreviewer: {info['feedback']}" if info.get("feedback") else "")
        board.post("system", "test_status", status, ev)
        if info.get("verdict") == "mismatch":
            route("reproducer",
                  "The reviewer rejected the writer's test: " + (info.get("feedback") or "")[:800]
                  + "\nRe-run the issue's example and post the correct expected behaviour under "
                    "expected_behaviour, quoting the issue (or UNSTATED if it does not say).",
                  tail)
        elif info["outcome"] == "error" or info.get("failure_reason") == "runtime":
            route("template", "The writer's test crashes before asserting (setup/import?). Post "
                              "the setup or import it needs under test_conventions.", tail)
            route("localizer", "The writer's test crashes before asserting (wrong call?). Post the "
                               "exact call that exercises the bug under entry_point.", tail)

    return before_check, on_check


def run_blackboard_brt_phase(inst: SWEBenchInstance, clients: dict[str, LLMClient], *,
                             max_iterations: int, max_total_tokens: int, max_tokens_per_turn: int,
                             scout_max_iterations: int = 50, brt_retries: int = 2,
                             max_mismatches: int = 3, scout_join_timeout: float = 300.0,
                             reactive: bool = False, soft_start: float = 90.0,
                             verbose: bool = False):
    """Scouts and writer concurrently around one board. `reactive=True` is the
    reactive blackboard: entries pushed into tool results, check_test results
    auto-posted, problems routed to scouts, a soft start. Returns
    (BlackboardResult, BrtPhaseResult | None)."""
    t0 = time.monotonic()
    board = Board()
    bb = BlackboardResult(board=board)
    before_check = on_check = None
    if reactive:
        before_check, on_check = reactive_hooks(board, soft_start=soft_start)
    print(f"{'Reactive blackboard' if reactive else 'Blackboard'}: {', '.join(SCOUT_ROLES)} "
          "+ writer, concurrently ...")
    threads = {}
    for role in SCOUT_ROLES:
        def target(role=role):
            bb.scouts[role] = _run_bb_scout(
                inst, role, clients[role], board, max_iterations=scout_max_iterations,
                max_total_tokens=max_total_tokens, max_tokens_per_turn=max_tokens_per_turn,
                verbose=verbose, reactive=reactive)
        th = threading.Thread(target=target, name=f"bb-{role}", daemon=True)
        th.start()
        threads[role] = th
    try:
        phase = run_brt_phase(
            inst, clients["writer"], max_iterations=max_iterations,
            max_total_tokens=max_total_tokens, max_tokens_per_turn=max_tokens_per_turn,
            brt_retries=brt_retries, verbose=verbose,
            writer_judge_client=clients["judge"], judge_context=board.render_all,
            max_mismatches=max_mismatches,
            writer_extra_tools=lambda: board_tools(board, "writer", scout=False),
            writer_tool_wrapper=lambda tools: with_notices(tools, board, "writer", scout=False,
                                                           push=reactive),
            writer_system_extra=WRITER_BOARD_ADDENDUM + (REACTIVE_WRITER_ADDENDUM if reactive else ""),
            writer_before_check=before_check, writer_on_check=on_check)
    finally:
        board.close()
        deadline = time.monotonic() + scout_join_timeout
        for role, th in threads.items():
            th.join(max(0.0, deadline - time.monotonic()))
            if th.is_alive():
                bb.scouts.setdefault(role, ScoutRun(role, clients[role].name, None, None,
                                                    time.monotonic() - t0,
                                                    "still running at the join deadline"))
    bb.seconds = time.monotonic() - t0
    s = board.summary()
    print(f"  board: {s['entries']} entries {s['by_author']}, questions {s['questions']} "
          f"(answered {s['answered']}, routed {s['routed']}/{s['routed_answered']} answered), "
          f"soft start {s['soft_start_wait']}s, reads {s['reads']}")
    for role, r in bb.scouts.items():
        print(f"  {role}: {r.agent.stop_cause if r.agent else r.error} "
              f"({r.seconds:.0f}s, ${r.cost_usd})")
    return bb, phase
