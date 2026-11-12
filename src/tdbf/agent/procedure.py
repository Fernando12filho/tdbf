"""The procedure agent: a controller written in Python around the model (design §0–§6).

The controller holds the current phase, decides which tools the model may call,
checks each phase's result with deterministic code (`procedure_gates`), and starts
each phase with a short handoff record instead of the whole transcript. A failed
check sends the run back to the phase that caused it. Every expected value the test
asserts must carry evidence the controller verified, or stay an unknown (D2: then it
appears only as a `# UNVERIFIED:` comment and the test is flagged).

    preflight (no model) -> understand -> reproduce -> pin_expected -> locate -> write -> finalise

Modes:
  strict  the procedure above, with gates and send-backs (V2);
  soft    the same steps and prompts as ONE prompt, all tools at all times, no gates
          (V1): separates the effect of the steps from the effect of enforcing them.

Budget (D3): 50 model turns per run, every turn counted (including a phase's closing
"finish now" turn). Phase budgets 2/12/10/4/15 and a reserve of 7 for sent-back
phases; turns of skipped phases go to the reserve.

Gold-free: the controller is given only the gold-free view of the instance
(`_SafeInstance`: id, repo, version, base commit, issue text). The offline L0-L4
label lives in `procedure_score.py` and runs after this module has finished.

The baseline agent (`pipeline.run_brt_phase`) is untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from ..harness.official import TestCommand
from ..models.llm import LLMClient
from .pipeline import billed_mark, phase_cost, serialise_messages, step_summary, take_served
from .preflight import DJANGO, Preflight, _keep, django_module, run_probe, run_preflight, run_snippet
from .procedure_gates import GateResult, check_write, quote_in, symptom_match
from .procedure_prompts import (
    LOCATE, PIN, REPRODUCE, SENT_BACK, SOFT_SYSTEM, SOFT_TASK, STYLE, SYSTEM, UNDERSTAND, WRITE,
    bullet, render_candidates, render_preflight,
)
from .procedure_tools import (
    ProcRunPython, ProcRunTest, RunLog, TestHome, finish_locate, finish_pin, finish_reproduce,
    finish_understand, finish_write, read_only_tools, run_test_at, submit_test_soft,
)
from .runtime import Agent, AgentResult
from .session import ContainerSession

TOTAL_TURNS = 50
PHASE_BUDGET = {"understand": 2, "reproduce": 12, "pin_expected": 10, "locate": 4, "write": 15}
MAX_VISITS = {"understand": 2, "reproduce": 2, "pin_expected": 2, "locate": 2, "write": 3}
PHASES = ("understand", "reproduce", "pin_expected", "locate", "write")


class _SafeInstance:
    """The gold-free view of a SWE-bench instance: no gold patch, no hidden tests."""

    def __init__(self, inst: object) -> None:
        safe = inst.context_safe_dict()  # type: ignore[attr-defined]
        self.instance_id = safe["instance_id"]
        self.repo = safe["repo"]
        self.base_commit = safe["base_commit"]
        self.problem_statement = safe["problem_statement"]
        self.version = getattr(inst, "version", None)

    def context_safe_dict(self) -> dict:
        return {"instance_id": self.instance_id, "repo": self.repo,
                "base_commit": self.base_commit, "problem_statement": self.problem_statement}


@dataclass
class Claim:
    id: str
    kind: str                 # observed | expected
    value: str
    quote: str
    source: str               # issue | docs | existing_test | observed_run | analogy
    where: str | None = None
    verified: bool = False


@dataclass
class Unknown:
    id: str
    question: str
    why: str = ""
    tried_value: str | None = None   # a value proposed without verified evidence
    tried_source: str | None = None


@dataclass
class Repro:
    run_id: str | None
    script: str
    output: str
    symptom: str              # matched | partial | none | unchecked
    tokens: list[str]
    explanation: str
    source: str               # preflight | model


@dataclass
class PhaseLog:
    phase: str
    visit: int
    allowance: int
    turns: int
    stop_cause: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    truncated_turns: int
    finish: dict | None
    task: str
    messages: list = field(default_factory=list)
    steps: list = field(default_factory=list)


@dataclass
class ProcedureResult:
    record: dict              # summary for results.jsonl (no transcripts)
    transcript: dict          # full: every phase's messages and the preflight
    test_code: str | None
    home: TestHome | None
    command: TestCommand | None
    outcome_buggy: str | None
    failure_reason_buggy: str | None
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    turns_used: int
    stop_cause: str


def phase_budgets(total: int) -> tuple[dict[str, int], int]:
    """Phase budgets scaled to `total` turns (design: 50 -> 2/12/10/4/15 + 7)."""
    if total == TOTAL_TURNS:
        b = dict(PHASE_BUDGET)
    else:
        b = {p: max(1, round(v * total / TOTAL_TURNS)) for p, v in PHASE_BUDGET.items()}
    return b, max(0, total - sum(b.values()))


class Procedure:
    def __init__(self, inst: object, client: LLMClient, *, mode: str = "strict",
                 total_turns: int = TOTAL_TURNS, max_total_tokens: int = 1_000_000,
                 max_tokens_per_turn: int = 4096, keep_container: bool = False,
                 verbose: bool = False, session: ContainerSession | None = None) -> None:
        if mode not in ("strict", "soft"):
            raise ValueError(f"mode must be strict or soft, got {mode!r}")
        self.inst = _SafeInstance(inst)
        self.client, self.mode = client, mode
        self.total_turns, self.max_total_tokens = total_turns, max_total_tokens
        self.max_tokens_per_turn, self.verbose = max_tokens_per_turn, verbose
        self.keep_container = keep_container
        self.session = session
        self.phase_left, self.reserve = phase_budgets(total_turns)
        self.turns_left = total_turns
        self.tok = {"in": 0, "out": 0, "cw": 0, "cr": 0}
        self.visits = {p: 0 for p in PHASES}
        self.feedback: dict[str, str] = {}
        self.log = RunLog()
        self.phases: list[PhaseLog] = []
        self.gates: list[dict] = []
        self.flags: list[str] = []
        self.claims: list[Claim] = []
        self.unknowns: list[Unknown] = []
        self.repro: Repro | None = None
        self.repro_attempt: Repro | None = None
        self.home: TestHome | None = None
        self.test: str | None = None
        self.test_run: dict | None = None
        self.best: dict | None = None
        self.pf: Preflight | None = None
        self.pfj: dict = {}
        self.cmd: TestCommand | None = None

    # ------------------------------------------------------------------ helpers

    @property
    def django(self) -> bool:
        return self.inst.repo == DJANGO

    def _say(self, msg: str) -> None:
        print(f"  [procedure] {msg}", flush=True)

    def _flag(self, f: str) -> None:
        if f not in self.flags:
            self.flags.append(f)

    def _tokens_used(self) -> int:
        return sum(self.tok.values())

    def _allowance(self, phase: str, revisit: bool) -> int:
        """First visit: the phase's own budget. A revisit, or a phase entered after it
        was skipped (its budget already moved to the reserve), may also use the reserve."""
        revisit = revisit or self.phase_left[phase] == 0
        base = self.phase_left[phase] + (self.reserve if revisit else 0)
        return max(0, min(base, self.turns_left))

    def _spend(self, phase: str, used: int) -> None:
        take = min(used, self.phase_left[phase])
        self.phase_left[phase] -= take
        self.reserve = max(0, self.reserve - (used - take))
        self.turns_left -= used

    def _skip(self, phase: str, why: str) -> None:
        self.reserve += self.phase_left[phase]
        self.phase_left[phase] = 0
        self._gate(GateResult(phase, True, "skipped", why))
        self._say(f"{phase}: skipped ({why})")

    def _gate(self, g: GateResult) -> GateResult:
        self.gates.append(asdict(g))
        return g

    def _issue(self) -> str:
        return self.inst.problem_statement

    def _run_agent(self, phase: str, system: str, task: str, tools: list, allowance: int
                   ) -> tuple[AgentResult | None, dict | None]:
        """One visit of a phase: an Agent limited to `allowance` turns in total."""
        if allowance <= 0 or self._tokens_used() >= self.max_total_tokens:
            if self.turns_left <= 0 or self._tokens_used() >= self.max_total_tokens:
                self._flag("budget_exhausted")
            else:
                self._flag(f"no_turns_left:{phase}")
            self.visits[phase] += 1  # counts, so a phase without turns cannot loop
            return None, None
        self.visits[phase] += 1
        agent = Agent(self.client, tools, system,
                      max_iterations=max(0, allowance - 1),
                      max_total_tokens=max(1, self.max_total_tokens - self._tokens_used()),
                      max_tokens_per_turn=self.max_tokens_per_turn,
                      submit_nudge_limit=1, submit_grace=1, verbose=self.verbose)
        res = agent.run(task)
        used = len(res.steps)
        self._spend(phase, used)
        self.tok["in"] += res.total_input_tokens
        self.tok["out"] += res.total_output_tokens
        self.tok["cw"] += res.total_cache_write_tokens
        self.tok["cr"] += res.total_cache_read_tokens
        finish = None
        if res.submission:
            try:
                finish = json.loads(res.submission)
            except ValueError:
                finish = None
        self.phases.append(PhaseLog(
            phase, self.visits[phase], allowance, used, res.stop_cause, res.total_input_tokens,
            res.total_output_tokens, res.total_cache_write_tokens, res.total_cache_read_tokens,
            res.truncated_turns, finish, task, serialise_messages(res.messages),
            step_summary(res.steps)))
        if res.truncated_turns:
            self._flag("truncated_turns")
        self._say(f"{phase} visit {self.visits[phase]}: {used} turn(s), stop={res.stop_cause}, "
                  f"turns left {self.turns_left}")
        return res, finish

    def _system(self, finish_tool: str) -> str:
        return SYSTEM.format(repo=self.inst.repo, base_commit=self.inst.base_commit,
                             finish_tool=finish_tool)

    def _with_feedback(self, phase: str, task: str) -> str:
        fb = self.feedback.pop(phase, None)
        return task + (f"\n\nNOTE FROM THE CONTROLLER (you were sent back to this step):\n{fb}"
                       if fb else "")

    def _observed(self) -> list[dict]:
        return [{"value": c.value, "quote": c.quote} for c in self.claims
                if c.kind == "observed" and c.verified]

    def _render_claims(self, kinds=("observed", "expected")) -> str:
        return bullet([f"{c.id} ({c.kind}, from {c.source}{', ' + c.where if c.where else ''}): "
                       f"{c.value} -- \"{c.quote[:160]}\"" for c in self.claims
                       if c.kind in kinds and c.verified])

    def _render_unknowns(self) -> str:
        return bullet([f"{u.id}: {u.question}" + (f" ({u.why})" if u.why else "") for u in self.unknowns])

    def _render_repro(self) -> str:
        r = self.repro
        if r is None:
            return ("Reproduction: none (the bug was not reproduced; write the test from the verified "
                    "expected values).")
        return (f"Reproduction ({r.symptom}: {r.explanation}):\n{r.script[:1500]}\nprinted:\n"
                f"{_keep(r.output, 1500)}")

    def _cands(self, key: str) -> list[dict]:
        return (self.pfj.get("candidates") or {}).get(key) or []

    def _home_check(self, test_dir: str, module: str | None) -> tuple[TestHome | None, str]:
        """Locate gate: the directory exists in HEAD and the trivial probe runs there."""
        test_dir = (test_dir or "").strip().strip("/") or "."
        if self.django:
            module = (module or django_module(test_dir) or "").strip().strip("/").replace("/", ".")
            if module.startswith("tests."):
                module = module[len("tests."):]
            if not module:
                return None, "Give the test package under tests/ as `module`."
            test_dir = "tests/" + module.replace(".", "/")
            res = self.session.exec('git cat-file -e "HEAD:$D/__init__.py"', env={"D": test_dir})
            if res.returncode != 0:
                return None, f"{test_dir}/ is not a test package in this repository (no __init__.py in HEAD)."
        elif test_dir != ".":
            res = self.session.exec('git cat-file -e "HEAD:$D"', env={"D": test_dir})
            if res.returncode != 0:
                return None, f"{test_dir!r} is not a directory of this repository."
        probe = run_probe(self.session, self.cmd, test_dir)
        if not probe["ok"]:
            return None, (f"A trivial test placed in {test_dir!r} does not run with the project's test "
                          f"command (pass -> {probe['pass']['outcome']}, fail -> {probe['fail']['outcome']}"
                          f"/{probe['fail']['reason']}). Choose another directory.")
        return TestHome(test_dir, module if self.django else None), ""

    def _default_home(self) -> TestHome:
        d = (self.pf.env.default_test_dir if self.pf else None) or "."
        return TestHome(d, django_module(d) if self.django else None, source="default")

    # ------------------------------------------------------------------ phases

    def p_understand(self) -> str:
        visit = self.visits["understand"] + 1
        task = self._with_feedback("understand", UNDERSTAND.format(
            issue=self._issue(), preflight=render_preflight(self.pfj)))
        tool = finish_understand()
        _, args = self._run_agent("understand", self._system(tool.name), task, [tool],
                                  self._allowance("understand", visit > 1))
        args = args or {}
        issue = self._issue()
        claims, unknowns, bad = [], [], []
        for k, c in enumerate(args.get("observed") or [], 1):
            if not isinstance(c, dict):
                continue
            ok = quote_in(issue, str(c.get("quote") or ""))
            claims.append(Claim(f"O{k}", "observed", str(c.get("value") or ""), str(c.get("quote") or ""),
                                "issue", verified=ok))
            if not ok:
                bad.append(str(c.get("quote") or "")[:120])
        n_e = 0
        for c in args.get("expected") or []:
            if not isinstance(c, dict):
                continue
            if quote_in(issue, str(c.get("quote") or "")):
                n_e += 1
                claims.append(Claim(f"E{n_e}", "expected", str(c.get("value") or ""),
                                    str(c.get("quote") or ""), "issue", verified=True))
            else:
                bad.append(str(c.get("quote") or "")[:120])
                unknowns.append(Unknown(f"U{len(unknowns) + 1}",
                                        f"What is the correct expected behaviour: {c.get('value')}?",
                                        "its quote is not in the issue",
                                        tried_value=str(c.get("value") or ""), tried_source="issue (quote not found)"))
        for u in args.get("unknowns") or []:
            if isinstance(u, dict) and u.get("question"):
                unknowns.append(Unknown(f"U{len(unknowns) + 1}", str(u["question"]), str(u.get("why") or "")))
        n_obs = sum(1 for c in claims if c.kind == "observed" and c.verified)
        self.claims = [c for c in claims if c.verified]
        self.unknowns = unknowns
        if not any(c.kind == "expected" for c in self.claims) and not self.unknowns:
            # nothing stated and nothing listed: the test still needs an expected value
            self.unknowns.append(Unknown("U1", "What is the correct behaviour the test should assert?",
                                         "the issue states no expected behaviour"))
        # Gate (design §3.2): every quote is in the issue and at least one observed claim.
        passed = bool(args) and n_obs >= 1 and not bad
        cause = ("ok" if passed else "no_finish" if not args else
                 "quote_missing" if bad else "no_observed_claim")
        msg = ("" if passed else "You did not call finish_understand." if not args else
               "These quotes are not in the issue (copy them verbatim):\n" + bullet(bad) if bad else
               "Report at least one observed behaviour, with a verbatim quote from the issue.")
        self._gate(GateResult("understand", passed, cause, msg,
                              {"observed_verified": n_obs, "expected_verified": n_e,
                               "unknowns": len(self.unknowns), "quotes_not_found": bad}))
        if not passed and visit < MAX_VISITS["understand"]:
            self.feedback["understand"] = msg
            return "understand"
        if n_obs == 0:  # repeated once already: keep the verified claims and go on
            self._flag("no_observed_claim")
        return "reproduce"

    def _after_reproduce(self) -> str:
        if self.unknowns and self.visits["pin_expected"] == 0:
            return "pin_expected"
        if self.visits["pin_expected"] == 0:
            self._skip("pin_expected", "every expected value is verified by an issue quote")
        return "write" if self.home else "locate"

    def p_reproduce(self) -> str:
        observed = self._observed()
        if self.visits["reproduce"] == 0 and self.pf and not self.feedback.get("reproduce"):
            for s in self.pf.snippets:
                if not s.ran:
                    continue
                status, toks, why = symptom_match(observed, s.output or "")
                if status == "matched":
                    self.repro = Repro(None, s.script or s.text, s.output or "", status, toks, why, "preflight")
                    self._skip("reproduce", f"preflight's run of the issue code matched: {why}")
                    return self._after_reproduce()
        visit = self.visits["reproduce"] + 1
        task = self._with_feedback("reproduce", REPRODUCE.format(
            observed=self._render_claims(("observed",)) if observed else
            "(no observed claim could be verified; reproduce what the issue describes)",
            preflight=render_preflight(self.pfj), code_candidates=render_candidates(self._cands("code")),
            issue=self._issue()))
        tools = read_only_tools(self.session) + [ProcRunPython(self.session, self.log),
                                                 finish_reproduce(self.log)]
        _, args = self._run_agent("reproduce", self._system("finish_reproduce"), task, tools,
                                  self._allowance("reproduce", visit > 1))
        rid = str((args or {}).get("run_id") or "").strip()
        if not rid:
            g = self._gate(GateResult("reproduce", False, "not_reproduced",
                                      "The model finished without a reproducing run."))
        else:
            run = self.log.runs[rid]
            rc, out = run_snippet(self.session, self.inst.repo, run["code"], run.get("module") or "brt_app")
            status, toks, why = symptom_match(observed, out)
            rep = Repro(rid, run["code"], out, status, toks, why, "model")
            passed = status in ("matched", "partial", "unchecked")
            g = self._gate(GateResult("reproduce", passed, status, why, {"run_id": rid}))
            if passed:
                self.repro = rep
                if status == "partial":
                    self._flag("symptom_partial")
                if status == "unchecked":
                    self._flag("symptom_unchecked")
                return self._after_reproduce()
            self.repro_attempt = rep
        if visit < MAX_VISITS["reproduce"] and self.phase_left["reproduce"] > 0:
            self.feedback["reproduce"] = (g.message or "The run does not show the bug.") + \
                " Try again: the run must show the observed behaviour."
            return "reproduce"
        self._flag("unreproduced")
        return self._after_reproduce()

    def p_pin(self) -> str:
        if not self.unknowns:
            self._skip("pin_expected", "no unknowns")
            return "write" if self.home else "locate"
        visit = self.visits["pin_expected"] + 1
        task = self._with_feedback("pin_expected", PIN.format(
            unknowns=self._render_unknowns(), claims=self._render_claims(), repro=self._render_repro(),
            code_candidates=render_candidates(self._cands("code"))))
        tools = read_only_tools(self.session) + [ProcRunPython(self.session, self.log), finish_pin()]
        _, args = self._run_agent("pin_expected", self._system("finish_pin"), task, tools,
                                  self._allowance("pin_expected", visit > 1))
        verified, rejected = [], []
        by_id = {u.id: u for u in self.unknowns}
        for r in (args or {}).get("resolved") or []:
            if not isinstance(r, dict):
                continue
            u = by_id.get(str(r.get("unknown_id") or "").strip())
            src, quote, where = r.get("source"), str(r.get("quote") or ""), str(r.get("where") or "").strip()
            ok, why = False, "source 'model' is never verified"
            if src in ("docs", "existing_test"):
                res = self.session.read_file(where) if where else None
                ok = bool(res and res.returncode == 0 and quote_in(res.output, quote))
                why = "" if ok else f"quote not found in {where or '(no file given)'}"
            elif src in ("observed_run", "analogy"):
                run = self.log.runs.get(where)
                if run:
                    _, out = run_snippet(self.session, self.inst.repo, run["code"],
                                         run.get("module") or "brt_app")
                    ok = quote_in(out, quote, min_len=1)
                    why = "" if ok else f"quote not in the output of re-running {where}"
                else:
                    why = f"run {where!r} does not exist"
            if u is None:
                rejected.append({"item": r, "why": "unknown_id does not exist"})
                continue
            if ok:
                n = sum(1 for c in self.claims if c.kind == "expected") + 1
                self.claims.append(Claim(f"E{n}", "expected", str(r.get("value") or ""), quote, src,
                                         where, verified=True))
                self.unknowns = [x for x in self.unknowns if x.id != u.id]
                verified.append(u.id)
            else:
                u.tried_value, u.tried_source = str(r.get("value") or ""), f"{src}: {why}"
                rejected.append({"unknown_id": u.id, "why": why})
        self._gate(GateResult("pin_expected", True, "ok", "",
                              {"verified": verified, "rejected": rejected,
                               "still_unknown": [u.id for u in self.unknowns]}))
        return "write" if self.home else "locate"

    def p_locate(self) -> str:
        tests = self._cands("tests")
        if self.visits["locate"] == 0 and tests and not tests[0]["path"].endswith("/"):
            top = tests[0]
            second = tests[1]["score"] if len(tests) > 1 else 0
            if top["score"] >= 4 and second <= top["score"] / 2:
                d = top["path"].rsplit("/", 1)[0] if "/" in top["path"] else "."
                home, _ = self._home_check(d, top.get("module"))
                if home:
                    home.source = "preflight_top"
                    self.home = home
                    self._skip("locate", f"one clear test-file candidate: {top['path']}")
                    return "write"
        visit = self.visits["locate"] + 1
        task = self._with_feedback("locate", LOCATE.format(
            test_candidates=bullet([f"{c['path']} (score {c['score']:.1f}: {', '.join(c['reasons'][:3])})"
                                    for c in tests[:6]]),
            code_candidates=render_candidates(self._cands("code")),
            django=" and the test package name (`module`, tests/<module>/)" if self.django else ""))
        tools = read_only_tools(self.session) + [finish_locate(self.django)]
        _, args = self._run_agent("locate", self._system("finish_locate"), task, tools,
                                  self._allowance("locate", visit > 1))
        if args:
            home, err = self._home_check(str(args.get("test_dir") or ""), args.get("module"))
            self._gate(GateResult("locate", home is not None, "ok" if home else "bad_home", err,
                                  {"test_dir": args.get("test_dir"), "module": args.get("module")}))
            if home:
                home.style_notes = str(args.get("style_notes") or "")[:500]
                self.home = home
                return "write"
            if visit < MAX_VISITS["locate"]:
                self.feedback["locate"] = err
                return "locate"
        else:
            self._gate(GateResult("locate", False, "no_finish", "The model did not choose a directory."))
        self.home = self._default_home()
        self._flag("default_location")
        return "write"

    def p_write(self) -> str:
        if self.home is None:
            self.home = self._default_home()
            self._flag("default_location")
        visit = self.visits["write"] + 1
        verified = [c for c in self.claims if c.verified]
        unknowns = "; ".join(f"{u.id}: {u.question}" + (f" (proposed without evidence: {u.tried_value})"
                                                        if u.tried_value else "") for u in self.unknowns)
        task = self._with_feedback("write", WRITE.format(
            repro=self._render_repro(),
            verified=bullet([f"{c.id} ({c.kind}, {c.source}): {c.value} -- \"{c.quote[:160]}\""
                             for c in verified]),
            unknowns=unknowns or "(none)", test_home=self.home.rel_path,
            style_notes=self.home.style_notes or "no notes", style=STYLE.get(self.cmd.framework, ""),
            issue=self._issue()))
        tools = read_only_tools(self.session) + [
            ProcRunTest(self.session, self.cmd, self.log, lambda _a: self.home), finish_write()]
        _, args = self._run_agent("write", self._system("finish_write"), task, tools,
                                  self._allowance("write", visit > 1))
        if not args:
            self._gate(GateResult("write", False, "no_finish", "No test was submitted."))
            if self.log.tests:
                self._consider_best(self.log.tests[-1])
            return "write" if visit < MAX_VISITS["write"] else "done"
        code = str(args["code"])
        run = run_test_at(self.session, self.cmd, self.home, code)
        self._consider_best(run)
        all_ids = {c.id for c in self.claims} | {u.id for u in self.unknowns}
        tokens = self.repro.tokens if self.repro and self.repro.symptom == "matched" else None
        g = self._gate(check_write(code, run, args.get("assertion_map") or [],
                                   {c.id for c in verified}, all_ids,
                                   accept_exception=self.django, symptom_tokens=tokens))
        g.details["run"] = {k: run[k] for k in ("outcome", "reason", "failing_line")}
        if g.passed:
            if g.details.get("symptom") == "not_checkable":
                self._flag("symptom_not_checked")
            self.test, self.test_run = code, run
            return "done"
        prev = f"\n\nYour previous test:\n{code}"
        if g.cause in ("passed_on_buggy", "symptom_mismatch") and self.visits["reproduce"] < MAX_VISITS["reproduce"]:
            self.feedback["reproduce"] = SENT_BACK[g.cause].format(
                repro_output=_keep(self.repro.output if self.repro else "(none)", 600),
                message=g.message) + prev
            self.feedback["write"] = g.message + prev
            self._say(f"write gate: {g.cause} -> back to reproduce")
            return "reproduce"
        if g.cause == "no_evidence" and self.visits["pin_expected"] < MAX_VISITS["pin_expected"]:
            line = g.details.get("failing_line")
            if not self.unknowns:
                text = code.splitlines()[line - 1].strip() if line and line <= len(code.splitlines()) else ""
                self.unknowns.append(Unknown(f"U{len(self.unknowns) + 1}",
                                             f"Is the value asserted at line {line} correct? `{text}`",
                                             "asserted without verified evidence"))
            self.feedback["pin_expected"] = SENT_BACK["no_evidence"].format(line=line) + prev
            self.feedback["write"] = g.message + prev
            self._say("write gate: no_evidence -> back to pin_expected")
            return "pin_expected"
        self.feedback["write"] = SENT_BACK.get(g.cause, "{message}").format(
            message=g.message, line=g.details.get("failing_line"),
            repro_output=_keep(self.repro.output if self.repro else "(none)", 600)) + prev
        self._say(f"write gate: {g.cause} -> stay in write")
        return "write"

    def _consider_best(self, run: dict) -> None:
        """Keep the best test seen, for a forced submission: one that fails at an
        assertion beats one that fails otherwise, which beats one that passes."""
        rank = {"fail": 2, "error": 1}.get(run.get("outcome"), 0) + (
            1 if run.get("reason") in ("assertion", "exception") else 0)
        if self.best is None or rank >= self.best["_rank"]:
            self.best = {**run, "_rank": rank}

    # ------------------------------------------------------------------ modes

    def _strict(self) -> str:
        handlers = {"understand": self.p_understand, "reproduce": self.p_reproduce,
                    "pin_expected": self.p_pin, "locate": self.p_locate, "write": self.p_write}
        phase = "understand"
        while phase != "done":
            if self.turns_left <= 0 or self._tokens_used() >= self.max_total_tokens:
                self._flag("budget_exhausted")
                break
            if self.visits[phase] >= MAX_VISITS[phase]:
                self._flag(f"degraded:{phase}")
                self.feedback.pop(phase, None)
                if phase != "write" and self.visits["write"] < MAX_VISITS["write"]:
                    phase = "write"
                    continue
                break
            phase = handlers[phase]()
        return "submit" if self.test else ("budget" if "budget_exhausted" in self.flags else "gates")

    def _soft(self) -> str:
        default = self._default_home()

        def resolve(args: dict):
            if not args.get("test_dir") and not args.get("module"):
                return default
            home, err = self._home_check(str(args.get("test_dir") or ""), args.get("module"))
            return home if home else err

        tools = read_only_tools(self.session) + [
            ProcRunPython(self.session, self.log),
            ProcRunTest(self.session, self.cmd, self.log, resolve, home_args=True),
            submit_test_soft(self.django)]
        task = SOFT_TASK.format(
            django_arg=" and `module`" if self.django else "", style=STYLE.get(self.cmd.framework, ""),
            issue=self._issue(), preflight=render_preflight(self.pfj),
            code_candidates=render_candidates(self._cands("code")),
            test_candidates=render_candidates(self._cands("tests")))
        system = SOFT_SYSTEM.format(repo=self.inst.repo, base_commit=self.inst.base_commit)
        self.phase_left = {p: 0 for p in PHASES}
        self.phase_left["write"] = self.total_turns
        self.reserve = 0
        res, args = self._run_agent("write", system, task, tools, self.total_turns)
        if args:
            home = resolve(args)
            if isinstance(home, str):
                self._flag("default_location")
                home = default
            self.home = home
            run = run_test_at(self.session, self.cmd, home, str(args["code"]))
            self.test, self.test_run = str(args["code"]), run
            if res is not None and res.stop_cause == "submit_forced":
                self._flag("forced_submit")
            return res.stop_cause if res else "submit"
        if self.log.tests:
            self._consider_best(self.log.tests[-1])
        return res.stop_cause if res else "budget"

    # ------------------------------------------------------------------ finalise

    def _header(self) -> str:
        lines = ["# Bug reproduction test written by the procedure agent.", "# Evidence:"]
        for c in self.claims:
            if c.verified:
                lines.append(f"#   {c.id} ({c.kind}, {c.source}{', ' + c.where if c.where else ''}): "
                             f"{' '.join(c.value.split())[:120]}")
        for u in self.unknowns:
            lines.append(f"# UNVERIFIED: {' '.join((u.tried_value or u.question).split())[:120]}"
                         f" ({u.why or 'no verified evidence'})")
        if self.flags:
            lines.append(f"# flags: {', '.join(self.flags)}")
        return "\n".join(lines) + "\n"

    def finalise(self, stop_cause: str) -> tuple[str | None, dict | None]:
        code = self.test
        if code is None and self.best is not None:
            code = self.best["code"]
            self.home = self.home or self._default_home()
            self._flag("forced_submit")
        if code is None:
            return None, None
        if self.unknowns:
            self._flag("unknown_expected")
        final = self._header() + "\n" + code
        run = run_test_at(self.session, self.cmd, self.home, final)  # gold-free, as submitted
        return final, run

    # ------------------------------------------------------------------ entry

    def run(self) -> ProcedureResult:
        bill0 = billed_mark(self.client)
        t0 = time.monotonic()
        own = self.session is None
        if own:
            self.session = ContainerSession(self.inst, keep=self.keep_container).start()
        stop = "environment_invalid"
        final = run = None
        try:
            self.pf = run_preflight(self.session, self.inst)
            self.pfj = self.pf.to_json()
            self.cmd = self.pf.runner()
            self._say(f"preflight: env={'INVALID' if self.pf.env.invalid else 'ok'} "
                      f"runner={(self.pf.env.command or {}).get('source')}")
            if not self.pf.env.invalid:
                stop = self._strict() if self.mode == "strict" else self._soft()
                final, run = self.finalise(stop)
        finally:
            if own:
                self.session.close()
        cost = phase_cost(self.client, bill0, self.tok["in"], self.tok["out"],
                          cache_write_tok=self.tok["cw"], cache_read_tok=self.tok["cr"])
        return self._result(stop, final, run, cost, time.monotonic() - t0)

    def _result(self, stop, final, run, cost, secs) -> ProcedureResult:
        turns = self.total_turns - self.turns_left
        outcome = ("environment_invalid" if stop == "environment_invalid" else
                   "no_test" if final is None else
                   "forced_test" if "forced_submit" in self.flags else "test")
        summary = {
            "mode": "agent_procedure" if self.mode == "strict" else "agent_procedure_soft",
            "outcome": outcome, "stop_cause": stop, "flags": self.flags,
            "turn_budget": self.total_turns, "turns_used": turns,
            "visits": self.visits,
            "phases": [{k: v for k, v in asdict(p).items() if k not in ("messages", "task")}
                       for p in self.phases],
            "gates": self.gates,
            "claims": [asdict(c) for c in self.claims],
            "unknowns": [asdict(u) for u in self.unknowns],
            "repro": asdict(self.repro) if self.repro else None,
            "test_home": ({**asdict(self.home), "rel_path": self.home.rel_path} if self.home else None),
            "runner": self.pfj.get("env", {}).get("command"),
            "candidates": {"code": self._cands("code")[:5], "tests": self._cands("tests")[:5]},
            "preflight": {"env_invalid": self.pfj.get("env", {}).get("invalid"),
                          "env_reason": self.pfj.get("env", {}).get("reason"),
                          "snippets_ran": sum(1 for s in self.pfj.get("snippets", []) if s.get("ran")),
                          "seconds": self.pfj.get("seconds")},
            "run_ids": len(self.log.runs), "test_runs": len(self.log.tests),
            "served": take_served(self.client),
            "seconds": round(secs, 1),
        }
        transcript = {
            "instance_id": self.inst.instance_id, "model": getattr(self.client, "name", "?"),
            "mode": summary["mode"], "preflight": self.pfj,
            "phases": [asdict(p) for p in self.phases], "gates": self.gates,
            "runs": self.log.runs, "test_runs": self.log.tests, "final_test": final,
            "final_run": run,
        }
        return ProcedureResult(
            record=summary, transcript=transcript, test_code=final, home=self.home, command=self.cmd,
            outcome_buggy=(run or {}).get("outcome"), failure_reason_buggy=(run or {}).get("reason"),
            cost_usd=cost, input_tokens=self.tok["in"], output_tokens=self.tok["out"],
            cache_write_tokens=self.tok["cw"], cache_read_tokens=self.tok["cr"], turns_used=turns,
            stop_cause=stop)


def run_procedure(inst: object, client: LLMClient, **kw) -> ProcedureResult:
    """Run the procedure agent (strict or soft) on one instance. Gold-free."""
    return Procedure(inst, client, **kw).run()


__all__ = ["Procedure", "ProcedureResult", "run_procedure", "phase_budgets", "TOTAL_TURNS",
           "PHASE_BUDGET", "MAX_VISITS"]
