"""Offline smoke for the best-of-N pipeline (generate N -> select -> retry fix).

No Docker, no LLM, no budget. A fake client generates 5 candidate BRTs (2 viable,
3 not), a fake Selector picks among the viable ones, and a fake fix worker fails
its first attempt then succeeds on the second. Asserts:

  - N candidates are generated and scored GOLD-FREE on the buggy tree; the
    viability filter keeps ONLY clean assertion-fails (passes=L1 and crashes are
    dropped);
  - the Selector chooses among viable candidates and the viable-local index maps
    back to the correct global candidate;
  - the chosen candidate is gold-barrier scored (L4) for the scoreboard;
  - the fix RETRY loop reverts after a gold-free failure and succeeds on attempt 2,
    aggregating tokens across attempts;
  - SubmitSelectionTool + selector registry are gold-free; no gold/test_patch leak.

    python3 scripts/agent_bestofn_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_tools_smoke import FakeSession

from tdbf.agent import pipeline
from tdbf.agent.session import ExecResult
from tdbf.agent.tools import (
    EditFileTool,
    SubmitSelectionTool,
    _SELECTOR_TOOL_NAMES,
    assert_selector_gold_free,
    selector_agent_tools,
)
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import ContainerResult, HarnessResult
from tdbf.models.messages import ToolCall, ToolUseResponse

_BUGGY = "def compute():\n    return 4\n"
_CAND_PASSES = "def test_brt():\n    assert compute() == 4  # asserts BUGGY value\n"     # L1
_CAND_VIABLE_A = "def test_brt():\n    assert compute() == 5  # VIABLE A\n"
_CAND_CRASH = "def test_brt():\n    undefined_symbol()  # CRASH\n"                        # runtime
_CAND_VIABLE_B = "def test_brt():\n    assert compute() == 5  # VIABLE B\n"
_CANDS = [_CAND_PASSES, _CAND_VIABLE_A, _CAND_CRASH, _CAND_VIABLE_B, _CAND_PASSES]

_INST = SWEBenchInstance(
    instance_id="demo__pkg-1", repo="demo/pkg", base_commit="abc123",
    gold_patch="GOLD MUST NOT BE USED", test_patch="TEST_PATCH MUST NOT BE USED",
    problem_statement="compute() returns 4 but should return 5.",
)


class BestOfNFakeSession(FakeSession):
    """Models the buggy tree + fix state. pytest on the correct BRT (== 5) passes
    only after a correct edit; the L1 test (== 4) always passes; the CRASH test
    raises a NameError (runtime, non-assertion)."""

    def __init__(self, repo: str = "demo/pkg") -> None:
        super().__init__(repo)
        self.core = _BUGGY
        self.edited = False
        self.fixed = False
        self.last_test = ""

    def start(self):
        return self

    def close(self):
        pass

    def read_file(self, path, *, max_bytes=0) -> ExecResult:
        if path.endswith("core.py"):
            return ExecResult(0, self.core)
        return ExecResult(1, f"cat: {path}: No such file or directory")

    def write_file(self, path, content) -> None:
        super().write_file(path, content)
        if path.endswith("core.py"):
            self.core = content
            self.edited = True
            self.fixed = "return 5" in content
        elif path.endswith("test_brt.py"):
            self.last_test = content

    def reset_tree(self) -> None:
        self.core, self.edited, self.fixed = _BUGGY, False, False

    def ensure_pytest(self) -> None:
        self._pytest_ready = True

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        self.execs.append(command)
        if "git diff" in command:
            return ExecResult(0, "diff --git a/core.py\n+    return 5\n" if self.edited else "")
        if "pytest" in command:
            t = self.last_test
            if "CRASH" in t:
                return ExecResult(1, "E   NameError: name 'undefined_symbol' is not defined\n1 failed")
            if "== 5" in t:  # correct BRT: passes only once the source is fixed
                return (ExecResult(0, "1 passed in 0.1s") if self.fixed
                        else ExecResult(1, "E   AssertionError: assert 4 == 5\n1 failed in 0.1s"))
            if "== 4" in t:  # asserts the buggy value -> passes on buggy (L1)
                return ExecResult(0, "1 passed in 0.1s")
            return ExecResult(1, "E   AssertionError\n1 failed")
        return super().exec(command, timeout=timeout, activate=activate, env=env)


class BestOfNFakeClient:
    name = "ollama:qwen2.5-coder:14b-instruct-q3_K_M"

    def __init__(self) -> None:
        self.brt = self.fix = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        names = {t.name for t in tools}
        if "submit_test" in names:  # ---- candidate generator ----
            self.brt += 1
            code = _CANDS[(self.brt - 1) % len(_CANDS)]
            return ToolUseResponse(text="candidate", tool_calls=[ToolCall(f"b{self.brt}", "submit_test", {"code": code})],
                                   stop_reason="tool_use", input_tokens=6000, output_tokens=150)
        if "submit_selection" in names:  # ---- Selector: pick viable index 1 (VIABLE_B) ----
            return ToolUseResponse(text="pick", tool_calls=[ToolCall("s1", "submit_selection",
                                   {"index": 1, "reason": "VIABLE B asserts the corrected value most directly."})],
                                   stop_reason="tool_use", input_tokens=4000, output_tokens=60)
        if "submit_fix" in names:  # ---- fix worker: wrong edit -> submit; correct edit -> submit ----
            self.fix += 1
            if self.fix == 1:
                return ToolUseResponse(text="edit (noop)", tool_calls=[ToolCall("e1", "edit_file",
                    {"path": "demo/pkg/core.py", "old_str": "return 4", "new_str": "return 4  # noop"})],
                    stop_reason="tool_use", input_tokens=7000, output_tokens=120)
            if self.fix == 2:
                return ToolUseResponse(text="submit", tool_calls=[ToolCall("f1", "submit_fix", {})],
                                       stop_reason="tool_use", input_tokens=7100, output_tokens=40)
            if self.fix == 3:
                return ToolUseResponse(text="edit (real)", tool_calls=[ToolCall("e2", "edit_file",
                    {"path": "demo/pkg/core.py", "old_str": "return 4", "new_str": "return 5"})],
                    stop_reason="tool_use", input_tokens=7200, output_tokens=120)
            return ToolUseResponse(text="submit", tool_calls=[ToolCall("f2", "submit_fix", {})],
                                   stop_reason="tool_use", input_tokens=7300, output_tokens=40)
        raise AssertionError(f"unexpected tool set: {sorted(names)}")


def test_selector_registry_and_tool() -> None:
    sess = BestOfNFakeSession()
    tools = selector_agent_tools(sess)
    assert {t.name for t in tools} <= set(_SELECTOR_TOOL_NAMES)
    assert "submit_selection" in {t.name for t in tools}
    for banned in ("submit_test", "edit_file", "run_test", "submit_fix"):
        assert banned not in {t.name for t in tools}, banned
    try:
        assert_selector_gold_free([EditFileTool(sess)])
        raise AssertionError("assert_selector_gold_free must reject edit_file")
    except ValueError:
        pass
    t = SubmitSelectionTool()
    assert t.execute({}).is_error                       # index required
    assert json.loads(t.execute({"index": 2, "reason": "x"}).content) == {"index": 2, "reason": "x"}


def test_bestofn_generate_select() -> None:
    pipeline.ContainerSession = lambda instance, keep=False: BestOfNFakeSession(instance.repo)

    def fake_run_test(instance, test_patch, *, gold_patch=None, test_path=None, **kw):
        # chosen candidate: fail (assertion) on buggy, pass on gold-fixed -> L4
        if gold_patch is None:
            return ContainerResult("fail", failure_reason="assertion", output="E AssertionError")
        return ContainerResult("pass", output="1 passed")
    pipeline.run_test_in_container = fake_run_test
    client = BestOfNFakeClient()

    bon = pipeline.run_brt_best_of_n(
        _INST, client, n_candidates=5, max_iterations=4, max_total_tokens=300_000,
        selector_client=client, score_fixed=True,
    )
    assert bon is not None
    # 2 viable (indices 1 and 3); 3 non-viable (0,4 pass; 2 crash)
    assert bon.n_viable == 2, [(c.index, c.viable, c.outcome_buggy, c.failure_reason) for c in bon.candidates]
    assert [c.viable for c in bon.candidates] == [False, True, False, True, False]
    # selector picked viable-local 1 -> global candidate index 3
    assert bon.chosen_index == 3, bon.chosen_index
    assert bon.selection is not None and bon.selection.index == 1
    assert bon.brt is not None and bon.brt.level == 4
    assert "VIABLE B" in bon.brt.submission
    # tokens aggregate 5 candidate runs + 1 selector run
    assert bon.total_input_tokens == 5 * 6000 + 4000, bon.total_input_tokens

    rec = pipeline.brt_run_record(_INST, client.name, split="lite",
                                  flags={"bestofn": True, "n_viable": bon.n_viable}, phase=bon.brt)
    assert rec["best_level"] == 4
    assert "MUST NOT BE USED" not in json.dumps(rec), "gold/test_patch leaked!"


def test_fix_retry_recovers() -> None:
    pipeline.ContainerSession = lambda instance, keep=False: BestOfNFakeSession(instance.repo)
    pipeline.write_predictions = lambda recs, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline._verify_scored = lambda *a, **k: None  # stubbed harness: no report on disk
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True,
        fail_to_pass_passed=["test_brt"], fail_to_pass_failed=[],
        pass_to_pass_passed=["t"], pass_to_pass_failed=[], applied=True,
    )
    client = BestOfNFakeClient()

    with tempfile.TemporaryDirectory() as td:
        fix = pipeline.run_fix_phase(
            _INST, _CAND_VIABLE_B, client, split="lite",
            max_iterations=6, max_total_tokens=300_000, run_dir=Path(td), max_attempts=3,
        )
    assert fix is not None
    assert fix.resolved is True
    assert "return 5" in fix.fix_patch
    # attempt 1 (edit-noop + submit) failed the gold-free check -> retry; attempt 2 succeeded.
    # tokens aggregate both attempts: (7000+7100) + (7200+7300)
    assert fix.total_input_tokens == (7000 + 7100) + (7200 + 7300), fix.total_input_tokens
    assert client.fix == 4, client.fix  # exactly two attempts, two calls each


def main() -> int:
    tests = [test_selector_registry_and_tool, test_bestofn_generate_select, test_fix_retry_recovers]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nBest-of-N pipeline glue: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
