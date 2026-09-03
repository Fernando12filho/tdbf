"""Offline glue test for scripts/05_agent_fix.py (Phase B fix agent).

No Docker, no LLM, no budget. Two layers:
  1. Unit-test the fix tools (edit_file / run_brt / run_tests / revert / submit_fix)
     and the gold-free fix guard against a fake session.
  2. Drive the real `run_one` with a scripted fake client (edit -> run_brt(pass) ->
     submit_fix); fake the harness scoring to resolved=True. Confirms the gold
     barrier ordering, the diff capture, the results record, and that the held-out
     test_patch never reaches the agent or the transcript.

    python3 scripts/agent_fix_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_tools_smoke import FakeSession  # Docker-free session fake

from tdbf.agent import pipeline  # phase logic now lives here (patched below)
from tdbf.agent.session import ExecResult
from tdbf.agent.tools import (
    SubmitFixTool,
    _FIX_TOOL_NAMES,
    assert_fix_gold_free,
    fix_agent_tools,
)
from tdbf.agent.types import FunctionTool
from tdbf.data.instances import SWEBenchInstance
from tdbf.harness.runner import HarnessResult
from tdbf.models.messages import ToolCall, ToolUseResponse

_SRC = "def compute():\n    return 4\n"
_FAKE_DIFF = (
    "diff --git a/demo/pkg/core.py b/demo/pkg/core.py\n"
    "--- a/demo/pkg/core.py\n+++ b/demo/pkg/core.py\n"
    "@@ -1,2 +1,2 @@\n def compute():\n-    return 4\n+    return 5\n"
)


class FixFakeSession(FakeSession):
    """FakeSession specialised for the fix tools + lifecycle hooks."""

    def __init__(self, repo: str = "demo/pkg") -> None:
        super().__init__(repo)
        self.reverted = False

    def start(self):
        return self

    def close(self):
        pass

    def read_file(self, path, *, max_bytes=0) -> ExecResult:
        if path == "demo/pkg/core.py":
            return ExecResult(0, _SRC)
        return ExecResult(1, f"cat: {path}: No such file or directory")

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        self.execs.append(command)
        if "git diff" in command:
            return ExecResult(0, _FAKE_DIFF)
        if "git checkout" in command:
            return ExecResult(0, "")
        if "pytest" in command:  # run_brt / run_tests both see a green tree here
            return ExecResult(0, "1 passed in 0.1s")
        return super().exec(command, timeout=timeout, activate=activate, env=env)

    def reset_tree(self) -> None:
        self.reverted = True


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


# ----------------------------- unit tests --------------------------------- #

def test_fix_registry_is_gold_free_and_terminal() -> None:
    tools = fix_agent_tools(FixFakeSession(), brt_code="def test_brt():\n    assert compute() == 5\n")
    names = {t.name for t in tools}
    assert names == set(_FIX_TOOL_NAMES), names
    terminal = [t for t in tools if t.terminal]
    assert len(terminal) == 1 and terminal[0].name == "submit_fix"
    assert_fix_gold_free(tools)


def test_fix_guard_rejects_off_allowlist_tool() -> None:
    rogue = FunctionTool("apply_gold_patch", "applies the gold fix", {"type": "object", "properties": {}}, func=lambda a: "x")
    try:
        assert_fix_gold_free([rogue])
    except ValueError as e:
        assert "gold-free allowlist" in str(e)
    else:
        raise AssertionError("fix guard should have rejected an off-allowlist tool")


def test_edit_file_replaces_and_reports() -> None:
    sess = FixFakeSession()
    tools = fix_agent_tools(sess, brt_code="x")
    res = _tool(tools, "edit_file").execute(
        {"path": "demo/pkg/core.py", "old_str": "return 4", "new_str": "return 5"}
    )
    assert not res.is_error and "1 occurrence" in res.content, res.content
    assert sess.writes["/testbed/demo/pkg/core.py"] == "def compute():\n    return 5\n"


def test_edit_file_missing_old_str_is_recoverable_error() -> None:
    tools = fix_agent_tools(FixFakeSession(), brt_code="x")
    res = _tool(tools, "edit_file").execute(
        {"path": "demo/pkg/core.py", "old_str": "not present", "new_str": "y"}
    )
    assert res.is_error and "not found" in res.content


def test_run_brt_uses_fixed_brt_and_parses_pass() -> None:
    sess = FixFakeSession()
    brt = "def test_brt():\n    assert compute() == 5\n"
    tools = fix_agent_tools(sess, brt_code=brt)
    res = _tool(tools, "run_brt").execute({})
    assert "outcome=pass" in res.content
    # BRT written to the repo-relative path; agent never supplied it
    assert any(p.endswith("test_brt.py") for p in sess.writes), sess.writes.keys()
    assert any("def test_brt" in v for v in sess.writes.values())


def test_revert_resets_tree() -> None:
    sess = FixFakeSession()
    tools = fix_agent_tools(sess, brt_code="x")
    res = _tool(tools, "revert").execute({})
    assert sess.reverted and not res.is_error


def test_submit_fix_is_terminal() -> None:
    assert SubmitFixTool.terminal is True
    assert not SubmitFixTool().execute({}).is_error


# --------------------------- orchestration glue --------------------------- #

class FakeFixClient:
    """Drives: edit_file -> run_brt -> submit_fix."""

    name = "anthropic:claude-sonnet-4-6"

    def __init__(self) -> None:
        self.turn = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.turn += 1
        if self.turn == 1:
            return ToolUseResponse(
                text="Editing the root cause.",
                tool_calls=[ToolCall("e1", "edit_file", {"path": "demo/pkg/core.py", "old_str": "return 4", "new_str": "return 5"})],
                stop_reason="tool_use", input_tokens=9000, output_tokens=200,
            )
        if self.turn == 2:
            return ToolUseResponse(
                text="Running the BRT.",
                tool_calls=[ToolCall("r1", "run_brt", {})],
                stop_reason="tool_use", input_tokens=9100, output_tokens=80,
            )
        return ToolUseResponse(
            text="BRT passes; submitting.",
            tool_calls=[ToolCall("s1", "submit_fix", {})],
            stop_reason="tool_use", input_tokens=9200, output_tokens=60,
        )


def _load_script_module():
    spec = importlib.util.spec_from_file_location("agent_fix", ROOT / "scripts" / "05_agent_fix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_orchestration_glue() -> None:
    mod = _load_script_module()
    inst = SWEBenchInstance(
        instance_id="demo__pkg-1",
        repo="demo/pkg",
        base_commit="abc123",
        gold_patch="GOLD MUST NOT BE USED",
        test_patch="TEST_PATCH MUST NOT BE USED",
        problem_statement="compute() returns 4 but should return 5.",
    )
    captured: dict = {}

    # build_client/load_best_brt/save_run are called in the script; the container
    # + harness scoring now live in tdbf.agent.pipeline (patch there).
    mod.build_client = lambda spec, **kw: FakeFixClient()
    mod.load_best_brt = lambda iid, min_level=4: {
        "test_code": "def test_brt():\n    assert compute() == 5\n", "level": 4, "model": "anthropic:claude-sonnet-4-6",
    }
    mod.save_run = lambda record: captured.setdefault("record", record)
    pipeline.ContainerSession = lambda instance, keep=False: FixFakeSession(instance.repo)
    pipeline.write_predictions = lambda records, path: Path(path)
    pipeline.run_harness = lambda *a, **k: 0
    pipeline.read_report = lambda run_id, model, instance: HarnessResult(
        instance_id=instance.instance_id, resolved=True,
        fail_to_pass_passed=["test_brt"], fail_to_pass_failed=[],
        pass_to_pass_passed=["test_existing"], pass_to_pass_failed=[], applied=True,
    )

    with tempfile.TemporaryDirectory() as td:
        resolved = mod.run_one(
            inst, "anthropic:claude-sonnet-4-6",
            split="lite", max_iterations=8, max_total_tokens=300_000,
            think=False, brt_override=None, keep_container=False, verbose=False,
            transcript_dir=Path(td),
        )
        assert resolved is True, resolved

        rec = captured["record"]
        assert rec["mode"] == "fix_agent" and rec["resolved"] is True
        att = rec["attempts"][0]
        assert att["stop_cause"] == "submit", att["stop_cause"]
        assert att["num_tool_calls"] == 3
        assert att["applied"] is True
        assert "return 5" in att["fix_patch"]
        # cost: sonnet 3/15 per Mtok over (27300 in, 340 out)
        assert abs(att["cost_usd"] - (27300 / 1e6 * 3 + 340 / 1e6 * 15)) < 1e-9, att["cost_usd"]

        transcript = json.loads((Path(td) / "demo__pkg-1.json").read_text())
        blob = json.dumps(transcript)
        assert "MUST NOT BE USED" not in blob, "gold_patch / test_patch leaked into the agent run!"
        assert transcript["resolved"] is True and "return 5" in transcript["fix_patch"]


def main() -> int:
    tests = [
        test_fix_registry_is_gold_free_and_terminal,
        test_fix_guard_rejects_off_allowlist_tool,
        test_edit_file_replaces_and_reports,
        test_edit_file_missing_old_str_is_recoverable_error,
        test_run_brt_uses_fixed_brt_and_parses_pass,
        test_revert_resets_tree,
        test_submit_fix_is_terminal,
        test_orchestration_glue,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nPhase B fix-agent glue: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
