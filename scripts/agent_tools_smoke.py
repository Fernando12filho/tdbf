"""Smoke test for the container session + gold-free tools (Phase 0, build 2).

Offline by default: validates tool schemas, the gold-free registry guard, and
input-validation paths with a fake session (no Docker). The conversion/loop
mechanics are covered by agent_runtime_smoke.py; this focuses on the tool layer.

Live mode (needs Docker + a built instance image) exercises the *real* session
against the buggy checkout — read a file, run a probe, run a trivial test:

    export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock
    python3 scripts/01_validate_gold.py --split lite --id astropy__astropy-12907  # build image first
    python3 scripts/agent_tools_smoke.py --id astropy__astropy-12907
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.agent.session import ContainerSession, ExecResult
from tdbf.agent.tools import (
    SubmitTestTool,
    _GOLD_FREE_TOOL_NAMES,
    assert_gold_free,
    brt_agent_tools,
)
from tdbf.agent.types import FunctionTool, Tool


# --------------------------------------------------------------------------- #
# A fake session: records commands and returns canned output, so the tool     #
# wrappers can be unit-tested without Docker.                                  #
# --------------------------------------------------------------------------- #


class FakeSession:
    def __init__(self, repo: str = "demo/pkg") -> None:
        self.repo = repo
        self.instance_id = "demo__pkg-1"
        self.writes: dict[str, str] = {}
        self.execs: list[str] = []
        self._pytest_ready = False

    def read_file(self, path, *, max_bytes=0) -> ExecResult:
        if path == "exists.py":
            return ExecResult(0, "print('hello')\n")
        if path == "multi.py":
            return ExecResult(0, "a = 1\nb = 2\nc = 3\nd = 4\n")
        return ExecResult(1, f"cat: {path}: No such file or directory")

    def write_file(self, path, content) -> None:
        self.writes[path] = content

    def exec(self, command, *, timeout=120, activate=True, env=None) -> ExecResult:
        self.execs.append(command)
        if "git ls-files" in command:
            return ExecResult(0, "demo/pkg/core.py\ndemo/pkg/util.py\n")
        if command.startswith("grep"):
            return ExecResult(0, "demo/pkg/core.py:10:def take(seq):\n")
        if "tdbf_probe.py" in command:
            return ExecResult(0, "observed: 42\n")
        if "pytest" in command:
            return ExecResult(1, "E   AssertionError: assert 1 == 2\n1 failed in 0.1s")
        return ExecResult(0, "")

    def ensure_pytest(self) -> None:
        self._pytest_ready = True


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def test_registry_is_gold_free_and_complete() -> None:
    tools = brt_agent_tools(FakeSession())  # type: ignore[arg-type]
    names = {t.name for t in tools}
    assert names == set(_GOLD_FREE_TOOL_NAMES), names
    # the submit tool is the single terminal tool
    terminal = [t for t in tools if t.terminal]
    assert len(terminal) == 1 and terminal[0].name == "submit_test"
    # guard passes for the standard set
    assert_gold_free(tools)


def test_guard_rejects_off_allowlist_tool() -> None:
    rogue = FunctionTool("run_on_fixed_tree", "runs against the gold tree", {"type": "object", "properties": {}}, func=lambda a: "x")
    try:
        assert_gold_free([rogue])
    except ValueError as e:
        assert "gold-free allowlist" in str(e)
    else:
        raise AssertionError("guard should have rejected an off-allowlist tool")


def test_schemas_are_object_schemas() -> None:
    tools = brt_agent_tools(FakeSession())  # type: ignore[arg-type]
    for t in tools:
        sch = t.schema().input_schema
        assert sch.get("type") == "object", (t.name, sch)
        assert "properties" in sch, t.name


def test_read_file_tool() -> None:
    tools = brt_agent_tools(FakeSession())  # type: ignore[arg-type]
    ok = _tool(tools, "read_file").execute({"path": "exists.py"})
    assert not ok.is_error and "hello" in ok.content
    assert ok.content.startswith("1\t"), f"expected line-number prefix, got {ok.content!r}"
    missing = _tool(tools, "read_file").execute({"path": "nope.py"})
    assert missing.is_error
    empty = _tool(tools, "read_file").execute({"path": "  "})
    assert empty.is_error


def test_read_file_line_range() -> None:
    tools = brt_agent_tools(FakeSession())  # type: ignore[arg-type]
    res = _tool(tools, "read_file").execute({"path": "multi.py", "start_line": 2, "end_line": 3})
    assert res.content == "2\tb = 2\n3\tc = 3", repr(res.content)
    # string args coerce; out-of-range start is a recoverable error
    past = _tool(tools, "read_file").execute({"path": "multi.py", "start_line": "99"})
    assert past.is_error


def test_run_test_tool_parses_outcome_and_writes_file() -> None:
    sess = FakeSession()
    tools = brt_agent_tools(sess)  # type: ignore[arg-type]
    res = _tool(tools, "run_test").execute({"code": "def test_x():\n    assert 1 == 2\n"})
    assert "outcome=fail" in res.content
    assert "failure_reason=assertion" in res.content
    # test was written to the repo-relative BRT path (root for non-django repo)
    assert any(p.endswith("test_brt.py") for p in sess.writes), sess.writes.keys()
    assert sess._pytest_ready  # ensure_pytest was called


def test_run_python_tool() -> None:
    sess = FakeSession()
    tools = brt_agent_tools(sess)  # type: ignore[arg-type]
    res = _tool(tools, "run_python").execute({"code": "print(40 + 2)"})
    assert "observed: 42" in res.content
    assert "/tmp/tdbf_probe.py" in sess.writes


def test_submit_tool_is_terminal_and_captures_code() -> None:
    code = "def test_brt():\n    assert foo() == 3\n"
    res = SubmitTestTool().execute({"code": code})
    assert res.content == code and not res.is_error
    assert SubmitTestTool.terminal is True


def run_offline() -> None:
    tests = [
        test_registry_is_gold_free_and_complete,
        test_guard_rejects_off_allowlist_tool,
        test_schemas_are_object_schemas,
        test_read_file_tool,
        test_read_file_line_range,
        test_run_test_tool_parses_outcome_and_writes_file,
        test_run_python_tool,
        test_submit_tool_is_terminal_and_captures_code,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} offline tool-layer checks passed.")


def run_live(instance_id: str, split: str) -> None:
    from tdbf.data.instances import InstanceStore

    store = InstanceStore(split=split)
    instance = next(i for i in store.iter_instances() if i.instance_id == instance_id)

    print(f"\n--- live session on {instance_id} ({instance.repo}) ---")
    with ContainerSession(instance) as sess:
        tools = brt_agent_tools(sess)
        glob = _tool(tools, "glob").execute({"pattern": r"\.py$"})
        print(f"glob '.py$' -> {len(glob.content.splitlines())} files (head):")
        print("  " + "\n  ".join(glob.content.splitlines()[:3]))

        probe = _tool(tools, "run_python").execute({"code": "import sys; print('py', sys.version.split()[0])"})
        print(f"\nrun_python -> {probe.content!r}")

        test = _tool(tools, "run_test").execute(
            {"code": "def test_smoke_passes():\n    assert 1 + 1 == 2\n"}
        )
        print(f"\nrun_test (trivial pass) -> first line: {test.content.splitlines()[0]}")
        ok = "outcome=pass" in test.content and probe.content.startswith("py ")
        print("\nRESULT:", "PASS (session + tools live)" if ok else "CHECK output above")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", help="Instance id for a live Docker run (image must be built).")
    ap.add_argument("--split", default="lite", choices=["lite", "verified"])
    args = ap.parse_args()

    run_offline()
    if args.id:
        run_live(args.id, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
