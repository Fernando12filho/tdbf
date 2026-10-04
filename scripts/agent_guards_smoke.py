"""Offline checks for the fixes made before the extension runs (2026-10-04
code review). No Docker, no API.

  1. Colour codes no longer hide a crash from the gate's failure-reason parser
     (scikit-learn >= ~1.3 forces `--color=yes`).
  2. edit_file refuses paths outside the repository instead of "editing" a copy.
  3. Gate-retry attempts are kept: record (`brt_retries`) and transcript
     (`prior_attempts`); a run with no retry gains neither key.
  4. build_results leaves out an instance its environment record marks not-valid.

    python3 scripts/agent_guards_smoke.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_tools_smoke import FakeSession  # noqa: E402

from tdbf import metrics  # noqa: E402
from tdbf.agent import pipeline  # noqa: E402
from tdbf.agent.tools import EditFileTool  # noqa: E402
from tdbf.data.instances import SWEBenchInstance  # noqa: E402
from tdbf.harness.runner import ContainerResult, _parse_failure_reason  # noqa: E402
from tdbf.models.messages import ToolCall, ToolUseResponse  # noqa: E402


def check_colour() -> None:
    red, bold, reset = "\x1b[31m", "\x1b[1m", "\x1b[0m"
    crash = f"{bold}{red}E   ValueError: could not broadcast{reset}\n1 failed"
    fired = (f"{bold}{red}E   ValueError: boom{reset}\n"
             f"{bold}{red}E   AssertionError: raised unexpectedly{reset}\n1 failed")
    assert _parse_failure_reason(crash) == "runtime", _parse_failure_reason(crash)
    assert _parse_failure_reason(fired) == "assertion"
    # plain output unchanged
    assert _parse_failure_reason("E   ImportError: no module x\n") == "runtime"
    assert _parse_failure_reason("E   assert 1 == 2\n") == "assertion"
    print("  ok  colour codes stripped before parsing (crash -> runtime)")


def check_edit_paths() -> None:
    s = FakeSession()
    tool = EditFileTool(s)  # type: ignore[arg-type]
    for bad in ("/opt/miniconda3/lib/python3.9/site.py", "../outside.py", "pkg/../../x.py"):
        res = tool.execute({"path": bad, "old_str": "a", "new_str": "b"})
        assert res.is_error and "inside the repository" in res.content, (bad, res.content)
    assert not s.writes, s.writes
    ok = tool.execute({"path": "exists.py", "old_str": "hello", "new_str": "bye"})
    assert not ok.is_error and s.writes == {"/testbed/exists.py": "print('bye')\n"}, s.writes
    print("  ok  edit_file rejects absolute / '..' paths; repo paths still edit")


class _SubmitEachTurn:
    """Each agent run: submit a test at once. The gate decides by call order."""

    name = "openrouter:demo/model:free"

    def __init__(self) -> None:
        self.n = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096):
        self.n += 1
        code = f"def test_brt():\n    assert compute() == {self.n}\n"
        return ToolUseResponse(text="", tool_calls=[ToolCall(f"c{self.n}", "submit_test",
                                                             {"code": code})],
                               stop_reason="tool_use", input_tokens=100, output_tokens=10)


def _phase(gate_outcomes: list[ContainerResult]):
    inst = SWEBenchInstance(instance_id="demo__pkg-1", repo="demo/pkg", base_commit="abc",
                            gold_patch="", test_patch="MUST NOT BE USED",
                            problem_statement="compute() is wrong")
    seq = iter(gate_outcomes)
    pipeline.ContainerSession = lambda instance, keep=False: _Live(instance.repo)
    pipeline._score_on_buggy_in_session = lambda session, sub: next(seq)
    return pipeline.run_brt_phase(inst, _SubmitEachTurn(), max_iterations=3,
                                  max_total_tokens=10_000, score_fixed=False), inst


class _Live(FakeSession):
    def start(self):
        return self

    def close(self):
        pass


def check_retries_recorded() -> None:
    m07 = _load_07()
    # attempt 1 passes on buggy (L1), attempt 2 crashes (runtime), attempt 3 reproduces
    phase, inst = _phase([ContainerResult("pass"),
                          ContainerResult("fail", failure_reason="runtime"),
                          ContainerResult("fail", failure_reason="assertion")])
    assert phase.reproduces_on_buggy and len(phase.prior_attempts) == 2
    rec = m07.brt_deployable_record(inst, "m", split="lite", flags={}, phase=phase,
                                    gate_accepted=True)
    r = rec["brt_retries"]
    assert [x["attempt"] for x in r] == [1, 2]
    assert [(x["outcome_buggy"], x["failure_reason_buggy"]) for x in r] == [
        ("pass", None), ("fail", "runtime")], r
    assert "== 1" in r[0]["test_code"] and "== 2" in r[1]["test_code"]
    assert "== 3" in rec["attempts"][0]["test_code"]  # final attempt unchanged in place
    assert pipeline.brt_run_record(inst, "m", split="lite", flags={},
                                   phase=phase)["brt_retries"] == r
    t = pipeline.brt_transcript_dict(inst, "m", phase)
    assert len(t["prior_attempts"]) == 2 and t["prior_attempts"][0]["messages"]
    assert "MUST NOT BE USED" not in str(t) + str(rec)

    # no retry -> neither key appears (records stay byte-identical to before)
    phase, inst = _phase([ContainerResult("fail", failure_reason="assertion")])
    assert phase.prior_attempts == []
    rec = m07.brt_deployable_record(inst, "m", split="lite", flags={}, phase=phase,
                                    gate_accepted=True)
    assert "brt_retries" not in rec
    assert "brt_retries" not in pipeline.brt_run_record(inst, "m", split="lite", flags={},
                                                        phase=phase)
    assert "prior_attempts" not in pipeline.brt_transcript_dict(inst, "m", phase)
    print("  ok  gate-retry attempts kept in record + transcript; absent when no retry")


def _load_07():
    spec = importlib.util.spec_from_file_location(
        "deployable_pipeline_guards", ROOT / "scripts" / "07_deployable_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def check_env_exclusion() -> None:
    def unit(iid, group, key, resolved):
        return {"instance_id": iid, "repo": "demo/pkg", "mode": "sweep_unit",
                "model": "m", "timestamp": "2026-10-05T00:00:00",
                "flags": {"run_group": group}, key: {"resolved": resolved}, "calls": 1}

    recs = []
    for iid in ("a__a-1", "a__a-2"):
        recs += [unit(iid, "rq1-sweep", "arm_b", True), unit(iid, "rq1-armA", "arm_a", False)]
    both = metrics.build_results(recs)
    assert both["n"] == 2
    res = metrics.build_results(recs, sample_ids=["a__a-1", "a__a-2"],
                                env_states={"a__a-1": "valid", "a__a-2": "environment_invalid"})
    assert res["n"] == 1 and res["paired"]["tf_only"] == ["a__a-1"], res["paired"]
    assert res["left_out"] == [{"instance_id": "a__a-2", "state": "environment_invalid"}]
    print("  ok  build_results leaves out env-invalid instances and lists them")

    # Extension: one AB unit per instance in rq1-ext pools with the base sample's
    # one-arm units, and each arm's calls come from its own records, not the
    # unit's combined total.
    ab = {"instance_id": "e__e-1", "repo": "demo/pkg", "mode": "sweep_unit", "model": "m",
          "timestamp": "2026-10-06T00:00:00", "calls": 70,
          "flags": {"run_group": "rq1-ext", "arms": "AB", "pair_id": "p1"},
          "arm_a": {"resolved": False}, "arm_b": {"resolved": True}}
    arm_recs = [
        {"instance_id": "e__e-1", "mode": "agent_deployable", "timestamp": "2026-10-06",
         "gate_accepted": True, "best_level": 4,
         "flags": {"run_group": "rq1-ext", "arm": "deployable", "pair_id": "p1",
                   "served_by": {"m|P": 20}}},
        {"instance_id": "e__e-1", "mode": "fix_agent", "timestamp": "2026-10-06",
         "resolved": True, "attempts": [{"stop_cause": "submit"}],
         "flags": {"run_group": "rq1-ext", "arm": "deployable", "pair_id": "p1",
                   "served_by": {"m|P": 25}}},
        {"instance_id": "e__e-1", "mode": "fix_agent", "timestamp": "2026-10-06",
         "resolved": False, "attempts": [{"stop_cause": "submit_forced"}],
         "flags": {"run_group": "rq1-ext", "arm": "reactive", "pair_id": "p1",
                   "served_by": {"m|P": 25}}},
    ]
    pooled = metrics.build_results(recs + [ab] + arm_recs)
    assert pooled["n"] == 3, pooled["n"]
    assert pooled["paired"]["tf_only"] == ["a__a-1", "a__a-2", "e__e-1"], pooled["paired"]
    arms = {a["label"]: a for a in pooled["arms"]}
    assert arms["test-first"]["calls_total"] == 1 + 1 + 45, arms["test-first"]["calls_total"]
    assert arms["reactive"]["calls_total"] == 1 + 1 + 25, arms["reactive"]["calls_total"]
    assert pooled["funnel"]["gate_rate"]["num"] == 1     # only e__e-1 has a test record
    print("  ok  extension AB units pool with the base sample; per-arm calls not doubled")


def main() -> int:
    check_colour()
    check_edit_paths()
    check_retries_recorded()
    check_env_exclusion()
    print("\nPre-extension guards: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
