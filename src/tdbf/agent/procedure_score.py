"""Offline L0-L4 label for a procedure-agent test (GOLD BARRIER).

Runs AFTER the agent has finished, in fresh containers, and never feeds back into
the agent: the gold patch is used only here, to tell L3 (fails for another reason)
from L4 (fails on the buggy code, passes once the real fix is applied).

The test runs where the agent placed it (its test home) with the same official
command it used, on the buggy tree and on the gold-fixed tree, through one path, so
the comparison is like for like. `classify` is the project's shared L0-L4 rule.
"""

from __future__ import annotations

from ..harness.official import TestCommand, parse_result
from ..harness.runner import ContainerResult
from ..validation.brt_levels import BRTClassification, classify
from .session import ContainerSession

LABEL_TIMEOUT = 300


def _run(inst: object, cmd: TestCommand, rel_path: str, code: str, *, gold: bool) -> ContainerResult:
    with ContainerSession(inst) as s:
        for step in cmd.setup:
            s.exec(step, timeout=300)
        if gold:
            s.write_file("/tmp/tdbf_gold.patch", inst.gold_patch)  # type: ignore[attr-defined]
            ap = s.exec("git apply /tmp/tdbf_gold.patch", timeout=120)
            if ap.returncode != 0:
                return ContainerResult("absent", None, f"gold patch did not apply: {ap.output[-500:]}")
        s.write_file(f"/testbed/{rel_path}", code)
        if cmd.framework in ("pytest", "tox-pytest"):
            s.ensure_pytest()
        res = s.exec(cmd.for_file(rel_path), timeout=LABEL_TIMEOUT)
        outcome, reason = parse_result(cmd, res.returncode, res.output)
        return ContainerResult(outcome, reason, res.output[-3000:])


def label_offline(inst: object, cmd: TestCommand, rel_path: str, code: str
                  ) -> tuple[ContainerResult, ContainerResult, BRTClassification]:
    """(buggy result, fixed result, L0-L4 classification) for a submitted test."""
    buggy = _run(inst, cmd, rel_path, code, gold=False)
    fixed = _run(inst, cmd, rel_path, code, gold=True)
    return buggy, fixed, classify(buggy, fixed)
