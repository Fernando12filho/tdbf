"""Live mechanics check of the procedure agent: real Docker, NO model (no tokens).

Runs the real controller, preflight, official runner, gates and the offline label on
development bugs sympy__sympy-20212 (strict and soft) and django__django-15498 (strict;
Django test package, run_python under runtests.py, failure by exception) from round 1,
not the 114, with a scripted
client in place of the model. Checks the plumbing end to end, not the agent's skill:

    python3 scripts/procedure_live_check.py            # strict and soft
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from procedure_smoke import ScriptedClient  # noqa: E402
from tdbf.agent.procedure import run_procedure  # noqa: E402
from tdbf.agent.procedure_score import label_offline  # noqa: E402
from tdbf.data.instances import InstanceStore  # noqa: E402

IID = "sympy__sympy-20212"
CODE = "from sympy import S, oo, zoo\n\n\ndef test_zero_to_minus_oo():\n    assert S.Zero**-oo == zoo\n"
AMAP = [{"line": 5, "evidence_id": "E1"}]

STRICT = {
    "finish_understand": [[[("finish_understand", {
        "observed": [{"value": "`0`", "quote": "evaluate `0**-oo` produces `0`."}],
        "expected": [{"value": "`zoo`",
                      "quote": "The documentation for the Pow class states that it should return "
                               "`ComplexInfinity`, aka `zoo`"}],
        "unknowns": []})]]],
    "finish_locate": [[[("finish_locate", {"test_dir": "sympy/core/tests", "style_notes": "plain functions"})]]],
    "finish_write": [[[("run_test", {"code": CODE})],
                      [("finish_write", {"code": CODE, "assertion_map": AMAP})]]],
}
SOFT = {"submit_test": [[[("run_test", {"code": CODE, "test_dir": "sympy/core/tests"})],
                         [("submit_test", {"code": CODE, "test_dir": "sympy/core/tests",
                                           "assertion_map": AMAP})]]]}


DJ_ID = "django__django-15498"
DJ_CODE = ("from django.test import SimpleTestCase\nfrom django.views.static import was_modified_since\n\n\n"
           "class EmptyIfModifiedSince(SimpleTestCase):\n    def test_empty_string_is_ignored(self):\n"
           "        self.assertTrue(was_modified_since(''))\n")
DJ_STRICT = {
    "finish_understand": [[[("finish_understand", {
        "observed": [{"value": "raises an exception",
                      "quote": "but now raises exception since d6aff369ad3."}],
        "expected": [{"value": "empty string is ignored",
                      "quote": "Empty string used to be ignored for If-Modified-Since header"}],
        "unknowns": []})]]],
    "finish_reproduce": [[[("run_python", {"code": "from django.views.static import was_modified_since\n"
                                                   "print(was_modified_since(''))"})],
                          [("finish_reproduce", {"run_id": "r1", "output_summary": "raises"})]]],
    "finish_locate": [[[("finish_locate", {"test_dir": "tests/view_tests", "module": "view_tests",
                                           "style_notes": "SimpleTestCase"})]]],
    "finish_write": [[[("run_test", {"code": DJ_CODE})],
                      [("finish_write", {"code": DJ_CODE, "assertion_map": [{"line": 7, "evidence_id": "E1"}]})]]],
}


def main() -> int:
    store = InstanceStore("lite")
    bad = 0
    for iid, mode, script in ((IID, "strict", STRICT), (IID, "soft", SOFT), (DJ_ID, "strict", DJ_STRICT)):
        inst = store.get(iid)
        print(f"\n=== {iid} {mode}")
        res = run_procedure(inst, ScriptedClient(script), mode=mode)
        s = res.record
        print("phases:", [(p["phase"], p["turns"], p["stop_cause"]) for p in s["phases"]])
        print("gates:", [(g["phase"], g["cause"]) for g in s["gates"]])
        print("outcome:", s["outcome"], "flags:", s["flags"], "turns:", res.turns_used,
              "home:", res.home.rel_path if res.home else None,
              "buggy run:", res.outcome_buggy, res.failure_reason_buggy)
        buggy, fixed, cls = label_offline(inst, res.command, res.home.rel_path, res.test_code)
        print("label: buggy", buggy, buggy.failure_reason, "| fixed", fixed, "->", cls.level_label)
        ok = (s["outcome"] == "test" and res.outcome_buggy == "fail"
              and res.failure_reason_buggy in ("assertion", "exception")
              and cls.level_label == "L4_VALIDATED")
        print("CHECK", "ok" if ok else "FAIL")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
