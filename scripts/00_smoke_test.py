"""Smoke test: offline, no API keys, no Docker.

Proves the Phase 1 wiring is correct end-to-end *except* the two parts
that genuinely need external resources (a live model, a Docker harness):

  1. SWEBenchInstance parsing handles both JSON-string and list forms
     of FAIL_TO_PASS / PASS_TO_PASS.
  2. The generation agent builds a prompt that does NOT leak the gold
     patch or the dataset test_patch (the fairness control).
  3. A fake model response is parsed into a GeneratedTest.
  4. make_test_patch produces a well-formed unified diff.
  5. The BRT level classifier returns the right level for every cell of
     the (buggy, fixed) outcome matrix — including the C2P rate.

Run:  python scripts/00_smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdbf.data.instances import SWEBenchInstance
from tdbf.generation.agent import TestGenerationAgent, _extract_code
from tdbf.harness.runner import _parse_failure_reason, make_test_patch
from tdbf.models.llm import LLMResponse
from tdbf.validation.brt_levels import BRTLevel, classify, c2p_rate


def test_instance_parsing():
    row = {
        "instance_id": "demo__pkg-1",
        "repo": "demo/pkg",
        "base_commit": "abc123",
        "patch": "PRODUCT FIX DIFF — must never reach the agent",
        "test_patch": "DATASET TEST DIFF — must never reach the agent",
        "problem_statement": "take() drops the last element.",
        "FAIL_TO_PASS": '["tests/test_x.py::test_a"]',  # JSON string form
        "PASS_TO_PASS": ["tests/test_x.py::test_b"],     # list form
    }
    inst = SWEBenchInstance.from_row(row)
    assert inst.fail_to_pass == ["tests/test_x.py::test_a"]
    assert inst.pass_to_pass == ["tests/test_x.py::test_b"]
    safe = inst.context_safe_dict()
    blob = repr(safe)
    assert "FIX DIFF" not in blob and "TEST DIFF" not in blob
    print("  [ok] instance parsing + fairness control")
    return inst


def test_prompt_no_leak(inst):
    class _NullClient:
        name = "null"

        def complete(self, system, user, *, max_tokens=4096):
            return LLMResponse(text="", model="null")

    agent = TestGenerationAgent(_NullClient())
    prompt = agent.build_user_prompt(inst)
    assert "FIX DIFF" not in prompt
    assert "TEST DIFF" not in prompt
    assert inst.problem_statement in prompt
    print("  [ok] generation prompt leaks neither gold patch nor test_patch")


def test_code_extraction():
    resp = (
        "Rationale: off-by-one.\n"
        "```python\n"
        "def test_last():\n"
        "    assert take([1,2,3],3)==[1,2,3]\n"
        "```\n"
    )
    code = _extract_code(resp)
    assert code is not None and "def test_last" in code
    print("  [ok] code-block extraction")
    return code


def test_patch_format(code):
    patch = make_test_patch("tests/test_brt_demo.py", code)
    assert patch.startswith("diff --git a/tests/test_brt_demo.py")
    assert "--- /dev/null" in patch
    assert "+++ b/tests/test_brt_demo.py" in patch
    assert "@@ -0,0 +1," in patch
    # every content line of the body is an addition line
    body_lines = [l for l in patch.splitlines() if l and l[0] in "+-@d"]
    assert any(l.startswith("+def test_last") for l in patch.splitlines())
    print("  [ok] test patch is a well-formed pure-addition diff")


def test_brt_matrix():
    # (buggy, fixed) -> expected level
    cases = {
        ("absent", "pass"): BRTLevel.L0_MALFORMED,
        ("pass", "pass"): BRTLevel.L1_NONREPRODUCING,
        ("pass", "fail"): BRTLevel.L1_NONREPRODUCING,
        ("error", "pass"): BRTLevel.L2_CRASHES,
        ("fail", "fail"): BRTLevel.L3_CANDIDATE,
        ("fail", "error"): BRTLevel.L3_CANDIDATE,
        ("fail", "pass"): BRTLevel.L4_VALIDATED,
    }
    for (b, f), expected in cases.items():
        got = classify(b, f).level
        assert got == expected, f"({b},{f}) -> {got.name}, want {expected.name}"
    print("  [ok] BRT level matrix (all 7 outcome pairs)")

    # C2P rate: 2 validated out of 3 candidates (L3,L3,L4) = 0.666...
    cls = [
        classify("fail", "fail"),   # L3
        classify("fail", "error"),  # L3
        classify("fail", "pass"),   # L4
        classify("pass", "pass"),   # L1 (not a candidate, excluded)
    ]
    rate = c2p_rate(cls)
    assert abs(rate - (1 / 3)) < 1e-9, rate
    assert c2p_rate([classify("pass", "pass")]) is None  # no candidates
    print("  [ok] C2P rate (and None when no candidates)")


def test_failure_reason_parsing():
    """The gold-free gate accepts a BRT iff failure_reason != 'runtime', so this
    parser decides which tests reach the fix phase. Both directions matter: a
    crash misread as an assertion dispatches the fix agent at a test that can
    never pass, and an assertion misread as a crash throws away a valid BRT."""
    cases = [
        # (label, pytest 'E' output, expected reason)
        ("bare rewritten assert",
         "E       assert [1, 2] == [1, 3]\nE         At index 1 diff: 2 != 3", "assertion"),
        ("AssertionError with message",
         "E       AssertionError: matrices differ", "assertion"),
        ("pytest.raises saw nothing",
         "E       Failed: DID NOT RAISE <class 'ValueError'>", "assertion"),
        ("assert next to a deprecation warning",
         "E       assert 1 == 2\nE       DeprecationWarning: old api", "assertion"),
        ("missing fixture file",
         "E       FileNotFoundError: [Errno 2] No such file or directory: 'test.qdp'", "runtime"),
        ("django FieldDoesNotExist",
         "E       django.core.exceptions.FieldDoesNotExist: no field named 'x'", "runtime"),
        ("model DoesNotExist",
         "E       myapp.models.Book.DoesNotExist: Book matching query does not exist.", "runtime"),
        ("MultipleObjectsReturned",
         "E       django.core.exceptions.MultipleObjectsReturned: got 2", "runtime"),
        ("EmptyResultSet (no message)",
         "E       django.core.exceptions.EmptyResultSet", "runtime"),
        ("SystemExit", "E       SystemExit: 2", "runtime"),
        ("StopIteration (no message)", "E       StopIteration", "runtime"),
        ("import error", "E       ImportError: cannot import name 'X'", "runtime"),
        ("no failure lines at all", "collected 1 item\n1 passed", None),
        ("empty output", "", None),
        # Verbatim from astropy__astropy-12907 on the buggy tree. This test is
        # offline-labelled L4 — a perfect BRT — yet an earlier parser read the
        # `Expected:` / `Got:` lines of its own assertion message as exception
        # class names and reported "runtime", so the gate rejected it.
        ("multi-line assertion message (astropy-12907, real output)",
         "E   AssertionError: separability_matrix for nested CompoundModel gave wrong result.\n"
         "E     Expected:\n"
         "E     [[ True  True False False]\n"
         "E      [False False False  True]]\n"
         "E     Got:\n"
         "E     [[ True  True False False]\n"
         "E      [False False  True  True]]\n"
         "E   assert False\n"
         "E    +  where False = <function array_equal at 0x7ffffbaa6530>(array([[ True]]))",
         "assertion"),
        ("prose line that is not an exception", "E     Something:", None),
    ]
    for label, output, expected in cases:
        got = _parse_failure_reason(output)
        assert got == expected, f"{label!r}: got {got!r}, want {expected!r}"
    print(f"  [ok] failure_reason parsing ({len(cases)} cases: crash vs assertion)")


if __name__ == "__main__":
    print("Phase 1 smoke test (offline)\n" + "-" * 40)
    inst = test_instance_parsing()
    test_prompt_no_leak(inst)
    code = test_code_extraction()
    test_patch_format(code)
    test_brt_matrix()
    test_failure_reason_parsing()
    print("-" * 40)
    print("ALL SMOKE CHECKS PASSED")
