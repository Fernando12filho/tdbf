"""BRT validation: classify a generated test into levels L0–L4.

This encodes the proposal's distinction between a *candidate* test and a
genuinely *validated* BRT. The classification is purely a function of two
observed outcomes:

    outcome_on_buggy  : how the generated test behaves on base_commit
    outcome_on_fixed  : how it behaves with the GOLD patch applied

Each outcome is one of:
    "pass"   test ran and all assertions passed
    "fail"   test ran and an assertion failed (the desired signal)
    "error"  test raised before/around assertions (collection/runtime)
    "absent" test could not be collected / file did not apply

The gold patch is the validation oracle: a true BRT must FAIL on buggy
code *and* PASS once the human's correct fix is applied. That fail->pass
flip (L3 -> L4) is the candidate-to-plausible (C2P) transition the whole
proposal is built around. Anything that passes on buggy code is a false
positive (L1) and is worse than useless — it would green-light a
non-fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

Outcome = str  # "pass" | "fail" | "error" | "absent"


class BRTLevel(IntEnum):
    L0_MALFORMED = 0      # never ran on buggy code
    L1_NONREPRODUCING = 1  # passed on buggy code (false positive)
    L2_CRASHES = 2        # errored on buggy code (no clean assertion)
    L3_CANDIDATE = 3      # failed on buggy, but not fixed by gold patch
    L4_VALIDATED = 4      # failed on buggy AND passes on gold-fixed (C2P)


@dataclass(frozen=True)
class BRTClassification:
    level: BRTLevel
    outcome_on_buggy: Outcome
    outcome_on_fixed: Outcome
    rationale: str
    l3_subtype: str | None = None  # "runtime" | "wrong_assertion" | None (L3 only)

    @property
    def is_valid_brt(self) -> bool:
        return self.level == BRTLevel.L4_VALIDATED

    @property
    def is_candidate(self) -> bool:
        # "Candidate" in the proposal's sense = reproduces a failure on
        # buggy code (L3 or L4). The C2P rate is L4 / (L3 + L4).
        return self.level >= BRTLevel.L3_CANDIDATE

    @property
    def level_label(self) -> str:
        """Human-readable level name, with L3 sub-type when available."""
        if self.level == BRTLevel.L3_CANDIDATE and self.l3_subtype:
            return f"L3_{self.l3_subtype.upper()}"
        return self.level.name


def classify(
    outcome_on_buggy: "Outcome | object", outcome_on_fixed: "Outcome | object"
) -> BRTClassification:
    """Map the (buggy, fixed) outcome pair to a BRT level.

    The buggy-side outcome alone determines L0–L2 vs. {L3,L4}. The
    fixed-side outcome only matters once the test actually reproduces a
    failure, and is what separates a real BRT (L4) from a test that fails
    for the wrong reason (L3).

    Accepts plain strings or ContainerResult objects (duck-typed via str()).
    """
    b = str(outcome_on_buggy)
    f = str(outcome_on_fixed)
    failure_reason: str | None = getattr(outcome_on_buggy, "failure_reason", None)

    if b == "absent":
        return BRTClassification(
            BRTLevel.L0_MALFORMED, b, f,
            "Test did not apply/collect on buggy code; not executable.",
        )
    if b == "pass":
        return BRTClassification(
            BRTLevel.L1_NONREPRODUCING, b, f,
            "Test PASSES on buggy code: false positive, never caught "
            "the defect. Would green-light a non-fix.",
        )
    if b == "error":
        return BRTClassification(
            BRTLevel.L2_CRASHES, b, f,
            "Test ERRORS on buggy code rather than failing an "
            "assertion: weak signal, no clean behavioral target.",
        )
    # b == "fail": the test reproduces *a* failure. Gold patch decides
    # whether it is the RIGHT failure.
    if f == "pass":
        return BRTClassification(
            BRTLevel.L4_VALIDATED, b, f,
            "Fail->pass under the gold patch: a validated BRT (C2P). "
            "This is the only scientifically valid case.",
        )
    l3_subtype: str | None = None
    if failure_reason == "runtime":
        l3_subtype = "runtime"
    elif failure_reason == "assertion":
        l3_subtype = "wrong_assertion"

    return BRTClassification(
        BRTLevel.L3_CANDIDATE, b, f,
        f"Fails on buggy code but is '{f}' under the gold patch: "
        "reproduces a failure, but not the one the human fix targets.",
        l3_subtype=l3_subtype,
    )


def c2p_rate(classifications: list[BRTClassification]) -> float | None:
    """Candidate-to-Plausible rate = L4 / (L3 + L4).

    Returns None if there are no candidates (denominator zero) rather
    than a misleading 0.0.
    """
    candidates = [c for c in classifications if c.is_candidate]
    if not candidates:
        return None
    validated = sum(1 for c in candidates if c.is_valid_brt)
    return validated / len(candidates)
