"""SWE-bench instance loading.

Loads instances from the HuggingFace datasets the proposal commits to
(SWE-bench Lite for pilots, Verified for scoring) and exposes exactly the
fields the proposal enumerates: instance_id, repo, base_commit, patch,
test_patch, problem_statement, FAIL_TO_PASS, PASS_TO_PASS.

No network or heavy deps are imported at module load so the smoke test
can import this without `datasets` installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# The two splits the proposal names. Lite is the pilot/tuning split;
# Verified is the primary scoring split. BugsInPy / GitBugs are
# deliberately *not* wired here — they are later adapters behind the
# same SWEBenchInstance interface, not Phase 1 work.
DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
}


def _parse_test_list(value: Any) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS are JSON-encoded lists in the dataset.

    They sometimes arrive already-parsed (list) and sometimes as a JSON
    string. Handle both rather than assuming.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            # Fall back to a single-element list rather than silently
            # dropping data we don't recognize.
            return [value]
    return [str(value)]


@dataclass(frozen=True)
class SWEBenchInstance:
    """One SWE-bench task instance.

    Fields mirror the proposal's enumerated key fields. `gold_patch` and
    `test_patch` are kept distinct: the gold patch is the human fix used
    for BRT validation; the test_patch is the dataset's own test edit and
    must NOT be shown to the generation agent (it would leak the answer).
    """

    instance_id: str
    repo: str
    base_commit: str
    gold_patch: str
    test_patch: str
    problem_statement: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    version: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SWEBenchInstance":
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            gold_patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            problem_statement=row.get("problem_statement", ""),
            fail_to_pass=_parse_test_list(row.get("FAIL_TO_PASS")),
            pass_to_pass=_parse_test_list(row.get("PASS_TO_PASS")),
            version=row.get("version"),
        )

    def context_safe_dict(self) -> dict[str, Any]:
        """The subset of fields the generation agent is allowed to see.

        Crucially excludes gold_patch and test_patch — exposing either
        would invalidate the experiment (the agent would be copying the
        known fix or the known test). This is the fairness control the
        proposal's Section 5.3 describes.
        """
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }


class InstanceStore:
    """Lazily loads a SWE-bench split via HuggingFace datasets."""

    def __init__(self, split: str = "lite") -> None:
        if split not in DATASETS:
            raise ValueError(
                f"Unknown split {split!r}. Options: {sorted(DATASETS)}"
            )
        self.split = split
        self.dataset_name = DATASETS[split]
        self._rows: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            # Imported here so importing this module never requires
            # `datasets` (keeps the offline smoke test importable).
            from datasets import load_dataset

            ds = load_dataset(self.dataset_name, split="test")
            self._rows = list(ds)
        return self._rows

    def get(self, instance_id: str) -> SWEBenchInstance:
        for row in self._load():
            if row["instance_id"] == instance_id:
                return SWEBenchInstance.from_row(row)
        raise KeyError(
            f"{instance_id!r} not found in {self.dataset_name}"
        )

    def iter_instances(
        self, limit: int | None = None
    ) -> Iterable[SWEBenchInstance]:
        for i, row in enumerate(self._load()):
            if limit is not None and i >= limit:
                return
            yield SWEBenchInstance.from_row(row)

    def first(self) -> SWEBenchInstance:
        return SWEBenchInstance.from_row(self._load()[0])
