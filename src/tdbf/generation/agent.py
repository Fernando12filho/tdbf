"""Phase 1: test-generation agent.

Produces candidate Bug Reproduction Tests from issue text + repo context.
The prompt design follows the proposal's literature review: Qureshi et
al. found structural/technical scaffolding (signatures, test skeletons,
near-identical exemplars) is far more predictive of success than
narrative description, so the prompt emphasizes structure and includes a
compact few-shot exemplar slot (the proposal's "compact few-shot
exemplars of high-quality reproduction tests").

The agent does NOT see the gold patch or the dataset's test_patch — only
context_safe_dict() — preserving the experiment's fairness control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tdbf.data.instances import SWEBenchInstance
from tdbf.models.llm import LLMClient

SYSTEM_PROMPT = """\
You are a software testing agent. Your sole job in this phase is to write \
ONE executable test that REPRODUCES the bug described in an issue: the \
test must FAIL on the current (buggy) code and would PASS once the bug is \
correctly fixed.

Hard rules:
- Output exactly one self-contained test, in a single fenced code block.
- Do NOT attempt to fix the bug. Do NOT modify product code.
- Prefer the repository's existing test framework and conventions.
- The test must assert on the SPECIFIC incorrect behavior in the issue, \
not merely that some exception is raised.
- No network, no nondeterminism (no real time, no random without a seed).
"""

USER_TEMPLATE = """\
Repository: {repo}
Base commit (buggy): {base_commit}\

## Issue / bug report
{problem_statement}

## Task
Write a single test that fails on the buggy code above because of the bug \
described, and that would pass once the bug is fixed. Assert on the exact \
described behavior.

Respond with a brief one-line rationale, then the test in ONE fenced \
python code block.
"""

# Slot for the proposal's "compact few-shot exemplars". Kept tiny and
# generic on purpose; per-repo exemplars are an RQ2/RQ3 tuning knob, not
# a Phase 1 scaffold concern.
FEWSHOT_EXEMPLAR = """\
## Example of the expected shape (illustrative only)
Rationale: the reported off-by-one means index N is dropped.

```python
def test_includes_last_element():
    from mypkg.slicer import take
    assert take([1, 2, 3], 3) == [1, 2, 3]   # buggy code returns [1, 2]
```
"""

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class GeneratedTest:
    instance_id: str
    model: str
    raw_response: str
    test_code: str | None  # None if no code block could be extracted

    @property
    def extracted(self) -> bool:
        return self.test_code is not None and self.test_code.strip() != ""


def _extract_code(text: str) -> str | None:
    matches = _CODE_BLOCK.findall(text)
    if not matches:
        return None
    # Take the longest block — guards against the model emitting a tiny
    # illustrative snippet before the real test.
    return max(matches, key=len).strip()


class TestGenerationAgent:
    def __init__(
        self,
        client: LLMClient,
        use_fewshot: bool = True,
        max_problem_chars: int = 6000,
    ) -> None:
        self.client = client
        self.use_fewshot = use_fewshot
        self.max_problem_chars = max_problem_chars

    def build_user_prompt(self, inst: SWEBenchInstance) -> str:
        ctx = inst.context_safe_dict()
        problem = ctx["problem_statement"].strip()
        if len(problem) > self.max_problem_chars:
            problem = problem[: self.max_problem_chars] + "\n...[truncated]"
        body = USER_TEMPLATE.format(
            repo=ctx["repo"],
            base_commit=ctx["base_commit"],
            problem_statement=problem,
        )
        if self.use_fewshot:
            body = FEWSHOT_EXEMPLAR + "\n" + body
        return body

    def generate(self, inst: SWEBenchInstance) -> GeneratedTest:
        user = self.build_user_prompt(inst)
        resp = self.client.complete(SYSTEM_PROMPT, user)
        return GeneratedTest(
            instance_id=inst.instance_id,
            model=resp.model,
            raw_response=resp.text,
            test_code=_extract_code(resp.text),
        )
