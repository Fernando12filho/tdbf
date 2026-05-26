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
Base commit (buggy): {base_commit}

## Issue / bug report
{problem_statement}
{code_context_section}
## Task
Before writing any code, reason through the bug in two lines:
  Buggy output:    what does the code currently return or do wrong?
  Expected output: what should it return or do once the bug is fixed?

Then write a single test that:
- Calls the relevant function with a minimal input that triggers the bug.
- Asserts the EXPECTED output (the correct value, not the buggy one).
- Only asserts values you derived in the two lines above — a wrong \
assertion that still fails after the fix is worse than no assertion.

Respond in this exact format:
Buggy output: <what the code returns now>
Expected output: <what it should return after the fix>
Rationale: <one line>
```python
<test>
```
"""

_CODE_CONTEXT_TEMPLATE = """\

## Relevant source files (buggy version — do NOT fix them, only test them)
{files}
"""

# Slot for the proposal's "compact few-shot exemplars". Kept tiny and
# generic on purpose; per-repo exemplars are an RQ2/RQ3 tuning knob, not
# a Phase 1 scaffold concern.
FEWSHOT_EXEMPLAR = """\
## Example of the expected format (illustrative only)
Buggy output: take([1, 2, 3], 3) returns [1, 2] — drops the last element.
Expected output: should return [1, 2, 3].
Rationale: off-by-one in the slice upper bound excludes index N.
```python
def test_includes_last_element():
    from mypkg.slicer import take
    assert take([1, 2, 3], 3) == [1, 2, 3]
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

    def build_user_prompt(
        self,
        inst: SWEBenchInstance,
        code_context: dict[str, str] | None = None,
    ) -> str:
        ctx = inst.context_safe_dict()
        problem = ctx["problem_statement"].strip()
        if len(problem) > self.max_problem_chars:
            problem = problem[: self.max_problem_chars] + "\n...[truncated]"

        if code_context:
            files_block = "\n".join(
                f"### {path}\n```python\n{content}\n```"
                for path, content in code_context.items()
            )
            code_context_section = _CODE_CONTEXT_TEMPLATE.format(
                files=files_block
            )
        else:
            code_context_section = ""

        body = USER_TEMPLATE.format(
            repo=ctx["repo"],
            base_commit=ctx["base_commit"],
            problem_statement=problem,
            code_context_section=code_context_section,
        )
        if self.use_fewshot:
            body = FEWSHOT_EXEMPLAR + "\n" + body
        return body

    def generate(
        self,
        inst: SWEBenchInstance,
        code_context: dict[str, str] | None = None,
    ) -> GeneratedTest:
        user = self.build_user_prompt(inst, code_context=code_context)
        resp = self.client.complete(SYSTEM_PROMPT, user)
        return GeneratedTest(
            instance_id=inst.instance_id,
            model=resp.model,
            raw_response=resp.text,
            test_code=_extract_code(resp.text),
        )
