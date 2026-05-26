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

_REPL_INPUT = re.compile(r"^>>> (.+)$", re.MULTILINE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

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

# ── two-phase prompts ─────────────────────────────────────────────────────────

PROBE_SYSTEM = """\
You are a software testing agent. Your job right now is observation only — \
NOT writing a test yet.
"""

PROBE_USER_TEMPLATE = """\
Repository: {repo}
Base commit (buggy): {base_commit}

## Issue / bug report
{problem_statement}
{code_context_section}
## Task
Write a SHORT Python script (no test function, no assertions) that:
- Imports the relevant module(s)
- If the issue compares a CORRECT reference call vs a BUGGY call, print BOTH \
so we can see the difference. Echo each expression before its result, e.g.:
    print(">>> some_call(args)")
    print(repr(some_call(args)))
- Otherwise, just print the one buggy call the same way.
- Wrap any call that might raise in try/except and print the exception.

Output ONLY a single fenced python code block.
"""

PHASE2_USER_TEMPLATE = """\
Repository: {repo}
Base commit (buggy): {base_commit}

## Issue / bug report
{problem_statement}
{code_context_section}
## Observed output from the probe (run on BUGGY code)

```python
{probe_code}
```

```
{observed_output}
```

Each output line is preceded by the expression that produced it (>>> ...).
If multiple calls appear, use the issue text to decide which is the CORRECT \
reference and which is the BUG:
- The CORRECT value is what the fixed code should return (often the first call \
  shown in the issue as a working comparison, or derivable from the issue description).
- The BUG is the wrong value currently produced (typically the last call).

## Task
Write a single test function that:
- Uses the same inputs as the buggy call in the probe
- Asserts the CORRECT expected value so the test FAILS on the buggy output above \
  and PASSES once the bug is fixed
- If the probe captured an exception instead of a return value, assert that the \
  fixed code either raises the RIGHT exception or returns the right value instead

Only assert values you are certain are correct in the fixed version. \
A wrong assertion that still fails after the fix is worse than no assertion.

Buggy output:   <identify from the probe output above>
Expected output: <derive from the issue or the reference call>
Rationale: <one line>
```python
<test>
```
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

    # ── two-phase generation ──────────────────────────────────────────────────

    def extract_probe_from_issue(self, problem_statement: str) -> str | None:
        """Pull runnable probe code from the issue's example snippets.

        Looks for fenced code blocks, converts REPL-style (>>>) lines into
        print(repr(...)) calls, and combines with any setup lines found.
        Returns None if no usable code is found.
        """
        blocks = _FENCE.findall(problem_statement)
        if not blocks:
            return None

        setup: list[str] = []
        repl_calls: list[str] = []
        seen_setup: set[str] = set()

        for block in blocks:
            for line in block.splitlines():
                m = _REPL_INPUT.match(line.rstrip())
                if m:
                    expr = m.group(1).strip()
                    if expr and "(" in expr:
                        # Echo the expression so Phase 2 can match each output to its call.
                        # repr() ensures the label is valid Python regardless of quotes in expr.
                        label = repr(f">>> {expr}")
                        repl_calls.append(f"print({label})")
                        repl_calls.append(f"print(repr({expr}))")
                elif not line.startswith("..."):
                    stripped = line.strip()
                    # skip array/output lines from REPL display
                    if stripped and not stripped.startswith(("array", "[", "False", "True")):
                        if stripped not in seen_setup:
                            setup.append(line)
                            seen_setup.add(stripped)

        if not repl_calls:
            return None

        return "\n".join(setup + [""] + repl_calls)

    def _build_probe_prompt(
        self,
        inst: SWEBenchInstance,
        code_context: dict[str, str] | None,
    ) -> str:
        ctx = inst.context_safe_dict()
        problem = ctx["problem_statement"].strip()[: self.max_problem_chars]
        code_context_section = self._code_context_section(code_context)
        return PROBE_USER_TEMPLATE.format(
            repo=ctx["repo"],
            base_commit=ctx["base_commit"],
            problem_statement=problem,
            code_context_section=code_context_section,
        )

    def _build_phase2_prompt(
        self,
        inst: SWEBenchInstance,
        probe_code: str,
        observed_output: str,
        code_context: dict[str, str] | None,
    ) -> str:
        ctx = inst.context_safe_dict()
        problem = ctx["problem_statement"].strip()[: self.max_problem_chars]
        code_context_section = self._code_context_section(code_context)
        return PHASE2_USER_TEMPLATE.format(
            repo=ctx["repo"],
            base_commit=ctx["base_commit"],
            problem_statement=problem,
            code_context_section=code_context_section,
            probe_code=probe_code.strip(),
            observed_output=observed_output.strip(),
        )

    def _code_context_section(self, code_context: dict[str, str] | None) -> str:
        if not code_context:
            return ""
        files_block = "\n".join(
            f"### {path}\n```python\n{content}\n```"
            for path, content in code_context.items()
        )
        return _CODE_CONTEXT_TEMPLATE.format(files=files_block)

    def generate_with_observation(
        self,
        inst: SWEBenchInstance,
        run_snippet_fn: "object",
        code_context: dict[str, str] | None = None,
    ) -> tuple["GeneratedTest", str, str]:
        """Two-phase generation: observe buggy output, then write the test.

        Returns (GeneratedTest, probe_code, observed_output).
        probe_source is 'issue' or 'model'.
        """
        # Phase 1: get a probe snippet
        probe_code = self.extract_probe_from_issue(inst.problem_statement)
        probe_source = "issue"
        if not probe_code:
            print("  [observe] no example code in issue — asking model for probe ...")
            resp = self.client.complete(
                PROBE_SYSTEM,
                self._build_probe_prompt(inst, code_context),
            )
            probe_code = _extract_code(resp.text)
            probe_source = "model"

        observed_output: str | None = None
        if probe_code:
            print(f"  [observe] running probe ({probe_source}) in container ...")
            observed_output = run_snippet_fn(inst, probe_code)  # type: ignore[call-arg]
            if observed_output:
                print(f"  [observe] captured output ({len(observed_output)} chars)")
            else:
                print("  [observe] probe failed or produced no output")

        # Phase 2: generate the test
        if probe_code and observed_output:
            user = self._build_phase2_prompt(
                inst, probe_code, observed_output, code_context
            )
        else:
            # fall back to one-shot if observation failed
            print("  [observe] falling back to one-shot generation")
            user = self.build_user_prompt(inst, code_context=code_context)

        resp = self.client.complete(SYSTEM_PROMPT, user)
        return (
            GeneratedTest(
                instance_id=inst.instance_id,
                model=resp.model,
                raw_response=resp.text,
                test_code=_extract_code(resp.text),
            ),
            probe_code or "",
            observed_output or "",
        )
