"""The vendor-agnostic agent loop.

One loop, driven entirely by the model: it picks tools and decides when it's
done. Identical behaviour on Claude and qwen — only the `LLMClient` swaps, which
is what keeps the RQ1 comparison clean.

Termination (per the agreed design): a successful call to a *terminal* tool
(`submit_*`) ends the run and captures its artifact; as a fallback, an assistant
turn with no tool calls also ends it. Hard caps on iterations and cumulative
tokens guard the budget so a flailing model (expected for qwen, that's RQ1) can't
burn the run open-ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.llm import LLMClient
from ..models.messages import Message, ToolCall
from .types import Tool, ToolResult


@dataclass
class AgentStep:
    """One model turn plus the tool results it triggered (telemetry/trace)."""

    iteration: int
    text: str
    tool_calls: list[ToolCall]
    tool_results: list[str]
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float | None


@dataclass
class AgentResult:
    """Outcome of a full agent run.

    `stop_cause` is one of: "submit" (terminal tool fired), "natural" (no tool
    calls), "max_iterations", "max_tokens". `submission` is the terminal tool's
    artifact, if any. Token totals are real (accumulated across turns) — this is
    the usage capture the plan flags as missing from the old Ollama runs.
    """

    final_text: str
    submission: str | None
    stop_cause: str
    steps: list[AgentStep]
    messages: list[Message]
    total_input_tokens: int
    total_output_tokens: int
    iterations: int

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class Agent:
    def __init__(
        self,
        client: LLMClient,
        tools: list[Tool],
        system: str,
        *,
        max_iterations: int = 12,
        max_total_tokens: int = 200_000,
        max_tokens_per_turn: int = 4096,
        verbose: bool = False,
    ) -> None:
        names = [t.name for t in tools]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate tool names: {sorted(dupes)}")
        self.client = client
        self.tools = {t.name: t for t in tools}
        self.schemas = [t.schema() for t in tools]
        self.system = system
        self.max_iterations = max_iterations
        self.max_total_tokens = max_total_tokens
        self.max_tokens_per_turn = max_tokens_per_turn
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            import sys

            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    def run(self, task: str) -> AgentResult:
        messages: list[Message] = [Message(role="user", content=task)]
        steps: list[AgentStep] = []
        in_tok = out_tok = 0
        submission: str | None = None
        stop_cause = "max_iterations"
        final_text = ""

        for i in range(self.max_iterations):
            if in_tok + out_tok >= self.max_total_tokens:
                stop_cause = "max_tokens"
                break

            resp = self.client.run_tools(
                system=self.system,
                messages=messages,
                tools=self.schemas,
                max_tokens=self.max_tokens_per_turn,
            )
            in_tok += resp.input_tokens or 0
            out_tok += resp.output_tokens or 0
            final_text = resp.text
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )
            self._log(
                f"[iter {i}] stop={resp.stop_reason} "
                f"tools={[tc.name for tc in resp.tool_calls]} "
                f"tok={resp.input_tokens}/{resp.output_tokens}"
            )

            # No tool calls -> the model is done (natural-stop fallback).
            if not resp.tool_calls:
                stop_cause = "natural"
                steps.append(
                    AgentStep(i, resp.text, [], [], resp.input_tokens, resp.output_tokens, resp.duration_seconds)
                )
                break

            results: list[str] = []
            terminated = False
            for tc in resp.tool_calls:
                res = self._execute(tc)
                messages.append(
                    Message(role="tool", tool_call_id=tc.id, content=res.content)
                )
                results.append(res.content)
                self._log(f"    -> {tc.name}({tc.arguments}) => {'ERR ' if res.is_error else ''}{res.content[:120]!r}")
                tool = self.tools.get(tc.name)
                if tool is not None and tool.terminal and not res.is_error:
                    submission = res.content
                    terminated = True

            steps.append(
                AgentStep(i, resp.text, resp.tool_calls, results, resp.input_tokens, resp.output_tokens, resp.duration_seconds)
            )

            if terminated:
                stop_cause = "submit"
                break

        return AgentResult(
            final_text=final_text,
            submission=submission,
            stop_cause=stop_cause,
            steps=steps,
            messages=messages,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            iterations=len(steps),
        )

    def _execute(self, tc: ToolCall) -> ToolResult:
        """Run one tool call, turning every failure into a corrective message
        the model can recover from rather than an exception that kills the run."""
        if tc.malformed:
            return ToolResult(
                f"Arguments for {tc.name!r} were not a valid JSON object: "
                f"{tc.raw_arguments!r}. Re-call the tool with a valid JSON object.",
                is_error=True,
            )
        tool = self.tools.get(tc.name)
        if tool is None:
            return ToolResult(
                f"Unknown tool {tc.name!r}. Available tools: {sorted(self.tools)}.",
                is_error=True,
            )
        try:
            return tool.execute(tc.arguments)
        except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash
            return ToolResult(f"Tool {tc.name!r} raised {type(exc).__name__}: {exc}", is_error=True)
