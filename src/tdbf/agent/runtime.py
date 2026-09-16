"""The vendor-agnostic agent loop.

One loop, driven entirely by the model: it picks tools and decides when it's
done. Identical behaviour on Claude and qwen — only the `LLMClient` swaps, which
is what keeps the RQ1 comparison clean.

Termination (per the agreed design): a successful call to a *terminal* tool
(`submit_*`) ends the run and captures its artifact; as a fallback, an assistant
turn with no tool calls also ends it. Before accepting that fallback, weaker
models that narrate a result in prose instead of firing the terminal tool get a
bounded "submit-nudge" (`submit_nudge_limit`) prompting them to actually call it
— this removes a scaffolding confound that otherwise auto-scores such runs L0
(see RQ1). Hard caps on iterations and cumulative tokens guard the budget so a
flailing model can't burn the run open-ended.
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
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


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
    # Prompt-caching split of the input side (0 when the provider doesn't cache).
    # Kept separate from total_input_tokens because they are billed at different
    # rates — folding them together would misstate the cost either way.
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    # No-tool turns that ended because they hit max_tokens_per_turn. Separates a
    # budget artefact from a model that stopped acting (RQ3's stage question).
    truncated_turns: int = 0

    @property
    def total_tokens(self) -> int:
        return (self.total_input_tokens + self.total_output_tokens
                + self.total_cache_write_tokens + self.total_cache_read_tokens)


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
        submit_nudge_limit: int = 2,
        submit_grace: int = 3,
        verbose: bool = False,
    ) -> None:
        names = [t.name for t in tools]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate tool names: {sorted(dupes)}")
        self.client = client
        self.tools = {t.name: t for t in tools}
        self.schemas = [t.schema() for t in tools]
        self.terminal_tools = [t.name for t in tools if t.terminal]
        self.system = system
        self.max_iterations = max_iterations
        self.max_total_tokens = max_total_tokens
        self.max_tokens_per_turn = max_tokens_per_turn
        self.submit_nudge_limit = submit_nudge_limit
        self.submit_grace = submit_grace
        self.verbose = verbose

    def _nudge_text(self) -> str:
        names = " or ".join(repr(n) for n in self.terminal_tools)
        return (
            "You ended your turn without calling a tool. A plain-text answer is "
            f"not recorded — to finish you MUST call {names} with your result. "
            "If you are not finished yet, keep investigating with the available tools."
        )

    def _truncated_text(self) -> str:
        return (
            "Your response was cut off at the per-turn output limit before you called "
            "a tool, so nothing happened. Keep your reasoning short and call a tool "
            "in this turn."
        )

    def _force_submit_text(self) -> str:
        names = " or ".join(repr(n) for n in self.terminal_tools)
        return (
            "You have used your full investigation budget. Do NOT read, grep, or run "
            f"anything else — call {names} NOW with the best result you can produce from "
            "what you have already learned. A submitted result is REQUIRED; an imperfect "
            "one is far better than none (a non-submission counts as a total failure)."
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            import sys

            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    def run(self, task: str) -> AgentResult:
        messages: list[Message] = [Message(role="user", content=task)]
        steps: list[AgentStep] = []
        in_tok = out_tok = 0
        cache_w = cache_r = 0
        submission: str | None = None
        stop_cause = "max_iterations"
        final_text = ""
        nudges_used = 0  # total, for telemetry
        truncations = 0  # no-tool turns cut off by max_tokens_per_turn
        consecutive_no_tool = 0  # what the nudge limit actually bounds

        for i in range(self.max_iterations):
            # Count cached input against the budget too. With prompt caching on,
            # most of a turn's input arrives as cache_read rather than input, so
            # a budget of `in_tok + out_tok` would quietly stop binding and the
            # arms would no longer be running under matched budgets.
            if in_tok + out_tok + cache_w + cache_r >= self.max_total_tokens:
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
            cache_w += resp.cache_write_tokens or 0
            cache_r += resp.cache_read_tokens or 0
            final_text = resp.text
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )
            self._log(
                f"[iter {i}] stop={resp.stop_reason} "
                f"tools={[tc.name for tc in resp.tool_calls]} "
                f"tok={resp.input_tokens}/{resp.output_tokens}"
            )

            # No tool calls. Weaker models often narrate the result in prose
            # instead of firing the terminal tool; nudge them to actually call
            # it, up to a bounded limit, before accepting the natural stop.
            #
            # The limit counts CONSECUTIVE no-tool turns and resets on any turn
            # that calls a tool. Counting cumulatively ended runs that were
            # working: a verbose local model narrates occasionally mid-trajectory
            # and recovers when nudged (observed: 9 clean tool calls, 2 scattered
            # narration turns, terminated as "natural" with 19 iterations of
            # budget unused). The nudge exists to catch a model that has STOPPED
            # acting, which is a run of consecutive no-tool turns, not a tally.
            if not resp.tool_calls:
                steps.append(
                    AgentStep(i, resp.text, [], [], resp.input_tokens, resp.output_tokens,
                              resp.duration_seconds, resp.cache_write_tokens or 0,
                              resp.cache_read_tokens or 0)
                )
                if consecutive_no_tool < self.submit_nudge_limit and self.terminal_tools:
                    consecutive_no_tool += 1
                    nudges_used += 1
                    # A reasoning model that exhausts max_tokens_per_turn while still
                    # thinking did not choose to stop; telling it that it "ended its
                    # turn" is false and pushes it toward submitting too early.
                    truncated = resp.stop_reason == "max_tokens"
                    messages.append(Message(role="user", content=(
                        self._truncated_text() if truncated else self._nudge_text())))
                    truncations += truncated
                    self._log(
                        f"    -> [nudge {consecutive_no_tool}/{self.submit_nudge_limit} "
                        f"consecutive, {nudges_used} total] no tool call"
                        + ("; turn TRUNCATED at the output limit" if truncated
                           else "; prompting to submit")
                    )
                    continue
                stop_cause = "natural"
                break
            consecutive_no_tool = 0  # the model is acting again

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
                AgentStep(i, resp.text, resp.tool_calls, results, resp.input_tokens,
                          resp.output_tokens, resp.duration_seconds,
                          resp.cache_write_tokens or 0, resp.cache_read_tokens or 0)
            )

            if terminated:
                stop_cause = "submit"
                break

        # Explore budget exhausted without a terminal submission. The caller
        # REQUIRES a submitted artifact whenever one can be produced, so spend a
        # few closing turns forcing the model to commit its best result via the
        # terminal tool instead of returning an empty L0. Bounded by submit_grace
        # and the token cap; if the model still refuses, we accept max_iterations.
        # These closing turns do NOT count toward `iterations` (the explore budget).
        main_iterations = len(steps)
        if stop_cause == "max_iterations" and submission is None and self.terminal_tools:
            for g in range(self.submit_grace):
                if in_tok + out_tok + cache_w + cache_r >= self.max_total_tokens:
                    stop_cause = "max_tokens"
                    break
                messages.append(Message(role="user", content=self._force_submit_text()))
                resp = self.client.run_tools(
                    system=self.system,
                    messages=messages,
                    tools=self.schemas,
                    max_tokens=self.max_tokens_per_turn,
                )
                in_tok += resp.input_tokens or 0
                out_tok += resp.output_tokens or 0
                cache_w += resp.cache_write_tokens or 0
                cache_r += resp.cache_read_tokens or 0
                final_text = resp.text
                messages.append(
                    Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
                )
                self._log(
                    f"[closing {g + 1}/{self.submit_grace}] "
                    f"tools={[tc.name for tc in resp.tool_calls]}"
                )
                results = []
                terminated = False
                for tc in resp.tool_calls:
                    res = self._execute(tc)
                    messages.append(Message(role="tool", tool_call_id=tc.id, content=res.content))
                    results.append(res.content)
                    tool = self.tools.get(tc.name)
                    if tool is not None and tool.terminal and not res.is_error:
                        submission = res.content
                        terminated = True
                steps.append(
                    AgentStep(main_iterations + g, resp.text, resp.tool_calls, results,
                              resp.input_tokens, resp.output_tokens, resp.duration_seconds,
                              resp.cache_write_tokens or 0, resp.cache_read_tokens or 0)
                )
                if terminated:
                    stop_cause = "submit_forced"
                    break

        return AgentResult(
            final_text=final_text,
            submission=submission,
            stop_cause=stop_cause,
            steps=steps,
            messages=messages,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            iterations=main_iterations,
            total_cache_write_tokens=cache_w,
            total_cache_read_tokens=cache_r,
            truncated_turns=truncations,
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
