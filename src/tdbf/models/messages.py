"""Vendor-neutral message and tool types for the agent runtime.

The agent loop must run unchanged on Claude (Anthropic native tool use) and
qwen (Ollama's OpenAI-compatible function calling). To keep that comparison
clean (RQ1: same architecture, different model), neither the loop nor the tools
ever see a provider's wire format. They speak in the normalized types defined
here; each `LLMClient.run_tools` adapter converts to/from its native format.

These types live in the models package (not the agent package) because they are
the *interface contract* of the model layer. Keeping them here lets both
`llm.py` and `agent/` import them with no circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolSchema:
    """A tool's name + description + JSON Schema for its arguments.

    Rendered to Anthropic's `input_schema` or OpenAI's `function.parameters`
    by the respective adapter.
    """

    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    """A single tool invocation requested by the model.

    `arguments` is always a parsed dict. Ollama/OpenAI return arguments as a
    JSON *string*; the adapter parses it. If a weak model emits invalid JSON,
    the adapter sets `malformed=True` and stashes the original in
    `raw_arguments` so the runtime can feed a corrective message back to the
    model instead of crashing.
    """

    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    raw_arguments: str | None = None
    malformed: bool = False


@dataclass
class Message:
    """One turn in the normalized conversation.

    role == "user"      -> `content` is the text (task / corrective notes).
    role == "assistant" -> `content` is any text the model emitted; `tool_calls`
                           holds the tools it requested this turn.
    role == "tool"      -> result of a tool call; `tool_call_id` ties it back to
                           the originating ToolCall, `content` is the result text.

    The Anthropic adapter merges consecutive tool messages into one user turn of
    `tool_result` blocks; the OpenAI adapter emits one `role:"tool"` message each.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class ToolUseResponse:
    """Normalized result of one `run_tools` call.

    `stop_reason` is one of: "tool_use" (model wants to call tools),
    "end_turn" (model is done), "max_tokens" (hit the per-turn cap).
    Token counts are surfaced on every call so the agent loop can accumulate
    real usage — fixing the `null` tokens the old Ollama path logged.
    """

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float | None = None
    raw: object = None
