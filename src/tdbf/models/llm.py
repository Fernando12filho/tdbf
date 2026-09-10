"""Multi-model LLM interface.

The proposal requires holding the *architecture* fixed while varying the
model, and explicitly uses both a frontier tool-using model and an
open/comparison model. So generation must not bind to one vendor SDK.

This module defines one `LLMClient` protocol and two adapters (Anthropic,
OpenAI). The generation agent depends only on the protocol, so swapping
or A/B-ing models is a config change, not a code change — which is what
makes RQ1's "same architecture, different model" comparison clean.

SDKs are imported lazily inside each adapter so the offline smoke test
can import this module with no SDKs and no keys.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .messages import Message, ToolCall, ToolSchema, ToolUseResponse


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float | None = None
    raw: object = None  # provider-native response, kept for logging


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the generation agent and the agent runtime need."""

    name: str

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096
    ) -> LLMResponse: ...

    def run_tools(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> ToolUseResponse:
        """Multi-turn tool-calling step. The agent runtime (not the workflow
        generation agent) drives this; `complete` stays the single-turn path."""
        ...


# --------------------------------------------------------------------------- #
# Format converters: normalized <-> vendor-native.                            #
# Two wire formats only. Anthropic is native; OpenAI / Ollama / Gemini all    #
# speak the same OpenAI function-calling format and share one converter pair. #
# --------------------------------------------------------------------------- #


def _to_anthropic_tools(tools: list[ToolSchema]) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    """Render normalized messages as Anthropic content-block messages.

    Anthropic requires all tool results for a turn in a single user message of
    `tool_result` blocks, so consecutive `role=="tool"` messages are merged.
    """
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        if m.role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content or "",
                }
            )
            continue
        flush_results()
        if m.role == "assistant":
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:  # user
            out.append({"role": "user", "content": m.content or ""})
    flush_results()
    return out


def _parse_anthropic_response(resp, model_name: str, duration: float) -> ToolUseResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            )
    stop_map = {
        "tool_use": "tool_use",
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "stop_sequence": "end_turn",
    }
    return ToolUseResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=stop_map.get(resp.stop_reason, resp.stop_reason or "end_turn"),
        input_tokens=getattr(resp.usage, "input_tokens", None),
        output_tokens=getattr(resp.usage, "output_tokens", None),
        # Billed separately from input_tokens when prompt caching is on; absent
        # (None -> 0) on responses that never touched the cache.
        cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", None) or 0,
        cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", None) or 0,
        duration_seconds=duration,
        raw=resp,
    )


def _to_openai_tools(tools: list[ToolSchema]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _to_openai_messages(system: str, messages: list[Message]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content or "",
                }
            )
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments
                            if tc.raw_arguments is not None
                            else json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        else:  # user
            out.append({"role": "user", "content": m.content or ""})
    return out


def _iter_json_objects(text: str):
    """Yield top-level JSON objects embedded in free text (string-aware brace
    matching), so we can recover tool calls a model wrote into its prose instead
    of the structured field. Handles fenced code blocks transparently (the fence
    chars sit outside the braces)."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    obj = None
                if isinstance(obj, dict):
                    yield obj
                start = None


def _coerce_text_tool_call(obj: dict, tool_names: set[str]):
    """If `obj` looks like a call to a *registered* tool, return (name, args).

    Requiring the name to be a known tool is the guard against treating an
    arbitrary JSON example in prose as a tool call.
    """
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    args = fn.get("arguments", fn.get("parameters", {}))
    if isinstance(args, str):  # arguments sometimes a JSON string
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _recover_tool_calls_from_text(text: str, tool_names: set[str]) -> list[ToolCall]:
    """Extract the FIRST tool call a model emitted as text rather than via the
    structured channel (e.g. qwen2.5-coder through Ollama). First-only keeps the
    agent loop genuinely iterative when a weak model dumps a multi-step plan."""
    if not text or not tool_names:
        return []
    for obj in _iter_json_objects(text):
        coerced = _coerce_text_tool_call(obj, tool_names)
        if coerced is not None:
            name, args = coerced
            return [ToolCall(id="call_text_0", name=name, arguments=args)]
    return []


def _parse_openai_response(
    resp, model_name: str, duration: float, tool_names: set[str] | None = None
) -> ToolUseResponse:
    choice = resp.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        raw_args = tc.function.arguments or "{}"
        try:
            parsed = json.loads(raw_args)
            malformed = not isinstance(parsed, dict)
            if malformed:
                parsed = {}
        except (json.JSONDecodeError, TypeError):
            parsed, malformed = {}, True
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=parsed,
                raw_arguments=raw_args,
                malformed=malformed,
            )
        )

    stop_map = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}
    stop_reason = stop_map.get(choice.finish_reason, choice.finish_reason or "end_turn")

    # Fallback: some models (notably qwen2.5-coder via Ollama) write tool calls
    # into the message content instead of the structured tool_calls field. If
    # nothing came through structurally, try to recover a call from the text so
    # the agent measures the model's *reasoning*, not its wire-format compliance.
    if not tool_calls and tool_names:
        recovered = _recover_tool_calls_from_text(msg.content or "", tool_names)
        if recovered:
            tool_calls = recovered
            stop_reason = "tool_use"

    usage = getattr(resp, "usage", None)
    return ToolUseResponse(
        text=msg.content or "",
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        duration_seconds=duration,
        raw=resp,
    )


def _to_ollama_tools(tools: list[ToolSchema]) -> list[dict]:
    """Ollama's native tool schema is the OpenAI shape."""
    return _to_openai_tools(tools)


def _to_ollama_messages(system: str, messages: list[Message]) -> list[dict]:
    """Normalized -> Ollama native /api/chat messages.

    Close to OpenAI's format with two differences that matter: tool-call
    arguments are a JSON *object* (not a string), and a tool result carries the
    tool's NAME rather than a tool_call_id, so results are matched by name.
    """
    out: list[dict] = [{"role": "system", "content": system}]
    id_to_name: dict[str, str] = {}
    for m in messages:
        if m.role == "tool":
            out.append({
                "role": "tool",
                "tool_name": id_to_name.get(m.tool_call_id or "", ""),
                "content": m.content or "",
            })
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in m.tool_calls
                ]
                for tc in m.tool_calls:
                    id_to_name[tc.id] = tc.name
            out.append(msg)
        else:
            out.append({"role": "user", "content": m.content or ""})
    return out


def _parse_ollama_response(
    body: dict, model_name: str, duration: float, tool_names: set[str] | None = None
) -> ToolUseResponse:
    """Parse a native /api/chat response into the normalized ToolUseResponse."""
    msg = body.get("message") or {}
    text = msg.get("content") or ""
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):  # some builds still send a JSON string
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        tool_calls.append(
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=fn.get("name", ""),
                arguments=args if isinstance(args, dict) else {},
                malformed=not isinstance(args, dict),
            )
        )

    done_reason = body.get("done_reason") or "stop"
    stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(done_reason, "end_turn")

    # Same rescue as the OpenAI path: a model that writes the call as prose is
    # measured on its reasoning, not its wire-format compliance. The id marks it.
    if not tool_calls and tool_names:
        recovered = _recover_tool_calls_from_text(text, tool_names)
        if recovered:
            tool_calls = recovered
    if tool_calls:
        stop_reason = "tool_use"

    return ToolUseResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        input_tokens=body.get("prompt_eval_count"),
        output_tokens=body.get("eval_count"),
        duration_seconds=duration,
        raw=body,
    )


class _OpenAIToolMixin:
    """`run_tools` for any client speaking the OpenAI function-calling format.

    Shared by OpenAIClient, GeminiClient, and OllamaClient. Each must provide
    `_ensure()`, `model`, and `name`; `_tool_extra_body()` lets Ollama inject
    `num_ctx`. Non-streaming by design — see OllamaClient.run_tools note.
    """

    def _tool_extra_body(self) -> dict:
        return {}

    def run_tools(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> ToolUseResponse:
        client = self._ensure()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=_to_openai_messages(system, messages),
            tools=_to_openai_tools(tools),
            extra_body=self._tool_extra_body(),
        )
        duration = time.perf_counter() - t0
        return _parse_openai_response(resp, self.name, duration, {t.name for t in tools})


class AnthropicClient:
    """Frontier tool-using model side of the comparison."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self._api_key_env = api_key_env
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic  # lazy

            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self._api_key_env} not set; cannot call Anthropic."
                )
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096
    ) -> LLMResponse:
        client = self._ensure()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        duration = time.perf_counter() - t0
        text = "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )
        return LLMResponse(
            text=text,
            model=self.name,
            input_tokens=getattr(resp.usage, "input_tokens", None),
            output_tokens=getattr(resp.usage, "output_tokens", None),
            duration_seconds=duration,
            raw=resp,
        )

    def run_tools(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> ToolUseResponse:
        client = self._ensure()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=_to_anthropic_tools(tools),
            messages=_to_anthropic_messages(messages),
            # Prompt caching. An agent turn re-sends the entire transcript, so
            # without this a 12-iteration phase bills the conversation ~12 times:
            # measured across the 07 arm, input was 29x output and 85% of spend.
            # The transcript is append-only and the tool list is fixed, which is
            # the ideal prefix shape — auto-caching puts the breakpoint at the
            # last cacheable block, so each turn reads back the previous prefix.
            # Behaviour is unchanged; only billing and the usage split differ.
            cache_control={"type": "ephemeral"},
        )
        duration = time.perf_counter() - t0
        return _parse_anthropic_response(resp, self.name, duration)


class OpenAIClient(_OpenAIToolMixin):
    """Comparison model side (GPT-4 family)."""

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        self._api_key_env = api_key_env
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self._api_key_env} not set; cannot call OpenAI."
                )
            self._client = OpenAI(api_key=key)
        return self._client

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096
    ) -> LLMResponse:
        client = self._ensure()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        duration = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model=self.name,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            duration_seconds=duration,
            raw=resp,
        )


class GeminiClient(_OpenAIToolMixin):
    """Google Gemini via its OpenAI-compatible endpoint.

    Reads GOOGLE_API_KEY from the environment. Use model names like
    'gemini-2.0-flash', 'gemini-1.5-pro', or 'gemma-3-27b-it'.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key_env: str = "GOOGLE_API_KEY",
    ) -> None:
        self.model = model
        self.name = f"gemini:{model}"
        self._api_key_env = api_key_env
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy — reuses the openai SDK

            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self._api_key_env} not set; cannot call Gemini."
                )
            self._client = OpenAI(base_url=self._BASE_URL, api_key=key)
        return self._client

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096
    ) -> LLMResponse:
        client = self._ensure()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        duration = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model=self.name,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            duration_seconds=duration,
            raw=resp,
        )


class OllamaClient(_OpenAIToolMixin):
    """Local Ollama instance via its OpenAI-compatible API.

    Ollama does not require an API key. Pass base_url pointing at the
    machine running Ollama, e.g. 'http://192.168.1.192:11434'.

    num_ctx is the WHOLE window (prompt + generation), not just the prompt:
    Ollama truncates the prompt to `num_ctx - max_tokens`. With the old
    defaults (8192 / 4096) an agent transcript was silently cut to 4096
    tokens, which is under the measured peak of a hard run — the model lost
    its history exactly when it needed it most. Default is 16384 so a real
    agent trajectory fits; pair it with a smaller max_tokens_per_turn (an
    agent turn is a tool call, not an essay).

    VRAM: weights + KV cache share the card. A 12B at q4_K_M plus a 16k
    window does not fit in 8 GB — either drop to a smaller quant or set
    OLLAMA_KV_CACHE_TYPE=q8_0 on the server (roughly halves KV memory).
    `ollama ps` reports the GPU/CPU split; any CPU fraction means the model
    spilled and decode speed collapses.

    temperature/seed are None by default (server defaults apply, typically
    temperature 0.7+). Pin both for agent runs: tool selection is a decision,
    not a sampling problem, and RQ2's determinism analysis is meaningless
    while sampling is unpinned.
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 16384,
        timeout: float = 900.0,
        think: bool = False,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self._base_url = base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._timeout = timeout
        self._think = think
        self._temperature = temperature
        self._seed = seed
        self._client = None

    def _options(self) -> dict:
        """Ollama `options` block shared by complete() and run_tools().

        num_ctx must be threaded through on EVERY call or the window silently
        drops back to the model default (often 2048), truncating the growing
        agent transcript mid-trajectory.
        """
        opts: dict = {"num_ctx": self._num_ctx}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if self._seed is not None:
            opts["seed"] = self._seed
        return opts

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy — reuses the openai SDK

            self._client = OpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="ollama",  # required by SDK; ignored by Ollama
                timeout=self._timeout,
            )
        return self._client

    def complete(
        self, system: str, user: str, *, max_tokens: int = 2048
    ) -> LLMResponse:
        client = self._ensure()
        # /no_think disables Qwen3's extended reasoning mode — much faster
        # and avoids read timeouts caused by long silent thinking phases.
        # With think=True the model reasons first; raise max_tokens accordingly.
        if not self._think:
            user = "/no_think\n" + user
        else:
            max_tokens = max(max_tokens, 4096)

        # Stream to avoid ReadTimeout during the silent first-token phase.
        # stream_options includes usage so the final chunk carries token counts.
        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"options": self._options()},
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[str] = []
        usage = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                usage = chunk.usage
        duration = time.perf_counter() - t0
        text = "".join(chunks)
        return LLMResponse(
            text=text,
            model=self.name,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            duration_seconds=duration,
            raw=None,
        )

    def _tool_extra_body(self) -> dict:
        return {"options": self._options()}

    # run_tools does NOT use the inherited OpenAI-compatible path. Ollama's
    # /v1/chat/completions ignores the `options` block entirely (verified against
    # a live server: a request carrying options.num_ctx=16384 loaded the model at
    # its default ctx=65536, while native /api/chat with options.num_ctx=4096
    # loaded at ctx=4096). Since num_ctx is the local arm's main fairness knob,
    # the agent path talks to the native endpoint, which honours it.
    #
    # Non-streaming, like the inherited path was: qwen/gemma have no silent
    # thinking phase here, and a generous `timeout` covers slow prompt eval on a
    # spilled model. Revisit with delta reassembly if a thinking model is used.
    def run_tools(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
    ) -> ToolUseResponse:
        import json as _json
        import urllib.request

        opts = self._options()
        opts["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "messages": _to_ollama_messages(system, messages),
            "tools": _to_ollama_tools(tools),
            "stream": False,
            "options": opts,
        }
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = _json.loads(resp.read().decode())
        duration = time.perf_counter() - t0
        return _parse_ollama_response(
            body, self.name, duration, {t.name for t in tools}
        )


def build_client(
    spec: str,
    *,
    num_ctx: int | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> LLMClient:
    """`spec` formats:
      'ollama:qwen3.5:9b@http://192.168.1.192:11434'
      'anthropic:claude-opus-4-7'
      'openai:gpt-4-turbo'

    The keyword options apply to Ollama only (they are the local arm's fairness
    knobs) and are ignored by the hosted vendors. Passing None keeps the
    OllamaClient default, so callers that don't care are unaffected.
    """
    vendor, _, rest = spec.partition(":")
    if vendor == "ollama":
        # rest is '<model>@<base_url>' or just '<model>'
        model, _, base_url = rest.partition("@")
        kw: dict = {}
        if num_ctx is not None:
            kw["num_ctx"] = num_ctx
        if timeout is not None:
            kw["timeout"] = timeout
        if temperature is not None:
            kw["temperature"] = temperature
        if seed is not None:
            kw["seed"] = seed
        return OllamaClient(
            model=model or "qwen3.5:9b",
            base_url=base_url or "http://localhost:11434",
            **kw,
        )
    if vendor == "anthropic":
        return AnthropicClient(model=rest or "claude-opus-4-7")
    if vendor == "openai":
        return OpenAIClient(model=rest or "gpt-4-turbo")
    if vendor == "gemini":
        return GeminiClient(model=rest or "gemini-2.0-flash")
    raise ValueError(f"Unknown vendor in spec {spec!r}")
