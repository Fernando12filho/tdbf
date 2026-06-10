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

import os
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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
    """Minimal surface the generation agent needs."""

    name: str

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096
    ) -> LLMResponse: ...


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


class OpenAIClient:
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


class GeminiClient:
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


class OllamaClient:
    """Local Ollama instance via its OpenAI-compatible API.

    Ollama does not require an API key. Pass base_url pointing at the
    machine running Ollama, e.g. 'http://192.168.1.192:11434'.

    num_ctx caps the KV-cache window. On an 8 GB card with Q4_K_M keep
    this at 8192 — pushing higher spills to system RAM and kills decode
    speed.
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        timeout: float = 300.0,
        think: bool = False,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self._base_url = base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._timeout = timeout
        self._think = think
        self._client = None

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
            extra_body={"options": {"num_ctx": self._num_ctx}},
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


def build_client(spec: str) -> LLMClient:
    """`spec` formats:
      'ollama:qwen3.5:9b@http://192.168.1.192:11434'
      'anthropic:claude-opus-4-7'
      'openai:gpt-4-turbo'
    """
    vendor, _, rest = spec.partition(":")
    if vendor == "ollama":
        # rest is '<model>@<base_url>' or just '<model>'
        model, _, base_url = rest.partition("@")
        return OllamaClient(
            model=model or "qwen3.5:9b",
            base_url=base_url or "http://localhost:11434",
        )
    if vendor == "anthropic":
        return AnthropicClient(model=rest or "claude-opus-4-7")
    if vendor == "openai":
        return OpenAIClient(model=rest or "gpt-4-turbo")
    if vendor == "gemini":
        return GeminiClient(model=rest or "gemini-2.0-flash")
    raise ValueError(f"Unknown vendor in spec {spec!r}")
