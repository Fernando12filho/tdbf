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
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
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
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )
        return LLMResponse(text=text, model=self.name, raw=resp)


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
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=self.name, raw=resp)


def build_client(spec: str) -> LLMClient:
    """`spec` like 'anthropic:claude-opus-4-7' or 'openai:gpt-4-turbo'."""
    vendor, _, model = spec.partition(":")
    if vendor == "anthropic":
        return AnthropicClient(model=model or "claude-opus-4-7")
    if vendor == "openai":
        return OpenAIClient(model=model or "gpt-4-turbo")
    raise ValueError(f"Unknown vendor in spec {spec!r}")
