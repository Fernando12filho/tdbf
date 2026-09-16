"""Offline checks for the OpenRouter client (the no-budget, open-weight arm).

No network, no key needed: a fake OpenAI SDK client stands in for OpenRouter, so
these exercise the real OpenRouterClient code paths.

  - `openrouter:<id>` parses, keeping a `:free` suffix inside the model id
  - auto-routers (`openrouter/free`, `openrouter/auto`) are refused — a run made
    with them could be neither named nor reproduced
  - `:free` models are priced at an explicit $0.00, never None ("unpriced")
  - a 200 with no choices is retried, then fails loudly instead of crashing on
    `choices[0]`
  - the served model + provider is counted per call and handed off by
    `take_served`, which leaves non-tracking clients untouched
  - a tool call comes back through the normal OpenAI-format parser

    python3 scripts/agent_openrouter_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdbf.agent.pipeline import take_served
from tdbf.metrics import short_model
from tdbf.models.llm import AnthropicClient, OpenRouterClient, build_client
from tdbf.models.messages import Message, ToolSchema
from tdbf.pricing import cost_usd, rates_for

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
TOOLS = [ToolSchema(name="submit_answer", description="Submit.",
                    input_schema={"type": "object",
                                  "properties": {"answer": {"type": "string"}}})]


def _completion(*, choices=True, model=MODEL, provider="Nvidia", error=None):
    """A stand-in for an openai ChatCompletion carrying OpenRouter's extra fields."""
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="submit_answer", arguments='{"answer": "42"}'),
        )],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")] if choices else None,
        model=model,
        provider=provider,
        model_extra={"provider": provider, **({"error": error} if error else {})},
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15),
    )


class _FakeSDK:
    """Replays a script of responses from chat.completions.create."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _client(responses, **kw) -> tuple[OpenRouterClient, _FakeSDK]:
    c = OpenRouterClient(MODEL, empty_backoff=0.0, **kw)
    sdk = _FakeSDK(responses)
    c._client = sdk  # bypass _ensure(): no key, no network
    return c, sdk


def test_spec_parsing() -> None:
    c = build_client(f"openrouter:{MODEL}")
    assert isinstance(c, OpenRouterClient), type(c)
    assert c.model == MODEL, c.model  # the ':free' suffix survives the vendor split
    assert c.name == f"openrouter:{MODEL}", c.name
    assert short_model(c.name) == MODEL, short_model(c.name)


def test_auto_router_refused() -> None:
    for bad in ("openrouter/free", "openrouter/auto"):
        try:
            build_client(f"openrouter:{bad}")
        except ValueError as e:
            assert "reproduced" in str(e), e
        else:
            raise AssertionError(f"{bad} should have been refused")
    try:
        build_client("openrouter:")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty model id should have been refused")


def test_free_models_cost_exactly_zero() -> None:
    assert rates_for(f"openrouter:{MODEL}") == (0.0, 0.0)
    got = cost_usd(f"openrouter:{MODEL}", 200_000, 8_000)
    assert got == 0.0 and got is not None, got  # $0.00, not "unpriced"
    # ':free' must win even when the id names a family that is priced elsewhere
    assert rates_for("openrouter:someone/opus-like-sonnet:free") == (0.0, 0.0)
    # a paid OpenRouter id is NOT silently free — it stays unpriced
    assert cost_usd("openrouter:nvidia/nemotron-3-super-120b-a12b", 1, 1) is None


def test_missing_key_is_a_clear_error() -> None:
    import os
    saved = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        OpenRouterClient(MODEL)._ensure()
    except RuntimeError as e:
        assert "OPENROUTER_API_KEY" in str(e), e
    else:
        raise AssertionError("expected a RuntimeError without a key")
    finally:
        if saved is not None:
            os.environ["OPENROUTER_API_KEY"] = saved


def test_tool_call_round_trip() -> None:
    c, sdk = _client([_completion()])
    resp = c.run_tools(system="sys", messages=[Message(role="user", content="go")],
                       tools=TOOLS, max_tokens=512)
    assert [tc.name for tc in resp.tool_calls] == ["submit_answer"], resp.tool_calls
    assert resp.tool_calls[0].arguments == {"answer": "42"}
    assert resp.input_tokens == 120 and resp.output_tokens == 15
    assert resp.cache_read_tokens in (None, 0)  # no caching on this path
    sent = sdk.calls[0]
    assert sent["model"] == MODEL and sent["max_tokens"] == 512
    assert sent["tools"][0]["function"]["name"] == "submit_answer"


def test_empty_choices_retried_then_succeeds() -> None:
    c, sdk = _client([_completion(choices=False, error={"code": 502}),
                      _completion(choices=False),
                      _completion()], empty_retries=3)
    resp = c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    assert len(sdk.calls) == 3, len(sdk.calls)
    assert resp.tool_calls, "the third response should have been used"
    assert c.served_snapshot() == {f"{MODEL}|Nvidia": 1}, c.served_snapshot()  # only the good one


def test_empty_choices_exhausted_fails_loudly() -> None:
    c, sdk = _client([_completion(choices=False, error={"code": 503, "message": "down"})] * 3,
                     empty_retries=2)
    try:
        c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    except RuntimeError as e:
        assert "no choices" in str(e) and "down" in str(e), e
    else:
        raise AssertionError("expected RuntimeError after exhausting retries")
    assert len(sdk.calls) == 3, len(sdk.calls)


def test_provenance_counted_and_handed_off() -> None:
    c, _ = _client([_completion(provider="Nvidia"),
                    _completion(provider="Chutes"),
                    _completion(provider="Nvidia", model="nvidia/nemotron-3-super-120b-a12b-0915")])
    for _ in range(3):
        c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    served = take_served(c)
    assert served == {
        f"{MODEL}|Nvidia": 1,
        f"{MODEL}|Chutes": 1,
        "nvidia/nemotron-3-super-120b-a12b-0915|Nvidia": 1,  # a silently re-pointed id shows up
    }, served
    assert take_served(c) == {}, "take_served must reset between phases"
    # clients that don't track provenance are left alone (records stay byte-identical)
    assert take_served(AnthropicClient("claude-sonnet-4-6")) is None


def main() -> int:
    tests = [
        test_spec_parsing,
        test_auto_router_refused,
        test_free_models_cost_exactly_zero,
        test_missing_key_is_a_clear_error,
        test_tool_call_round_trip,
        test_empty_choices_retried_then_succeeds,
        test_empty_choices_exhausted_fails_loudly,
        test_provenance_counted_and_handed_off,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nOpenRouter client: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
