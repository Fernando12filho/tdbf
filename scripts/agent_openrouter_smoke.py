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
from tdbf.models.llm import AnthropicClient, OpenRouterClient, build_client, is_upstream_throttle
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


class _Http429(Exception):
    status_code = 429

    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body


def _upstream_429():
    # the shape OpenRouter returned live on 2026-09-16 (gemma-4-31b, Google AI Studio)
    return _Http429(
        "Error code: 429 - Provider returned error: google/gemma-4-31b-it:free is "
        "temporarily rate-limited upstream. Please retry shortly",
        body={"message": "Provider returned error", "code": 429,
              "metadata": {"limit_source": "upstream_provider_shared_pool"}})


class _FlakySDK(_FakeSDK):
    """Raises the scripted exceptions first, then replays responses."""

    def __init__(self, errors, responses):
        super().__init__(responses)
        self._errors = list(errors)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return self._responses.pop(0)


def test_upstream_throttle_is_waited_out() -> None:
    c = OpenRouterClient(MODEL, empty_backoff=0.0, upstream_waits=(0.0, 0.0, 0.0))
    sdk = _FlakySDK([_upstream_429(), _upstream_429()], [_completion()])
    c._client = sdk
    resp = c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    assert resp.tool_calls and len(sdk.calls) == 3, len(sdk.calls)


def test_upstream_throttle_gives_up_eventually() -> None:
    c = OpenRouterClient(MODEL, empty_backoff=0.0, upstream_waits=(0.0, 0.0))
    c._client = _FlakySDK([_upstream_429()] * 3, [])
    try:
        c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    except _Http429 as e:
        assert is_upstream_throttle(e)
    else:
        raise AssertionError("expected the throttle to surface after the waits ran out")


def test_own_quota_429_is_not_retried() -> None:
    c = OpenRouterClient(MODEL, empty_backoff=0.0, upstream_waits=(0.0, 0.0, 0.0))
    sdk = _FlakySDK([_Http429("Rate limit exceeded: free-models-per-day")], [_completion()])
    c._client = sdk
    try:
        c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    except _Http429 as e:
        assert not is_upstream_throttle(e)
    else:
        raise AssertionError("an account 429 must surface immediately")
    assert len(sdk.calls) == 1, "and must not be retried (it would burn quota)"


PAID = "nvidia/nemotron-3-super-120b-a12b"


def _billed(cost, provider="DeepInfra"):
    r = _completion(model=PAID, provider=provider)
    r.usage = SimpleNamespace(prompt_tokens=120, completion_tokens=15, cost=cost)
    return r


def test_paid_id_needs_a_pinned_provider() -> None:
    try:
        build_client(f"openrouter:{PAID}")
    except ValueError as e:
        assert "@<provider>" in str(e), e
    else:
        raise AssertionError("an unpinned paid id must be refused")
    c = build_client(f"openrouter:{PAID}@deepinfra")
    assert (c.model, c.provider) == (PAID, "deepinfra"), (c.model, c.provider)
    assert c.name == f"openrouter:{PAID}@deepinfra", c.name
    assert short_model(c.name) == PAID, "metrics group on the model, not the provider"
    assert build_client(f"openrouter:{MODEL}").provider is None  # free ids stay unpinned


def test_pin_is_sent_without_fallbacks() -> None:
    c = OpenRouterClient(PAID, provider="deepinfra", empty_backoff=0.0)
    sdk = _FakeSDK([_billed(0.001), _billed(0.002)])
    c._client = sdk
    c.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    c.complete("s", "u")
    for sent in sdk.calls:
        assert sent["extra_body"] == {"provider": {"order": ["deepinfra"],
                                                   "allow_fallbacks": False}}, sent
    unpinned_sdk = _FakeSDK([_completion()])
    u = OpenRouterClient(MODEL, empty_backoff=0.0)
    u._client = unpinned_sdk
    u.run_tools(system="s", messages=[Message(role="user", content="x")], tools=TOOLS)
    assert unpinned_sdk.calls[0]["extra_body"] == {}, "unpinned requests are unchanged"


def test_billed_cost_is_what_a_phase_records() -> None:
    from tdbf.agent.pipeline import billed_mark, phase_cost
    c = OpenRouterClient(PAID, provider="deepinfra", empty_backoff=0.0)
    c._client = _FakeSDK([_billed(0.0012), _billed(0.0030), _billed(0.0100)])
    msg = [Message(role="user", content="x")]
    c.run_tools(system="s", messages=msg, tools=TOOLS)       # before the phase
    mark = billed_mark(c)
    c.run_tools(system="s", messages=msg, tools=TOOLS)
    c.run_tools(system="s", messages=msg, tools=TOOLS)
    assert phase_cost(c, mark, 240, 30) == 0.013, phase_cost(c, mark, 240, 30)
    # the cost survives take_served's reset (it is cumulative, read by difference)
    take_served(c)
    assert phase_cost(c, mark, 240, 30) == 0.013
    # no billed call in the phase: the pricing table decides (paid id: unpriced)
    assert phase_cost(c, billed_mark(c), 1, 1) is None
    # a free response reporting no cost leaves the table in charge ($0.00)
    f = OpenRouterClient(MODEL, empty_backoff=0.0)
    f._client = _FakeSDK([_completion()])
    m = billed_mark(f)
    f.run_tools(system="s", messages=msg, tools=TOOLS)
    assert phase_cost(f, m, 120, 15) == 0.0
    # clients that report no cost: the table, as before
    a = AnthropicClient("claude-sonnet-4-6")
    assert billed_mark(a) is None
    assert phase_cost(a, None, 1_000_000, 0) == 3.0


def main() -> int:
    tests = [
        test_paid_id_needs_a_pinned_provider,
        test_pin_is_sent_without_fallbacks,
        test_billed_cost_is_what_a_phase_records,
        test_spec_parsing,
        test_auto_router_refused,
        test_free_models_cost_exactly_zero,
        test_missing_key_is_a_clear_error,
        test_tool_call_round_trip,
        test_empty_choices_retried_then_succeeds,
        test_empty_choices_exhausted_fails_loudly,
        test_provenance_counted_and_handed_off,
        test_upstream_throttle_is_waited_out,
        test_upstream_throttle_gives_up_eventually,
        test_own_quota_429_is_not_retried,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nOpenRouter client: all {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
