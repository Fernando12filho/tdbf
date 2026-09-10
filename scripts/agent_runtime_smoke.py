"""Smoke test for the Phase 0 agent runtime.

Offline by default (no API keys, no Docker): a scripted FakeClient drives the
loop to verify the mechanics that matter — terminal-tool submission, natural
stop, malformed-argument recovery, unknown-tool recovery, the iteration cap, and
token accumulation — plus the vendor message converters.

Optionally, pass --model to drive a *real* agent over the same tiny tool set on a
live model, proving the same loop runs unchanged across providers:

    export DOCKER_HOST=unix://${HOME}/.colima/default/docker.sock  # not needed here
    python3 scripts/agent_runtime_smoke.py
    python3 scripts/agent_runtime_smoke.py --model anthropic:claude-opus-4-8
    python3 scripts/agent_runtime_smoke.py --model ollama:qwen2.5-coder@http://192.168.1.192:11434
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from types import SimpleNamespace

from tdbf.agent import Agent, FunctionTool, ToolResult
from tdbf.models.llm import _parse_openai_response, _to_anthropic_messages, _to_openai_messages
from tdbf.models.messages import Message, ToolCall, ToolSchema, ToolUseResponse


# --------------------------------------------------------------------------- #
# Tools shared by the offline and live paths.                                 #
# --------------------------------------------------------------------------- #

ADD = FunctionTool(
    name="add",
    description="Add two integers and return their sum.",
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
    func=lambda args: str(int(args["a"]) + int(args["b"])),
)

SUBMIT = FunctionTool(
    name="submit_answer",
    description="Submit the final numeric answer and finish.",
    input_schema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
    func=lambda args: ToolResult(str(args["answer"])),
    terminal=True,
)


# --------------------------------------------------------------------------- #
# A scripted client: yields a predetermined sequence of ToolUseResponses,     #
# ignoring its input. Exercises loop mechanics with zero network.             #
# --------------------------------------------------------------------------- #


class FakeClient:
    name = "fake:scripted"

    def __init__(self, responses: list[ToolUseResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096) -> ToolUseResponse:
        self.calls += 1
        # If the script runs out, mimic a model that keeps requesting tools so
        # the iteration cap is what stops the loop.
        if not self._responses:
            return ToolUseResponse(
                text="still working",
                tool_calls=[ToolCall(id=f"x{self.calls}", name="add", arguments={"a": 1, "b": 1})],
                stop_reason="tool_use",
                input_tokens=5,
                output_tokens=3,
            )
        return self._responses.pop(0)


def _r(text, tool_calls, stop="tool_use", it=10, ot=7):
    return ToolUseResponse(text=text, tool_calls=tool_calls, stop_reason=stop, input_tokens=it, output_tokens=ot)


def test_submit_terminates() -> None:
    client = FakeClient([
        _r("let me add", [ToolCall("c1", "add", {"a": 17, "b": 25})]),
        _r("done", [ToolCall("c2", "submit_answer", {"answer": "42"})]),
    ])
    agent = Agent(client, [ADD, SUBMIT], "calc agent", max_iterations=6)
    res = agent.run("What is 17 + 25?")
    assert res.stop_cause == "submit", res.stop_cause
    assert res.submission == "42", res.submission
    assert res.iterations == 2, res.iterations
    # tokens accumulated across both turns (10+7)*2
    assert res.total_input_tokens == 20 and res.total_output_tokens == 14
    assert res.total_tokens == 34
    # add tool result threaded back as a tool message
    tool_msgs = [m for m in res.messages if m.role == "tool"]
    assert tool_msgs[0].content == "42" or any(m.content == "42" for m in tool_msgs)
    assert any(m.content == "42" for m in tool_msgs)


def test_natural_stop() -> None:
    # With nudging disabled, a no-tool turn stops immediately (original fallback).
    client = FakeClient([_r("the answer is 42, no tools needed", [], stop="end_turn")])
    res = Agent(client, [ADD, SUBMIT], "calc agent", submit_nudge_limit=0).run("trivial")
    assert res.stop_cause == "natural", res.stop_cause
    assert res.submission is None
    assert res.final_text.endswith("no tools needed")


def test_submit_nudge_recovers() -> None:
    # Model narrates an answer (no tool call), gets nudged, then actually submits.
    client = FakeClient([
        _r("the answer is 42", [], stop="end_turn"),
        _r("ok submitting", [ToolCall("c1", "submit_answer", {"answer": "42"})]),
    ])
    res = Agent(client, [ADD, SUBMIT], "calc agent").run("x")
    assert res.stop_cause == "submit", res.stop_cause
    assert res.submission == "42", res.submission
    nudges = [m for m in res.messages if m.role == "user" and "MUST call" in m.content]
    assert len(nudges) == 1, nudges
    assert "submit_answer" in nudges[0].content


def test_submit_nudge_exhausted() -> None:
    # Model never fires a tool; after the nudge budget is spent, accept natural stop.
    client = FakeClient([_r("nope", [], stop="end_turn") for _ in range(3)])
    res = Agent(client, [ADD, SUBMIT], "calc agent", submit_nudge_limit=2).run("x")
    assert res.stop_cause == "natural", res.stop_cause
    assert res.submission is None
    nudges = [m for m in res.messages if m.role == "user" and "MUST call" in m.content]
    assert len(nudges) == 2, nudges


def test_malformed_args_recovery() -> None:
    bad = ToolCall("b1", "add", arguments={}, raw_arguments="{a:17, b:", malformed=True)
    client = FakeClient([
        _r("oops bad json", [bad]),
        _r("fixed", [ToolCall("c2", "submit_answer", {"answer": "99"})]),
    ])
    res = Agent(client, [ADD, SUBMIT], "calc agent").run("x")
    assert res.stop_cause == "submit"
    corrective = next(m.content for m in res.messages if m.role == "tool")
    assert "valid JSON" in corrective, corrective


def test_unknown_tool_recovery() -> None:
    client = FakeClient([
        _r("call ghost", [ToolCall("g1", "nonexistent", {"x": 1})]),
        _r("submit", [ToolCall("c2", "submit_answer", {"answer": "7"})]),
    ])
    res = Agent(client, [ADD, SUBMIT], "calc agent").run("x")
    assert res.stop_cause == "submit"
    err = next(m.content for m in res.messages if m.role == "tool")
    assert "Unknown tool" in err, err


def test_iteration_cap() -> None:
    client = FakeClient([])  # always requests a non-terminal tool
    res = Agent(client, [ADD, SUBMIT], "calc agent", max_iterations=3).run("loop forever")
    assert res.stop_cause == "max_iterations", res.stop_cause
    assert res.iterations == 3


def test_forced_submit_on_budget_exhaustion() -> None:
    # The model burns the whole explore budget on non-terminal tools, then submits
    # only when the forced-submit grace kicks in. The run MUST end with a captured
    # submission, not an empty max_iterations (the caller requires a test/fix).
    client = FakeClient([
        _r("investigating", [ToolCall("a1", "add", {"a": 1, "b": 1})]),
        _r("still investigating", [ToolCall("a2", "add", {"a": 2, "b": 2})]),
        _r("ok, submitting my best", [ToolCall("s1", "submit_answer", {"answer": "55"})]),
    ])
    res = Agent(client, [ADD, SUBMIT], "calc agent", max_iterations=2, submit_grace=3).run("x")
    assert res.stop_cause == "submit_forced", res.stop_cause
    assert res.submission == "55", res.submission
    assert res.iterations == 2, res.iterations  # grace turns don't inflate the explore count


def test_forced_submit_gives_up_if_never_submits() -> None:
    # If the model refuses to submit even during the grace window, we still stop
    # cleanly as max_iterations (no crash, no infinite loop).
    client = FakeClient([])  # always requests a non-terminal tool, forever
    res = Agent(client, [ADD, SUBMIT], "calc agent", max_iterations=2, submit_grace=3).run("x")
    assert res.stop_cause == "max_iterations", res.stop_cause
    assert res.submission is None
    assert res.iterations == 2, res.iterations


def test_tool_raises_is_recoverable() -> None:
    boom = FunctionTool(
        "boom", "always raises", {"type": "object", "properties": {}},
        func=lambda args: (_ for _ in ()).throw(ValueError("kaboom")),
    )
    client = FakeClient([
        _r("trigger", [ToolCall("e1", "boom", {})]),
        _r("submit", [ToolCall("c2", "submit_answer", {"answer": "0"})]),
    ])
    res = Agent(client, [boom, SUBMIT], "x").run("x")
    assert res.stop_cause == "submit"
    err = next(m.content for m in res.messages if m.role == "tool")
    assert "kaboom" in err and "ValueError" in err, err


def test_anthropic_converter_merges_tool_results() -> None:
    msgs = [
        Message(role="user", content="task"),
        Message(role="assistant", content="calling", tool_calls=[
            ToolCall("t1", "add", {"a": 1, "b": 2}),
            ToolCall("t2", "add", {"a": 3, "b": 4}),
        ]),
        Message(role="tool", tool_call_id="t1", content="3"),
        Message(role="tool", tool_call_id="t2", content="7"),
    ]
    out = _to_anthropic_messages(msgs)
    # user, assistant(text+2 tool_use), then ONE merged user with 2 tool_result
    assert out[1]["role"] == "assistant"
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert out[2]["role"] == "user"
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result"]
    assert out[2]["content"][0]["tool_use_id"] == "t1"


def test_openai_converter_one_message_per_result() -> None:
    msgs = [
        Message(role="user", content="task"),
        Message(role="assistant", content="calling", tool_calls=[ToolCall("t1", "add", {"a": 1, "b": 2})]),
        Message(role="tool", tool_call_id="t1", content="3"),
    ]
    out = _to_openai_messages("sys", msgs)
    assert out[0]["role"] == "system"
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "add"
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "t1"


def _fake_openai_resp(content, *, tool_calls=None, finish="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_text_fallback_recovers_known_tool_call() -> None:
    # qwen-style: tool call written into content, structured field empty.
    content = 'Let me look:\n```json\n{"name": "read_file", "arguments": {"path": "a/b.py"}}\n```'
    resp = _fake_openai_resp(content, tool_calls=None, finish="stop")
    out = _parse_openai_response(resp, "ollama:qwen", 0.0, {"read_file", "run_python"})
    assert out.stop_reason == "tool_use", out.stop_reason
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "read_file"
    assert out.tool_calls[0].arguments == {"path": "a/b.py"}


def test_text_fallback_ignores_unknown_name() -> None:
    # A JSON example that is NOT a registered tool must not be hijacked.
    content = 'Example payload: {"name": "delete_everything", "arguments": {}}'
    resp = _fake_openai_resp(content, tool_calls=None, finish="stop")
    out = _parse_openai_response(resp, "ollama:qwen", 0.0, {"read_file"})
    assert out.tool_calls == []
    assert out.stop_reason == "end_turn"


def test_text_fallback_takes_first_when_second_is_malformed() -> None:
    # Mirrors the real astropy run: a valid read_file block, then a run_python
    # block whose triple-quoted code is invalid JSON.
    content = (
        '{"name": "read_file", "arguments": {"path": "astropy/modeling/mappings.py"}}\n'
        '{"name": "run_python", "arguments": {"code": """bad json"""}}'
    )
    resp = _fake_openai_resp(content, tool_calls=None, finish="stop")
    out = _parse_openai_response(resp, "ollama:qwen", 0.0, {"read_file", "run_python"})
    assert len(out.tool_calls) == 1 and out.tool_calls[0].name == "read_file"


def test_cache_tokens_accumulate_and_price() -> None:
    """Prompt caching splits input into three separately-billed counts. If the
    runtime tracked only `input_tokens`, a cached run would look nearly free —
    the same class of silent-zero bug that recorded 21 fable runs at $0."""
    from tdbf.pricing import cost_usd

    resp = ToolUseResponse(
        text="", tool_calls=[ToolCall(id="s1", name="submit_answer", arguments={"answer": "ok"})],
        stop_reason="tool_use", input_tokens=100, output_tokens=50,
        cache_write_tokens=1000, cache_read_tokens=8000,
    )
    agent = Agent(FakeClient([resp]), [ADD, SUBMIT], "calc agent", max_iterations=4)
    result = agent.run("go")
    assert result.total_input_tokens == 100, result.total_input_tokens
    assert result.total_cache_write_tokens == 1000, result.total_cache_write_tokens
    assert result.total_cache_read_tokens == 8000, result.total_cache_read_tokens
    assert result.total_tokens == 9150, result.total_tokens

    # sonnet-4-6 = $3/$15 per 1M; cache write 1.25x input, cache read 0.10x input
    got = cost_usd("anthropic:claude-sonnet-4-6", 100, 50,
                   cache_write_tok=1000, cache_read_tok=8000)
    want = round(100 / 1e6 * 3 + 50 / 1e6 * 15 + 1000 / 1e6 * 3 * 1.25 + 8000 / 1e6 * 3 * 0.10, 4)
    assert got == want, (got, want)
    # A cache-blind cost would be strictly smaller — that is the bug being guarded.
    assert got > cost_usd("anthropic:claude-sonnet-4-6", 100, 50)


def test_token_budget_counts_cached_tokens() -> None:
    """The budget must count cached input too. Otherwise enabling caching moves
    most input into cache_read, the cap silently stops binding, and the arms are
    no longer running under matched budgets."""
    heavy = ToolUseResponse(
        text="working", tool_calls=[ToolCall(id="a1", name="add", arguments={"a": 1, "b": 1})],
        stop_reason="tool_use", input_tokens=10, output_tokens=10,
        cache_write_tokens=0, cache_read_tokens=5_000,
    )
    agent = Agent(FakeClient([heavy] * 10), [ADD, SUBMIT], "calc agent",
                  max_iterations=10, max_total_tokens=6_000, submit_grace=0)
    result = agent.run("go")
    assert result.stop_cause == "max_tokens", result.stop_cause
    assert result.iterations <= 2, result.iterations


def test_pricing_table_distinguishes_model_tiers() -> None:
    """Substring matching is order-sensitive: Sonnet 5 ($2/$10) must not be
    priced at Sonnet 4.6's rate ($3/$15), and every hosted model we actually
    run must be priced at all (an unpriced model silently costs $0)."""
    from tdbf.pricing import cost_usd, rates_for

    assert rates_for("anthropic:claude-sonnet-5") == (2.0, 10.0)
    assert rates_for("anthropic:claude-sonnet-4-6") == (3.0, 15.0)
    assert rates_for("anthropic:claude-fable-5") == (10.0, 50.0)
    assert rates_for("anthropic:claude-opus-5") == (5.0, 25.0)
    assert rates_for("ollama:qwen2.5-coder:14b") is None  # local = free
    assert cost_usd("ollama:qwen2.5-coder:14b", 10_000, 500) is None


def run_offline() -> None:
    tests = [
        test_submit_terminates,
        test_natural_stop,
        test_submit_nudge_recovers,
        test_submit_nudge_exhausted,
        test_malformed_args_recovery,
        test_unknown_tool_recovery,
        test_iteration_cap,
        test_forced_submit_on_budget_exhaustion,
        test_forced_submit_gives_up_if_never_submits,
        test_tool_raises_is_recoverable,
        test_anthropic_converter_merges_tool_results,
        test_openai_converter_one_message_per_result,
        test_text_fallback_recovers_known_tool_call,
        test_text_fallback_ignores_unknown_name,
        test_text_fallback_takes_first_when_second_is_malformed,
        test_cache_tokens_accumulate_and_price,
        test_token_budget_counts_cached_tokens,
        test_pricing_table_distinguishes_model_tiers,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} offline runtime checks passed.")


def run_live(spec: str) -> None:
    from tdbf.models.llm import build_client

    client = build_client(spec)
    system = (
        "You are a calculator agent. To add numbers you MUST call the `add` tool "
        "(do not compute sums yourself). When you have the final answer, call "
        "`submit_answer` with it. Do not give the answer in plain text."
    )
    agent = Agent(client, [ADD, SUBMIT], system, max_iterations=8, verbose=True)
    print(f"\n--- live run on {client.name} ---")
    res = agent.run("What is 17 + 25? Then add 100 to that. Use your tools.")
    print(f"\nstop_cause   : {res.stop_cause}")
    print(f"submission   : {res.submission!r}")
    print(f"iterations   : {res.iterations}")
    print(f"tokens (in/out): {res.total_input_tokens}/{res.total_output_tokens}")
    print(f"final_text   : {res.final_text!r}")
    ok = res.stop_cause == "submit" and res.submission and res.submission.strip() == "142"
    print("\nRESULT:", "PASS (agent submitted 142)" if ok else "CHECK trace above")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="LLM spec for a live run, e.g. anthropic:claude-opus-4-8")
    args = ap.parse_args()

    run_offline()
    if args.model:
        run_live(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
