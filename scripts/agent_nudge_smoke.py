"""Nudge policy: consecutive, not cumulative.

Scripted client replays the exact shape observed in the gemma4 pilot -- clean
tool calls, scattered narration turns that recover when nudged, then a submit.
Under the old cumulative rule this terminated as `natural` with most of the
iteration budget unused.
"""
from tdbf.agent.runtime import Agent
from tdbf.agent.types import FunctionTool
from tdbf.models.messages import ToolCall, ToolUseResponse


class ScriptedClient:
    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def run_tools(self, *, system, messages, tools, max_tokens=4096):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if step is None:  # narration turn, no tool call
            return ToolUseResponse(text="Let me think about this at length...",
                                   tool_calls=[], stop_reason="end_turn",
                                   input_tokens=10, output_tokens=4096)
        return ToolUseResponse(text="", tool_calls=[ToolCall(id=f"c{self.calls}", name=step,
                                                             arguments={})],
                               stop_reason="tool_use", input_tokens=10, output_tokens=20)


def build(script, max_iterations=30):
    tools = [
        FunctionTool("grep", "search", {"type": "object", "properties": {}}, lambda a: "hit"),
        FunctionTool("submit_test", "submit", {"type": "object", "properties": {}},
                     lambda a: "TEST", terminal=True),
    ]
    return Agent(ScriptedClient(script), tools, "sys",
                 max_iterations=max_iterations, submit_nudge_limit=2, verbose=False)


# The pilot's shape: work, narrate, recover, narrate, recover, then submit.
PILOT = ["grep"] * 9 + [None] + ["grep"] + [None] + ["grep", "submit_test"]
r = build(PILOT).run("task")
print(f"pilot shape      : stop={r.stop_cause:8} iters={r.iterations:2} "
      f"submission={'yes' if r.submission else 'NO'}")
assert r.stop_cause == "submit", "scattered narration must not end a working run"
assert r.submission == "TEST"

# A model that has genuinely stopped acting must still terminate.
STOPPED = ["grep", "grep"] + [None] * 10
r2 = build(STOPPED).run("task")
print(f"genuinely stopped: stop={r2.stop_cause:8} iters={r2.iterations:2} "
      f"submission={'yes' if r2.submission else 'NO'}")
assert r2.stop_cause == "natural", "3 consecutive no-tool turns must still stop"
assert r2.iterations == 5, f"should stop promptly, got {r2.iterations}"

# Unchanged: a model that submits immediately.
r3 = build(["submit_test"]).run("task")
print(f"immediate submit : stop={r3.stop_cause:8} iters={r3.iterations:2} "
      f"submission={'yes' if r3.submission else 'NO'}")
assert r3.stop_cause == "submit"

print("\nnudge policy OK: recovers scattered narration, still stops a dead run")
