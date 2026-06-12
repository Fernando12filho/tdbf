"""Tool abstraction the agent runtime executes.

A `Tool` is the only surface the model touches. Phase A wraps `runner.py`
functions as Tools; here we define the base class plus a `FunctionTool`
convenience used by the smoke test and any thin callable wrapper.

GOLD-FREE INVARIANT: no Tool may run against the gold-fixed tree. Tools wrap
only buggy-tree / read-only operations. `gold_patch` and `test_patch` stay out
of this package entirely (offline scoring only). Keep the registry auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..models.messages import ToolSchema


@dataclass
class ToolResult:
    """What a tool returns to the loop.

    `content` is fed back to the model as the tool result. `is_error=True` marks
    a recoverable failure (bad args, tool raised) so the model can correct course
    rather than the run aborting. For a terminal tool, `content` is the captured
    artifact (e.g. the submitted test code).
    """

    content: str
    is_error: bool = False


class Tool:
    """Base class. Subclass and override `execute`, or use `FunctionTool`.

    `terminal=True` means a successful call ends the agent loop and its result
    is captured as the run's submission (the `submit_*` tool the design uses as
    the primary done-signal).
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {"type": "object", "properties": {}}
    terminal: bool = False

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def execute(self, arguments: dict) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


class FunctionTool(Tool):
    """Wrap a plain Python callable as a Tool.

    The callable receives the parsed arguments dict and returns either a str
    (taken as the result content) or a ToolResult.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        func: Callable[[dict], "str | ToolResult"],
        *,
        terminal: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.terminal = terminal
        self._func = func

    def execute(self, arguments: dict) -> ToolResult:
        out = self._func(arguments)
        return out if isinstance(out, ToolResult) else ToolResult(str(out))
