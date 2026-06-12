"""TDBF agent runtime: a vendor-agnostic tool-use loop.

This package holds the *real agent* (model-driven control), kept separate from
`generation/agent.py`, which is the non-agentic workflow baseline. The loop runs
unchanged across providers via the normalized types in `models/messages.py`.
"""

from .runtime import Agent, AgentResult, AgentStep
from .session import ContainerSession
from .tools import assert_gold_free, brt_agent_tools
from .types import FunctionTool, Tool, ToolResult

__all__ = [
    "Agent",
    "AgentResult",
    "AgentStep",
    "Tool",
    "FunctionTool",
    "ToolResult",
    "ContainerSession",
    "brt_agent_tools",
    "assert_gold_free",
]
