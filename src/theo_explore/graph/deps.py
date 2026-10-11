"""Shared runtime dependencies injected into every node via closures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theo_explore.agents.llm import LLMFactory
from theo_explore.config import Settings
from theo_explore.io.artifacts import Workspace, open_workspace, resolve_workspace_root
from theo_explore.models.schemas import NodeUsage
from theo_explore.observability import RunLogger
from theo_explore.tools.mcp_client import ToolProvider


@dataclass
class Deps:
    settings: Settings
    llm: LLMFactory
    tools: ToolProvider
    logger: RunLogger | None = None
    flags: dict[str, Any] = field(default_factory=dict)

    def workspace(self, project_id: str) -> Workspace:
        root = resolve_workspace_root(self.settings.workspace_root)
        return open_workspace(root, project_id)

    def log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log(event, **fields)

    def budget_delta(self, node: str, usage: NodeUsage, *, tool_calls: int = 0) -> dict[str, Any]:
        return {
            "budget": {
                "per_node": {node: usage.model_dump()},
                "tokens_in": usage.tokens_in,
                "tokens_out": usage.tokens_out,
                "cost_usd": usage.cost_usd,
                "tool_calls": tool_calls,
            }
        }
