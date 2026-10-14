"""Runtime assembly: dependencies, graph, checkpointer and interrupt-aware driving."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scholastra.agents.fake import FakeBackend
from scholastra.agents.llm import LLMFactory
from scholastra.config import Settings, get_settings, load_mcp_config
from scholastra.graph.build import build_graph
from scholastra.graph.deps import Deps
from scholastra.io.artifacts import Workspace, open_workspace, resolve_workspace_root
from scholastra.observability import RunLogger
from scholastra.tools.mcp_client import MCPToolProvider, ToolProvider, stub_server_specs

STUB_ALLOWLIST = ["stub_search", "stub_get_paper", "stub_fetch"]


def build_tool_provider(
    settings: Settings,
    *,
    fake: bool,
    workspace: Path | None = None,
) -> tuple[ToolProvider, list[str]]:
    """Return (provider, research_allowlist). Env-gated; falls back to the stub."""
    if fake:
        provider = MCPToolProvider(specs=stub_server_specs())
        return provider, list(STUB_ALLOWLIST)
    config = load_mcp_config(settings)
    provider = MCPToolProvider.from_config(config, workspace=workspace, env=os.environ)
    allowlist = list((config.get("agent_allowlists") or {}).get("research") or [])
    return provider, allowlist


def build_deps(
    settings: Settings | None = None,
    *,
    fake: bool = False,
    logger: RunLogger | None = None,
    flags: dict[str, Any] | None = None,
    workspace: Path | None = None,
) -> Deps:
    settings = settings or get_settings()
    backend = FakeBackend() if fake or settings.fake_llm else None
    llm = LLMFactory(backend=backend, settings=settings)
    tools, allowlist = build_tool_provider(settings, fake=fake, workspace=workspace)
    merged_flags = {
        "research_allowlist": allowlist,
        "tool_timeout_seconds": 60,
        **(flags or {}),
    }
    return Deps(settings=settings, llm=llm, tools=tools, logger=logger, flags=merged_flags)


def make_initial_state(
    *,
    project_id: str,
    topic: str,
    doc_type: str = "article",
    preset: str = "article-short",
    audience: str | None = None,
    constraints: list[str] | None = None,
    auto: bool = False,
    synthesize: bool = False,
    export_formats: list[str] | None = None,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "topic": topic,
        "doc_type": doc_type,
        "preset": preset,
        "audience": audience,
        "constraints": constraints or [],
        "auto": auto,
        "synthesize": synthesize,
        "export_formats": export_formats or [],
        "max_cost_usd": max_cost_usd,
        "status": "intake",
    }


@dataclass
class Project:
    """A prepared project: workspace + deps + compiled graph + thread config."""

    workspace: Workspace
    deps: Deps
    graph: Any
    config: dict[str, Any]

    async def astream_values(self, payload: Any):
        async for chunk in self.graph.astream(payload, self.config, stream_mode="values"):
            yield chunk

    async def invoke(self, payload: Any) -> Any:
        await self.graph.ainvoke(payload, self.config)
        return await self.graph.aget_state(self.config)

    async def snapshot(self) -> Any:
        return await self.graph.aget_state(self.config)

    async def aclose(self) -> None:
        await self.deps.tools.aclose()
        if self.deps.logger is not None:
            self.deps.logger.close()


def pending_interrupt(snapshot: Any) -> Any:
    tasks = getattr(snapshot, "tasks", ()) or ()
    for task in tasks:
        for intr in getattr(task, "interrupts", ()) or ():
            return intr.value
    return None


def open_project(
    project_id: str,
    *,
    settings: Settings | None = None,
    fake: bool = False,
    flags: dict[str, Any] | None = None,
    checkpointer: Any = None,
) -> Project:
    settings = settings or get_settings()
    root = resolve_workspace_root(settings.workspace_root)
    workspace = open_workspace(root, project_id)
    logger = RunLogger(workspace.run_log_path)
    deps = build_deps(settings, fake=fake, logger=logger, flags=flags, workspace=workspace.dir)
    graph = build_graph(deps, checkpointer)
    config = {"configurable": {"thread_id": project_id}}
    return Project(workspace=workspace, deps=deps, graph=graph, config=config)
