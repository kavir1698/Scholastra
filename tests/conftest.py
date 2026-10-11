"""Shared pytest fixtures and lightweight test doubles."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

import pytest

from theo_explore.agents.fake import FakeBackend, ScriptedResponder
from theo_explore.agents.llm import LLMFactory
from theo_explore.config import Settings, get_settings
from theo_explore.graph.deps import Deps


class FakeTool:
    """Minimal stand-in for a LangChain tool used by the research subgraph."""

    def __init__(self, name: str, fn: Callable[[dict[str, Any]], Any], description: str = "stub") -> None:
        self.name = name
        self.description = description
        self.args = {"query": "str"}
        self._fn = fn

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self._fn):
            return await self._fn(arguments)
        return self._fn(arguments)


class TrackingToolProvider:
    """In-memory tool provider that records concurrency of tool invocations."""

    def __init__(self, records: list[dict[str, Any]] | None = None, delay: float = 0.0) -> None:
        self.records = records if records is not None else _default_records()
        self.delay = delay
        self.current = 0
        self.max_concurrent = 0

    async def _search(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        self.current += 1
        self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return [dict(record, query=arguments.get("query", "")) for record in self.records]
        finally:
            self.current -= 1

    async def get_tools(self, allowlist: list[str]) -> list[FakeTool]:
        tools = [
            FakeTool("stub_search", self._search, "search the stub corpus"),
            FakeTool("stub_get_paper", self._search, "fetch a paper"),
        ]
        if not allowlist:
            return tools
        from theo_explore.tools.mcp_client import matches_allowlist

        return [t for t in tools if matches_allowlist(t.name, allowlist)]

    async def aclose(self) -> None:
        return None


def _default_records() -> list[dict[str, Any]]:
    return [
        {
            "id": "arxiv:2401.00001",
            "kind": "preprint",
            "title": "A Survey of the Topic",
            "authors": ["A. Researcher"],
            "year": 2024,
            "url": "https://arxiv.org/abs/2401.00001",
            "arxiv_id": "2401.00001",
        },
        {
            "id": "web:example.com/overview",
            "kind": "web",
            "title": "An Overview",
            "authors": [],
            "year": 2023,
            "url": "https://example.com/overview",
        },
    ]


class ThreeRootResponder(ScriptedResponder):
    """Outline with three independent roots so fan-out can be observed."""

    def _node_outline(self, prompt: str) -> dict[str, Any]:
        def section(sid: str) -> dict[str, Any]:
            return {
                "id": sid,
                "title": sid.replace("-", " ").title(),
                "purpose": f"Establish {sid}.",
                "key_questions": [f"What is {sid}?"],
                "target_words": 200,
                "depends_on": [],
                "source_domains": ["arxiv"],
                "required": True,
                "subsections": [],
            }

        return {
            "title": "Parallel Test",
            "abstract_target_words": 80,
            "sections": [section("alpha"), section("beta"), section("gamma")],
        }


class RevisionResponder(ScriptedResponder):
    """Emits one blocking issue (round 0) targeting introduction, then clean."""

    def __init__(self) -> None:
        self.critique_calls = 0

    def _node_critique(self, prompt: str) -> dict[str, Any]:
        self.critique_calls += 1
        if self.critique_calls == 1:
            return {
                "issues": [
                    {
                        "id": "issue-1",
                        "severity": "blocking",
                        "category": "factual",
                        "target_section": "introduction",
                        "description": "Introduction needs tightening.",
                        "evidence": "n/a",
                    }
                ]
            }
        return {"issues": []}


@pytest.fixture
def settings_factory(tmp_path):
    def _make(**overrides: Any) -> Settings:
        return get_settings().model_copy(update={"workspace_root": str(tmp_path), **overrides})

    return _make


@pytest.fixture
def deps_factory(settings_factory):
    def _make(
        *,
        responder: ScriptedResponder | None = None,
        tools: Any = None,
        settings: Settings | None = None,
        flags: dict[str, Any] | None = None,
        backend: Any = None,
    ) -> Deps:
        settings = settings or settings_factory()
        backend = backend or FakeBackend(responder or ScriptedResponder())
        llm = LLMFactory(backend=backend, settings=settings)
        provider = tools or TrackingToolProvider()
        merged_flags = {
            "research_allowlist": ["stub_search", "stub_get_paper"],
            "tool_timeout_seconds": 5,
            **(flags or {}),
        }
        return Deps(settings=settings, llm=llm, tools=provider, flags=merged_flags)

    return _make


@pytest.fixture
def no_capture(capfd):
    """Disable pytest output capture.

    Needed for tests that spawn MCP stdio subprocesses: pytest's fd capture
    replaces ``sys.stdin`` with an object whose ``fileno()`` raises
    ``UnsupportedOperation`` on Windows, which anyio's process spawn touches.
    """
    with capfd.disabled():
        yield


def json_list(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Encode records as MCP text content blocks (realistic tool output shape)."""
    return [
        {"type": "text", "text": json.dumps(record), "id": f"blk-{i}"}
        for i, record in enumerate(records)
    ]
