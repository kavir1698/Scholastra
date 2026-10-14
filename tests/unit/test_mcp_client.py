"""MCP client layer: gating, Windows wrapping, allowlists and a live stdio stub."""

from __future__ import annotations

import sys
from pathlib import Path

from scholastra.tools.mcp_client import (
    MCPToolProvider,
    build_connection,
    is_enabled,
    load_server_specs,
    matches_allowlist,
    normalize_pattern,
    stub_server_specs,
    wrap_stdio_command,
)


def test_wrap_stdio_command_on_windows():
    assert wrap_stdio_command("npx", ["-y", "x"], platform="win32") == ("cmd", ["/c", "npx", "-y", "x"])
    assert wrap_stdio_command("uvx", ["pkg"], platform="win32") == ("uvx", ["pkg"])


def test_wrap_stdio_command_on_posix():
    assert wrap_stdio_command("npx", ["-y", "x"], platform="linux") == ("npx", ["-y", "x"])


def test_is_enabled_requires_env():
    assert is_enabled({"requires_env": ["EXA_API_KEY"]}, env={}) is False
    assert is_enabled({"requires_env": ["EXA_API_KEY"]}, env={"EXA_API_KEY": "k"}) is True
    assert is_enabled({"enabled": False}, env={"EXA_API_KEY": "k"}) is False
    assert is_enabled({"requires_env": []}, env={}) is True


def test_build_connection_substitutes_placeholders(tmp_path: Path):
    conn = build_connection(
        {
            "transport": "stdio",
            "command": "{python}",
            "args": ["-m", "x", "--path", "{workspace}/papers"],
        },
        workspace=tmp_path,
        platform="linux",
    )
    assert conn["command"] == sys.executable
    assert conn["args"][-1].endswith("papers")
    assert str(tmp_path) in conn["args"][-1]


def test_build_connection_http_headers_from_env():
    conn = build_connection(
        {
            "transport": "streamable_http",
            "url": "https://mcp.exa.ai/mcp",
            "headers_env": {"x-api-key": "EXA_API_KEY"},
        },
        env={"EXA_API_KEY": "secret"},
        platform="linux",
    )
    assert conn["headers"]["x-api-key"] == "secret"


def test_load_server_specs_skips_missing_keys():
    config = {
        "servers": {
            "exa": {
                "transport": "streamable_http",
                "url": "https://mcp.exa.ai/mcp",
                "requires_env": ["EXA_API_KEY"],
            },
            "fetch": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]},
        }
    }
    enabled = load_server_specs(config, env={}, platform="linux")
    assert [s["name"] for s in enabled] == ["fetch"]


def test_allowlist_normalization_and_matching():
    assert normalize_pattern("arxiv__search") == "arxiv_search"
    assert matches_allowlist("arxiv_search", ["arxiv__*"]) is True
    assert matches_allowlist("pubmed_search", ["arxiv__*"]) is False
    assert matches_allowlist("fetch", ["fetch"]) is True


async def test_stdio_stub_server_end_to_end(no_capture):
    provider = MCPToolProvider(specs=stub_server_specs())
    try:
        tools = await provider.get_tools(["stub_search", "stub_fetch"])
    finally:
        await provider.aclose()
    names = sorted(t.name for t in tools)
    assert names == ["stub_fetch", "stub_search"], f"failed servers: {provider.failed_servers}"

    search = next(t for t in tools if t.name == "stub_search")
    result = await search.ainvoke({"query": "anything"})
    from scholastra.sources.registry import normalize_tool_result

    sources = normalize_tool_result(search.name, result)
    assert sources, "stub search should yield at least one source"
    await provider.aclose()
