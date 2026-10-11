"""MCP tool layer: client, gating, fetch fallback, stub server."""

from __future__ import annotations

from theo_explore.tools.fetch_fallback import fetch_markdown
from theo_explore.tools.mcp_client import (
    MCPToolProvider,
    ToolProvider,
    build_connection,
    is_enabled,
    load_server_specs,
    matches_allowlist,
    normalize_pattern,
    wrap_stdio_command,
)

__all__ = [
    "MCPToolProvider",
    "ToolProvider",
    "build_connection",
    "fetch_markdown",
    "is_enabled",
    "load_server_specs",
    "matches_allowlist",
    "normalize_pattern",
    "wrap_stdio_command",
]
