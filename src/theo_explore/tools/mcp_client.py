"""MCP tool layer: config loading, env gating, Windows stdio wrapping, allowlists.

The client is deliberately thin and bounded:

* servers are loaded from ``config/mcp_servers.json`` (single source of truth);
* a server is only started when all of its ``requires_env`` variables are present;
* Node-based servers are wrapped as ``cmd /c npx …`` on Windows;
* per-agent tool allowlists are enforced by filtering the discovered tools;
* failures are isolated per server so a broken/absent server degrades gracefully.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from typing_extensions import TypedDict

# Importing this eagerly binds ``mcp.client.stdio``'s ``errlog=sys.stderr`` default
# while the real stderr is available. Importing it lazily (e.g. from inside a test
# runner that has swapped ``sys.stderr``) binds a stream without a usable
# ``fileno()`` and breaks stdio subprocess creation on Windows.
# Importing this eagerly binds ``mcp.client.stdio``'s ``errlog=sys.stderr`` default
# while the real stderr is available. Importing it lazily (e.g. from inside a test
# runner that has swapped ``sys.stderr``) binds a stream without a usable
# ``fileno()`` and breaks stdio subprocess creation on Windows.
with contextlib.suppress(Exception):
    import mcp.client.stdio  # noqa: F401


class ServerSpec(TypedDict, total=False):
    name: str
    connection: dict[str, Any]
    purpose: str
    tool_allow: list[str]
    enabled: bool


# Node-based launchers that must be wrapped with `cmd /c` on Windows.
_NODE_LAUNCHERS = {"npx", "npm", "pnpm", "bunx", "yarn", "node"}


def wrap_stdio_command(
    command: str, args: list[str], *, platform: str | None = None
) -> tuple[str, list[str]]:
    """Wrap ``npx``-style commands as ``cmd /c`` on Windows; leave others intact."""
    platform = platform or sys.platform
    launcher = Path(command).name
    if platform.startswith("win") and launcher in _NODE_LAUNCHERS:
        return "cmd", ["/c", command, *args]
    return command, list(args)


def _resolve_env_headers(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for header, env_var in (raw.get("headers_env") or {}).items():
        if env.get(env_var):
            headers[header] = env[env_var]
    return headers


def build_connection(
    raw: dict[str, Any],
    *,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """Convert a config entry into a ``langchain-mcp-adapters`` connection dict."""
    env = env if env is not None else os.environ
    transport = raw.get("transport", "stdio")
    if transport == "stdio":
        args = [
            str(a)
            .replace("{workspace}", str(workspace) if workspace else ".")
            .replace("{python}", sys.executable)
            for a in raw.get("args", [])
        ]
        command = str(raw["command"]).replace("{python}", sys.executable)
        command, args = wrap_stdio_command(command, args, platform=platform)
        conn: dict[str, Any] = {"transport": "stdio", "command": command, "args": args}
        child_env = {var: env[var] for var in raw.get("pass_env", []) if env.get(var)}
        if child_env:
            conn["env"] = {**child_env}
        return conn
    if transport in {"streamable_http", "sse"}:
        conn = {"transport": transport, "url": raw["url"]}
        headers = _resolve_env_headers(raw, env)
        if headers:
            conn["headers"] = headers
        if raw.get("timeout"):
            conn["timeout"] = raw["timeout"]
        return conn
    if transport == "websocket":
        return {"transport": "websocket", "url": raw["url"]}
    raise ValueError(f"unsupported MCP transport: {transport}")


def is_enabled(raw: dict[str, Any], env: Mapping[str, str] | None = None) -> bool:
    env = env if env is not None else os.environ
    if raw.get("enabled") is False:
        return False
    required = raw.get("requires_env") or []
    return all(env.get(var) for var in required)


def load_server_specs(
    config: dict[str, Any],
    *,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    include_disabled: bool = False,
) -> list[ServerSpec]:
    specs: list[ServerSpec] = []
    env = env if env is not None else os.environ
    for name, raw in (config.get("servers") or {}).items():
        enabled = is_enabled(raw, env)
        if not enabled and not include_disabled:
            continue
        specs.append(
            {
                "name": name,
                "connection": build_connection(
                    raw, workspace=workspace, env=env, platform=platform
                ),
                "purpose": raw.get("purpose", ""),
                "tool_allow": list(raw.get("tool_allow") or []),
                "enabled": enabled,
            }
        )
    return specs


def normalize_pattern(pattern: str) -> str:
    """Accept ``server__tool`` or ``server_tool``; both normalise to single ``_``."""
    return pattern.replace("__", "_")


def stub_server_specs() -> list[ServerSpec]:
    """Server spec for the in-repo stdio stub (offline mode / tests)."""
    return [
        {
            "name": "stub",
            "connection": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "theo_explore.tools.stub_server"],
            },
            "purpose": "offline stub research server",
            "tool_allow": [],
            "enabled": True,
        }
    ]


def matches_allowlist(tool_name: str, allowlist: list[str]) -> bool:
    if not allowlist:
        # Empty allowlist on a server means "all tools of that server are allowed",
        # which is handled at the server level; here an empty list matches nothing.
        return False
    return any(fnmatch.fnmatch(tool_name, normalize_pattern(pat)) for pat in allowlist)


@runtime_checkable
class ToolProvider(Protocol):
    """Interface implemented by the real MCP client and by test stubs."""

    async def get_tools(self, allowlist: list[str]) -> list[Any]: ...

    async def aclose(self) -> None: ...


@dataclass
class MCPToolProvider:
    """Wraps :class:`MultiServerMCPClient` with gating, allowlists and isolation."""

    specs: list[ServerSpec]
    connect_timeout: float = 30.0
    tool_timeout: float = 60.0
    _client: Any = None
    _tools: list[Any] = field(default_factory=list)
    _loaded: bool = False
    _failed_servers: list[str] = field(default_factory=list)
    _failures: list[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        workspace: Path | None = None,
        env: Mapping[str, str] | None = None,
        platform: str | None = None,
    ) -> MCPToolProvider:
        settings = config.get("settings") or {}
        specs = load_server_specs(config, workspace=workspace, env=env, platform=platform)
        return cls(
            specs=specs,
            connect_timeout=float(settings.get("connect_timeout_seconds", 30)),
            tool_timeout=float(settings.get("tool_timeout_seconds", 60)),
        )

    @property
    def failed_servers(self) -> list[str]:
        return list(self._failed_servers)

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Double-checked locking: parallel Send branches must not observe an
        # empty tool list while another branch is still connecting.
        async with self._lock:
            if self._loaded:
                return
            connections = {
                spec["name"]: spec["connection"] for spec in self.specs if spec.get("enabled")
            }
            if not connections:
                self._loaded = True
                return
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._client = MultiServerMCPClient(connections, tool_name_prefix=True)  # type: ignore[arg-type]
            collected: list[Any] = []
            for name in connections:
                for attempt in range(2):
                    try:
                        tools = await asyncio.wait_for(
                            self._client.get_tools(server_name=name),
                            timeout=self.connect_timeout,
                        )
                        collected.extend(tools)
                        break
                    except Exception as exc:  # noqa: BLE001 - isolate & retry per-server failures
                        if attempt == 1:
                            self._failed_servers.append(name)
                            self._failures.append(f"{name}: {exc!r}")
                        else:
                            await asyncio.sleep(0.5)
            self._tools = collected
            self._loaded = True

    async def get_tools(self, allowlist: list[str]) -> list[Any]:
        await self._ensure_loaded()
        if not allowlist:
            return list(self._tools)
        return [t for t in self._tools if matches_allowlist(t.name, allowlist)]

    async def aclose(self) -> None:
        client, self._client = self._client, None
        self._tools = []
        self._loaded = False
        if client is None:
            return
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - best-effort shutdown
            pass
