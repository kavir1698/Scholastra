"""Native fallback used when no ``fetch`` MCP server is available.

Uses ``httpx`` + ``trafilatura`` to retrieve a URL and convert it to Markdown.
Returns ``None`` on any failure so callers can degrade gracefully.
"""

from __future__ import annotations

import httpx

_USER_AGENT = (
    "Mozilla/5.0 (compatible; scholastra/0.1; +https://example.invalid/bot)"
)


async def fetch_markdown(url: str, *, timeout: float = 30.0) -> str | None:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception:  # noqa: BLE001 - network failure -> graceful None
        return None

    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_comments=False,
            url=url,
        )
    except Exception:  # noqa: BLE001 - extraction failure -> graceful None
        return None
    return text or None
