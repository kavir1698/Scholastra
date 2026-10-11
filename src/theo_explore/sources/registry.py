"""Source registry: stable SourceIds, normalisation from tool output, dedupe."""

from __future__ import annotations

import json
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

from theo_explore.models.schemas import Source
from theo_explore.utils import stable_id

_VALID_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/._-+@%")


def sanitize_id(raw: str) -> str:
    cleaned = "".join(ch for ch in raw.strip() if ch in _VALID_ID_CHARS)
    return cleaned or f"ref:{stable_id(raw)}"


def normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme or "https", host, path, "", ""))


def make_source_id(
    *,
    kind: str = "other",
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmid: str | None = None,
    url: str | None = None,
    title: str | None = None,
    year: int | None = None,
    authors: list[str] | None = None,
) -> str:
    if doi:
        return sanitize_id(f"doi:{doi.lower()}")
    if arxiv_id:
        return sanitize_id(f"arxiv:{arxiv_id}")
    if pmid:
        return sanitize_id(f"pmid:{pmid}")
    if url:
        return sanitize_id(f"url:{normalize_url(url)}")
    key = "|".join([title or "", str(year or ""), ",".join(authors or [])])
    return sanitize_id(f"ref:{stable_id(key)}")


def _coerce_source(data: dict) -> Source | None:
    title = data.get("title") or data.get("name")
    if not title:
        return None
    authors = data.get("authors") or data.get("author") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]
    elif isinstance(authors, list):
        authors = [
            a.get("name") if isinstance(a, dict) else str(a) for a in authors
        ]
    year = data.get("year") or data.get("published_year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    doi = data.get("doi") or data.get("DOI")
    arxiv_id = data.get("arxiv_id") or data.get("arxiv")
    pmid = data.get("pmid") or data.get("pmid_id")
    url = data.get("url") or data.get("link") or data.get("href")
    kind = data.get("kind")
    if not kind:
        if arxiv_id:
            kind = "preprint"
        elif pmid:
            kind = "paper"
        elif url:
            kind = "web"
        else:
            kind = "other"
    sid = data.get("id")
    expected = make_source_id(
        kind=kind, doi=doi, arxiv_id=arxiv_id, pmid=pmid, url=url, title=title, year=year
    )
    sid = expected if (not sid or sid != expected) else sanitize_id(str(sid))
    valid_kinds = {"paper", "preprint", "web", "book", "encyclopedia", "other"}
    return Source(
        id=sid,
        kind=kind if kind in valid_kinds else "other",
        title=str(title),
        authors=[str(a) for a in authors if a],
        year=year,
        venue=data.get("venue") or data.get("journal") or data.get("container_title"),
        url=url,
        doi=doi,
        arxiv_id=str(arxiv_id) if arxiv_id else None,
        pmid=str(pmid) if pmid else None,
        bibtex=data.get("bibtex"),
        accessed=data.get("accessed"),
    )


def _payloads_from_obj(obj: object) -> Iterable[object]:
    if isinstance(obj, dict):
        if "title" in obj or "name" in obj:
            yield obj
            return
        for key in ("results", "items", "entries", "data", "papers", "sources"):
            nested = obj.get(key)
            if isinstance(nested, (list, dict)):
                yield from _payloads_from_obj(nested)
                return
    elif isinstance(obj, list):
        for item in obj:
            yield from _payloads_from_obj(item)


def _parse_result_text(text: str) -> object | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _extract_texts(result: object) -> list[str]:
    """Pull text out of a string or a list of MCP content blocks."""
    if isinstance(result, str):
        return [result]
    if isinstance(result, list):
        texts: list[str] = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and isinstance(block.get("resource"), dict):
                texts.append(str(block["resource"].get("text", "")))
            elif isinstance(block, str):
                texts.append(block)
        return texts
    return []


def normalize_tool_result(tool_name: str, result: object) -> list[Source]:
    """Best-effort extraction of :class:`Source` records from a tool result.

    Handles a bare JSON string, a list of MCP ``{"type": "text"}`` content
    blocks (each carrying its own JSON record), and lists of record dicts.
    """
    payloads: list[object] = []
    texts = _extract_texts(result)
    if texts:
        for text in texts:
            parsed = _parse_result_text(text)
            if parsed is not None:
                payloads.extend(_payloads_from_obj(parsed))
    else:
        payloads = list(_payloads_from_obj(result))

    sources: list[Source] = []
    for payload in payloads:
        if isinstance(payload, dict):
            source = _coerce_source(payload)
            if source is not None:
                sources.append(source)
    return sources


def register_sources(existing: dict[str, Source], new: Iterable[Source]) -> dict[str, Source]:
    merged = dict(existing)
    for source in new:
        if source.id in merged:
            current = merged[source.id]
            merged[source.id] = current.model_copy(
                update={k: v for k, v in source.model_dump().items() if v not in (None, [], "")}
            )
        else:
            merged[source.id] = source
    return merged


def dedupe(sources: Iterable[Source]) -> dict[str, Source]:
    return {source.id: source for source in sources}


def catalog_lines(sources: dict[str, Source]) -> list[str]:
    """Human/LLM-readable catalog lines: ``- <id> :: <title> — <authors> (<year>)``."""
    lines = []
    for sid, source in sources.items():
        author = source.authors[0] if source.authors else "n.a."
        year = source.year or "n.d."
        lines.append(f"- {sid} :: {source.title} — {author} ({year})")
    return lines
