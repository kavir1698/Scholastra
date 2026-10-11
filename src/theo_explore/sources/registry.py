"""Source registry: stable SourceIds, normalisation from tool output, dedupe."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import get_args
from urllib.parse import urlsplit, urlunsplit

from theo_explore.models.schemas import AccessMode, Source, SourceRole
from theo_explore.utils import stable_id

_VALID_ROLES = set(get_args(SourceRole))
_VALID_ACCESS = set(get_args(AccessMode))

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


_KIND_ALIASES = {
    "article": "paper",
    "journal-article": "paper",
    "review": "paper",
    "proceedings-article": "paper",
    "preprint": "preprint",
    "posted-content": "preprint",
    "book": "book",
    "monograph": "book",
    "book-chapter": "book",
    "reference-entry": "encyclopedia",
    "encyclopedia": "encyclopedia",
    "webpage": "web",
    "website": "web",
}


def _coerce_authors(authors: object) -> list[str]:
    if isinstance(authors, str):
        return [a.strip() for a in authors.split(";") if a.strip()]
    if isinstance(authors, list):
        out: list[str] = []
        for author in authors:
            if isinstance(author, dict):
                raw_nested = author.get("author")
                nested: dict = raw_nested if isinstance(raw_nested, dict) else {}
                name = author.get("name")
                given = author.get("given") or nested.get("given")
                family = author.get("family") or nested.get("family")
                if not name:
                    name = " ".join(part for part in (given, family) if part)
                if not name and given:
                    name = given
                if name:
                    out.append(str(name))
            elif author:
                out.append(str(author))
        return out
    return []


def _doi_from_url(url: str | None) -> str | None:
    if not url:
        return None
    lowered = url.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if lowered.startswith(prefix):
            return url[len(prefix) :]
    return None


def _coerce_source(data: dict) -> Source | None:
    title = data.get("title") or data.get("name") or data.get("display_name")
    if not title:
        return None
    authors = _coerce_authors(data.get("authors") or data.get("author") or data.get("authorships"))
    year = (
        data.get("year")
        or data.get("published_year")
        or data.get("publication_year")
        or data.get("publication_date")
    )
    if isinstance(year, str):
        year = year[:4]
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    url = data.get("url") or data.get("link") or data.get("href")
    doi = data.get("doi") or data.get("DOI")
    doi = (_doi_from_url(str(doi)) or str(doi)) if doi else _doi_from_url(url)
    arxiv_id = data.get("arxiv_id") or data.get("arxiv")
    pmid = data.get("pmid") or data.get("pmid_id")
    kind = data.get("kind") or data.get("type") or data.get("publication_type")
    if isinstance(kind, str):
        kind = _KIND_ALIASES.get(kind.lower(), kind.lower())
    if not kind:
        if arxiv_id:
            kind = "preprint"
        elif pmid:
            kind = "paper"
        elif url:
            kind = "web"
        else:
            kind = "other"

    raw_abstract = data.get("abstract") or data.get("abstract_text")
    if not raw_abstract and isinstance(data.get("abstract_inverted_index"), dict):
        raw_abstract = " ".join(data["abstract_inverted_index"].keys()) or None
    has_full_text = any(
        data.get(key) for key in ("full_text", "fulltext", "latex", "sections", "content", "body")
    )
    full_text_available = bool(data.get("full_text_available")) or bool(has_full_text)
    raw_mode = data.get("access_mode")
    # Only explicit, recognised access modes are trusted here; inferred modes are
    # owned solely by sources.policy.infer_access_mode so the two cannot diverge.
    access_mode = raw_mode if raw_mode in _VALID_ACCESS else "unknown"
    abstract_only = bool(data.get("abstract_only")) or bool(
        raw_abstract and not full_text_available
    )

    sid = data.get("id")
    expected = make_source_id(
        kind=kind, doi=doi, arxiv_id=arxiv_id, pmid=pmid, url=url, title=title, year=year
    )
    sid = expected if (not sid or sid != expected) else sanitize_id(str(sid))
    valid_kinds = {"paper", "preprint", "web", "book", "encyclopedia", "other"}
    tier = data.get("tier")
    try:
        tier = int(tier) if tier is not None and 1 <= int(tier) <= 4 else None
    except (TypeError, ValueError):
        tier = None
    return Source(
        id=sid,
        kind=kind if kind in valid_kinds else "other",
        title=str(title),
        authors=authors,
        year=year,
        venue=data.get("venue") or data.get("journal") or data.get("container_title"),
        url=url,
        doi=doi,
        arxiv_id=str(arxiv_id) if arxiv_id else None,
        pmid=str(pmid) if pmid else None,
        bibtex=data.get("bibtex"),
        accessed=data.get("accessed"),
        tier=tier,
        role=(str(data["role"]) if data.get("role") in _VALID_ROLES else "unknown"),
        peer_reviewed=data.get("peer_reviewed"),
        access_mode=access_mode,
        full_text_available=full_text_available,
        abstract_only=abstract_only,
        retrieved_url=data.get("retrieved_url") or url,
        bibliographic_confidence=data.get("bibliographic_confidence"),
    )


def _payloads_from_obj(obj: object) -> Iterable[object]:
    if isinstance(obj, dict):
        if "title" in obj or "name" in obj or "display_name" in obj:
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


def normalize_tool_result(
    tool_name: str, result: object, policy: dict | None = None
) -> list[Source]:
    """Best-effort extraction of :class:`Source` records from a tool result.

    Handles a bare JSON string, a list of MCP ``{"type": "text"}`` content
    blocks (each carrying its own JSON record), and lists of record dicts.
    When ``policy`` is given, provenance tier/role/access metadata is filled in
    after normalisation.
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

    if policy:
        from theo_explore.sources.policy import apply_source_policy

    sources: list[Source] = []
    for payload in payloads:
        if isinstance(payload, dict):
            source = _coerce_source(payload)
            if source is not None:
                sources.append(apply_source_policy(source, policy) if policy else source)
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
    """Human/LLM-readable catalog lines: ``- <id> :: <title> — <authors> (<year>)``.

    Provenance (tier/access) is appended when known so the extractor can prefer
    full-text, higher-tier sources.
    """
    lines = []
    for sid, source in sources.items():
        author = source.authors[0] if source.authors else "n.a."
        year = source.year or "n.d."
        line = f"- {sid} :: {source.title} — {author} ({year})"
        provenance: list[str] = []
        if source.tier is not None:
            provenance.append(f"tier={source.tier}")
        if source.access_mode and source.access_mode != "unknown":
            provenance.append(f"access={source.access_mode}")
        if source.abstract_only:
            provenance.append("abstract_only")
        if provenance:
            line += " [" + ", ".join(provenance) + "]"
        lines.append(line)
    return lines
