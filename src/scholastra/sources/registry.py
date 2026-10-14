"""Source registry: stable SourceIds, normalisation from tool output, dedupe."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, get_args
from urllib.parse import urlsplit, urlunsplit

from scholastra.models.schemas import AccessMode, Source, SourceKind, SourceRole
from scholastra.utils import stable_id

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
    url = data.get("url") or data.get("link") or data.get("href")
    if not title:
        # Web/search results frequently identify a record by URL or DOI alone.
        # Fall back to those so the record still becomes a usable Source instead
        # of being dropped for lacking a title.
        title = url or data.get("doi") or data.get("DOI")
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


# A dict is treated as a *record* (rather than a wrapper to descend into) when it
# carries any of these; otherwise its wrapper keys are traversed.
_RECORD_KEYS = ("title", "name", "display_name", "url", "link", "href", "doi")
_WRAPPER_KEYS = (
    "results",
    "items",
    "entries",
    "data",
    "papers",
    "sources",
    "documents",
    "hits",
    "records",
    "web",
    "organic",
)


def _payloads_from_obj(obj: object) -> Iterable[object]:
    if isinstance(obj, dict):
        if any(key in obj for key in _RECORD_KEYS):
            yield obj
            return
        for key in _WRAPPER_KEYS:
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


# The Exa and Tavily MCP search servers return human-readable blocks rather than
# JSON, one per hit:
#   Title: ...\nURL: https://...\nPublished: 2000-08-17...\nAuthor: ...\nHighlights:\n<text>
#   Title: ...\nID: ...\nURL: ...\nContent: <text>
# Records are separated by a blank line and/or a ``---`` rule.
_FORMATTED_FIELD_RE = re.compile(
    r"^(Title|URL|Published|Author|Authors|ID|Highlights|Content):\s?(.*)$"
)


def _parse_formatted_records(text: str) -> list[dict[str, Any]]:
    """Parse the plain-text result format used by Exa/Tavily search tools.

    Falls back on nothing else: without this, every Exa/Tavily hit is discarded
    (their text is not JSON), leaving the section with an empty source catalog
    and therefore no citable evidence. Returns one payload per hit for
    :func:`_coerce_source`; only blocks carrying a ``Title:`` start a record.
    """
    records: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    field: str | None = None

    def flush() -> None:
        nonlocal field
        title = (current.get("title") or "").strip()
        url = (current.get("url") or "").strip()
        # Require a URL so a stray ``Title:`` line inside fetched full text cannot
        # mint a junk title-only Source (which would also suppress the URL-based
        # full-text Source). Every real Exa/Tavily search hit carries ``URL:``.
        if title and url:
            records.append(_formatted_payload(current))
        current.clear()
        field = None

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        match = _FORMATTED_FIELD_RE.match(line)
        if match:
            key = match.group(1).lower()
            if key == "title" and current.get("title"):
                flush()
            field = key
            current[key] = match.group(2).strip()
            continue
        if field and line.strip():
            current[field] = f"{current.get(field, '')}\n{line}".strip()
    flush()
    return records


def _split_formatted_authors(raw: str) -> list[str]:
    """Normalise the author string of a formatted search hit to "Given Family".

    Exa emits ``Author: McKenna, Michael, Coates, D. Justin`` (comma-separated
    ``Family, Given`` pairs); Tavily omits authors. Semicolons are accepted as a
    plain separator. Names are reordered so ``_source_lookup``'s
    ``authors[0].split()[-1]`` family derivation and reference rendering are
    correct.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if ";" in raw:
        return [part.strip() for part in raw.split(";") if part.strip()]
    if "," in raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        authors: list[str] = []
        for index in range(0, len(parts), 2):
            family = parts[index]
            given = parts[index + 1] if index + 1 < len(parts) else ""
            authors.append(f"{given} {family}".strip())
        return authors
    return [raw]


def _formatted_payload(record: Mapping[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    title = (record.get("title") or "").strip()
    url = (record.get("url") or "").strip()
    if title:
        payload["title"] = title
    if url:
        payload["url"] = url
    published = record.get("published")
    if published:
        year = re.search(r"\d{4}", published)
        if year:
            payload["year"] = int(year.group(0))
    author = record.get("author") or record.get("authors")
    if author and author.strip().lower() not in {"n/a", "none", "unknown"}:
        authors = _split_formatted_authors(author)
        if authors:
            payload["authors"] = authors
    return payload


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
                payloads.extend(_parse_formatted_records(text))
    else:
        payloads = list(_payloads_from_obj(result))

    if policy:
        from scholastra.sources.policy import apply_source_policy

    sources: list[Source] = []
    for payload in payloads:
        if isinstance(payload, dict):
            source = _coerce_source(payload)
            if source is not None:
                sources.append(apply_source_policy(source, policy) if policy else source)
    return sources


_URL_ARG_KEYS = ("url", "urls", "uri", "link", "href")


def _host_of(url: str) -> str:
    try:
        parts = urlsplit(url if "://" in url else f"https://{url}")
    except ValueError:
        return ""
    return parts.netloc.lower().removeprefix("www.")


def url_from_arguments(arguments: Mapping[str, Any] | None) -> str | None:
    """Return the first HTTP(S) URL found among a tool call's arguments.

    Handles a bare ``url``/``uri`` string as well as the ``urls: [...]`` list
    shape used by the Exa/Tavily fetch/extract tools, so their page text can be
    minted into a citable :class:`Source` instead of being discarded.
    """
    if not arguments:
        return None
    ordered = [arguments.get(key) for key in _URL_ARG_KEYS if key in arguments]
    ordered += [value for key, value in arguments.items() if key not in _URL_ARG_KEYS]
    for value in ordered:
        candidate: str | None = None
        if isinstance(value, str):
            candidate = value
        elif isinstance(value, (list, tuple)):
            candidate = next((item for item in value if isinstance(item, str)), None)
        if candidate and candidate.strip().lower().startswith(("http://", "https://")):
            return candidate.strip()
    return None


def title_from_markdown(text: str, *, max_len: int = 200) -> str | None:
    """Best-effort title from fetched Markdown: first ATX heading, else first line."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:max_len] or None
        return stripped[:max_len]
    return None


def _kind_from_url(url: str) -> SourceKind:
    host = _host_of(url)
    if "arxiv.org" in host:
        return "preprint"
    if "doi.org" in host or "/doi/" in url.lower():
        return "paper"
    return "web"


def source_from_text_result(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    result: object,
    policy: dict | None = None,
) -> Source | None:
    """Mint a :class:`Source` for a tool that returned page text rather than records.

    Search/registry tools return structured records handled by
    :func:`normalize_tool_result`; fetch/convert tools (``fetch``, ``markitdown``,
    the native fetch fallback) return Markdown or plain text for a single URL.
    Those results carry no bibliographic record, so without this a page read at
    full text can never back a citation and every claim drawn from it is later
    dropped as an unknown cite. The URL is taken from the tool arguments (the only
    reliable place), the title from the fetched text, and provenance metadata is
    applied from the source policy so tiers/roles are correct.
    """
    texts = [t for t in _extract_texts(result) if t and t.strip()]
    text = "\n".join(texts).strip()
    url = url_from_arguments(arguments)
    if not url or not text:
        return None
    source = Source(
        id=make_source_id(url=url),
        kind=_kind_from_url(url),
        title=title_from_markdown(text) or url,
        url=url,
        retrieved_url=url,
        full_text_available=True,
    )
    if policy is None:
        return source
    from scholastra.sources.policy import apply_source_policy

    return apply_source_policy(source, policy)


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
