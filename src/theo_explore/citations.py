"""Shared inline-citation syntax and rendering.

One citation rule is defined here and reused by drafting (strip unknown keys),
verification (detect unresolved keys) and assembly (render author-date), so the
three can never diverge. Grouped citations such as ``[@a; @b]`` are handled as a
unit with each member checked independently.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from theo_explore.models.schemas import Source

# A bracketed citation group: ``[@a]`` or ``[@a; @b]``.
CITATION_GROUP_RE = re.compile(r"\[((?:@[^\]\s;]+)(?:\s*;\s*@[^\]\s;]+)*)\]")
_CITE_KEY_RE = re.compile(r"@([^\]\s;]+)")


def iter_citation_keys(markdown: str) -> list[str]:
    """All inline cite keys in ``markdown``, expanding groups."""
    keys: list[str] = []
    for group in CITATION_GROUP_RE.finditer(markdown):
        keys.extend(_CITE_KEY_RE.findall(group.group(1)))
    return keys


def filter_citations(markdown: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Remove cites not in ``allowed``; return (markdown, removed_keys)."""
    removed: list[str] = []

    def repl(match: re.Match[str]) -> str:
        keys = _CITE_KEY_RE.findall(match.group(1))
        kept = [k for k in keys if k in allowed]
        removed.extend(k for k in keys if k not in allowed)
        if not kept:
            return ""
        return "[" + "; ".join(f"@{k}" for k in kept) + "]"

    return CITATION_GROUP_RE.sub(repl, markdown), removed


def _author_year(source: Source) -> str:
    if source.authors:
        first = source.authors[0]
        surname = first.split()[-1] if first.split() else first
        if len(source.authors) > 1:
            surname += " et al."
    else:
        surname = source.title.split(":")[0][:40] if source.title else "Anon."
    return f"{surname} {source.year or 'n.d.'}"


def render_inline_citations(markdown: str, sources: dict[str, Source] | Iterable[Source]) -> str:
    """Replace ``[@id]`` / ``[@a; @b]`` with author-date text using stored sources."""
    source_map = sources if isinstance(sources, dict) else {s.id: s for s in sources}

    def repl(match: re.Match[str]) -> str:
        keys = _CITE_KEY_RE.findall(match.group(1))
        labels = [_author_year(source_map[k]) for k in keys if k in source_map]
        return f"({'; '.join(labels)})" if labels else ""

    return CITATION_GROUP_RE.sub(repl, markdown)
