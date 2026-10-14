"""BibTeX / CSL-JSON conversion and a deterministic reference-list fallback."""

from __future__ import annotations

from collections.abc import Iterable

from scholastra.models.schemas import Source

_ENTRY_TYPE = {
    "paper": "article",
    "preprint": "article",
    "book": "book",
    "web": "misc",
    "encyclopedia": "misc",
    "other": "misc",
}


def bibtex_key(source: Source) -> str:
    return source.id


def _escape(value: str) -> str:
    return value.replace("{", "\\{").replace("}", "\\}")


def bibtex_entry(source: Source) -> str:
    kind = _ENTRY_TYPE.get(source.kind, "misc")
    fields: list[tuple[str, str]] = []
    if source.authors:
        fields.append(("author", " and ".join(_escape(a) for a in source.authors)))
    fields.append(("title", _escape(source.title)))
    if source.year:
        fields.append(("year", str(source.year)))
    if source.venue:
        key = "journal" if kind == "article" else "booktitle"
        fields.append((key, _escape(source.venue)))
    if source.doi:
        fields.append(("doi", source.doi))
    if source.arxiv_id:
        fields.append(("eprint", source.arxiv_id))
        fields.append(("archivePrefix", "arXiv"))
    if source.url:
        fields.append(("url", source.url))
    if source.accessed:
        fields.append(("urldate", source.accessed))
    body = ",\n  ".join(f"{name} = {{{value}}}" for name, value in fields)
    return f"@{kind}{{{bibtex_key(source)},\n  {body}\n}}\n"


def collect_bibtex(sources: Iterable[Source]) -> str:
    return "\n".join(bibtex_entry(s) for s in sources)


def source_to_csl_json(source: Source) -> dict:
    csl_type = {
        "paper": "article-journal",
        "preprint": "article",
        "book": "book",
        "web": "webpage",
        "encyclopedia": "entry-encyclopedia",
        "other": "article",
    }.get(source.kind, "article")
    entry: dict = {
        "id": source.id,
        "type": csl_type,
        "title": source.title,
    }
    if source.authors:
        entry["author"] = [{"literal": name} for name in source.authors]
    if source.year:
        entry["issued"] = {"date-parts": [[source.year]]}
    if source.venue:
        key = "container-title" if csl_type == "article-journal" else "publisher"
        entry[key] = source.venue
    if source.url:
        entry["URL"] = source.url
    if source.doi:
        entry["DOI"] = source.doi
    return entry


def collect_csl_json(sources: Iterable[Source]) -> list[dict]:
    return [source_to_csl_json(s) for s in sources]


def parse_bibtex_entry(text: str) -> dict | None:
    """Parse a single BibTeX entry into a flat dict (best-effort).

    Supports both ``bibtexparser`` v1 (``loads``) and v2 (``parse_string``).
    """
    try:
        import bibtexparser
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        if hasattr(bibtexparser, "parse_string"):
            library = bibtexparser.parse_string(text)
            if not library.entries:
                return None
            entry = library.entries[0]
            return {field.key: field.value for field in entry.fields}
        legacy = bibtexparser.loads(text)  # type: ignore[attr-defined]
        if not legacy.entries:
            return None
        legacy_entry = dict(legacy.entries[0])
        legacy_entry.pop("ID", None)
        legacy_entry.pop("ENTRYTYPE", None)
        return legacy_entry
    except Exception:  # noqa: BLE001 - tolerate malformed bibtex
        return None


def render_reference_list(sources: Iterable[Source]) -> str:
    """Deterministic author-date reference list (used when Pandoc/citeproc is absent)."""

    def sort_key(source: Source) -> tuple[str, int]:
        first = source.authors[0] if source.authors else source.title
        return (first.lower(), source.year or 0)

    lines: list[str] = []
    for source in sorted(sources, key=sort_key):
        author = ", ".join(source.authors) if source.authors else "Anonymous"
        year = source.year or "n.d."
        parts = [f"{author}. {year}. "]
        emphasized = source.kind in {"book", "preprint"}
        parts.append(f"*{source.title}*." if emphasized else f"“{source.title}.”")
        if source.venue:
            parts.append(f" {source.venue}.")
        link = source.doi or source.arxiv_id or source.url
        if link:
            parts.append(f" {link}.")
        lines.append("".join(parts))
    return "\n\n".join(lines)
