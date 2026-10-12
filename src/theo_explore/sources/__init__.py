"""Source store, evidence binding and bibliography helpers."""

from __future__ import annotations

from theo_explore.sources.bibtex import (
    bibtex_entry,
    collect_bibtex,
    collect_csl_json,
    parse_bibtex_entry,
    render_reference_list,
    source_to_csl_json,
)
from theo_explore.sources.registry import (
    catalog_lines,
    dedupe,
    make_source_id,
    normalize_tool_result,
    normalize_url,
    register_sources,
    sanitize_id,
    source_from_text_result,
)

__all__ = [
    "bibtex_entry",
    "catalog_lines",
    "collect_bibtex",
    "collect_csl_json",
    "dedupe",
    "make_source_id",
    "normalize_tool_result",
    "normalize_url",
    "parse_bibtex_entry",
    "register_sources",
    "render_reference_list",
    "sanitize_id",
    "source_from_text_result",
    "source_to_csl_json",
]
