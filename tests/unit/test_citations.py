"""Shared citation syntax: group handling, stripping and rendering."""

from __future__ import annotations

from scholastra.citations import (
    filter_citations,
    iter_citation_keys,
    render_inline_citations,
)
from scholastra.models.schemas import Source


def test_iter_citation_keys_expands_groups():
    md = "See [@a] and [@b; @c], plus [@d; @e]."
    assert iter_citation_keys(md) == ["a", "b", "c", "d", "e"]


def test_filter_citations_strips_unknown_in_group():
    md = "See [@known] and [@known; @unknown] and [@unknown]."
    filtered, removed = filter_citations(md, {"known"})
    assert filtered == "See [@known] and [@known] and ."
    assert removed == ["unknown", "unknown"]


def test_filter_citations_drops_fully_unknown_group():
    filtered, _removed = filter_citations("X [@a; @b] Y", set())
    assert filtered == "X  Y"


def test_render_inline_citations_group_author_date():
    sources = {
        "doi:1": Source(id="doi:1", kind="paper", title="T1", authors=["Jane Doe"], year=2020),
        "doi:2": Source(id="doi:2", kind="paper", title="T2", authors=["John Roe"], year=2021),
    }
    out = render_inline_citations("See [@doi:1; @doi:2].", sources)
    assert out == "See (Doe 2020; Roe 2021)."


def test_render_inline_citations_accepts_iterable():
    sources = [Source(id="doi:1", kind="paper", title="T", authors=["Jane Doe"], year=2020)]
    assert render_inline_citations("[@doi:1]", sources) == "(Doe 2020)"
