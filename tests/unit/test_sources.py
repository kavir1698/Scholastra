"""Source registry, dedupe, BibTeX/CSL conversion and citation rendering."""

from __future__ import annotations

from theo_explore.models.schemas import Source
from theo_explore.sources.bibtex import collect_bibtex, render_reference_list, source_to_csl_json
from theo_explore.sources.registry import (
    make_source_id,
    normalize_tool_result,
    normalize_url,
    register_sources,
)


def test_make_source_id_precedence():
    assert make_source_id(doi="10.1/ABC") == "doi:10.1/abc"
    assert make_source_id(arxiv_id="2401.1", url="http://x") == "arxiv:2401.1"
    assert make_source_id(pmid="123") == "pmid:123"
    assert make_source_id(url="https://www.Example.com/a/") == "url:https://example.com/a"


def test_normalize_url_strips_www_slash_query():
    assert normalize_url("https://WWW.Example.com/path/?q=1") == "https://example.com/path"


def test_normalize_tool_result_content_blocks():
    result = [
        {"type": "text", "text": '{"id": "x", "title": "T1", "url": "https://a.com/1"}', "id": "b1"},
        {"type": "text", "text": '{"id": "y", "title": "T2", "arxiv_id": "2401.2"}', "id": "b2"},
    ]
    sources = normalize_tool_result("stub_search", result)
    ids = {s.id for s in sources}
    assert ids == {"url:https://a.com/1", "arxiv:2401.2"}


def test_normalize_tool_result_plain_list_of_dicts():
    sources = normalize_tool_result("t", [{"title": "Only title", "year": 2020}])
    assert len(sources) == 1
    assert sources[0].id.startswith("ref:")


def test_normalize_tool_result_unparseable_returns_empty():
    assert normalize_tool_result("t", "not json") == []


def test_register_sources_merges_by_id():
    a = Source(id="doi:1", kind="paper", title="A")
    b = Source(id="doi:1", kind="paper", title="A", year=2024, url="https://x")
    merged = register_sources({}, [a, b])
    assert merged["doi:1"].year == 2024


def test_bibtex_and_csl_rendering():
    source = Source(
        id="arxiv:2401.00001",
        kind="preprint",
        title="A Survey",
        authors=["A. Researcher", "B. Scholar"],
        year=2024,
        arxiv_id="2401.00001",
    )
    bib = collect_bibtex([source])
    assert "@article{arxiv:2401.00001" in bib
    assert "archivePrefix = {arXiv}" in bib
    csl = source_to_csl_json(source)
    assert csl["id"] == "arxiv:2401.00001"
    assert csl["author"][0]["literal"] == "A. Researcher"


def test_reference_list_renders():
    source = Source(id="doi:1", kind="paper", title="Title", authors=["Jane Doe"], year=2020)
    rendered = render_reference_list([source])
    assert "Jane Doe" in rendered and "2020" in rendered
