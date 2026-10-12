"""Source registry, dedupe, BibTeX/CSL conversion and citation rendering."""

from __future__ import annotations

from theo_explore.models.schemas import Source
from theo_explore.sources.bibtex import collect_bibtex, render_reference_list, source_to_csl_json
from theo_explore.sources.registry import (
    make_source_id,
    normalize_tool_result,
    normalize_url,
    register_sources,
    source_from_text_result,
    url_from_arguments,
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


def test_normalize_tool_result_url_only_record():
    # Web/search results may identify a record by URL alone; it must not be dropped.
    sources = normalize_tool_result("t", [{"url": "https://example.com/a/"}])
    assert len(sources) == 1
    assert sources[0].id == "url:https://example.com/a"
    assert sources[0].title == "https://example.com/a/"


def test_normalize_tool_result_nested_wrapper_key():
    result = '{"web": {"results": [{"title": "Nested", "url": "https://a.com/x"}]}}'
    sources = normalize_tool_result("t", result)
    assert [s.id for s in sources] == ["url:https://a.com/x"]


def test_source_from_text_result_mints_fetch_source():
    markdown = "# Are We Free to Break the Laws?\n\nSome body text about the topic."
    source = source_from_text_result("fetch", {"url": "https://plato.stanford.edu/x"}, markdown)
    assert source is not None
    assert source.id == "url:https://plato.stanford.edu/x"
    assert source.title == "Are We Free to Break the Laws?"
    assert source.full_text_available is True


def test_source_from_text_result_requires_url_and_text():
    assert source_from_text_result("fetch", {"query": "no url"}, "# Title") is None
    assert source_from_text_result("fetch", {"url": "https://a.com"}, "") is None


def test_source_from_text_result_does_not_mint_for_search_tool():
    # A search tool with no URL argument (records failed to parse) must not mint.
    assert source_from_text_result("exa_web_search_exa", {"query": "free will"}, "[]") is None


def test_source_from_text_result_applies_policy_tier():
    source = source_from_text_result(
        "fetch", {"url": "https://en.wikipedia.org/wiki/Free_will"}, "# Free will\n\nText.", {}
    )
    assert source is not None and source.tier == 4  # orientation-only domain


# --- Exa/Tavily plain-text result formats ---------------------------------
# These MCP search servers return human-readable blocks, not JSON. Without a
# text parser every hit is discarded and the section ends up with no evidence.

_EXA_TEXT = (
    "Title: Compatibilism (Stanford Encyclopedia of Philosophy)\n"
    "URL: https://plato.stanford.edu/entries/compatibilism/\n"
    "Published: 2004-04-26T02:29:23.000Z\n"
    "Author: McKenna, Michael, Coates, D. Justin\n"
    "Highlights:\n"
    "Compatibilism is the thesis that free will is compatible with determinism.\n"
    "\n"
    "---\n"
    "\n"
    "Title: Arguments for Incompatibilism\n"
    "URL: https://plato.stanford.edu/entries/incompatibilism-arguments/\n"
    "Published: N/A\n"
    "Author: N/A\n"
    "Highlights:\n"
    "Incompatibilism is the thesis that no deterministic world is a free will world.\n"
)

_TAVILY_TEXT = (
    "Detailed Results:\n\n"
    "Title: Libertarianism about Free Will\n"
    "ID: 7f5e23-00\n"
    "URL: https://philpapers.org/browse/libertarianism-about-free-will\n"
    "Content: The fundamental intuition of incompatibilism is that the future is not "
    "predetermined.\n\n"
    "Title: Compatibilism and Incompatibilism\n"
    "ID: 0340ff-02\n"
    "URL: https://example.edu/free-will\n"
    "Content: Incompatibilism is the view that actions cannot be free if determinism is true.\n"
)


def test_normalize_tool_result_exa_formatted_text():
    sources = normalize_tool_result("exa_web_search_exa", [{"type": "text", "text": _EXA_TEXT}])
    by_id = {s.id: s for s in sources}
    assert set(by_id) == {
        "url:https://plato.stanford.edu/entries/compatibilism",
        "url:https://plato.stanford.edu/entries/incompatibilism-arguments",
    }
    first = by_id["url:https://plato.stanford.edu/entries/compatibilism"]
    assert first.title == "Compatibilism (Stanford Encyclopedia of Philosophy)"
    assert first.year == 2004
    assert first.authors == ["Michael McKenna", "D. Justin Coates"]
    assert by_id["url:https://plato.stanford.edu/entries/incompatibilism-arguments"].year is None
    assert by_id["url:https://plato.stanford.edu/entries/incompatibilism-arguments"].authors == []


def test_formatted_parser_requires_url():
    # A stray ``Title:`` line in fetched full text must not mint a title-only Source.
    text = "Title: Stray heading in fetched page\nbody line\n"
    assert normalize_tool_result("fetch_fetch", [{"type": "text", "text": text}]) == []


def test_normalize_tool_result_tavily_formatted_text():
    sources = normalize_tool_result("tavily_tavily_search", [{"type": "text", "text": _TAVILY_TEXT}])
    ids = {s.id for s in sources}
    assert ids == {
        "url:https://philpapers.org/browse/libertarianism-about-free-will",
        "url:https://example.edu/free-will",
    }


def test_url_from_arguments_handles_urls_list():
    assert url_from_arguments({"urls": ["https://a.com/x", "https://b.com/y"]}) == "https://a.com/x"
    assert url_from_arguments({"url": "https://a.com/x"}) == "https://a.com/x"
    assert url_from_arguments({"query": "no url here"}) is None


def test_source_from_text_result_mints_markitdown_pdf():
    """markitdown takes a ``uri`` and returns page text; that text must become a
    citable Source so PDF full text can back claims."""
    markdown = "Learning Tractable Probabilistic Models\n\nAbstract\nMoral responsibility..."
    source = source_from_text_result(
        "markitdown_convert_to_markdown", {"uri": "https://arxiv.org/pdf/1810.03736v3"}, markdown
    )
    assert source is not None
    assert source.id == "url:https://arxiv.org/pdf/1810.03736v3"
    assert source.full_text_available is True



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
