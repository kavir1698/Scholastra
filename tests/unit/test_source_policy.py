"""Tiered source policy: classification, provenance and quality summary."""

from __future__ import annotations

from theo_explore.models.schemas import Source
from theo_explore.sources.policy import (
    apply_source_policy,
    classify_role,
    classify_tier,
    source_quality_summary,
    tier_allows_claim_support,
)
from theo_explore.sources.registry import normalize_tool_result, register_sources


def test_tier_classification_by_kind():
    assert classify_tier(Source(id="a", kind="paper", title="T", venue="Nature")) == 1
    assert classify_tier(Source(id="b", kind="preprint", title="T")) == 2
    assert (
        classify_tier(
            Source(id="c", kind="book", title="T", venue="Oxford University Press")
        )
        == 1
    )
    assert classify_tier(Source(id="d", kind="book", title="T")) == 3
    assert classify_tier(Source(id="e", kind="encyclopedia", title="T")) == 3


def test_wikipedia_is_orientation_only():
    wiki = Source(
        id="w", kind="web", title="T", url="https://en.wikipedia.org/wiki/Topic"
    )
    tier = classify_tier(wiki)
    assert tier == 4
    assert classify_role(wiki, tier) == "orientation"
    assert tier_allows_claim_support(tier) is False
    assert tier_allows_claim_support(tier, {"allow_orientation_support": True}) is True


def test_asserted_tier_cannot_upgrade_a_source():
    spoof = Source(
        id="s",
        kind="web",
        title="T",
        url="https://en.wikipedia.org/wiki/X",
        tier=1,
    )
    assert classify_tier(spoof) == 4


def test_retrieved_url_cannot_downgrade_the_domain():
    spoof = Source(
        id="s",
        kind="web",
        title="T",
        url="https://en.wikipedia.org/wiki/X",
        retrieved_url="https://mit.edu/x",
    )
    assert classify_tier(spoof) == 4


def test_registry_does_not_infer_access_mode():
    sources = normalize_tool_result("t", [{"title": "A", "url": "https://a.com", "kind": "book"}])
    assert sources[0].access_mode == "unknown"
    applied = apply_source_policy(sources[0])
    assert applied.access_mode == "book_metadata"


def test_low_bibliographic_confidence_is_discovery_only():
    source = apply_source_policy(
        Source(
            id="x",
            kind="paper",
            title="T",
            full_text_available=True,
            bibliographic_confidence=0.2,
        )
    )
    assert source.access_mode == "full_text"
    assert source.role == "bibliographic"


def test_authoritative_web_is_tier_three():
    edu = Source(id="u", kind="web", title="T", url="https://physics.mit.edu/page")
    assert classify_tier(edu) == 3
    assert tier_allows_claim_support(3) is True


def test_abstract_only_record_is_bibliographic():
    source = Source(
        id="m",
        kind="paper",
        title="T",
        abstract_only=True,
    )
    applied = apply_source_policy(source)
    assert applied.access_mode == "abstract"
    assert applied.role == "bibliographic"
    assert applied.abstract_only is True


def test_full_text_preferred_over_metadata():
    full = apply_source_policy(
        Source(id="f", kind="paper", title="T", full_text_available=True)
    )
    meta = apply_source_policy(Source(id="mm", kind="paper", title="T"))
    assert full.access_mode == "full_text"
    assert meta.access_mode == "metadata"


def test_source_quality_summary_counts():
    sources = {
        "a": Source(id="a", kind="paper", title="T"),
        "b": Source(id="b", kind="web", title="T", url="https://en.wikipedia.org/wiki/X"),
        "c": Source(id="c", kind="paper", title="T", abstract_only=True),
        "d": Source(id="d", kind="paper", title="T", full_text_available=True),
    }
    summary = source_quality_summary(sources)
    assert summary["total"] == 4
    assert summary["tier_1"] == 3
    assert summary["tier_4"] == 1
    assert summary["abstract_only"] == 1
    assert summary["full_text"] >= 1


def test_openalex_style_record_is_normalized():
    record = {
        "id": "https://openalex.org/W1",
        "display_name": "A Study",
        "publication_year": 2021,
        "doi": "https://doi.org/10.5/ABC",
        "type": "article",
        "authorships": [{"author": {"given": "Jane", "family": "Doe"}}],
        "abstract_inverted_index": {"We": [0], "show": [1]},
    }
    sources = normalize_tool_result("openalex__search", [record])
    assert len(sources) == 1
    source = sources[0]
    assert source.doi == "10.5/ABC"
    assert source.id == "doi:10.5/abc"
    assert source.year == 2021
    assert source.authors == ["Jane Doe"]
    assert source.abstract_only is True


def test_register_sources_dedupes_and_merges():
    a = Source(id="doi:1", kind="paper", title="A")
    b = Source(id="doi:1", kind="paper", title="A", full_text_available=True)
    merged = register_sources({}, [a, b])
    assert list(merged) == ["doi:1"]
    assert merged["doi:1"].full_text_available is True
