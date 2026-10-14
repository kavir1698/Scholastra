"""Draft sentence/citation parsing and factual-claim heuristics."""

from __future__ import annotations

from scholastra.claims import (
    is_interpretive,
    looks_factual,
    parse_sentences,
    split_sentences,
    uncited_factual_units,
)


def test_parse_single_and_grouped_citations():
    units = parse_sentences(
        "Alpha is established [@doi:1]. Beta is contested [@doi:2; @doi:3].", section_id="s1"
    )
    assert [u.id for u in units] == ["s1:s1", "s1:s2"]
    assert units[0].cite_keys == ("doi:1",)
    assert units[1].cite_keys == ("doi:2", "doi:3")
    assert all(u.has_citation for u in units)


def test_parse_multiline_paragraph_and_list_items():
    markdown = (
        "Intro prose without citations here.\n"
        "Second sentence has a cite [@doi:1].\n"
        "- First bullet finding [@doi:2].\n"
        "- Second bullet finding [@doi:3].\n"
    )
    units = parse_sentences(markdown, section_id="s2")
    assert [u.text for u in units] == [
        "Intro prose without citations here.",
        "Second sentence has a cite [@doi:1].",
        "First bullet finding [@doi:2].",
        "Second bullet finding [@doi:3].",
    ]
    assert units[0].has_citation is False
    assert units[2].cite_keys == ("doi:2",)


def test_headings_are_skipped():
    units = parse_sentences("## Heading\n\nBody sentence [@a].", section_id="s3")
    assert [u.text for u in units] == ["Body sentence [@a]."]


def test_split_sentences_tolerates_abbreviations_and_decimals():
    parts = split_sentences("Smith et al. report a value of 1.5 units. Other work disagrees.")
    assert parts == ["Smith et al. report a value of 1.5 units.", "Other work disagrees."]


def test_sentence_final_abbreviation_still_splits():
    parts = split_sentences("Entropy grows, etc. The rate rose forty percent.")
    assert parts == ["Entropy grows, etc.", "The rate rose forty percent."]


def test_midsentence_abbreviation_does_not_split():
    parts = split_sentences("See e.g. Smith et al. report more.")
    assert parts == ["See e.g. Smith et al. report more."]


def test_numbered_abbreviation_is_not_split():
    parts = split_sentences("Fig. 3 shows the trend clearly.")
    assert parts == ["Fig. 3 shows the trend clearly."]


def test_uncited_factual_detection_excludes_interpretation():
    markdown = (
        "The measured concentration increased by forty percent across the trial. "
        "We may argue that this is contested. "
        "The ungrounded assertion has no source here. "
        "A grounded statement [@doi:1] holds."
    )
    units = parse_sentences(markdown, section_id="s4")
    flagged = uncited_factual_units(units)
    texts = [u.text for u in flagged]
    assert any("measured concentration" in t for t in texts)
    assert any("ungrounded assertion" in t for t in texts)
    assert texts, "expected at least one uncited factual unit"
    assert not any("may argue" in t for t in texts)
    assert not any("grounded statement" in t for t in texts)


def test_factual_and_interpretive_helpers():
    assert looks_factual("The study found a 20% increase.")
    assert not is_interpretive("The study found a 20% increase.")
    assert is_interpretive("We believe this interpretation is plausible.")
