"""Document contract: defaulting, compaction, terminology normalisation."""

from __future__ import annotations

from scholastra.graph.contract import (
    as_contract,
    bounded_drafts,
    compact_contract,
    find_terminology_violations,
    normalize_terminology,
    relevant_glossary,
)
from scholastra.models.schemas import (
    ContractContribution,
    DocumentContract,
    GlossaryEntry,
    SectionHandoff,
)


def _contract() -> DocumentContract:
    return DocumentContract(
        thesis="X follows from Y under stated assumptions.",
        argument_arc=["Define X", "Relate X to Y", "Derive the consequence"],
        glossary=[
            GlossaryEntry(
                term="paradigm",
                definition="a shared research framework",
                aliases=["framework"],
                forbidden_variants=["paradime"],
                owner_section="intro",
            ),
            GlossaryEntry(term="entropy", owner_section="body"),
        ],
        handoffs=[SectionHandoff(section_id="body", from_sections=["intro"], handoff_goal="Build on X.")],
        contributions=[
            ContractContribution(section_id="body", role="compare", expected_contribution="Compare Y.")
        ],
        repetition_budget="Define once.",
        terminology_rules=["Use canonical terms."],
        source_policy=["Tier 1 preferred."],
    )


def test_as_contract_defaults_and_passthrough():
    assert as_contract({}) == DocumentContract()
    model = _contract()
    assert as_contract(model) is model


def test_compact_contract_includes_section_context():
    block = compact_contract(_contract(), section_id="body")
    assert "THESIS" in block
    assert "SECTION HANDOFF (body)" in block
    assert "YOUR ROLE: compare" in block
    assert "entropy" in block
    assert "SOURCE POLICY" in block


def test_compact_contract_respects_character_budget():
    huge = DocumentContract(
        thesis="t" * 500,
        argument_arc=["step " + "x" * 200 for _ in range(20)],
        glossary=[GlossaryEntry(term=f"term{i}", definition="d" * 100) for i in range(40)],
    )
    block = compact_contract(huge, max_chars=300)
    assert len(block) <= 300


def test_relevant_glossary_orders_owned_terms_first():
    entries = relevant_glossary(_contract(), section_id="intro", max_terms=1)
    assert entries[0].term == "paradigm"


def test_normalize_terminology_canonicalises_variants():
    text = "This framework is a paradigm shift; the paradime is wrong."
    normalized, changes = normalize_terminology(text, _contract())
    assert "framework" not in normalized
    assert "paradime" not in normalized
    assert normalized.count("paradigm") >= 3
    assert any("paradime -> paradigm" in c for c in changes)


def test_bounded_drafts_truncates_without_mutating_input():
    drafts = {"a": {"markdown": "x" * 100}, "b": {"markdown": "y" * 100}}
    bounded, truncated = bounded_drafts(
        drafts, max_chars_per_section=50, max_total_chars=10**6
    )
    assert bounded["a"]["markdown"].endswith("…[truncated]")
    assert set(truncated) == {"a", "b"}
    assert drafts["a"]["markdown"] == "x" * 100


def test_find_terminology_violations_reports_forbidden_variants():
    violations = find_terminology_violations("The paradime is obsolete.", _contract())
    assert violations == ["paradime -> paradigm"]
    assert find_terminology_violations("The paradigm holds.", _contract()) == []
