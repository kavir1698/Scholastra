"""Citation verification: mechanical resolution + claim-support checks."""

from __future__ import annotations

from theo_explore.agents.fake import ScriptedResponder
from theo_explore.graph.nodes.assembly import _build_gate
from theo_explore.graph.nodes.quality import make_route_after_critique, make_verify_citations_node
from theo_explore.models.schemas import (
    Claim,
    SectionDraft,
    SectionEvidence,
    SectionVerification,
    Source,
)


def _draft(markdown: str) -> dict:
    return SectionDraft(section_id="s1", title="S1", markdown=markdown).model_dump()


async def test_unresolved_citation_is_blocking(deps_factory):
    node = make_verify_citations_node(deps_factory())
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@missing-key] here.")},
            "sources": {},
            "evidence": {},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["unresolved_cites"] == ["missing-key"]
    assert out["verification"]["s1"]["passed"] is False
    assert any(i["severity"] == "blocking" for i in out["issues"])


async def test_all_resolved_passes(deps_factory):
    source = Source(id="doi:1", kind="paper", title="T")
    node = make_verify_citations_node(deps_factory())
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@doi:1] here.")},
            "sources": {"doi:1": source.model_dump()},
            "evidence": {},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["passed"] is True
    assert out["issues"] == []


class _UnsupportedResponder(ScriptedResponder):
    def _node_verify(self, prompt: str) -> dict:
        return {"unsupported_claims": ["c0"], "notes": ""}


async def test_unsupported_claim_flagged(deps_factory):
    node = make_verify_citations_node(deps_factory(responder=_UnsupportedResponder()))
    evidence = SectionEvidence(
        section_id="s1",
        claims=[
            Claim(id="c0", text="unsupported", source_ids=["doi:1"], excerpt="x", locator="p1")
        ],
    )
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@doi:1].")},
            "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
            "evidence": {"s1": evidence.model_dump()},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["unsupported_claims"] == ["c0"]
    assert any(i["category"] == "factual" and i["severity"] == "major" for i in out["issues"])


async def test_grouped_citation_unresolved_flagged(deps_factory):
    node = make_verify_citations_node(deps_factory())
    out = await node(
        {
            "drafts": {"s1": _draft("A [@doi:1; @missing] claim.")},
            "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
            "evidence": {},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["unresolved_cites"] == ["missing"]
    assert any(i["severity"] == "blocking" for i in out["issues"])


class _PartialResponder(ScriptedResponder):
    def _node_verify(self, prompt: str) -> dict:
        return {"unsupported_claims": [], "partial_claims": ["s1:s1"], "notes": ""}


async def test_partial_claim_flagged(deps_factory):
    node = make_verify_citations_node(deps_factory(responder=_PartialResponder()))
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@doi:1].")},
            "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
            "evidence": {},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["partial_claims"] == ["s1:s1"]
    assert any(i["severity"] == "minor" and i["category"] == "factual" for i in out["issues"])


class _UncitedResponder(ScriptedResponder):
    def _node_verify(self, prompt: str) -> dict:
        return {"unsupported_claims": [], "uncited_factual": ["s1:s1"], "notes": ""}


async def test_uncited_factual_flagged(deps_factory):
    node = make_verify_citations_node(deps_factory(responder=_UncitedResponder()))
    out = await node(
        {
            "drafts": {
                "s1": _draft("The measured value increased by forty percent in the study.")
            },
            "sources": {},
            "evidence": {},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["uncited_factual"] == ["s1:s1"]
    assert out["verification"]["s1"]["uncited_candidates"]
    assert any(i["severity"] == "major" and i["category"] == "factual" for i in out["issues"])


async def test_known_source_without_excerpt_is_reported(deps_factory):
    node = make_verify_citations_node(deps_factory())
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@doi:1].")},
            "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
            "evidence": {},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["sources_without_excerpt"] == ["doi:1"]


async def test_low_tier_support_is_flagged(deps_factory):
    source = Source(
        id="w", kind="web", title="W", url="https://en.wikipedia.org/wiki/X", tier=4
    )
    evidence = SectionEvidence(
        section_id="s1",
        claims=[
            Claim(
                id="c0",
                text="t",
                source_ids=["w"],
                excerpt="x",
                locator="p",
                origin="web",
                tier=4,
            )
        ],
    )
    node = make_verify_citations_node(deps_factory())
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@w].")},
            "sources": {"w": source.model_dump()},
            "evidence": {"s1": evidence.model_dump()},
            "revision_round": 0,
        }
    )
    assert out["verification"]["s1"]["low_tier_support"] == ["s1:s1"]
    assert any(i["category"] == "source_policy" for i in out["issues"])


def test_quality_gate_blocks_low_tier_support():
    verification = {
        "s1": SectionVerification(section_id="s1", low_tier_support=["s1:s1"], passed=False)
    }
    gate = _build_gate({"drafts": {"s1": {}}, "editorial": []}, verification, strict=True)
    assert gate.blocked is True
    assert any("claim-support tier" in r for r in gate.reasons)


def test_quality_gate_counts_current_round_blocking_issues():
    issues = [
        {"severity": "blocking", "round": 1, "category": "editorial", "description": "live"},
        {"severity": "blocking", "round": 0, "category": "editorial", "description": "stale"},
    ]
    state = {"drafts": {"s1": {}}, "editorial": [], "revision_round": 1, "issues": issues}
    gate = _build_gate(state, {}, strict=True)
    assert gate.blocked is True
    assert any("blocking issue" in r for r in gate.reasons)

    stale_only = {**state, "drafts": {}, "issues": [issues[1]]}
    assert _build_gate(stale_only, {}, strict=False).clean is True


def test_quality_gate_flags_synthesis_verification_failure():
    state = {"drafts": {}, "editorial": [], "synthesis_verification": {"verified": False}}
    gate = _build_gate(state, {}, strict=True)
    assert gate.verification_incomplete is True
    assert gate.blocked is True


def test_quality_gate_blocks_unsupported_when_strict():
    verification = {
        "s1": SectionVerification(section_id="s1", unsupported_claims=["c0"], passed=False)
    }
    state = {"drafts": {"s1": {}}, "editorial": []}
    strict = _build_gate(state, verification, strict=True)
    assert strict.blocked is True and strict.clean is False
    assert strict.unsupported_count == 1
    audit = _build_gate(state, verification, strict=False)
    assert audit.blocked is False and audit.clean is False


def test_quality_gate_flags_incomplete_verification():
    verification = {"s1": SectionVerification(section_id="s1", verified=False)}
    gate = _build_gate({"drafts": {"s1": {}}, "editorial": []}, verification, strict=True)
    assert gate.verification_incomplete is True
    assert gate.blocked is True


class _BrokenVerifyResponder(ScriptedResponder):
    """Returns a shape that can never validate, forcing the verifier to fail."""

    def _node_verify(self, prompt: str) -> dict:
        return {"results": "not-a-list"}


async def test_verifier_error_is_recorded_distinctly_from_a_clean_pass(deps_factory):
    node = make_verify_citations_node(deps_factory(responder=_BrokenVerifyResponder()))
    out = await node(
        {
            "drafts": {"s1": _draft("A claim [@doi:1].")},
            "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
            "evidence": {},
            "revision_round": 0,
        }
    )
    verification = out["verification"]["s1"]
    assert verification["verified"] is False
    assert verification["verifier_error"]
    # Mechanical resolution still passed, so the two states are genuinely distinct.
    assert verification["passed"] is True


def test_quality_gate_counts_verifier_errors():
    verification = {
        "s1": SectionVerification(
            section_id="s1", verified=False, verifier_error="boom"
        )
    }
    gate = _build_gate({"drafts": {"s1": {}}, "editorial": []}, verification, strict=True)
    assert gate.verification_incomplete is True
    assert gate.verifier_error_count == 1
    assert any("verifier call failed" in r for r in gate.reasons)


def test_route_after_critique_ignores_unknown_section(deps_factory):
    route = make_route_after_critique(deps_factory())
    state = {
        "revision_round": 0,
        "max_revision_rounds": 1,
        "drafts": {"introduction": {"title": "Introduction"}},
        "issues": [
            {
                "round": 0,
                "severity": "blocking",
                "target_section": "ghost-section",
                "description": "x",
                "category": "factual",
            }
        ],
    }
    # When no valid section is affected the document moves to the editorial pass.
    assert route(state) == "editor"

