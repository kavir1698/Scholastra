"""Citation verification: mechanical resolution + claim-support checks."""

from __future__ import annotations

from theo_explore.agents.fake import ScriptedResponder
from theo_explore.graph.nodes.quality import make_route_after_critique, make_verify_citations_node
from theo_explore.models.schemas import Claim, SectionDraft, SectionEvidence, Source


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
    assert route(state) == "synthesize"

