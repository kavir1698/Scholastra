"""Synthesis consumes edited drafts/contract and its citations are verified."""

from __future__ import annotations

from theo_explore.agents.fake import ScriptedResponder
from theo_explore.graph.nodes.synthesize import make_synthesize_node
from theo_explore.models.schemas import DocumentContract, SectionDraft, Source


class _SynthResponder(ScriptedResponder):
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.prompt = ""

    def _node_synthesize(self, prompt: str) -> dict:
        self.prompt = prompt
        return {"markdown": self.markdown, "cite_keys": ["doi:1"]}


def _state() -> dict:
    drafts = {
        "body": SectionDraft(
            section_id="body",
            title="Body",
            markdown="## Body\n\nEdited prose with grounding [@doi:1].",
            revision=1,
        ).model_dump()
    }
    return {
        "project_id": "syn-1",
        "synthesize": True,
        "drafts": drafts,
        "evidence": {},
        "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
        "verification": {"body": {"resolved_cites": ["doi:1"]}},
        "contract": DocumentContract(thesis="A THESIS SENTENCE.").model_dump(),
    }


async def test_synthesis_receives_edited_drafts_and_contract(deps_factory):
    responder = _SynthResponder("## Synthesis\n\nA cross-cutting theme [@doi:1].")
    node = make_synthesize_node(deps_factory(responder=responder))
    out = await node(_state())

    assert "Edited prose with grounding" in responder.prompt
    assert "A THESIS SENTENCE." in responder.prompt
    assert "A cross-cutting theme" in out["synthesis"]
    assert out["synthesis_verification"]


async def test_synthesis_unknown_citation_is_removed(deps_factory):
    responder = _SynthResponder("## Synthesis\n\nClaim [@ghost].")
    node = make_synthesize_node(deps_factory(responder=responder))
    out = await node(_state())
    assert "[@ghost]" not in out["synthesis"]


async def test_synthesis_disabled_is_noop(deps_factory):
    node = make_synthesize_node(deps_factory())
    out = await node({"synthesize": False})
    assert out["synthesis"] is None
