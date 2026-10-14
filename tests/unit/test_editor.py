"""Whole-document editor: sibling visibility, edits, citation safety, failures."""

from __future__ import annotations

from scholastra.agents.fake import ScriptedResponder
from scholastra.graph.nodes.editor import make_editor_node
from scholastra.models.schemas import (
    DocumentContract,
    GlossaryEntry,
    Outline,
    Section,
    SectionDraft,
    Source,
)


class _EditorResponder(ScriptedResponder):
    def __init__(self, *, edited: str | None, section: str = "body", changed: bool = True,
                 findings: list | None = None, bad: bool = False) -> None:
        self.edited = edited
        self.section = section
        self.changed = changed
        self.findings = findings or []
        self.bad = bad
        self.prompt = ""

    def _node_editor(self, prompt: str) -> dict:
        self.prompt = prompt
        if self.bad:
            return {"findings": "not-a-list"}
        edits = []
        if self.edited is not None:
            edits.append(
                {
                    "section_id": self.section,
                    "revised_markdown": self.edited,
                    "changed": self.changed,
                    "rationale": "editorial cleanup",
                }
            )
        return {"findings": self.findings, "edits": edits, "notes": ""}


def _state() -> dict:
    outline = Outline(
        title="T",
        sections=[
            Section(id="intro", title="Introduction", purpose="p"),
            Section(id="body", title="Body", purpose="p"),
        ],
    )
    drafts = {
        "intro": SectionDraft(
            section_id="intro",
            title="Introduction",
            markdown="## Introduction\n\nX is the core concept [@doi:1].",
        ).model_dump(),
        "body": SectionDraft(
            section_id="body",
            title="Body",
            markdown="## Body\n\nX is the core concept [@doi:1]. More on Y.",
        ).model_dump(),
    }
    return {
        "project_id": "ed-1",
        "outline": outline.model_dump(),
        "drafts": drafts,
        "sources": {"doi:1": Source(id="doi:1", kind="paper", title="T").model_dump()},
        "evidence": {},
        "verification": {},
        "contract": {},
        "revision_round": 0,
        "editor_round": 0,
    }


async def test_editor_sees_sibling_sections(deps_factory):
    responder = _EditorResponder(edited="## Body\n\nMore on Y [@doi:1].", changed=True)
    node = make_editor_node(deps_factory(responder=responder))
    await node(_state())
    assert "X is the core concept" in responder.prompt  # intro prose visible
    assert "More on Y" in responder.prompt  # sibling body prose visible
    assert "## Introduction" in responder.prompt


async def test_editor_removes_duplicate_definition_and_preserves_citations(deps_factory):
    edited = "## Body\n\nMore on Y [@doi:1]."
    findings = [
        {
            "id": "f1",
            "kind": "duplicate_definition",
            "severity": "major",
            "sections": ["body"],
            "description": "X is defined in both intro and body.",
        }
    ]
    responder = _EditorResponder(edited=edited, findings=findings)
    node = make_editor_node(deps_factory(responder=responder))
    out = await node(_state())

    body = out["drafts"]["body"]
    assert "X is the core concept" not in body["markdown"]
    assert body["markdown"].startswith("## Body")
    assert "[@doi:1]" in body["markdown"]
    assert body["revision"] == 1
    assert out["editorial"] and out["editorial"][0]["kind"] == "duplicate_definition"
    assert any(i["category"] == "editorial:duplicate_definition" for i in out["issues"])


async def test_editor_unknown_citation_is_blocking(deps_factory):
    edited = "## Body\n\nY depends on Z [@ghost]."
    responder = _EditorResponder(edited=edited)
    node = make_editor_node(deps_factory(responder=responder))
    out = await node(_state())
    assert "[@ghost]" not in out["drafts"]["body"]["markdown"]
    assert any(
        i["severity"] == "blocking" and i["category"] == "citation" for i in out["issues"]
    )


async def test_editor_no_change_keeps_draft(deps_factory):
    responder = _EditorResponder(edited="## Body\n\nX is the core concept [@doi:1]. More on Y.",
                                 changed=False)
    node = make_editor_node(deps_factory(responder=responder))
    out = await node(_state())
    assert out["drafts"] == {}
    assert out["editorial"] == []


async def test_editor_failure_is_blocking(deps_factory):
    responder = _EditorResponder(edited=None, bad=True)
    node = make_editor_node(deps_factory(responder=responder))
    out = await node(_state())
    assert any(
        i["severity"] == "blocking" and i["category"] == "editorial" for i in out["issues"]
    )
    assert any(f["severity"] == "blocking" for f in out["editorial"])


async def test_editor_reports_terminology_drift(deps_factory):
    state = _state()
    state["drafts"]["body"]["markdown"] = "## Body\n\nThe thing is central [@doi:1]."
    state["contract"] = DocumentContract(
        glossary=[GlossaryEntry(term="topic", forbidden_variants=["the thing"])]
    ).model_dump()
    node = make_editor_node(deps_factory(responder=_EditorResponder(edited=None)))
    out = await node(state)
    assert any(f["kind"] == "terminology_drift" for f in out["editorial"])
    assert any(i["category"] == "editorial:terminology_drift" for i in out["issues"])
