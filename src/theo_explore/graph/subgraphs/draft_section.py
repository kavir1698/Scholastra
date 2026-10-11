"""``draft_section`` subgraph: prepare -> write -> post-check.

The post-check enforces the hard rule that a draft may not cite any source that
is absent from that section's evidence: unknown inline cite keys are stripped and
recorded, and the word count is recomputed from the sanitised Markdown.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.citations import filter_citations, iter_citation_keys
from theo_explore.graph.deps import Deps
from theo_explore.graph.reducers import merge_budget, merge_by_key
from theo_explore.models.schemas import SectionDraft, SectionEvidence
from theo_explore.utils import word_count


class DraftState(TypedDict, total=False):
    project_id: str
    section_id: str
    section_title: str
    section_purpose: str
    evidence: dict[str, Any]
    dep_summaries: list[str]
    revision_round: int
    revision_notes: str
    target_words: int
    available_keys: list[str]
    drafts: Annotated[dict[str, dict[str, Any]], merge_by_key]
    budget: Annotated[dict[str, Any], merge_budget]


def _evidence_from_state(state: DraftState) -> SectionEvidence:
    """Validate the section evidence, tolerating a missing/empty payload."""
    raw = state.get("evidence")
    if not raw:
        return SectionEvidence(section_id=state["section_id"])
    try:
        return SectionEvidence.model_validate(raw)
    except Exception:  # noqa: BLE001 - never fail drafting on malformed evidence
        return SectionEvidence(section_id=state["section_id"])


def build_draft_subgraph(deps: Deps):
    async def prepare(state: DraftState) -> dict[str, Any]:
        evidence = _evidence_from_state(state)
        keys = sorted({sid for claim in evidence.claims for sid in claim.source_ids})
        return {"available_keys": keys}

    async def write(state: DraftState) -> dict[str, Any]:
        evidence = _evidence_from_state(state)
        prompt = render(
            "drafter",
            section_id=state["section_id"],
            section_title=state.get("section_title", state["section_id"]),
            section_purpose=state.get("section_purpose", ""),
            revision_round=state.get("revision_round", 0),
            target_words=state.get("target_words", 400),
            cite_keys=state.get("available_keys", []),
            claims=[c.model_dump() for c in evidence.claims],
            dep_summaries=state.get("dep_summaries", []),
            revision_notes=state.get("revision_notes", ""),
        )
        draft, usage = await deps.llm.structured(
            "draft",
            SectionDraft,
            [{"role": "user", "content": prompt}],
            extra_system=system_preamble("Drafter", untrusted=True),
            max_repairs=1,
        )
        draft = draft.model_copy(
            update={
                "section_id": state["section_id"],
                "revision": state.get("revision_round", 0),
            }
        )
        return {
            "drafts": {state["section_id"]: draft.model_dump()},
            **deps.budget_delta("draft", usage),
        }

    async def postcheck(state: DraftState) -> dict[str, Any]:
        draft = SectionDraft.model_validate((state.get("drafts") or {})[state["section_id"]])
        allowed = set(state.get("available_keys", []))
        markdown, removed = filter_citations(draft.markdown, allowed)
        keys = sorted(set(iter_citation_keys(markdown)))
        draft = draft.model_copy(
            update={
                "markdown": markdown,
                "cite_keys": keys,
                "word_count": word_count(markdown),
            }
        )
        if removed:
            deps.log("draft_citations_stripped", section=state["section_id"], keys=removed)
        workspace = deps.workspace(state["project_id"])
        workspace.save_draft(draft)
        return {"drafts": {state["section_id"]: draft.model_dump()}}

    graph = StateGraph(DraftState)
    graph.add_node("prepare", prepare)
    graph.add_node("write", write)
    graph.add_node("postcheck", postcheck)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "write")
    graph.add_edge("write", "postcheck")
    graph.add_edge("postcheck", END)
    return graph.compile(name="draft_section")
