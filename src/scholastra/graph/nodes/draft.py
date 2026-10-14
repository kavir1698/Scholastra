"""Draft scheduling: topological-wave ``Send`` fan-out by ``depends_on``."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from scholastra.graph.deps import Deps
from scholastra.graph.validation import ready_sections
from scholastra.models.schemas import Outline, Section
from scholastra.observability import budget_exceeded


def _draftable_sections(outline: Outline) -> list[Section]:
    return [s for s in outline.sections if s.target_words > 0]


def make_dispatch_draft(deps: Deps):
    def dispatch_draft(state: dict[str, Any]) -> list[Send] | str:
        if budget_exceeded(state.get("budget")):
            deps.log("budget_abort", stage="draft")
            return "verify_citations"
        outline = Outline.model_validate(state["outline"])
        sections = _draftable_sections(outline)
        drafted = set((state.get("drafts") or {}).keys())
        ready = ready_sections(sections, drafted)
        if not ready:
            return "verify_citations"

        by_id = {s.id: s for s in sections}
        evidence = state.get("evidence") or {}
        drafts = state.get("drafts") or {}
        sends: list[Send] = []
        for sid in ready:
            section = by_id[sid]
            dep_summaries = [
                (drafts[dep].get("summary") or drafts[dep].get("title") or "")
                for dep in section.depends_on
                if dep in drafts
            ]
            sends.append(
                Send(
                    "draft_section",
                    {
                        "project_id": state["project_id"],
                        "section_id": sid,
                        "section_title": section.title,
                        "section_purpose": section.purpose,
                        "target_words": section.target_words,
                        "subsections": [sub.model_dump() for sub in section.subsections],
                        "evidence": evidence.get(sid, {}),
                        "dep_summaries": dep_summaries,
                        "contract": state.get("contract") or {},
                        "revision_round": state.get("revision_round", 0),
                    },
                )
            )
        deps.log("dispatch_draft", wave=state.get("draft_wave", 0), sections=ready)
        return sends

    return dispatch_draft


def make_draft_join_node(deps: Deps):
    async def draft_join(state: dict[str, Any]) -> dict[str, Any]:
        return {"draft_wave": state.get("draft_wave", 0) + 1, "status": "draft"}

    return draft_join
