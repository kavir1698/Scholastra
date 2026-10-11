"""Intake node: normalise the raw request into a :class:`Brief` and seed defaults."""

from __future__ import annotations

from typing import Any

from theo_explore.config import load_preset
from theo_explore.graph.deps import Deps
from theo_explore.models.schemas import Brief
from theo_explore.observability import new_budget


def make_intake_node(deps: Deps):
    async def intake(state: dict[str, Any]) -> dict[str, Any]:
        brief = Brief(
            topic=str(state.get("topic", "")).strip(),
            doc_type=state.get("doc_type", "article"),
            preset=state.get("preset", "article-short"),
            audience=state.get("audience"),
            constraints=list(state.get("constraints") or []),
        )
        preset = load_preset(brief.preset, deps.settings)
        deps.workspace(state["project_id"]).save_brief(brief)
        deps.log("intake", topic=brief.topic, doc_type=brief.doc_type, preset=brief.preset)
        return {
            "brief": brief.model_dump(),
            "status": "scope",
            "revision_round": 0,
            "max_revision_rounds": int(preset["quality"]["max_revision_rounds"]),
            "draft_wave": 0,
            "evidence": {},
            "sources": {},
            "drafts": {},
            "verification": {},
            "issues": [],
            "errors": [],
            "exports": {},
            "budget": new_budget(state.get("max_cost_usd")),
        }

    return intake
