"""Outline node + human-in-the-loop interrupt."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.config import load_preset
from theo_explore.graph.deps import Deps
from theo_explore.graph.validation import sanitize_outline
from theo_explore.models.schemas import Outline, ScopeProposal


def make_outline_node(deps: Deps):
    async def outline(state: dict[str, Any]) -> dict[str, Any]:
        scope = ScopeProposal.model_validate(state["scope"])
        preset = load_preset(scope.length_preset, deps.settings)
        prompt = render("planner", scope=scope, preset=preset)
        messages = [{"role": "user", "content": prompt}]
        proposal, usage = await deps.llm.structured(
            "outline",
            Outline,
            messages,
            extra_system=system_preamble("Planner"),
        )
        proposal, warnings = sanitize_outline(proposal)
        for warning in warnings:
            deps.log("outline_warning", warning=warning)
        deps.workspace(state["project_id"]).save_outline(proposal)
        deps.log("outline", sections=[s.id for s in proposal.sections])
        return {
            "outline": proposal.model_dump(),
            "status": "await_outline",
            "errors": warnings,
            **deps.budget_delta("outline", usage),
        }

    return outline


def make_await_outline_node(deps: Deps):
    async def await_outline(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("auto"):
            deps.log("await_outline", auto=True)
            return {"status": "plan_research"}

        decision = interrupt({"kind": "outline_review", "outline": state.get("outline", {})})
        edits = decision.get("outline") if isinstance(decision, dict) else None
        if edits:
            proposal, _ = sanitize_outline(Outline.model_validate(edits))
            deps.workspace(state["project_id"]).save_outline(proposal)
            deps.log("await_outline", edited=True)
            return {"outline": proposal.model_dump(), "status": "plan_research"}
        deps.log("await_outline", approved=True)
        return {"status": "plan_research"}

    return await_outline
