"""Scope node + human-in-the-loop interrupt."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.config import load_preset
from theo_explore.graph.deps import Deps
from theo_explore.models.schemas import Brief, ScopeProposal


def make_scope_node(deps: Deps):
    async def scope(state: dict[str, Any]) -> dict[str, Any]:
        brief = Brief.model_validate(state["brief"])
        preset = load_preset(brief.preset, deps.settings)
        prompt = render("scoper", brief=brief, preset=preset)
        messages = [{"role": "user", "content": prompt}]
        proposal, usage = await deps.llm.structured(
            "scope",
            ScopeProposal,
            messages,
            extra_system=system_preamble("Scoper"),
        )
        proposal = proposal.model_copy(update={"length_preset": brief.preset})
        deps.workspace(state["project_id"]).save_scope(proposal)
        deps.log("scope", refined_question=proposal.refined_question)
        return {
            "scope": proposal.model_dump(),
            "status": "await_scope",
            **deps.budget_delta("scope", usage),
        }

    return scope


def make_await_scope_node(deps: Deps):
    async def await_scope(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("auto"):
            deps.log("await_scope", auto=True)
            return {"status": "outline"}

        decision = interrupt({"kind": "scope_review", "scope": state.get("scope", {})})
        edits = decision.get("scope") if isinstance(decision, dict) else None
        if edits:
            proposal = ScopeProposal.model_validate(edits)
            deps.workspace(state["project_id"]).save_scope(proposal)
            deps.log("await_scope", edited=True)
            return {"scope": proposal.model_dump(), "status": "outline"}
        deps.log("await_scope", approved=True)
        return {"status": "outline"}

    return await_scope
