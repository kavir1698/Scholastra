"""Research planner, fan-out dispatch and join."""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import Send

from scholastra.agents.prompting import render, system_preamble
from scholastra.config import load_preset, load_source_policy
from scholastra.graph.contract import as_contract, compact_contract
from scholastra.graph.deps import Deps
from scholastra.models.schemas import (
    DocumentContract,
    NodeUsage,
    Outline,
    ResearchBrief,
    ScopeProposal,
    Source,
)
from scholastra.observability import budget_exceeded
from scholastra.sources.policy import policy_lines


def _deterministic_brief(section, budget: dict[str, Any]) -> ResearchBrief:
    queries = list(section.key_questions)[:3] or [section.title]
    return ResearchBrief(
        section_id=section.id,
        queries=queries,
        source_domains=list(section.source_domains),
        max_tool_calls=int(budget.get("max_tool_calls_per_section", 8)),
        max_sources=int(budget.get("max_sources_per_section", 6)),
        wall_clock_seconds=int(budget.get("wall_clock_seconds_per_section", 180)),
    )


def make_plan_research_node(deps: Deps):
    async def plan_research(state: dict[str, Any]) -> dict[str, Any]:
        outline = Outline.model_validate(state["outline"])
        scope = ScopeProposal.model_validate(state["scope"])
        preset = load_preset(scope.length_preset, deps.settings)
        budget = dict(preset.get("research_budget", {}))
        contract: DocumentContract = as_contract(state.get("contract"))
        policy = load_source_policy(deps.settings)
        policy_block = policy_lines(policy)
        plan: dict[str, dict[str, Any]] = {}
        usage = NodeUsage()
        for section in outline.sections:
            if section.target_words <= 0:
                continue
            limits = {
                "max_tool_calls": budget.get("max_tool_calls_per_section", 8),
                "max_sources": budget.get("max_sources_per_section", 6),
            }
            fallback = _deterministic_brief(section, budget)
            contract_block = compact_contract(
                contract,
                section_id=section.id,
                policy_lines=policy_block,
            )
            prompt = render(
                "research_brief",
                section=section,
                budget=limits,
                contract=contract_block,
            )
            try:
                brief, node_usage = await deps.llm.structured(
                    "plan_research",
                    ResearchBrief,
                    [{"role": "user", "content": prompt}],
                    extra_system=system_preamble("Research Planner"),
                    max_repairs=1,
                )
                # Clamp to the preset limits so the LLM cannot inflate the budget.
                brief = brief.model_copy(
                    update={
                        "section_id": section.id,
                        "max_tool_calls": min(brief.max_tool_calls, fallback.max_tool_calls),
                        "max_sources": min(brief.max_sources, fallback.max_sources),
                        "wall_clock_seconds": min(
                            brief.wall_clock_seconds, fallback.wall_clock_seconds
                        ),
                    }
                )
                usage = _add(usage, node_usage)
            except Exception as exc:  # noqa: BLE001 - fall back to deterministic brief
                deps.log("plan_research_fallback", section=section.id, error=str(exc))
                brief = fallback
            plan[section.id] = brief.model_dump()
        deps.log("plan_research", sections=list(plan))
        return {
            "research_plan": plan,
            "status": "research",
            **deps.budget_delta("plan_research", usage),
        }

    return plan_research


def _add(left: NodeUsage, right: NodeUsage) -> NodeUsage:
    return NodeUsage(
        calls=left.calls + right.calls,
        tokens_in=left.tokens_in + right.tokens_in,
        tokens_out=left.tokens_out + right.tokens_out,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def make_dispatch_research(deps: Deps):
    def dispatch_research(state: dict[str, Any]) -> list[Send] | str:
        if budget_exceeded(state.get("budget")):
            deps.log("budget_abort", stage="research")
            return "research_join"
        outline = Outline.model_validate(state["outline"])
        sends: list[Send] = []
        started_at = time.time()
        for section in outline.sections:
            if section.target_words <= 0:
                continue
            sends.append(
                Send(
                    "research_section",
                    {
                        "project_id": state["project_id"],
                        "section_id": section.id,
                        "section_title": section.title,
                        "section_purpose": section.purpose,
                        "brief": state["research_plan"].get(section.id, {}),
                        "contract": state.get("contract") or {},
                        "started_at": started_at,
                    },
                )
            )
        return sends or "research_join"

    return dispatch_research


def make_research_join_node(deps: Deps):
    async def research_join(state: dict[str, Any]) -> dict[str, Any]:
        sources = {sid: Source.model_validate(s) for sid, s in (state.get("sources") or {}).items()}
        deps.workspace(state["project_id"]).save_sources(sources)
        failed = getattr(deps.tools, "failed_servers", None)
        if failed:
            # A dead MCP server silently removes evidence; make it visible.
            deps.log("mcp_servers_failed", servers=list(failed))
        deps.log("research_join", sources=len(sources), sections=len(state.get("evidence") or {}))
        return {"status": "draft"}

    return research_join
