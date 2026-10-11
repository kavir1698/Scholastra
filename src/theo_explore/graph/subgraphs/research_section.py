"""``research_section`` subgraph: a bounded ReAct loop over read-only MCP tools.

The subgraph is invoked once per researchable section (via ``Send``) and returns
updates for the parent ``evidence`` and ``sources`` channels. Tool output is
treated strictly as untrusted data: it is summarised back into the planner prompt
and normalised into :class:`Source` records, never executed or followed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.graph.deps import Deps
from theo_explore.graph.reducers import add_lists, merge_budget, merge_by_key
from theo_explore.models.schemas import (
    NodeUsage,
    ResearchBrief,
    SectionEvidence,
    ToolCall,
    ToolPlan,
)
from theo_explore.sources.registry import catalog_lines, normalize_tool_result, normalize_url


class ResearchState(TypedDict, total=False):
    project_id: str
    section_id: str
    section_title: str
    section_purpose: str
    brief: dict[str, Any]
    observations: Annotated[list[str], add_lists]
    next_plan: dict[str, Any]
    tool_calls_used: int
    steps: int
    sources: Annotated[dict[str, dict[str, Any]], merge_by_key]
    evidence: Annotated[dict[str, dict[str, Any]], merge_by_key]
    budget: Annotated[dict[str, Any], merge_budget]


def _tool_signature(tool: Any) -> str:
    args = getattr(tool, "args", None) or {}
    rendered = ", ".join(args.keys()) if isinstance(args, dict) else ""
    description = (getattr(tool, "description", "") or "").split("\n")[0]
    return f"{tool.name}({rendered}) — {description}"


def _fallback_plan(state: ResearchState, tools: list[Any], brief: ResearchBrief) -> ToolPlan:
    """Deterministic plan used when the LLM planning step fails.

    Issues the section's planned queries directly against the available search
    tools so a planning failure never silently yields an evidence-free section.
    """
    if state.get("tool_calls_used", 0) > 0 or not tools:
        return ToolPlan(done=True)
    search = [t for t in tools if "search" in t.name.lower()] or tools
    queries = (brief.queries or [state.get("section_title", "")])[: brief.max_tool_calls]
    calls = [
        ToolCall(tool=search[i % len(search)].name, arguments={"query": query})
        for i, query in enumerate(queries)
        if query
    ]
    return ToolPlan(tool_calls=calls, done=not calls, reasoning="deterministic fallback")


def _summarize_observation(result: Any, *, limit: int = 1200) -> str:
    text: str
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except TypeError:
            text = str(result)
    return text[:limit]


def build_research_subgraph(deps: Deps):
    allowlist: list[str] = list(deps.flags.get("research_allowlist") or [])
    tool_timeout = float(deps.flags.get("tool_timeout_seconds", 60))

    async def plan_step(state: ResearchState) -> dict[str, Any]:
        brief = ResearchBrief.model_validate(
            state.get("brief") or {"section_id": state["section_id"]}
        )
        tools = await deps.tools.get_tools(allowlist)
        prompt = render(
            "research_plan",
            section_id=state["section_id"],
            section_title=state.get("section_title", state["section_id"]),
            queries=brief.queries or [state.get("section_title", "")],
            step=state.get("steps", 0),
            tool_calls_used=state.get("tool_calls_used", 0),
            max_tool_calls=brief.max_tool_calls,
            tool_names=[t.name for t in tools],
            tool_signatures=[_tool_signature(t) for t in tools],
            observations="\n".join(state.get("observations", []))[:4000],
        )
        try:
            plan, usage = await deps.llm.structured(
                "research_plan",
                ToolPlan,
                [{"role": "user", "content": prompt}],
                extra_system=system_preamble("Research Agent", untrusted=True),
                max_repairs=1,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a run on one plan step
            deps.log("research_plan_failed", section=state["section_id"], error=str(exc))
            plan, usage = _fallback_plan(state, tools, brief), NodeUsage()
        return {
            "next_plan": plan.model_dump(),
            "steps": state.get("steps", 0) + 1,
            **deps.budget_delta("research_plan", usage),
        }

    async def execute(state: ResearchState) -> dict[str, Any]:
        brief = ResearchBrief.model_validate(
            state.get("brief") or {"section_id": state["section_id"]}
        )
        plan = ToolPlan.model_validate(state.get("next_plan") or {})
        tools = {t.name: t for t in await deps.tools.get_tools(allowlist)}
        used = state.get("tool_calls_used", 0)
        remaining = max(brief.max_tool_calls - used, 0)
        observations: list[str] = []
        new_sources: dict[str, dict[str, Any]] = {}
        executed = 0
        for call in plan.tool_calls[:remaining]:
            tool = tools.get(call.tool)
            if tool is None:
                continue
            try:
                result = await asyncio.wait_for(tool.ainvoke(call.arguments), timeout=tool_timeout)
            except Exception as exc:  # noqa: BLE001 - isolate tool failures
                deps.log("tool_error", section=state["section_id"], tool=call.tool, error=str(exc))
                continue
            executed += 1
            for source in normalize_tool_result(call.tool, result):
                new_sources[source.id] = source.model_dump()
            observations.append(
                f"[{call.tool}({call.arguments})] {_summarize_observation(result)}"
            )
        deps.log(
            "research_tools",
            section=state["section_id"],
            calls=executed,
            sources=len(new_sources),
        )
        return {
            "observations": observations,
            "sources": new_sources,
            "tool_calls_used": used + executed,
        }

    async def extract(state: ResearchState) -> dict[str, Any]:
        sources = state.get("sources") or {}
        prompt = render(
            "research_extract",
            section_id=state["section_id"],
            section_title=state.get("section_title", state["section_id"]),
            section_purpose=state.get("section_purpose", ""),
            catalog=catalog_lines({sid: _source_view(s) for sid, s in sources.items()}),
            observations="\n".join(state.get("observations", []))[:8000],
        )
        try:
            evidence, usage = await deps.llm.structured(
                "research",
                SectionEvidence,
                [{"role": "user", "content": prompt}],
                extra_system=system_preamble("Research Agent", untrusted=True),
                max_repairs=1,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to empty evidence
            deps.log("research_extract_failed", section=state["section_id"], error=str(exc))
            evidence, usage = SectionEvidence(section_id=state["section_id"]), NodeUsage()

        evidence = evidence.model_copy(
            update={
                "section_id": state["section_id"],
                "tool_calls": state.get("tool_calls_used", 0),
                "claims": _bind_claims(evidence, sources, deps.log),
            }
        )
        deps.workspace(state["project_id"]).save_evidence(evidence)
        return {
            "evidence": {state["section_id"]: evidence.model_dump()},
            **deps.budget_delta(
                "research", usage, tool_calls=state.get("tool_calls_used", 0)
            ),
        }

    def route_step(state: ResearchState) -> str:
        brief = ResearchBrief.model_validate(
            state.get("brief") or {"section_id": state["section_id"]}
        )
        plan = state.get("next_plan") or {}
        if plan.get("done") or not plan.get("tool_calls"):
            return "extract"
        if state.get("steps", 0) > brief.max_tool_calls:
            return "extract"
        if state.get("tool_calls_used", 0) >= brief.max_tool_calls:
            return "extract"
        return "execute"

    graph = StateGraph(ResearchState)
    graph.add_node("plan_step", plan_step)
    graph.add_node("execute", execute)
    graph.add_node("extract", extract)
    graph.add_edge(START, "plan_step")
    graph.add_conditional_edges(
        "plan_step", route_step, {"execute": "execute", "extract": "extract"}
    )
    graph.add_edge("execute", "plan_step")
    graph.add_edge("extract", END)
    return graph.compile(name="research_section")


def _source_view(data: dict[str, Any]):
    from theo_explore.models.schemas import Source

    return Source.model_validate(data)


def _source_lookup(sources: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for sid, data in sources.items():
        data = data or {}
        lookup[sid] = sid
        lookup[sid.lower()] = sid
        url = data.get("url")
        if url:
            norm = normalize_url(str(url))
            for key in (str(url), str(url).rstrip("/"), norm, f"url:{norm}"):
                lookup.setdefault(key.lower(), sid)
        title = str(data.get("title") or "").strip().lower()
        if title:
            lookup.setdefault(title, sid)
    return lookup


def _resolve_citation(key: str, lookup: dict[str, str]) -> str | None:
    raw = key.strip().lstrip("@")
    candidates = [raw, raw.lower()]
    if "://" in raw:
        candidates.append(f"url:{normalize_url(raw)}")
    else:
        candidates.append(f"url:{normalize_url('https://' + raw)}")
    for candidate in candidates:
        if candidate and candidate in lookup:
            return lookup[candidate]
    return None


def _bind_claims(evidence: SectionEvidence, sources: dict[str, Any], log: Any = None) -> list:
    """Resolve each cited source id leniently; drop only genuinely unknown cites."""
    lookup = _source_lookup(sources)
    bound = []
    dropped: list[str] = []
    for claim in evidence.claims:
        valid: list[str] = []
        for raw in claim.source_ids:
            resolved = _resolve_citation(str(raw), lookup)
            if resolved and resolved not in valid:
                valid.append(resolved)
            elif not resolved:
                dropped.append(str(raw))
        if valid:
            bound.append(claim.model_copy(update={"source_ids": valid}))
        else:
            dropped.append(claim.id)
    if dropped and log is not None:
        log(
            "claims_dropped",
            section=evidence.section_id,
            count=len(dropped),
            sample=dropped[:5],
        )
    return bound
