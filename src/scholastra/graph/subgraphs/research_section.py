"""``research_section`` subgraph: a bounded ReAct loop over read-only MCP tools.

The subgraph is invoked once per researchable section (via ``Send``) and returns
updates for the parent ``evidence`` and ``sources`` channels. Tool output is
treated strictly as untrusted data: it is summarised back into the planner prompt
and normalised into :class:`Source` records, never executed or followed.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from scholastra.agents.prompting import render, system_preamble
from scholastra.config import load_source_policy
from scholastra.graph.contract import as_contract, compact_contract
from scholastra.graph.deps import Deps
from scholastra.graph.reducers import add_lists, merge_budget, merge_by_key
from scholastra.models.schemas import (
    NodeUsage,
    ResearchBrief,
    SectionEvidence,
    ToolCall,
    ToolPlan,
)
from scholastra.sources.policy import policy_lines
from scholastra.sources.registry import (
    catalog_lines,
    normalize_tool_result,
    normalize_url,
    source_from_text_result,
)


class ResearchState(TypedDict, total=False):
    project_id: str
    section_id: str
    section_title: str
    section_purpose: str
    brief: dict[str, Any]
    contract: dict[str, Any]
    observations: Annotated[list[str], add_lists]
    next_plan: dict[str, Any]
    tool_calls_used: int
    steps: int
    started_at: float
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


def _wall_clock_expired(state: ResearchState, brief: ResearchBrief) -> bool:
    """Whether the section's per-brief wall clock has elapsed."""
    started = state.get("started_at")
    if not started or not brief.wall_clock_seconds:
        return False
    return (time.time() - float(started)) >= brief.wall_clock_seconds


def _summarize_observation(result: Any, *, limit: int = 6000) -> str:
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
    source_policy = load_source_policy(deps.settings)

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
            observations="\n".join(state.get("observations", []))[:8000],
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
        called_tools: list[str] = []
        minted = 0
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
            called_tools.append(call.tool)
            produced = normalize_tool_result(call.tool, result, policy=source_policy)
            if not produced:
                # Fetch/convert tools return page text rather than records; mint a
                # Source from the URL so the evidence can actually be cited.
                text_source = source_from_text_result(
                    call.tool, call.arguments, result, policy=source_policy
                )
                if text_source is not None:
                    produced = [text_source]
                    minted += 1
            for source in produced:
                new_sources[source.id] = source.model_dump()
            observations.append(
                f"[{call.tool}({call.arguments})] {_summarize_observation(result)}"
            )
        deps.log(
            "research_tools",
            section=state["section_id"],
            calls=executed,
            sources=len(new_sources),
            tools=called_tools,
            minted=minted,
        )
        return {
            "observations": observations,
            "sources": new_sources,
            "tool_calls_used": used + executed,
        }

    async def extract(state: ResearchState) -> dict[str, Any]:
        sources = state.get("sources") or {}
        contract = as_contract(state.get("contract"))
        contract_block = compact_contract(
            contract,
            section_id=state["section_id"],
            policy_lines=policy_lines(source_policy),
        )
        prompt = render(
            "research_extract",
            section_id=state["section_id"],
            section_title=state.get("section_title", state["section_id"]),
            section_purpose=state.get("section_purpose", ""),
            catalog=catalog_lines({sid: _source_view(s) for sid, s in sources.items()}),
            observations="\n".join(state.get("observations", []))[:20000],
            contract=contract_block,
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
        if _wall_clock_expired(state, brief):
            deps.log(
                "research_wall_clock",
                section=state["section_id"],
                elapsed=round(time.time() - float(state["started_at"]), 1),
            )
            return "extract"
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
    from scholastra.models.schemas import Source

    return Source.model_validate(data)


_SLUG_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "in",
        "on",
        "to",
        "for",
        "with",
        "is",
        "are",
        "be",
        "by",
        "at",
        "as",
        "from",
        "into",
        "that",
        "this",
        "its",
    }
)


def _slugify(text: str) -> str:
    return _SLUG_SPLIT_RE.sub("-", (text or "").strip().lower()).strip("-")


def _significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in _SLUG_SPLIT_RE.split((text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _source_lookup(sources: dict[str, Any]) -> dict[str, str]:
    """Lenient key -> SourceId index.

    Indexes each source by its id, URL (raw and normalised), title and title slug,
    DOI, arXiv id and a possible author-year bibkey, so an extractor that cites a
    source by any of those resolves instead of being dropped as unknown.
    """
    lookup: dict[str, str] = {}

    def index(key: Any, sid: str) -> None:
        if key:
            lookup.setdefault(str(key).strip().lower(), sid)

    for sid, data in sources.items():
        data = data or {}
        lookup[sid] = sid
        lookup[sid.lower()] = sid
        index(_slugify(sid), sid)
        url = data.get("url")
        if url:
            norm = normalize_url(str(url))
            for key in (str(url), str(url).rstrip("/"), norm, f"url:{norm}"):
                index(key, sid)
        title = str(data.get("title") or "").strip()
        if title:
            index(title, sid)
            index(_slugify(title), sid)
        doi = data.get("doi")
        if doi:
            index(str(doi), sid)
            index(f"doi:{str(doi).lower()}", sid)
        arxiv_id = data.get("arxiv_id")
        if arxiv_id:
            index(str(arxiv_id), sid)
            index(f"arxiv:{arxiv_id}", sid)
        authors = data.get("authors") or []
        year = data.get("year")
        if authors and year:
            first = str(authors[0]).split()
            if first:
                family = first[-1]
                index(f"{family}{year}", sid)
                index(f"{family} {year}", sid)
                if len(authors) > 1:
                    index(f"{family} et al {year}", sid)
    return lookup


def _fuzzy_resolve(raw: str, sources: dict[str, Any]) -> str | None:
    """Conservative title-token fallback for a cite key that is not an exact id.

    Binds only when the key shares at least two significant tokens with a source
    title and either the key's tokens are a subset of the title's or the Jaccard
    overlap is at least 0.6 — otherwise the claim keeps being dropped rather than
    bound to the wrong source.
    """
    key_tokens = _significant_tokens(raw)
    if len(key_tokens) < 2:
        return None
    best: str | None = None
    best_score = 0.0
    for sid, data in (sources or {}).items():
        title_tokens = _significant_tokens(str((data or {}).get("title") or ""))
        if len(title_tokens) < 2:
            continue
        overlap = key_tokens & title_tokens
        if len(overlap) < 2:
            continue
        score = len(overlap) / len(key_tokens | title_tokens)
        if key_tokens <= title_tokens:
            score = max(score, 0.75)
        if score > best_score:
            best_score, best = score, sid
    if best is not None and best_score >= 0.6:
        return best
    return None


def _resolve_citation(key: str, lookup: dict[str, str], sources: dict[str, Any]) -> str | None:
    raw = key.strip().lstrip("@")
    if not raw:
        return None
    candidates = [raw, raw.lower(), _slugify(raw)]
    if "://" in raw:
        candidates.append(f"url:{normalize_url(raw)}")
    else:
        candidates.append(f"url:{normalize_url('https://' + raw)}")
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in lookup:
            return lookup[candidate]
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return _fuzzy_resolve(raw, sources)


def _origin_for(sid: str, sources: dict[str, Any]) -> str:
    data = sources.get(sid) or {}
    mode = data.get("access_mode")
    if mode and mode != "unknown":
        return str(mode)
    return "unknown"


def _tier_for(sid: str, sources: dict[str, Any]) -> int | None:
    tier = (sources.get(sid) or {}).get("tier")
    try:
        return int(tier) if tier is not None else None
    except (TypeError, ValueError):
        return None


def _bind_claims(evidence: SectionEvidence, sources: dict[str, Any], log: Any = None) -> list:
    """Resolve each cited source id leniently; drop only genuinely unknown cites."""
    lookup = _source_lookup(sources)
    bound = []
    dropped: list[str] = []
    for claim in evidence.claims:
        valid: list[str] = []
        for raw in claim.source_ids:
            resolved = _resolve_citation(str(raw), lookup, sources)
            if resolved and resolved not in valid:
                valid.append(resolved)
            elif not resolved:
                dropped.append(str(raw))
        if valid:
            updates: dict[str, Any] = {"source_ids": valid}
            if not claim.origin or claim.origin == "unknown":
                updates["origin"] = _origin_for(valid[0], sources)
            if claim.tier is None:
                updates["tier"] = _tier_for(valid[0], sources)
            bound.append(claim.model_copy(update=updates))
        else:
            dropped.append(claim.id)
    if dropped and log is not None:
        log(
            "claims_dropped",
            section=evidence.section_id,
            count=len(dropped),
            sample=dropped[:5],
            sources=len(sources),
        )
    return bound
