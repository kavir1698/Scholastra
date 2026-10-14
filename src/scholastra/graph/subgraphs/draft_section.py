"""``draft_section`` subgraph: prepare -> write -> post-check.

Depth comes from drafting every LEAF unit (a subsection with no children) as its
own focused generation, concurrently, then joining the tree under the section
heading. Parents contribute a heading only. A short expand retry lifts any leaf
that falls short of its target. The post-check enforces the hard rule that a
draft may not cite any source absent from that section's evidence.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from scholastra.agents.prompting import render, system_preamble
from scholastra.citations import filter_citations, iter_citation_keys
from scholastra.graph.contract import as_contract, compact_contract, normalize_terminology
from scholastra.graph.deps import Deps
from scholastra.graph.reducers import merge_budget, merge_by_key
from scholastra.models.schemas import NodeUsage, SectionDraft, SectionEvidence
from scholastra.utils import word_count


class DraftState(TypedDict, total=False):
    project_id: str
    section_id: str
    section_title: str
    section_purpose: str
    subsections: list[dict[str, Any]]
    evidence: dict[str, Any]
    dep_summaries: list[str]
    contract: dict[str, Any]
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


def _add(left: NodeUsage, right: NodeUsage) -> NodeUsage:
    return NodeUsage(
        calls=left.calls + right.calls,
        tokens_in=left.tokens_in + right.tokens_in,
        tokens_out=left.tokens_out + right.tokens_out,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def walk_units(units: list[dict[str, Any]], depth: int = 1):
    """Yield ``(depth, unit, has_children)`` for the subtree, in document order."""
    for unit in units:
        children = unit.get("children") or []
        yield depth, unit, bool(children)
        if children:
            yield from walk_units(children, depth + 1)


def _strip_leading_heading(markdown: str) -> str:
    lines = markdown.lstrip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def build_draft_subgraph(deps: Deps):
    async def prepare(state: DraftState) -> dict[str, Any]:
        evidence = _evidence_from_state(state)
        keys = sorted({sid for claim in evidence.claims for sid in claim.source_ids})
        return {"available_keys": keys}

    async def _draft_one(prompt: str, target_words: int, section_id: str) -> tuple[str, NodeUsage]:
        draft, usage = await deps.llm.structured(
            "draft",
            SectionDraft,
            [{"role": "user", "content": prompt}],
            extra_system=system_preamble("Drafter", untrusted=True),
            max_repairs=1,
        )
        markdown = draft.markdown or ""
        # One bounded expansion pass when the unit falls short of its target.
        if target_words and word_count(markdown) < 0.6 * target_words:
            follow_up = (
                f"{prompt}\n\nYour previous draft was only {word_count(markdown)} words. "
                f"Expand it to about {target_words} words: add depth, mechanism, examples, "
                "numbers, competing views and limitations. Keep the same inline citations."
            )
            try:
                expanded, extra_usage = await deps.llm.structured(
                    "draft",
                    SectionDraft,
                    [{"role": "user", "content": follow_up}],
                    extra_system=system_preamble("Drafter", untrusted=True),
                    max_repairs=1,
                )
                usage = _add(usage, extra_usage)
                if word_count(expanded.markdown or "") > word_count(markdown):
                    markdown = expanded.markdown or markdown
            except Exception as exc:  # noqa: BLE001 - keep the original draft
                deps.log("draft_expand_failed", section=section_id, error=str(exc))
        return markdown, usage

    async def write(state: DraftState) -> dict[str, Any]:
        evidence = _evidence_from_state(state)
        keys = state.get("available_keys", [])
        section_title = state.get("section_title", state["section_id"])
        claims = [c.model_dump() for c in evidence.claims]
        dep_summaries = state.get("dep_summaries", [])
        revision_notes = state.get("revision_notes", "")
        revision_round = state.get("revision_round", 0)
        subsections = state.get("subsections") or []
        contract = as_contract(state.get("contract"))
        section_contract = compact_contract(contract, section_id=state["section_id"])

        async def draft_leaf(unit: dict[str, Any], index: int) -> tuple[str, NodeUsage]:
            target = int(unit.get("target_words") or 0)
            leaf_id = unit.get("id") or f"{state['section_id']}-leaf-{index + 1}"
            leaf_contract = compact_contract(
                contract, section_id=state["section_id"], subsection_id=leaf_id
            )
            prompt = render(
                "subsection",
                section_id=state["section_id"],
                section_title=section_title,
                subsection_id=leaf_id,
                subsection_title=unit.get("title") or "Discussion",
                subsection_purpose=unit.get("purpose", ""),
                key_questions=unit.get("key_questions", []),
                target_words=target or 400,
                revision_round=revision_round,
                cite_keys=keys,
                claims=claims,
                dep_summaries=dep_summaries,
                contract=leaf_contract,
                revision_notes=revision_notes,
            )
            return await _draft_one(prompt, target, state["section_id"])

        parts: list[str] = [f"## {section_title}"]
        usage = NodeUsage()

        if subsections:
            nodes = list(walk_units(subsections))
            leaves = [(depth, unit) for depth, unit, has_children in nodes if not has_children]
            leaf_results = await asyncio.gather(
                *(draft_leaf(unit, i) for i, (_, unit) in enumerate(leaves))
            )
            leaf_markdown = [markdown for markdown, _ in leaf_results]
            for _, node_usage in leaf_results:
                usage = _add(usage, node_usage)

            leaf_index = 0
            for depth, unit, has_children in nodes:
                level = min(depth + 2, 6)
                parts.append(f"{'#' * level} {unit.get('title') or 'Section'}")
                if not has_children:
                    body = _strip_leading_heading(leaf_markdown[leaf_index])
                    leaf_index += 1
                    if body:
                        parts.append(body)
        else:
            prompt = render(
                "drafter",
                section_id=state["section_id"],
                section_title=section_title,
                section_purpose=state.get("section_purpose", ""),
                revision_round=revision_round,
                target_words=state.get("target_words", 600),
                cite_keys=keys,
                claims=claims,
                dep_summaries=dep_summaries,
                contract=section_contract,
                revision_notes=revision_notes,
            )
            markdown, usage = await _draft_one(
                prompt, state.get("target_words", 600), state["section_id"]
            )
            if markdown.strip():
                parts.append(_strip_leading_heading(markdown))

        markdown = "\n\n".join(part for part in parts if part.strip())
        markdown, term_changes = normalize_terminology(markdown, contract)
        if term_changes:
            deps.log(
                "draft_terminology_normalized",
                section=state["section_id"],
                changes=term_changes[:5],
            )
        draft = SectionDraft(
            section_id=state["section_id"],
            title=section_title,
            markdown=markdown,
            revision=revision_round,
            word_count=word_count(markdown),
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
                "stripped_cites": sorted(set(removed)),
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
