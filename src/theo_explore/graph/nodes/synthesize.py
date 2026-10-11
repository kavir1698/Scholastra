"""Optional whole-document synthesis node."""

from __future__ import annotations

from typing import Any

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.graph.deps import Deps
from theo_explore.graph.subgraphs.draft_section import filter_citations
from theo_explore.models.schemas import NodeUsage, SynthesisResult


def make_synthesize_node(deps: Deps):
    async def synthesize(state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("synthesize"):
            return {"synthesis": None, "status": "assemble"}

        drafts = state.get("drafts") or {}
        evidence = state.get("evidence") or {}
        sources = state.get("sources") or {}
        cite_keys = sorted({sid for sid in sources})
        prompt = render("synthesizer", drafts=drafts, evidence=evidence, cite_keys=cite_keys)
        try:
            result, usage = await deps.llm.structured(
                "synthesize",
                SynthesisResult,
                [{"role": "user", "content": prompt}],
                extra_system=system_preamble("Synthesizer", untrusted=True),
                max_repairs=1,
            )
        except Exception as exc:  # noqa: BLE001 - synthesis is optional
            deps.log("synthesize_failed", error=str(exc))
            result, usage = SynthesisResult(markdown=""), NodeUsage()

        markdown, removed = filter_citations(result.markdown, set(sources))
        if removed:
            deps.log("synthesis_citations_stripped", keys=removed)
        deps.log("synthesize", words=len(markdown.split()))
        return {
            "synthesis": markdown,
            "status": "assemble",
            **deps.budget_delta("synthesize", usage),
        }

    return synthesize
