"""Optional whole-document synthesis node.

Synthesis is a cross-cutting conclusion generated from the *edited* drafts and
the contract. It is validated with the same sentence-level citation verifier
before assembly, and it can never be the sole coherence mechanism because the
editor already reconciles the drafts.
"""

from __future__ import annotations

from typing import Any

from scholastra.agents.prompting import render, system_preamble
from scholastra.citations import filter_citations
from scholastra.graph.contract import as_contract, bounded_drafts
from scholastra.graph.deps import Deps
from scholastra.graph.nodes.quality import _add, verify_text
from scholastra.models.schemas import (
    Claim,
    NodeUsage,
    SectionEvidence,
    Source,
    SynthesisResult,
)
from scholastra.sources.policy import source_quality_summary


def _combined_evidence(evidence: dict[str, Any]) -> SectionEvidence:
    claims: list[Claim] = []
    for raw in (evidence or {}).values():
        try:
            model = SectionEvidence.model_validate(raw)
        except Exception:  # noqa: BLE001 - tolerate malformed evidence
            continue
        claims.extend(model.claims)
    return SectionEvidence(section_id="synthesis", claims=claims)


def make_synthesize_node(deps: Deps):
    async def synthesize(state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("synthesize"):
            return {"synthesis": None, "status": "assemble"}

        drafts = state.get("drafts") or {}
        evidence = state.get("evidence") or {}
        sources = state.get("sources") or {}
        verification = state.get("verification") or {}
        contract = as_contract(state.get("contract"))
        source_models = {k: Source.model_validate(v) for k, v in sources.items()}
        cite_keys = sorted(source_models)
        summary = source_quality_summary(source_models)
        prompt_drafts, truncated = bounded_drafts(drafts)
        if truncated:
            deps.log("synthesize_prompt_truncated", sections=truncated)
        prompt = render(
            "synthesizer",
            drafts=prompt_drafts,
            evidence=evidence,
            cite_keys=cite_keys,
            contract=contract,
            source_summary=summary,
            verification=verification,
        )
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

        issues: list[dict[str, Any]] = []
        synthesis_verification: dict[str, Any] | None = None
        if markdown.strip():
            combined = _combined_evidence(evidence)
            synth_report, synth_issues, synth_usage = await verify_text(
                deps, "synthesis", markdown, source_models, combined
            )
            usage = _add(usage, synth_usage)
            synthesis_verification = synth_report
            issues.extend(synth_issues)

        deps.log("synthesize", words=len(markdown.split()), issues=len(issues))
        return {
            "synthesis": markdown,
            "synthesis_verification": synthesis_verification or {},
            "issues": issues,
            "status": "assemble",
            **deps.budget_delta("synthesize", usage),
        }

    return synthesize
