"""Assembly (merge, TOC, bibliography), Pandoc export and finalisation.

Assembly enforces the quality gate: when a strict preset still has blocking
citation or editorial issues, the artifact is produced but clearly marked
incomplete rather than silently repaired, and the run is never reported clean.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from theo_explore.citations import filter_citations, render_inline_citations
from theo_explore.config import load_preset
from theo_explore.graph.contract import as_contract
from theo_explore.graph.deps import Deps
from theo_explore.io.export import export_document
from theo_explore.models.schemas import (
    EditorialFinding,
    Outline,
    QualityGate,
    QualityReport,
    ScopeProposal,
    SectionVerification,
    Source,
)
from theo_explore.observability import build_run_report
from theo_explore.sources.bibtex import collect_bibtex, render_reference_list
from theo_explore.sources.policy import source_quality_summary


def _build_gate(
    state: dict[str, Any], verification: dict[str, SectionVerification], strict: bool
) -> QualityGate:
    editorial = [EditorialFinding.model_validate(f) for f in (state.get("editorial") or [])]
    drafted = set(state.get("drafts") or {})
    unsupported = sum(len(v.unsupported_claims) for v in verification.values())
    partial = sum(len(v.partial_claims) for v in verification.values())
    uncited = sum(len(v.uncited_factual) for v in verification.values())
    unresolved = sum(len(v.unresolved_cites) for v in verification.values())
    low_tier = sum(len(v.low_tier_support) for v in verification.values())
    missing_verification = sorted(drafted - set(verification))
    verifier_errors = sum(1 for v in verification.values() if v.verifier_error)
    incomplete = any(not v.verified for v in verification.values()) or bool(missing_verification)

    synth = state.get("synthesis_verification") or {}
    synth_blocking = bool(
        synth.get("unsupported_claims") or synth.get("unresolved_cites")
    ) if synth else False
    synth_incomplete = bool(synth) and not synth.get("verified", True)
    editorial_blocking = [f for f in editorial if f.severity in {"major", "blocking"}]

    # Blocking issues raised in the final pass (editor failures, editor-removed
    # unknown citations, unresolved verification issues) must gate assembly even
    # when they are not mirrored in the editorial findings or verification report.
    round_no = state.get("revision_round", 0)
    unresolved_issues = [
        i
        for i in (state.get("issues") or [])
        if i.get("severity") == "blocking" and i.get("round", round_no) == round_no
    ]

    reasons: list[str] = []
    if unresolved:
        reasons.append(f"{unresolved} unresolved citation(s)")
    if unsupported:
        reasons.append(f"{unsupported} unsupported claim(s)")
    if uncited:
        reasons.append(f"{uncited} uncited factual sentence(s)")
    if low_tier:
        reasons.append(f"{low_tier} claim(s) supported only below the claim-support tier")
    if editorial_blocking:
        reasons.append(f"{len(editorial_blocking)} blocking editorial finding(s)")
    if unresolved_issues:
        reasons.append(f"{len(unresolved_issues)} blocking issue(s) remain unresolved")
    if incomplete:
        detail: list[str] = []
        if missing_verification:
            detail.append(f"{len(missing_verification)} section(s) never verified")
        if verifier_errors:
            detail.append(f"{verifier_errors} section(s) where the verifier call failed")
        suffix = f" ({'; '.join(detail)})" if detail else ""
        reasons.append(f"citation verification was incomplete (verifier failed or skipped){suffix}")
    if synth_blocking:
        reasons.append("synthesis contains unsupported or unresolved citations")
    if synth_incomplete:
        reasons.append("synthesis verification failed or was skipped")

    blocked = strict and bool(reasons)
    return QualityGate(
        strict=strict,
        blocked=blocked,
        clean=not reasons,
        reasons=reasons,
        verification_incomplete=incomplete or synth_incomplete,
        verifier_error_count=verifier_errors,
        unsupported_count=unsupported,
        partial_count=partial,
        uncited_count=uncited,
        unresolved_count=unresolved,
        editorial_findings=len(editorial),
        sections_edited=sorted(
            sid
            for sid, raw in (state.get("drafts") or {}).items()
            if (raw or {}).get("revision", 0) > 0
        ),
    )


def make_assemble_node(deps: Deps):
    async def assemble(state: dict[str, Any]) -> dict[str, Any]:
        outline = Outline.model_validate(state["outline"])
        scope = ScopeProposal.model_validate(state.get("scope") or {})
        drafts = state.get("drafts") or {}
        sources = {sid: Source.model_validate(s) for sid, s in (state.get("sources") or {}).items()}
        verification = {
            sid: SectionVerification.model_validate(v)
            for sid, v in (state.get("verification") or {}).items()
        }
        preset = load_preset(scope.length_preset, deps.settings)
        gate_mode = str((preset.get("quality") or {}).get("citation_gate", "audit")).lower()
        strict = gate_mode == "strict"
        gate = _build_gate(state, verification, strict)

        parts: list[str] = [f"# {outline.title}\n"]
        if gate.blocked:
            parts.append(
                "> **INCOMPLETE DRAFT — UNRESOLVED QUALITY ISSUES**\n>\n"
                + "\n".join(f"> - {reason}" for reason in gate.reasons)
                + "\n"
            )
        if scope.refined_question:
            parts.append(f"*{scope.refined_question}*\n")

        toc = ["## Contents\n"]
        for section in outline.sections:
            if section.target_words > 0 and section.id in drafts:
                toc.append(f"- {section.title}")
        parts.append("\n".join(toc) + "\n")

        for section in outline.sections:
            if section.target_words <= 0:
                continue
            draft = drafts.get(section.id)
            if not draft:
                continue
            body, _ = filter_citations(draft["markdown"], set(sources))
            body = render_inline_citations(body, sources)
            parts.append(body.strip() + "\n")

        if state.get("synthesis"):
            parts.append("## Synthesis and Future Directions\n")
            parts.append(render_inline_citations(state["synthesis"], sources).strip() + "\n")

        parts.append("## References\n")
        parts.append(render_reference_list(sources.values()) + "\n")

        final_markdown = "\n".join(parts)
        words = len(final_markdown.split())
        workspace = deps.workspace(state["project_id"])
        workspace.write_text("artifacts/final.md", final_markdown)
        workspace.write_text("artifacts/refs.bib", collect_bibtex(sources.values()))

        source_summary = source_quality_summary(sources)
        gate.source_tiers = {
            str(tier): source_summary.get(f"tier_{tier}", 0) for tier in (1, 2, 3, 4)
        }
        gate.full_text_sources = source_summary.get("full_text", 0)
        gate.abstract_only_sources = source_summary.get("abstract_only", 0)
        gate.bibliographic_only_sources = source_summary.get("bibliographic_only", 0)

        synthesis_verification = state.get("synthesis_verification") or None
        report = QualityReport(
            contract=as_contract(state.get("contract")),
            gate=gate,
            verification=verification,
            editorial=[EditorialFinding.model_validate(f) for f in (state.get("editorial") or [])],
            synthesis_verification=(
                SectionVerification.model_validate(synthesis_verification)
                if synthesis_verification
                else None
            ),
            source_summary={k: int(v) for k, v in source_summary.items()},
        )
        workspace.save_quality_report(report)

        warnings: list[str] = []
        target = preset.get("target_words", {})
        minimum = int(target.get("min", 0) or 0)
        if minimum and words < minimum * 0.8:
            warning = (
                f"document is {words} words, below the {minimum}-word target for preset "
                f"'{scope.length_preset}'; consider a narrower scope or more subsections."
            )
            warnings.append(warning)
            deps.log("length_below_target", words=words, target=minimum, preset=scope.length_preset)
        if gate.blocked:
            warnings.extend(f"quality gate: {reason}" for reason in gate.reasons)
        deps.log(
            "assemble",
            words=words,
            sources=len(sources),
            blocked=gate.blocked,
            clean=gate.clean,
        )
        return {
            "final_markdown": final_markdown,
            "quality": report.model_dump(),
            "status": "export",
            "errors": warnings,
        }

    return assemble


def make_export_node(deps: Deps):
    async def export(state: dict[str, Any]) -> dict[str, Any]:
        workspace = deps.workspace(state["project_id"])
        formats = list(state.get("export_formats") or [])
        exports: dict[str, str] = {"md": str(workspace.path("artifacts/final.md"))}
        warnings: list[str] = []
        if formats:
            result = export_document(
                workspace.path("artifacts/final.md"),
                workspace.artifacts_dir,
                formats=formats,
                stem="final",
            )
            exports.update(result.outputs)
            warnings = result.warnings
        for warning in warnings:
            deps.log("export_warning", warning=warning)
        deps.log("export", formats=list(exports.keys()))
        return {"exports": exports, "errors": warnings, "status": "finalize"}

    return export


def make_finalize_node(deps: Deps):
    async def finalize(state: dict[str, Any]) -> dict[str, Any]:
        workspace = deps.workspace(state["project_id"])
        final_md = str(state.get("final_markdown") or "")
        words = len(final_md.split())
        gate = (state.get("quality") or {}).get("gate") or {}
        blocked = bool(gate.get("blocked"))
        status = "blocked" if blocked else "completed"
        preset_name = (state.get("brief") or {}).get("preset") or (
            state.get("scope") or {}
        ).get("length_preset")
        target: dict[str, Any] = {}
        if preset_name:
            try:
                target = dict(load_preset(preset_name, deps.settings).get("target_words", {}))
            except FileNotFoundError:
                target = {}
        manifest = workspace.update_manifest(
            status=status,
            topic=(state.get("brief") or {}).get("topic"),
            doc_type=(state.get("brief") or {}).get("doc_type"),
            preset=(state.get("brief") or {}).get("preset"),
            sections=list((state.get("outline") or {}).get("sections", [])),
            exports=state.get("exports", {}),
            issues=len(state.get("issues") or []),
            revision_rounds=state.get("revision_round", 0),
            editor_round=state.get("editor_round", 0),
            quality=gate,
            warnings=list(state.get("errors") or []),
            word_count=words,
            target_words=target,
        )
        report = build_run_report(
            manifest, state.get("budget"), quality=state.get("quality") or {}
        )
        workspace.write_json("run_report.json", report)
        workspace.update_manifest(run_report=report, finished_at=_utcnow())
        deps.log("finalize", status=status, cost_usd=report["cost_usd"], blocked=blocked)
        return {"status": status}

    return finalize


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
