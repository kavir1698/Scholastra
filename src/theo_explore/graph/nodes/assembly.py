"""Assembly (merge, TOC, bibliography), Pandoc export and finalisation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from theo_explore.citations import filter_citations, render_inline_citations
from theo_explore.graph.deps import Deps
from theo_explore.io.export import export_document
from theo_explore.models.schemas import Outline, ScopeProposal, Source
from theo_explore.observability import build_run_report
from theo_explore.sources.bibtex import collect_bibtex, render_reference_list


def make_assemble_node(deps: Deps):
    async def assemble(state: dict[str, Any]) -> dict[str, Any]:
        outline = Outline.model_validate(state["outline"])
        scope = ScopeProposal.model_validate(state.get("scope") or {})
        drafts = state.get("drafts") or {}
        sources = {sid: Source.model_validate(s) for sid, s in (state.get("sources") or {}).items()}

        parts: list[str] = [f"# {outline.title}\n"]
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
        workspace = deps.workspace(state["project_id"])
        workspace.write_text("artifacts/final.md", final_markdown)
        workspace.write_text("artifacts/refs.bib", collect_bibtex(sources.values()))
        deps.log("assemble", words=len(final_markdown.split()), sources=len(sources))
        return {"final_markdown": final_markdown, "status": "export"}

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
        manifest = workspace.update_manifest(
            status="completed",
            topic=(state.get("brief") or {}).get("topic"),
            doc_type=(state.get("brief") or {}).get("doc_type"),
            preset=(state.get("brief") or {}).get("preset"),
            sections=list((state.get("outline") or {}).get("sections", [])),
            exports=state.get("exports", {}),
            issues=len(state.get("issues") or []),
            revision_rounds=state.get("revision_round", 0),
            warnings=list(state.get("errors") or []),
        )
        report = build_run_report(manifest, state.get("budget"))
        workspace.write_json("run_report.json", report)
        workspace.update_manifest(run_report=report, finished_at=_utcnow())
        deps.log("finalize", status="completed", cost_usd=report["cost_usd"])
        return {"status": "completed"}

    return finalize


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
