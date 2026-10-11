"""Whole-document editorial pass (post-draft, post-verification, pre-synthesis).

The editor is the only node that sees every section draft at once, so it owns
cross-section coherence: duplicate definitions, terminology drift, transitions,
argument-arc breaks, contradictions, ordering and uneven depth. Edits are applied
through a bounded single pass, then mechanical citation checks and sentence-level
support checks are re-run on the changed sections. The editor may not introduce a
citation key that is not already registered.
"""

from __future__ import annotations

from typing import Any

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.citations import filter_citations, iter_citation_keys
from theo_explore.graph.contract import (
    as_contract,
    bounded_drafts,
    find_terminology_violations,
    normalize_terminology,
)
from theo_explore.graph.deps import Deps
from theo_explore.graph.nodes.quality import _add, verify_sections
from theo_explore.models.schemas import (
    EditorialFinding,
    EditorialResult,
    Issue,
    NodeUsage,
    Outline,
    SectionDraft,
)
from theo_explore.utils import stable_id, word_count


def make_editor_node(deps: Deps):
    async def edit(state: dict[str, Any]) -> dict[str, Any]:
        drafts: dict[str, Any] = state.get("drafts") or {}
        sources: dict[str, Any] = state.get("sources") or {}
        evidence: dict[str, Any] = state.get("evidence") or {}
        verification: dict[str, Any] = state.get("verification") or {}
        contract = as_contract(state.get("contract"))
        round_no = state.get("revision_round", 0)
        editor_round = state.get("editor_round", 0)

        if not drafts:
            deps.log("editor_skipped", reason="no drafts")
            return {"status": "synthesize", "editor_round": editor_round + 1}

        outline = Outline.model_validate(state.get("outline") or {"title": "", "sections": []})
        issues: list[dict[str, Any]] = []

        prompt_drafts, truncated = bounded_drafts(drafts)
        if truncated:
            deps.log("editor_prompt_truncated", sections=truncated)
        try:
            result, usage = await deps.llm.structured(
                "editor",
                EditorialResult,
                [
                    {
                        "role": "user",
                        "content": render(
                            "editor",
                            outline=outline,
                            contract=contract,
                            drafts=prompt_drafts,
                            evidence=evidence,
                            verification=verification,
                        ),
                    }
                ],
                extra_system=system_preamble("Document Editor", untrusted=True),
                max_repairs=1,
            )
        except Exception as exc:  # noqa: BLE001 - record a blocking editorial warning
            deps.log("editor_failed", error=str(exc))
            result, usage = EditorialResult(), NodeUsage()
            failure_finding = EditorialFinding(
                id=stable_id("editor", "failed"),
                kind="other",
                severity="blocking",
                sections=[],
                description=(
                    "Whole-document editor failed; cross-section coherence was not verified."
                ),
            )
            result = result.model_copy(update={"findings": [*result.findings, failure_finding]})
            issues.append(
                Issue(
                    id=failure_finding.id,
                    severity="blocking",
                    category="editorial",
                    target_section=None,
                    description=failure_finding.description,
                    round=round_no,
                ).model_dump()
            )

        allowed = set(sources)
        updates: dict[str, dict[str, Any]] = {}
        changed_sections: list[str] = []
        for editorial_edit in result.edits:
            sid = editorial_edit.section_id
            if not editorial_edit.changed or not editorial_edit.revised_markdown:
                continue
            if sid not in drafts:
                continue
            markdown, removed = filter_citations(editorial_edit.revised_markdown, allowed)
            if removed:
                issues.append(
                    Issue(
                        id=stable_id("editor-cite", sid, ",".join(removed)),
                        severity="blocking",
                        category="citation",
                        target_section=sid,
                        description=(
                            f"Editor introduced citation keys absent from the source registry "
                            f"({', '.join(removed)}); they were removed."
                        ),
                        round=round_no,
                    ).model_dump()
                )
            markdown, term_changes = normalize_terminology(markdown, contract)
            if term_changes:
                deps.log("editor_terminology_normalized", section=sid, changes=term_changes[:5])
            previous = SectionDraft.model_validate(drafts[sid])
            draft = previous.model_copy(
                update={
                    "markdown": markdown,
                    "cite_keys": sorted(set(iter_citation_keys(markdown))),
                    "word_count": word_count(markdown),
                    "revision": previous.revision + 1,
                }
            )
            deps.workspace(state["project_id"]).save_draft(draft)
            updates[sid] = draft.model_dump()
            changed_sections.append(sid)

        findings = list(result.findings)
        # Deterministic terminology-drift safety net: surface variants that the
        # mechanical normaliser did not rewrite (e.g. casing or multi-word forms).
        for sid, raw in drafts.items():
            markdown = str((raw or {}).get("markdown") or "")
            violations = find_terminology_violations(markdown, contract)
            if violations:
                findings.append(
                    EditorialFinding(
                        id=stable_id("term-drift", sid, ",".join(violations)),
                        kind="terminology_drift",
                        severity="minor",
                        sections=[sid],
                        description=(
                            f"Terminology drift in '{sid}': {', '.join(violations[:5])}"
                        ),
                    )
                )

        for finding in findings:
            issues.append(
                Issue(
                    id=finding.id
                    or stable_id("editor", finding.kind, finding.description),
                    severity=finding.severity,
                    category=f"editorial:{finding.kind}",
                    target_section=(finding.sections[0] if finding.sections else None),
                    description=finding.description,
                    evidence=finding.evidence,
                    round=round_no,
                ).model_dump()
            )

        verification_updates: dict[str, dict[str, Any]] = {}
        if changed_sections:
            merged = {**drafts, **updates}
            verification_updates, verify_issues, verify_usage = await verify_sections(
                deps, changed_sections, merged, sources, evidence, round_no
            )
            issues.extend(verify_issues)
            usage = _add(usage, verify_usage)

        deps.log(
            "editor",
            findings=len(findings),
            changed=changed_sections,
            round=editor_round,
        )
        return {
            "drafts": updates,
            "verification": verification_updates,
            "issues": issues,
            "editorial": [f.model_dump() for f in findings],
            "editor_round": editor_round + 1,
            "status": "synthesize",
            **deps.budget_delta("editor", usage),
        }

    return edit
