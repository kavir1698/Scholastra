"""Citation verification, adversarial critique and the bounded revision loop.

Verification is sentence-level: each draft sentence carrying a citation is checked
against the excerpts retrieved for those citations, with provenance included, and
uncited factual sentences are surfaced for judgment. Mechanical unresolved-key
checking is retained regardless of LLM availability so a failed verifier call can
never be mistaken for success.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from scholastra.agents.prompting import render, system_preamble
from scholastra.citations import iter_citation_keys
from scholastra.claims import parse_sentences, uncited_factual_units
from scholastra.config import load_source_policy
from scholastra.graph.contract import as_contract
from scholastra.graph.deps import Deps
from scholastra.models.schemas import (
    ClaimSupportReport,
    DocumentContract,
    Issue,
    IssueList,
    NodeUsage,
    Outline,
    SectionDraft,
    SectionEvidence,
    SectionVerification,
    Source,
)
from scholastra.sources.policy import tier_allows_claim_support
from scholastra.utils import stable_id


def _as_source(value: Any) -> Source:
    return value if isinstance(value, Source) else Source.model_validate(value)


def _add(left: NodeUsage, right: NodeUsage) -> NodeUsage:
    return NodeUsage(
        calls=left.calls + right.calls,
        tokens_in=left.tokens_in + right.tokens_in,
        tokens_out=left.tokens_out + right.tokens_out,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def _claims_by_source(claims: list[Any]) -> dict[str, list[Any]]:
    index: dict[str, list[Any]] = {}
    for claim in claims:
        for key in claim.source_ids:
            index.setdefault(key, []).append(claim)
    return index


def _sentences_payload(
    section_id: str, markdown: str, claims: list[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], set[str]]:
    """Build verifier payloads; return (cited, uncited, keys_without_excerpt, referenced)."""
    units = parse_sentences(markdown, section_id=section_id)
    index = _claims_by_source(claims)
    cited: list[dict[str, Any]] = []
    referenced: set[str] = set()
    for unit in units:
        if not unit.has_citation:
            continue
        matched: list[Any] = []
        for key in unit.cite_keys:
            referenced.add(key)
            for claim in index.get(key, []):
                if claim not in matched:
                    matched.append(claim)
        cited.append(
            {
                "id": unit.id,
                "text": unit.text,
                "cited_keys": list(unit.cite_keys),
                "evidence": [
                    {
                        "claim_id": claim.id,
                        "source_ids": list(claim.source_ids),
                        "excerpt": claim.excerpt,
                        "locator": claim.locator,
                        "origin": claim.origin,
                        "tier": claim.tier,
                    }
                    for claim in matched
                ],
            }
        )
    uncited = [{"id": unit.id, "text": unit.text} for unit in uncited_factual_units(units)]
    keys_with_excerpt = set(index)
    without_excerpt = sorted(k for k in referenced if k not in keys_with_excerpt)
    return cited, uncited, without_excerpt, referenced


def _low_tier_units(cited: list[dict[str, Any]], policy: dict[str, Any] | None) -> list[str]:
    """Sentence ids whose only supporting excerpts fall below the claim-support tier."""
    flagged: list[str] = []
    for unit in cited:
        known = [
            ev.get("tier") for ev in (unit.get("evidence") or []) if ev.get("tier") is not None
        ]
        if known and not any(tier_allows_claim_support(tier, policy) for tier in known):
            flagged.append(unit["id"])
    return flagged


def _derive_lists(report: ClaimSupportReport) -> tuple[list[str], list[str]]:
    unsupported = list(report.unsupported_claims)
    partial = list(report.partial_claims)
    if report.results:
        if not unsupported:
            unsupported = [r.claim_id for r in report.results if r.judgment == "unsupported"]
        if not partial:
            partial = [r.claim_id for r in report.results if r.judgment == "partial"]
    return unsupported, partial


async def _verify_one_section(
    deps: Deps,
    sid: str,
    raw_draft: dict[str, Any],
    sources: dict[str, Any],
    section_evidence: Any,
    round_no: int,
    policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], NodeUsage]:
    draft = SectionDraft.model_validate(raw_draft)
    cite_keys = sorted(set(iter_citation_keys(draft.markdown)))
    unresolved = [k for k in cite_keys if k not in sources]

    evidence = (
        SectionEvidence.model_validate(section_evidence)
        if section_evidence
        else SectionEvidence(section_id=sid)
    )
    claims = evidence.claims
    cited, uncited, without_excerpt, referenced = _sentences_payload(
        sid, draft.markdown, claims
    )
    # Only the sources this section actually cites are needed in the prompt.
    source_models = {k: _as_source(sources[k]) for k in referenced if k in sources}
    low_tier = _low_tier_units(cited, policy)

    unsupported: list[str] = []
    partial: list[str] = []
    uncited_ids: list[str] = []
    candidate_ids = [u["id"] for u in uncited]
    results: list = []
    verifier_error: str | None = None
    usage = NodeUsage()
    if cited or uncited:
        try:
            report, usage = await deps.llm.structured(
                "verify",
                ClaimSupportReport,
                [
                    {
                        "role": "user",
                        "content": render(
                            "verifier",
                            section_id=sid,
                            sentences=cited,
                            uncited=uncited,
                            sources=source_models,
                        ),
                    }
                ],
                extra_system=system_preamble("Citation Verifier", untrusted=True),
                max_repairs=2,
            )
            unsupported, partial = _derive_lists(report)
            uncited_ids = list(report.uncited_factual)
            results = list(report.results)
        except Exception as exc:  # noqa: BLE001 - mechanical checks still apply
            deps.log("verify_llm_failed", section=sid, error=str(exc))
            verifier_error = str(exc)

    passed = (
        not unresolved
        and not unsupported
        and not draft.stripped_cites
        and not low_tier
    )
    verification = SectionVerification(
        section_id=sid,
        resolved_cites=[k for k in cite_keys if k in sources],
        unresolved_cites=unresolved,
        unsupported_claims=unsupported,
        partial_claims=partial,
        uncited_factual=uncited_ids,
        uncited_candidates=candidate_ids,
        support_results=results,
        sources_without_excerpt=without_excerpt,
        low_tier_support=low_tier,
        verified=verifier_error is None,
        verifier_error=verifier_error,
        passed=passed,
    ).model_dump()

    issues: list[dict[str, Any]] = []
    for key in unresolved:
        issues.append(
            Issue(
                id=stable_id("cite", sid, key),
                severity="blocking",
                category="citation",
                target_section=sid,
                description=(
                    f"Inline citation [@{key}] does not resolve to any stored source."
                ),
                round=round_no,
            ).model_dump()
        )
    for key in draft.stripped_cites:
        issues.append(
            Issue(
                id=stable_id("stripped", sid, key),
                severity="blocking",
                category="citation",
                target_section=sid,
                description=(
                    f"Draft cited [@{key}], which has no evidence in this section; the "
                    f"citation was removed. Re-ground the sentence or drop it."
                ),
                round=round_no,
            ).model_dump()
        )
    for claim_id in low_tier:
        issues.append(
            Issue(
                id=stable_id("low-tier", sid, claim_id),
                severity="minor",
                category="source_policy",
                target_section=sid,
                description=(
                    f"Sentence '{claim_id}' is supported only by sources below the "
                    f"claim-support tier (e.g. orientation-only or abstract-only)."
                ),
                round=round_no,
            ).model_dump()
        )
    for claim_id in unsupported:
        issues.append(
            Issue(
                id=stable_id("claim", sid, claim_id),
                severity="major",
                category="factual",
                target_section=sid,
                description=(
                    f"Sentence '{claim_id}' is not entailed by its cited evidence "
                    f"(unsupported claim)."
                ),
                round=round_no,
            ).model_dump()
        )
    for claim_id in partial:
        issues.append(
            Issue(
                id=stable_id("partial", sid, claim_id),
                severity="minor",
                category="factual",
                target_section=sid,
                description=(
                    f"Sentence '{claim_id}' is only partially supported by its cited evidence."
                ),
                round=round_no,
            ).model_dump()
        )
    for claim_id in uncited_ids:
        issues.append(
            Issue(
                id=stable_id("uncited", sid, claim_id),
                severity="major",
                category="factual",
                target_section=sid,
                description=(
                    f"Factual sentence '{claim_id}' has no citation; add a supporting source "
                    f"or mark it explicitly as interpretation."
                ),
                round=round_no,
            ).model_dump()
        )
    return verification, issues, usage


async def verify_sections(
    deps: Deps,
    section_ids: list[str],
    drafts: dict[str, Any],
    sources: dict[str, Any],
    evidence: dict[str, Any],
    round_no: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], NodeUsage]:
    """Verify ``section_ids`` against their evidence; reusable by editor/synthesis."""
    verification: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    usage = NodeUsage()
    policy = load_source_policy(deps.settings)
    for sid in section_ids:
        raw_draft = drafts.get(sid)
        if not raw_draft:
            continue
        result, section_issues, section_usage = await _verify_one_section(
            deps, sid, raw_draft, sources, evidence.get(sid), round_no, policy
        )
        verification[sid] = result
        issues.extend(section_issues)
        usage = _add(usage, section_usage)
    return verification, issues, usage


async def verify_text(
    deps: Deps,
    section_id: str,
    markdown: str,
    sources: dict[str, Any],
    section_evidence: Any,
    round_no: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], NodeUsage]:
    """Verify a standalone markdown blob (e.g. the synthesis) as a pseudo-section."""
    draft = SectionDraft(
        section_id=section_id, title=section_id, markdown=markdown
    ).model_dump()
    policy = load_source_policy(deps.settings)
    return await _verify_one_section(
        deps, section_id, draft, sources, section_evidence, round_no, policy
    )


def make_verify_citations_node(deps: Deps):
    async def verify_citations(state: dict[str, Any]) -> dict[str, Any]:
        sources: dict[str, Any] = state.get("sources") or {}
        drafts: dict[str, Any] = state.get("drafts") or {}
        evidence: dict[str, Any] = state.get("evidence") or {}
        round_no = state.get("revision_round", 0)
        verification, issues, usage = await verify_sections(
            deps, sorted(drafts), drafts, sources, evidence, round_no
        )
        deps.log("verify_citations", sections=len(verification), issues=len(issues))
        return {
            "verification": verification,
            "issues": issues,
            "status": "critique",
            **deps.budget_delta("verify", usage),
        }

    return verify_citations


def make_critique_node(deps: Deps):
    async def critique(state: dict[str, Any]) -> dict[str, Any]:
        drafts = state.get("drafts") or {}
        evidence = state.get("evidence") or {}
        contract: DocumentContract = as_contract(state.get("contract"))
        round_no = state.get("revision_round", 0)
        prompt = render("critic", drafts=drafts, evidence=evidence, contract=contract)
        try:
            result, usage = await deps.llm.structured(
                "critique",
                IssueList,
                [{"role": "user", "content": prompt}],
                extra_system=system_preamble("Adversarial Critic", untrusted=True),
                max_repairs=1,
            )
        except Exception as exc:  # noqa: BLE001 - critique is best-effort
            deps.log("critique_failed", error=str(exc))
            result, usage = IssueList(), NodeUsage()
        issues = []
        for index, issue in enumerate(result.issues):
            issue = issue.model_copy(
                update={
                    "round": round_no,
                    "id": issue.id or stable_id("critic", str(index), issue.description),
                }
            )
            issues.append(issue.model_dump())
        deps.log("critique", issues=len(issues))
        return {
            "issues": issues,
            "status": "revise",
            **deps.budget_delta("critique", usage),
        }

    return critique


def make_route_after_critique(deps: Deps):
    def route_after_critique(state: dict[str, Any]) -> list[Send] | str:
        round_no = state.get("revision_round", 0)
        max_rounds = state.get("max_revision_rounds", 1)
        blocking = [
            i
            for i in state.get("issues", [])
            if i.get("round") == round_no and i.get("severity") in {"major", "blocking"}
        ]
        drafts = state.get("drafts") or {}
        valid_sections = set(drafts)
        affected = sorted(
            {
                i["target_section"]
                for i in blocking
                if i.get("target_section") in valid_sections
            }
        )
        if not blocking or not affected or round_no >= max_rounds:
            deps.log(
                "route_after_critique",
                decision="edit",
                blocking=len(blocking),
                round=round_no,
            )
            return "editor"

        evidence = state.get("evidence") or {}
        outline = Outline.model_validate(state.get("outline") or {"title": "", "sections": []})
        sections_by_id = {s.id: s for s in outline.sections}
        contract = state.get("contract") or {}
        sends: list[Send] = []
        for sid in affected:
            notes = "\n".join(
                f"- ({i['severity']}/{i['category']}) {i['description']}"
                for i in blocking
                if i.get("target_section") == sid
            )
            section = sections_by_id.get(sid)
            sends.append(
                Send(
                    "revise_section",
                    {
                        "project_id": state["project_id"],
                        "section_id": sid,
                        "section_title": (drafts.get(sid) or {}).get("title", sid),
                        "section_purpose": section.purpose if section else "",
                        "target_words": section.target_words if section else 400,
                        "subsections": (
                            [sub.model_dump() for sub in section.subsections] if section else []
                        ),
                        "evidence": evidence.get(sid, {}),
                        "dep_summaries": [],
                        "contract": contract,
                        "revision_round": round_no + 1,
                        "revision_notes": notes,
                    },
                )
            )
        deps.log("route_after_critique", decision="revise", sections=affected, round=round_no)
        return sends

    return route_after_critique


def make_revise_join_node(deps: Deps):
    async def revise_join(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "revision_round": state.get("revision_round", 0) + 1,
            "status": "verify",
        }

    return revise_join


def collect_sources(state: dict[str, Any]) -> dict[str, Source]:
    return {sid: Source.model_validate(s) for sid, s in (state.get("sources") or {}).items()}
