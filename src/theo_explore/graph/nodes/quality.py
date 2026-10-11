"""Citation verification, adversarial critique and the bounded revision loop."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.citations import iter_citation_keys
from theo_explore.graph.deps import Deps
from theo_explore.models.schemas import (
    ClaimSupport,
    Issue,
    IssueList,
    NodeUsage,
    SectionDraft,
    SectionEvidence,
    SectionVerification,
    Source,
)
from theo_explore.utils import stable_id


def make_verify_citations_node(deps: Deps):
    async def verify_citations(state: dict[str, Any]) -> dict[str, Any]:
        sources: dict[str, Any] = state.get("sources") or {}
        drafts: dict[str, Any] = state.get("drafts") or {}
        evidence: dict[str, Any] = state.get("evidence") or {}
        round_no = state.get("revision_round", 0)
        verification: dict[str, dict[str, Any]] = {}
        issues: list[dict[str, Any]] = []
        usage = NodeUsage()

        for sid, raw_draft in drafts.items():
            draft = SectionDraft.model_validate(raw_draft)
            cite_keys = sorted(set(iter_citation_keys(draft.markdown)))
            unresolved = [k for k in cite_keys if k not in sources]

            unsupported: list[str] = []
            section_evidence = evidence.get(sid)
            if section_evidence and SectionEvidence.model_validate(section_evidence).claims:
                ev = SectionEvidence.model_validate(section_evidence)
                try:
                    support, node_usage = await deps.llm.structured(
                        "verify",
                        ClaimSupport,
                        [
                            {
                                "role": "user",
                                "content": render(
                                    "verifier",
                                    section_id=sid,
                                    claims=[c.model_dump() for c in ev.claims],
                                ),
                            }
                        ],
                        extra_system=system_preamble("Citation Verifier", untrusted=True),
                        max_repairs=0,
                    )
                    unsupported = list(support.unsupported_claims)
                    usage = _add(usage, node_usage)
                except Exception as exc:  # noqa: BLE001 - mechanical checks still apply
                    deps.log("verify_llm_failed", section=sid, error=str(exc))

            passed = not unresolved and not unsupported
            verification[sid] = SectionVerification(
                section_id=sid,
                resolved_cites=[k for k in cite_keys if k in sources],
                unresolved_cites=unresolved,
                unsupported_claims=unsupported,
                passed=passed,
            ).model_dump()

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
            for claim_id in unsupported:
                issues.append(
                    Issue(
                        id=stable_id("claim", sid, claim_id),
                        severity="major",
                        category="factual",
                        target_section=sid,
                        description=f"Claim '{claim_id}' is not supported by its cited excerpt.",
                        round=round_no,
                    ).model_dump()
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
        round_no = state.get("revision_round", 0)
        prompt = render("critic", drafts=drafts, evidence=evidence)
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
                decision="continue",
                blocking=len(blocking),
                round=round_no,
            )
            return "synthesize"

        evidence = state.get("evidence") or {}
        sends: list[Send] = []
        for sid in affected:
            notes = "\n".join(
                f"- ({i['severity']}/{i['category']}) {i['description']}"
                for i in blocking
                if i.get("target_section") == sid
            )
            sends.append(
                Send(
                    "revise_section",
                    {
                        "project_id": state["project_id"],
                        "section_id": sid,
                        "section_title": (drafts.get(sid) or {}).get("title", sid),
                        "section_purpose": "",
                        "target_words": (drafts.get(sid) or {}).get("word_count", 400),
                        "evidence": evidence.get(sid, {}),
                        "dep_summaries": [],
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


def _add(left: NodeUsage, right: NodeUsage) -> NodeUsage:
    return NodeUsage(
        calls=left.calls + right.calls,
        tokens_in=left.tokens_in + right.tokens_in,
        tokens_out=left.tokens_out + right.tokens_out,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def collect_sources(state: dict[str, Any]) -> dict[str, Source]:
    return {sid: Source.model_validate(s) for sid, s in (state.get("sources") or {}).items()}
