"""Document-contract helpers: compact views, prompt sizing and terminology.

The contract is generated once, after outline approval, and then propagated as a
*bounded* compact view into every research extractor and leaf drafter. The full
document context is reserved for the editorial node.
"""

from __future__ import annotations

import re
from typing import Any

from scholastra.models.schemas import (
    ContractContribution,
    DocumentContract,
    GlossaryEntry,
    SectionHandoff,
)

_DEFAULT_MAX_CHARS = 1800


def as_contract(value: Any) -> DocumentContract:
    if isinstance(value, DocumentContract):
        return value
    if isinstance(value, dict):
        return DocumentContract.model_validate(value)
    return DocumentContract()


def _entry_sort_key(entry: GlossaryEntry, section_id: str | None, subsection_id: str | None) -> int:
    if subsection_id and subsection_id in (entry.aliases or []):
        return 0
    if section_id and entry.owner_section == section_id:
        return 1
    if entry.owner_section:
        return 2
    return 3


def _variant_list(entry: GlossaryEntry) -> list[str]:
    variants = [*(entry.aliases or []), *(entry.forbidden_variants or [])]
    seen: list[str] = []
    for variant in variants:
        v = (variant or "").strip()
        if v and v.lower() != entry.term.strip().lower() and v not in seen:
            seen.append(v)
    return seen


def relevant_glossary(
    contract: DocumentContract,
    *,
    section_id: str | None = None,
    subsection_id: str | None = None,
    max_terms: int = 12,
) -> list[GlossaryEntry]:
    entries = sorted(
        contract.glossary, key=lambda e: _entry_sort_key(e, section_id, subsection_id)
    )
    return entries[: max(0, max_terms)]


def compact_contract(
    contract: DocumentContract | dict[str, Any],
    *,
    section_id: str | None = None,
    subsection_id: str | None = None,
    policy_lines: list[str] | None = None,
    max_terms: int = 12,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """A deterministic, bounded prose view of the contract for a leaf prompt."""
    model = as_contract(contract)
    lines: list[str] = []
    if model.thesis:
        lines.append(f"THESIS: {model.thesis}")
    if model.argument_arc:
        lines.append("ARGUMENT ARC:")
        lines.extend(f"- {step}" for step in model.argument_arc[:10])
    handoff = _handoff_for(model.handoffs, section_id)
    if handoff:
        lines.append(f"SECTION HANDOFF ({handoff.section_id}): {handoff.handoff_goal}")
    contribution = _contribution_for(model.contributions, section_id, subsection_id)
    if contribution:
        lines.append(
            f"YOUR ROLE: {contribution.role} — {contribution.expected_contribution}".rstrip(" —")
        )
    glossary = relevant_glossary(
        model, section_id=section_id, subsection_id=subsection_id, max_terms=max_terms
    )
    if glossary:
        lines.append("CONTROLLED TERMINOLOGY (use canonical terms; do not redefine an owned term):")
        for entry in glossary:
            owner = f" [owned by {entry.owner_section}]" if entry.owner_section else ""
            variants = _variant_list(entry)
            variant_text = f" (avoid: {', '.join(variants)})" if variants else ""
            definition = f": {entry.definition}" if entry.definition else ""
            lines.append(f"- {entry.term}{owner}{definition}{variant_text}")
    if model.terminology_rules:
        lines.append("TERMINOLOGY RULES:")
        lines.extend(f"- {rule}" for rule in model.terminology_rules[:8])
    policy = policy_lines or model.source_policy
    if policy:
        lines.append("SOURCE POLICY:")
        lines.extend(f"- {line}" for line in policy[:8])
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def bounded_drafts(
    drafts: dict[str, Any],
    *,
    max_chars_per_section: int = 6000,
    max_total_chars: int = 48000,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically truncate drafts for a whole-document prompt.

    Returns ``(bounded_copy, truncated_section_ids)`` without mutating the input.
    """
    bounded: dict[str, Any] = {}
    truncated: list[str] = []
    total = 0
    for sid, raw in drafts.items():
        raw = raw or {}
        markdown = str(raw.get("markdown") or "")
        if len(markdown) > max_chars_per_section:
            markdown = markdown[:max_chars_per_section] + "\n\n…[truncated]"
            truncated.append(sid)
        remaining = max(max_total_chars - total, 0)
        if len(markdown) > remaining:
            markdown = markdown[:remaining] + "\n\n…[truncated]"
            if sid not in truncated:
                truncated.append(sid)
        total += len(markdown)
        bounded[sid] = {**raw, "markdown": markdown}
    return bounded, truncated


def _handoff_for(handoffs: list[SectionHandoff], section_id: str | None) -> SectionHandoff | None:
    if not section_id:
        return None
    for handoff in handoffs:
        if handoff.section_id == section_id:
            return handoff
    return None


def _contribution_for(
    contributions: list[ContractContribution], section_id: str | None, subsection_id: str | None
) -> ContractContribution | None:
    if subsection_id:
        for contribution in contributions:
            if contribution.subsection_id == subsection_id:
                return contribution
    if section_id:
        for contribution in contributions:
            if contribution.section_id == section_id and not contribution.subsection_id:
                return contribution
    return None


def normalize_terminology(
    text: str, contract: DocumentContract | dict[str, Any]
) -> tuple[str, list[str]]:
    """Canonicalise glossary aliases/variants; return (text, applied changes)."""
    model = as_contract(contract)
    changes: list[str] = []
    for entry in model.glossary:
        term = (entry.term or "").strip()
        if not term:
            continue
        for variant in _variant_list(entry):
            pattern = re.compile(r"(?<![\w-])" + re.escape(variant) + r"(?![\w-])", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(term, text)
                changes.append(f"{variant} -> {term}")
    return text, changes


def find_terminology_violations(
    text: str, contract: DocumentContract | dict[str, Any]
) -> list[str]:
    """Forbidden variants present in ``text`` (drift not yet normalised)."""
    model = as_contract(contract)
    violations: list[str] = []
    for entry in model.glossary:
        for variant in entry.forbidden_variants or []:
            if not variant:
                continue
            pattern = re.compile(
                r"(?<![\w-])" + re.escape(variant) + r"(?![\w-])", re.IGNORECASE
            )
            if pattern.search(text):
                violations.append(f"{variant} -> {entry.term}")
    return violations
