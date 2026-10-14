"""Observability: JSONL run log, token/cost accounting and budget guards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholastra.models.schemas import BudgetState, NodeUsage


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_budget(max_cost_usd: float | None = None) -> dict[str, Any]:
    return BudgetState(max_cost_usd=max_cost_usd).model_dump()


def as_budget(value: Any) -> BudgetState:
    if isinstance(value, BudgetState):
        return value
    return BudgetState.model_validate(value or {})


def add_usage(budget: dict | None, node: str, usage: NodeUsage) -> dict[str, Any]:
    """Return a new budget dict with ``usage`` folded in under ``node``."""
    current = as_budget(budget)
    per_node = {k: v.model_dump() for k, v in current.per_node.items()}
    bucket = per_node.setdefault(node, NodeUsage().model_dump())
    bucket["calls"] += usage.calls
    bucket["tokens_in"] += usage.tokens_in
    bucket["tokens_out"] += usage.tokens_out
    bucket["cost_usd"] += usage.cost_usd
    updated = current.model_copy(
        update={
            "tokens_in": current.tokens_in + usage.tokens_in,
            "tokens_out": current.tokens_out + usage.tokens_out,
            "cost_usd": current.cost_usd + usage.cost_usd,
            "tool_calls": current.tool_calls,
            "per_node": {k: NodeUsage.model_validate(v) for k, v in per_node.items()},
        }
    )
    if updated.max_cost_usd is not None and updated.cost_usd > updated.max_cost_usd:
        updated = updated.model_copy(update={"exceeded": True})
    return updated.model_dump()


def budget_exceeded(budget: dict | None) -> bool:
    model = as_budget(budget)
    if model.exceeded:
        return True
    return bool(model.max_cost_usd is not None and model.cost_usd > model.max_cost_usd)


@dataclass
class RunLogger:
    """Append-only JSONL event log for a project run."""

    path: Path
    _fh: Any = field(default=None, repr=False)

    def _open(self) -> Any:
        if self._fh is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 - long-lived handle
        return self._fh

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": _utcnow(), "event": event, **fields}
        handle = self._open()
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def build_run_report(
    manifest: dict[str, Any], budget: dict | None, quality: dict | None = None
) -> dict[str, Any]:
    model = as_budget(budget)
    quality = quality or {}
    gate = quality.get("gate") or {}
    verification = quality.get("verification") or {}
    support_total = 0
    unsupported_total = 0
    partial_total = 0
    section_rates: dict[str, float] = {}
    for sid, report in verification.items():
        unsupported = len(report.get("unsupported_claims") or [])
        partial = len(report.get("partial_claims") or [])
        resolved = len(report.get("resolved_cites") or [])
        total = resolved + len(report.get("unresolved_cites") or [])
        support_total += total
        unsupported_total += unsupported
        partial_total += partial
        if total:
            section_rates[sid] = round((total - unsupported - partial) / total, 3)
    return {
        "project_id": manifest.get("project_id"),
        "topic": manifest.get("topic"),
        "doc_type": manifest.get("doc_type"),
        "preset": manifest.get("preset"),
        "status": manifest.get("status"),
        "tokens_in": model.tokens_in,
        "tokens_out": model.tokens_out,
        "cost_usd": round(model.cost_usd, 6),
        "tool_calls": model.tool_calls,
        "per_node": {k: v.model_dump() for k, v in model.per_node.items()},
        "budget_exceeded": model.exceeded,
        "quality": {
            "clean": gate.get("clean", True),
            "blocked": gate.get("blocked", False),
            "strict": gate.get("strict", False),
            "reasons": gate.get("reasons", []),
            "verification_incomplete": gate.get("verification_incomplete", False),
            "verifier_error_sections": gate.get("verifier_error_count", 0),
            "unsupported_claims": gate.get("unsupported_count", unsupported_total),
            "partial_claims": gate.get("partial_count", partial_total),
            "uncited_factual": gate.get("uncited_count", 0),
            "unresolved_citations": gate.get("unresolved_count", 0),
            "editorial_findings": gate.get("editorial_findings", 0),
            "sections_edited": gate.get("sections_edited", []),
            "source_tiers": gate.get("source_tiers", {}),
            "full_text_sources": gate.get("full_text_sources", 0),
            "abstract_only_sources": gate.get("abstract_only_sources", 0),
            "bibliographic_only_sources": gate.get("bibliographic_only_sources", 0),
            "citation_support_rate": (
                round((support_total - unsupported_total) / support_total, 3)
                if support_total
                else None
            ),
            "per_section_support_rate": section_rates,
        },
        "generated_at": _utcnow(),
    }
