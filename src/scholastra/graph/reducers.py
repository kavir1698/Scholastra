"""State reducers for the LangGraph map-reduce pattern.

State values are kept JSON-serialisable (plain dicts/lists) so the SQLite
checkpointer can persist them without custom encoders. Reducers therefore
operate on dicts, and nodes convert to/from Pydantic at their boundaries.

``merge_by_key`` is the critical one: parallel ``Send`` branches each write a
different key of ``evidence``/``drafts``/``sources``; a plain assignment would
clobber, so every branch must merge by key instead.
"""

from __future__ import annotations

from typing import Any


def merge_by_key(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def add_lists(left: list | None, right: list | None) -> list:
    return [*(left or []), *(right or [])]


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def merge_budget(left: dict | None, right: dict | None) -> dict:
    left = left or {}
    right = right or {}
    per_node: dict[str, dict] = {}
    for source in (left.get("per_node") or {}, right.get("per_node") or {}):
        for node, usage in source.items():
            bucket = per_node.setdefault(
                node, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
            )
            bucket["calls"] += int(usage.get("calls", 0) or 0)
            bucket["tokens_in"] += int(usage.get("tokens_in", 0) or 0)
            bucket["tokens_out"] += int(usage.get("tokens_out", 0) or 0)
            bucket["cost_usd"] += _num(usage.get("cost_usd"))
    incoming_cap = right.get("max_cost_usd")
    if incoming_cap is None:
        incoming_cap = left.get("max_cost_usd")
    max_cost: float | None = float(incoming_cap) if incoming_cap is not None else None
    merged: dict[str, Any] = {
        "tokens_in": int(_num(left.get("tokens_in")) + _num(right.get("tokens_in"))),
        "tokens_out": int(_num(left.get("tokens_out")) + _num(right.get("tokens_out"))),
        "cost_usd": _num(left.get("cost_usd")) + _num(right.get("cost_usd")),
        "tool_calls": int(_num(left.get("tool_calls")) + _num(right.get("tool_calls"))),
        "per_node": per_node,
        "max_cost_usd": max_cost,
        "exceeded": bool(left.get("exceeded") or right.get("exceeded")),
    }
    if max_cost is not None and float(merged["cost_usd"]) > max_cost:
        merged["exceeded"] = True
    return merged
