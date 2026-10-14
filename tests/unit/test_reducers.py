"""Reducer merge semantics (the critical parallel-fan-out behaviour)."""

from __future__ import annotations

import pytest

from scholastra.graph.reducers import add_lists, merge_budget, merge_by_key


def test_merge_by_key_does_not_clobber():
    left = {"a": {"v": 1}}
    right = {"b": {"v": 2}}
    assert merge_by_key(left, right) == {"a": {"v": 1}, "b": {"v": 2}}


def test_merge_by_key_handles_none_and_overwrite():
    assert merge_by_key(None, {"a": 1}) == {"a": 1}
    assert merge_by_key({"a": 1}, None) == {"a": 1}
    assert merge_by_key({"a": 1}, {"a": 2}) == {"a": 2}


def test_add_lists():
    assert add_lists(None, [1]) == [1]
    assert add_lists([1], [2, 3]) == [1, 2, 3]


def test_merge_budget_sums_fields_and_per_node():
    left = {"tokens_in": 10, "tokens_out": 5, "cost_usd": 0.1, "per_node": {"a": {"calls": 1, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.1}}}
    right = {"tokens_in": 3, "tokens_out": 2, "cost_usd": 0.05, "per_node": {"b": {"calls": 1, "tokens_in": 3, "tokens_out": 2, "cost_usd": 0.05}}}
    merged = merge_budget(left, right)
    assert merged["tokens_in"] == 13
    assert merged["tokens_out"] == 7
    assert merged["cost_usd"] == pytest.approx(0.15)
    assert set(merged["per_node"]) == {"a", "b"}


def test_merge_budget_marks_exceeded():
    merged = merge_budget({"cost_usd": 0.0, "max_cost_usd": 0.01}, {"cost_usd": 0.5})
    assert merged["exceeded"] is True


def test_merge_budget_preserves_cap_when_incoming_is_none():
    # Subgraph budgets carry an explicit ``max_cost_usd: None``; it must not
    # overwrite the parent's configured cap.
    left = {"max_cost_usd": 1.0, "cost_usd": 0.1, "per_node": {}}
    right = {"max_cost_usd": None, "cost_usd": 0.2, "per_node": {}}
    merged = merge_budget(left, right)
    assert merged["max_cost_usd"] == 1.0
    assert merged["cost_usd"] == pytest.approx(0.3)


def test_merge_budget_carries_exceeded_forward():
    merged = merge_budget({"exceeded": True, "cost_usd": 1.0}, {"cost_usd": 0.1})
    assert merged["exceeded"] is True
