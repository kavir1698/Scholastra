"""Model routing, capability flags and budget guards."""

from __future__ import annotations

import pytest

from theo_explore.agents.llm import LLMFactory
from theo_explore.config import get_settings
from theo_explore.models.schemas import NodeUsage
from theo_explore.observability import (
    add_usage,
    as_budget,
    budget_exceeded,
    build_run_report,
    new_budget,
)


def test_node_config_resolves_tiers():
    config = {
        "tiers": {
            "strong": "openrouter/deepseek/deepseek-v3.2",
            "cheap": "openrouter/deepseek/deepseek-chat",
        },
        "defaults": {},
        "nodes": {"scope": {"model": "strong"}, "research": {"model": "cheap"}},
    }
    factory = LLMFactory(models_config=config)
    assert factory.node_config("scope").model == config["tiers"]["strong"]
    assert factory.node_config("research").model == config["tiers"]["cheap"]


def test_node_config_defaults_for_unknown_node():
    factory = LLMFactory()
    cfg = factory.node_config("totally-new-node")
    assert cfg.model.split("/", 1)[0] in {"openrouter", "deepseek"}


def test_resolve_model_normalizes_bare_openrouter_ids():
    factory = LLMFactory()
    assert (
        factory.resolve_model("anthropic/claude-haiku-5.5")
        == "openrouter/anthropic/claude-haiku-5.5"
    )
    assert (
        factory.resolve_model("~deepseek/deepseek-flash-latest")
        == "openrouter/~deepseek/deepseek-flash-latest"
    )
    # Already-prefixed ids are left untouched.
    assert (
        factory.resolve_model("openrouter/deepseek/deepseek-chat")
        == "openrouter/deepseek/deepseek-chat"
    )
    assert factory.resolve_model("deepseek/deepseek-chat") == "deepseek/deepseek-chat"


def test_node_config_normalizes_bare_openrouter_ids():
    config = {
        "tiers": {"strong": "anthropic/claude-haiku-5.5", "cheap": "~deepseek/deepseek-flash-latest"},
        "defaults": {},
        "nodes": {"scope": {"model": "strong"}, "research": {"model": "cheap"}},
    }
    factory = LLMFactory(models_config=config)
    assert factory.node_config("scope").model == "openrouter/anthropic/claude-haiku-5.5"
    assert factory.node_config("research").model == "openrouter/~deepseek/deepseek-flash-latest"


def test_empty_allowed_providers_rejected():
    settings = get_settings().model_copy(update={"allowed_providers": ""})
    with pytest.raises(ValueError, match="not allowed"):
        LLMFactory(settings=settings)


def test_capability_flags_present():
    # json_schema is off by default (most providers reject it); json_object is used.
    factory = LLMFactory()
    assert factory.node_config("draft").supports_json_schema is False
    assert factory.node_config("draft").supports_temperature is True
    assert factory.node_config("draft").reasoning_effort == "high"
    assert factory.node_config("draft").supports_reasoning_effort is True


def test_reasoning_effort_defaults_high_for_all_nodes():
    factory = LLMFactory()
    for node in ("draft", "verify", "outline", "intake"):
        cfg = factory.node_config(node)
        assert cfg.reasoning_effort == "high"
        assert factory._reasoning_effort(cfg) == "high"


def test_reasoning_effort_overridable_and_disableable_per_node():
    config = {
        "tiers": {
            "strong": "openrouter/deepseek/deepseek-chat",
            "cheap": "openrouter/deepseek/deepseek-chat",
        },
        "defaults": {"reasoning_effort": "high", "supports_reasoning_effort": False},
        "nodes": {
            "scope": {"model": "strong"},
            "verify": {
                "model": "cheap",
                "reasoning_effort": "low",
                "supports_reasoning_effort": True,
            },
        },
    }
    factory = LLMFactory(models_config=config)
    assert factory._reasoning_effort(factory.node_config("scope")) is None
    assert factory._reasoning_effort(factory.node_config("verify")) == "low"


def test_add_usage_and_budget():
    budget = new_budget(max_cost_usd=1.0)
    usage = NodeUsage(calls=1, tokens_in=100, tokens_out=50, cost_usd=0.25)
    budget = add_usage(budget, "scope", usage)
    budget = add_usage(budget, "scope", usage)
    model = as_budget(budget)
    assert model.tokens_in == 200
    assert model.per_node["scope"].calls == 2
    assert model.cost_usd == 0.5
    assert not budget_exceeded(budget)


def test_budget_exceeded_guard():
    budget = add_usage(new_budget(max_cost_usd=0.1), "draft", NodeUsage(cost_usd=0.2))
    assert budget_exceeded(budget)


def test_run_report_contains_per_node_costs():
    report = build_run_report(
        {"project_id": "p", "status": "completed"},
        add_usage(new_budget(), "draft", NodeUsage(tokens_in=1, cost_usd=0.02)),
    )
    assert report["per_node"]["draft"]["cost_usd"] == 0.02
    assert report["project_id"] == "p"
