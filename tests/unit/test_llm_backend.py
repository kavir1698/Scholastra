"""LiteLLM backend parameter-degradation, reasoning-effort caching and observability."""

from __future__ import annotations

import logging

import litellm
import pytest

from theo_explore.agents.llm import LiteLLMBackend


def _response(text: str = "{}") -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


async def _complete(backend: LiteLLMBackend, **overrides):
    kwargs = {
        "node": "draft",
        "model": "openrouter/anthropic/claude-haiku-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3,
        "max_tokens": 100,
        "json_object": True,
        "json_schema": None,
        "reasoning_effort": "high",
    }
    kwargs.update(overrides)
    return await backend.complete(**kwargs)


async def test_reasoning_effort_sent_when_accepted(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(dict(kwargs))
        return _response()

    monkeypatch.setattr(litellm, "acompletion", fake)
    backend = LiteLLMBackend()
    await _complete(backend)

    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["temperature"] == 0.3
    assert backend._effort_supported["openrouter/anthropic/claude-haiku-5.5"] is True


async def test_rejected_effort_downgrades_warns_and_is_cached(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(dict(kwargs))
        if "reasoning_effort" in kwargs:
            raise ValueError("reasoning_effort is not supported")
        return _response()

    monkeypatch.setattr(litellm, "acompletion", fake)
    backend = LiteLLMBackend()

    with caplog.at_level(logging.WARNING):
        await _complete(backend)
        after_first = len(calls)
        await _complete(backend)

    # One warning, and the effort probe is not repeated on the second call.
    assert sum("reasoning_effort" in r.getMessage() for r in caplog.records) == 1
    assert after_first < len(calls)
    assert all("reasoning_effort" not in call for call in calls[after_first:])
    assert backend._effort_supported["openrouter/anthropic/claude-haiku-5.5"] is False


async def test_effort_kept_when_only_temperature_rejected(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise ValueError("temperature is not supported with reasoning")
        return _response()

    monkeypatch.setattr(litellm, "acompletion", fake)
    backend = LiteLLMBackend()

    with caplog.at_level(logging.WARNING):
        await _complete(backend)

    assert any(call.get("reasoning_effort") == "high" for call in calls)
    assert not any("reasoning_effort" in r.getMessage() for r in caplog.records)
    assert backend._effort_supported["openrouter/anthropic/claude-haiku-5.5"] is True


async def test_finish_reason_is_surfaced(monkeypatch: pytest.MonkeyPatch):
    async def fake(**kwargs):
        return {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(litellm, "acompletion", fake)
    backend = LiteLLMBackend()
    result = await _complete(backend)

    assert result.finish_reason == "length"
