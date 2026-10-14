"""Schema round-trips, JSON extraction and the repair-retry loop."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scholastra.agents.llm import LLMFactory, RawCompletion, SchemaValidationError
from scholastra.models.schemas import (
    Brief,
    DocumentContract,
    Issue,
    IssueList,
    Outline,
    ScopeProposal,
    Section,
    SectionEvidence,
)
from scholastra.utils import extract_json


def test_brief_round_trip():
    brief = Brief(topic="time", doc_type="article", preset="article-short", constraints=["x"])
    again = Brief.model_validate(brief.model_dump())
    assert again == brief


def test_outline_round_trip():
    outline = Outline(
        title="T",
        sections=[
            Section(id="intro", title="Intro", purpose="p", depends_on=[]),
            Section(id="body", title="Body", purpose="p", depends_on=["intro"]),
        ],
    )
    assert Outline.model_validate(outline.model_dump()).sections[1].depends_on == ["intro"]


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Here you go: {"a": 3} — done') == {"a": 3}
    assert extract_json('text [1, 2, 3] tail') == [1, 2, 3]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("not json at all")


def test_extract_json_rejects_truncated_object_with_balanced_inner_array():
    # An object truncated before its closing brace, containing a balanced array.
    # The old extractor fell through to that array and returned a wrong-shaped list.
    truncated = '{"thesis": "t", "argument_arc": ["Introduction", "History"]'
    with pytest.raises(ValueError, match="incomplete"):
        extract_json(truncated)


def test_extract_json_prefers_object_over_bracketed_prose():
    assert extract_json('[1] Here is the JSON: {"a": 1}') == {"a": 1}
    assert extract_json('[Note] {"a": 1}') == {"a": 1}


class _FlakyBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def complete(self, **kwargs) -> RawCompletion:
        return RawCompletion(text=self.responses.pop(0))


async def test_structured_repairs_invalid_then_valid():
    backend = _FlakyBackend(["not json", '{"refined_question": "Q", "length_preset": "article-short"}'])
    factory = LLMFactory(backend=backend)
    obj, usage = await factory.structured("scope", ScopeProposal, [{"role": "user", "content": "x"}])
    assert isinstance(obj, ScopeProposal)
    assert obj.refined_question == "Q"
    assert usage.calls == 2


async def test_structured_raises_after_max_repairs():
    backend = _FlakyBackend(["bad", "still bad", "nope"])
    factory = LLMFactory(backend=backend)
    with pytest.raises(SchemaValidationError):
        await factory.structured("scope", ScopeProposal, [{"role": "user", "content": "x"}], max_repairs=2)


async def test_fake_backend_returns_valid_issue_list():
    from scholastra.agents.fake import FakeBackend

    factory = LLMFactory(backend=FakeBackend())
    obj, _ = await factory.structured("critique", IssueList, [{"role": "user", "content": "x"}])
    assert isinstance(obj, IssueList)
    assert obj.issues == []


def test_issue_defaults():
    issue = Issue(id="i", description="d")
    assert issue.severity == "minor"


class _FixedBackend:
    """Always returns the same raw text (optionally with a finish_reason)."""

    def __init__(self, text: str, *, finish_reason: str | None = None) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.max_tokens: list[int] = []
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, **kwargs) -> RawCompletion:
        self.max_tokens.append(kwargs["max_tokens"])
        self.messages.append(list(kwargs["messages"]))
        return RawCompletion(text=self.text, finish_reason=self.finish_reason)


async def test_structured_wraps_bare_list_into_single_array_field():
    payload: list[dict[str, Any]] = [
        {"id": "c0", "text": "a claim", "excerpt": "verbatim"}
    ]
    factory = LLMFactory(backend=_FixedBackend(json.dumps(payload)))
    obj, _ = await factory.structured(
        "research", SectionEvidence, [{"role": "user", "content": "x"}]
    )
    assert isinstance(obj, SectionEvidence)
    assert obj.claims[0].id == "c0"


async def test_structured_wraps_bare_list_for_issue_list():
    factory = LLMFactory(backend=_FixedBackend(json.dumps([{"id": "i1", "description": "d"}])))
    obj, _ = await factory.structured("critique", IssueList, [{"role": "user", "content": "x"}])
    assert isinstance(obj, IssueList)
    assert obj.issues[0].id == "i1"


async def test_structured_does_not_guess_ambiguous_array_shape():
    # Two array fields -> coercion must not misfile the array; validation should fail.
    factory = LLMFactory(backend=_FixedBackend(json.dumps([{"id": "i1"}])))
    with pytest.raises(SchemaValidationError):
        await factory.structured(
            "contract", DocumentContract, [{"role": "user", "content": "x"}], max_repairs=0
        )


class _TruncatingThenValidBackend:
    """First call reports finish_reason=length with cut-off JSON, then a valid object."""

    def __init__(self) -> None:
        self.max_tokens: list[int] = []

    async def complete(self, **kwargs) -> RawCompletion:
        self.max_tokens.append(kwargs["max_tokens"])
        if len(self.max_tokens) == 1:
            return RawCompletion(text='{"refined_question": "Q', finish_reason="length")
        return RawCompletion(
            text='{"refined_question": "Q", "length_preset": "article-short"}'
        )


async def test_structured_grows_token_budget_after_truncation():
    backend = _TruncatingThenValidBackend()
    factory = LLMFactory(backend=backend)
    obj, usage = await factory.structured(
        "scope", ScopeProposal, [{"role": "user", "content": "x"}]
    )
    assert obj.refined_question == "Q"
    assert usage.calls == 2
    assert backend.max_tokens[1] > backend.max_tokens[0]


async def test_structured_stops_early_on_identical_failure():
    # A deterministic repeated failure must not pay for the full repair budget.
    backend = _FixedBackend("not json at all")
    factory = LLMFactory(backend=backend)
    with pytest.raises(SchemaValidationError):
        await factory.structured(
            "research", SectionEvidence, [{"role": "user", "content": "x"}]
        )
    assert len(backend.max_tokens) == 2  # one attempt + one retry, not three


async def test_structured_caps_echoed_output_in_repair_turn():
    # A huge prior response must not be echoed verbatim into the retry prompt.
    huge = '{"claims": [' + ("x" * 20000)
    backend = _FixedBackend(huge)
    factory = LLMFactory(backend=backend)
    with pytest.raises(SchemaValidationError):
        await factory.structured(
            "research", SectionEvidence, [{"role": "user", "content": "x"}], max_repairs=1
        )
    echoed = [m for m in backend.messages[1] if m["role"] == "assistant"]
    assert echoed and len(echoed[-1]["content"]) < 6100
