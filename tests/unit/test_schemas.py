"""Schema round-trips, JSON extraction and the repair-retry loop."""

from __future__ import annotations

import pytest

from theo_explore.agents.llm import LLMFactory, RawCompletion, SchemaValidationError
from theo_explore.models.schemas import Brief, Issue, IssueList, Outline, ScopeProposal, Section
from theo_explore.utils import extract_json


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
    from theo_explore.agents.fake import FakeBackend

    factory = LLMFactory(backend=FakeBackend())
    obj, _ = await factory.structured("critique", IssueList, [{"role": "user", "content": "x"}])
    assert isinstance(obj, IssueList)
    assert obj.issues == []


def test_issue_defaults():
    issue = Issue(id="i", description="d")
    assert issue.severity == "minor"
