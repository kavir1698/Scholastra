"""Research-section citation binding and per-section guards."""

from __future__ import annotations

import re
import time

from tests.conftest import FakeTool
from theo_explore.agents.fake import ScriptedResponder
from theo_explore.graph.subgraphs.research_section import (
    _bind_claims,
    _fuzzy_resolve,
    _source_lookup,
    _wall_clock_expired,
    build_research_subgraph,
)
from theo_explore.models.schemas import Claim, ResearchBrief, SectionEvidence, Source


def _sources(*records: Source) -> dict:
    return {s.id: s.model_dump() for s in records}


def _farewell() -> Source:
    return Source(
        id="url:https://plato.stanford.edu/entries/compatibilism",
        kind="web",
        title="Compatibilism (Stanford Encyclopedia of Philosophy)",
        url="https://plato.stanford.edu/entries/compatibilism",
        tier=3,
    )


def _claim(*source_ids: str) -> SectionEvidence:
    return SectionEvidence(
        section_id="s1",
        claims=[
            Claim(
                id="c1",
                text="t",
                source_ids=list(source_ids),
                excerpt="verbatim excerpt",
                locator="p1",
            )
        ],
    )


def _bind(evidence: SectionEvidence, sources: dict, log=None) -> list:
    return _bind_claims(evidence, sources, log)


def test_bind_resolves_bare_and_prefixed_url():
    sources = _sources(_farewell())
    for key in (
        "https://plato.stanford.edu/entries/compatibilism/",
        "url:https://plato.stanford.edu/entries/compatibilism/",
        "plato.stanford.edu/entries/compatibilism",
    ):
        bound = _bind(_claim(key), sources)
        assert len(bound) == 1, key
        assert bound[0].source_ids == ["url:https://plato.stanford.edu/entries/compatibilism"]
        assert bound[0].tier == 3


def test_bind_resolves_title_slug():
    sources = _sources(_farewell())
    bound = _bind(_claim("compatibilism-stanford-encyclopedia-of-philosophy"), sources)
    assert len(bound) == 1


def test_bind_resolves_author_year_bibkey():
    source = Source(
        id="doi:10.1/x", kind="paper", title="Responsibility and Control", authors=["John Fischer"], year=1998
    )
    bound = _bind(_claim("Fischer1998"), _sources(source))
    assert len(bound) == 1 and bound[0].source_ids == ["doi:10.1/x"]


def test_bind_fuzzy_title_tokens():
    source = Source(
        id="doi:10.2/y",
        kind="paper",
        title="Manipulation and Moral Responsibility",
        authors=["A. Author"],
        year=2001,
    )
    bound = _bind(_claim("manipulation responsibility"), _sources(source))
    assert len(bound) == 1


def test_bind_drops_genuinely_unknown_and_logs():
    logged: list[tuple] = []
    evidence = _claim("zz-999", "cc-01")
    bound = _bind(evidence, _sources(_farewell()), lambda event, **kw: logged.append((event, kw)))
    assert bound == []
    assert any(event == "claims_dropped" for event, _ in logged)
    dropped = next(kw for event, kw in logged if event == "claims_dropped")
    assert dropped["sources"] == 1


def test_fuzzy_requires_two_token_overlap():
    sources = _sources(_farewell())
    assert _fuzzy_resolve("compatibilism", sources) is None  # single token
    assert _fuzzy_resolve("goldfish umbrella", sources) is None


def test_lookup_indexes_url_title_and_bibkey():
    lookup = _source_lookup(_sources(_farewell()))
    assert lookup["url:https://plato.stanford.edu/entries/compatibilism"].startswith("url:")
    assert lookup["compatibilism-stanford-encyclopedia-of-philosophy"].startswith("url:")


def test_wall_clock_expiry():
    brief = ResearchBrief(section_id="s1", wall_clock_seconds=10)
    assert _wall_clock_expired({"started_at": time.time() - 100.0}, brief) is True
    assert _wall_clock_expired({"started_at": time.time()}, brief) is False
    assert _wall_clock_expired({}, brief) is False


class _FetchProvider:
    """Tool provider exposing a single fetch-style tool that returns page text."""

    async def get_tools(self, allowlist):
        return [FakeTool("fetch__fetch", lambda args: "# Page\n\nBody text about the topic.", "fetch")]

    async def aclose(self):
        return None


class _FetchPlanResponder(ScriptedResponder):
    def _node_research_plan(self, prompt: str) -> dict:
        step = int(re.search(r"STEP:\s*(\d+)", prompt).group(1))
        if step == 0:
            return {
                "tool_calls": [
                    {
                        "tool": "fetch__fetch",
                        "arguments": {"url": "https://plato.stanford.edu/entries/compatibilism"},
                    }
                ],
                "done": False,
                "reasoning": "fetch the entry",
            }
        return {"tool_calls": [], "done": True, "reasoning": "done"}


async def test_fetched_page_becomes_citable_evidence(deps_factory):
    """Regression: text returned by a fetch tool must be minted into a Source so
    the extractor's claims bind instead of being dropped as unknown cites."""
    deps = deps_factory(
        responder=_FetchPlanResponder(),
        tools=_FetchProvider(),
        flags={"research_allowlist": ["fetch__fetch"]},
    )
    subgraph = build_research_subgraph(deps)
    final = await subgraph.ainvoke(
        {
            "project_id": "mint-1",
            "section_id": "s1",
            "section_title": "Section One",
            "section_purpose": "p",
            "brief": {"section_id": "s1", "max_tool_calls": 2, "queries": ["q"]},
            "contract": {},
            "started_at": time.time(),
        }
    )
    minted = "url:https://plato.stanford.edu/entries/compatibilism"
    assert minted in final["sources"]
    evidence = final["evidence"]["s1"]
    assert evidence["claims"], "fetched evidence must bind, not be dropped"
    assert evidence["claims"][0]["source_ids"] == [minted]

