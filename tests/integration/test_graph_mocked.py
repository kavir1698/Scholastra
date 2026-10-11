"""Integration tests: full mocked graph runs (interrupts, fan-out, revision)."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.conftest import RevisionResponder, ThreeRootResponder, TrackingToolProvider
from theo_explore.graph.build import build_graph
from theo_explore.models.schemas import Outline, ScopeProposal
from theo_explore.runtime import make_initial_state, pending_interrupt


async def _run(deps, project_id: str, **overrides):
    graph = build_graph(deps, InMemorySaver())
    state = make_initial_state(project_id=project_id, topic="a test topic", **overrides)
    config = {"configurable": {"thread_id": project_id}}
    await graph.ainvoke(state, config)
    return graph, config


async def test_interrupts_fire_then_resume_to_completion(deps_factory):
    deps = deps_factory()
    graph, config = await _run(deps, "int-1", auto=False)

    snapshot = await graph.aget_state(config)
    review = pending_interrupt(snapshot)
    assert review is not None and review["kind"] == "scope_review"

    await graph.ainvoke(Command(resume={"approve": True}), config)
    snapshot = await graph.aget_state(config)
    review = pending_interrupt(snapshot)
    assert review is not None and review["kind"] == "outline_review"

    await graph.ainvoke(Command(resume={"approve": True}), config)
    snapshot = await graph.aget_state(config)
    assert not snapshot.next
    assert snapshot.values["status"] == "completed"
    assert snapshot.values["final_markdown"]


async def test_edited_scope_flows_into_outline(deps_factory):
    deps = deps_factory()
    graph, config = await _run(deps, "int-2", auto=False)

    snapshot = await graph.aget_state(config)
    scope = ScopeProposal.model_validate(pending_interrupt(snapshot)["scope"])
    edited = scope.model_copy(update={"refined_question": "EDITED QUESTION"})
    await graph.ainvoke(Command(resume={"scope": edited.model_dump()}), config)

    snapshot = await graph.aget_state(config)
    assert snapshot.values["scope"]["refined_question"] == "EDITED QUESTION"

    await graph.ainvoke(Command(resume={"approve": True}), config)
    snapshot = await graph.aget_state(config)
    assert snapshot.values["status"] == "completed"
    assert Outline.model_validate(snapshot.values["outline"]).sections


async def test_parallel_fanout_overlaps(deps_factory):
    tools = TrackingToolProvider(delay=0.15)
    deps = deps_factory(responder=ThreeRootResponder(), tools=tools)
    graph, config = await _run(deps, "par-1", auto=True)
    assert tools.max_concurrent >= 2

    snapshot = await graph.aget_state(config)
    assert set(snapshot.values["evidence"]) == {"alpha", "beta", "gamma"}
    assert snapshot.values["status"] == "completed"


async def test_revision_loop_is_bounded(deps_factory):
    responder = RevisionResponder()
    deps = deps_factory(responder=responder)
    graph, config = await _run(deps, "rev-1", auto=True)
    values = (await graph.aget_state(config)).values

    assert values["revision_round"] == 1
    assert responder.critique_calls == 2
    assert values["drafts"]["introduction"]["revision"] == 1
    assert values["drafts"]["key-concepts"]["revision"] == 0


async def test_preset_shapes_outline(deps_factory):
    deps = deps_factory()
    graph, config = await _run(deps, "pres-book", preset="mini-book", doc_type="mini-book", auto=True)
    book_sections = (await graph.aget_state(config)).values["outline"]["sections"]

    graph2, config2 = await _run(deps, "pres-article", preset="article-short", auto=True)
    article_sections = (await graph2.aget_state(config2)).values["outline"]["sections"]

    assert len(book_sections) > len(article_sections)


async def test_unresolved_citation_never_reaches_assembly(deps_factory):
    """The draft post-check strips cite keys absent from the section's evidence."""
    deps = deps_factory()
    graph, config = await _run(deps, "cite-1", auto=True)
    values = (await graph.aget_state(config)).values
    sources = set(values["sources"])
    for draft in values["drafts"].values():
        assert set(draft["cite_keys"]) <= sources


async def test_budget_guard_aborts_cleanly(deps_factory):
    from theo_explore.agents.fake import FakeBackend

    deps = deps_factory(backend=FakeBackend(cost_per_call=0.01))
    graph, config = await _run(deps, "budget-1", auto=True, max_cost_usd=0.005)
    values = (await graph.aget_state(config)).values

    # The run still finishes, but no research spend happens past the budget.
    assert values["status"] == "completed"
    assert values["sources"] == {}
    assert values["budget"]["exceeded"] is True
    assert values["final_markdown"]


async def test_budget_cap_survives_fanout(deps_factory):
    """The configured cap must not be overwritten by subgraph budget merges."""
    deps = deps_factory()
    graph, config = await _run(deps, "cap-1", auto=True, max_cost_usd=100.0)
    values = (await graph.aget_state(config)).values
    assert values["budget"]["max_cost_usd"] == 100.0
    assert values["status"] == "completed"


async def test_mini_book_drafts_nested_leaf_units(deps_factory):
    """Sub-subsections are drafted and rendered as deeper headings."""
    deps = deps_factory()
    graph, config = await _run(
        deps, "deep-1", preset="mini-book", doc_type="mini-book", auto=True
    )
    md = (await graph.aget_state(config)).values["final_markdown"]
    assert "#### " in md  # level-4 heading for a drafted sub-subsection leaf
    assert "### " in md  # level-3 heading for a subsection
