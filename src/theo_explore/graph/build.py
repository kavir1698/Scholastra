"""Graph construction: wires nodes, subgraphs, fan-out and the revision loop."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from theo_explore.graph.deps import Deps
from theo_explore.graph.nodes.assembly import (
    make_assemble_node,
    make_export_node,
    make_finalize_node,
)
from theo_explore.graph.nodes.contract import make_contract_node
from theo_explore.graph.nodes.draft import make_dispatch_draft, make_draft_join_node
from theo_explore.graph.nodes.editor import make_editor_node
from theo_explore.graph.nodes.intake import make_intake_node
from theo_explore.graph.nodes.outline import make_await_outline_node, make_outline_node
from theo_explore.graph.nodes.quality import (
    make_critique_node,
    make_revise_join_node,
    make_route_after_critique,
    make_verify_citations_node,
)
from theo_explore.graph.nodes.research import (
    make_dispatch_research,
    make_plan_research_node,
    make_research_join_node,
)
from theo_explore.graph.nodes.scope import make_await_scope_node, make_scope_node
from theo_explore.graph.nodes.synthesize import make_synthesize_node
from theo_explore.graph.state import DocState
from theo_explore.graph.subgraphs.draft_section import build_draft_subgraph
from theo_explore.graph.subgraphs.research_section import build_research_subgraph


def _research_node(subgraph: Any):
    async def node(payload: dict[str, Any]) -> dict[str, Any]:
        final = await subgraph.ainvoke(payload)
        return {
            "evidence": final.get("evidence", {}),
            "sources": final.get("sources", {}),
            "budget": final.get("budget", {}),
        }

    return node


def _draft_node(subgraph: Any):
    async def node(payload: dict[str, Any]) -> dict[str, Any]:
        final = await subgraph.ainvoke(payload)
        return {
            "drafts": final.get("drafts", {}),
            "budget": final.get("budget", {}),
        }

    return node


def build_graph(deps: Deps, checkpointer: Any = None) -> Any:
    dispatch_research = make_dispatch_research(deps)
    dispatch_draft = make_dispatch_draft(deps)
    route_after_critique = make_route_after_critique(deps)

    graph = StateGraph(DocState)
    graph.add_node("intake", make_intake_node(deps))
    graph.add_node("scope", make_scope_node(deps))
    graph.add_node("await_scope", make_await_scope_node(deps))
    graph.add_node("outline", make_outline_node(deps))
    graph.add_node("await_outline", make_await_outline_node(deps))
    graph.add_node("contract", make_contract_node(deps))
    graph.add_node("plan_research", make_plan_research_node(deps))

    graph.add_node("research_section", _research_node(build_research_subgraph(deps)))
    graph.add_node("research_join", make_research_join_node(deps))

    graph.add_node("draft_section", _draft_node(build_draft_subgraph(deps)))
    graph.add_node("revise_section", _draft_node(build_draft_subgraph(deps)))
    graph.add_node("draft_join", make_draft_join_node(deps))

    graph.add_node("verify_citations", make_verify_citations_node(deps))
    graph.add_node("critique", make_critique_node(deps))
    graph.add_node("revise_join", make_revise_join_node(deps))
    graph.add_node("editor", make_editor_node(deps))

    graph.add_node("synthesize", make_synthesize_node(deps))
    graph.add_node("assemble", make_assemble_node(deps))
    graph.add_node("export", make_export_node(deps))
    graph.add_node("finalize", make_finalize_node(deps))

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "scope")
    graph.add_edge("scope", "await_scope")
    graph.add_edge("await_scope", "outline")
    graph.add_edge("outline", "await_outline")
    graph.add_edge("await_outline", "contract")
    graph.add_edge("contract", "plan_research")

    graph.add_conditional_edges(
        "plan_research", dispatch_research, ["research_section", "research_join"]
    )
    graph.add_edge("research_section", "research_join")

    graph.add_conditional_edges(
        "research_join", dispatch_draft, ["draft_section", "verify_citations"]
    )
    graph.add_edge("draft_section", "draft_join")
    graph.add_conditional_edges(
        "draft_join", dispatch_draft, ["draft_section", "verify_citations"]
    )

    graph.add_edge("verify_citations", "critique")
    graph.add_conditional_edges("critique", route_after_critique, ["revise_section", "editor"])
    graph.add_edge("revise_section", "revise_join")
    graph.add_edge("revise_join", "verify_citations")
    graph.add_edge("editor", "synthesize")

    graph.add_edge("synthesize", "assemble")
    graph.add_edge("assemble", "export")
    graph.add_edge("export", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer, name="theo_explore")
