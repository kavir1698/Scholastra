"""Pydantic domain schemas shared across nodes, agents and subgraphs.

Every LLM-facing model carries field descriptions because these schemas double as
the structured-output contract for the LLM factory.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["article", "mini-book"]
Coverage = Literal["high", "medium", "low"]
Severity = Literal["info", "minor", "major", "blocking"]
SourceKind = Literal["paper", "preprint", "web", "book", "encyclopedia", "other"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Brief(_Base):
    """Normalised intake of the raw user request."""

    topic: str = Field(description="The theoretical question or topic to explore.")
    doc_type: DocType = Field(default="article", description="Target artifact type.")
    preset: str = Field(default="article-short", description="Length/quality preset name.")
    audience: str | None = Field(default=None, description="Intended reader, if stated.")
    constraints: list[str] = Field(
        default_factory=list, description="Explicit user constraints (scope, style, sources)."
    )


class ScopeProposal(_Base):
    """The refined research scope, produced by the Scoper and human-approved."""

    refined_question: str = Field(description="A single, sharpened research question.")
    sub_questions: list[str] = Field(
        default_factory=list, description="3-7 decomposable sub-questions the document must answer."
    )
    in_scope: list[str] = Field(default_factory=list, description="What the document covers.")
    out_of_scope: list[str] = Field(default_factory=list, description="Explicit non-goals.")
    audience: str = Field(default="informed general reader", description="Target reader.")
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions made in the absence of clarification."
    )
    length_preset: str = Field(default="article-short", description="Length preset to apply.")
    scope_rationale: str = Field(
        default="",
        description="Why the scope admits comprehensive, authoritative depth at the target length.",
    )


class Subsection(_Base):
    id: str = Field(description="Stable slug id, unique within the document.")
    title: str = Field(description="Short subsection heading.")
    purpose: str = Field(description="One sentence on what this subsection establishes.")
    key_questions: list[str] = Field(default_factory=list)
    target_words: int = Field(default=350, ge=0)
    children: list[Subsection] = Field(
        default_factory=list,
        description="Nested sub-subsections. Each leaf (a subsection with no children) is "
        "drafted by its own generation; parents contribute a heading and structure only.",
    )


class Section(_Base):
    id: str = Field(description="Stable slug id, unique within the document.")
    title: str = Field(description="Section heading.")
    purpose: str = Field(description="What this section establishes and why.")
    key_questions: list[str] = Field(default_factory=list)
    target_words: int = Field(default=600, ge=0)
    depends_on: list[str] = Field(
        default_factory=list,
        description="Section ids that must be drafted before this one (topological order).",
    )
    source_domains: list[str] = Field(
        default_factory=list,
        description="Preferred source domains/servers, e.g. ['arxiv','pubmed','exa'].",
    )
    required: bool = Field(default=True, description="Whether the section is mandatory.")
    subsections: list[Subsection] = Field(default_factory=list)


class Outline(_Base):
    title: str = Field(description="Working document title.")
    abstract_target_words: int = Field(default=150, ge=0)
    sections: list[Section] = Field(default_factory=list)


class ResearchBrief(_Base):
    """Per-section research plan produced by the research planner."""

    section_id: str
    queries: list[str] = Field(default_factory=list, description="Search queries to issue.")
    source_domains: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(default=8, ge=1)
    max_sources: int = Field(default=6, ge=1)
    wall_clock_seconds: int = Field(default=180, ge=10)


class Claim(_Base):
    """An atomic, source-bound finding extracted by a research agent."""

    id: str = Field(description="Stable id for the claim.")
    text: str = Field(description="The claim, stated in one or two sentences.")
    source_ids: list[str] = Field(
        default_factory=list, description="One or more SourceId values backing the claim."
    )
    excerpt: str = Field(description="Verbatim excerpt retrieved from the cited source.")
    locator: str | None = Field(
        default=None, description="Page/section/URL anchor for the excerpt."
    )


class SectionEvidence(_Base):
    section_id: str
    claims: list[Claim] = Field(default_factory=list)
    notes: str = Field(default="", description="Synthesis notes, tensions, gaps.")
    coverage: Coverage = Field(default="medium")
    tool_calls: int = Field(default=0, ge=0)


class Source(_Base):
    """A normalised, deduplicated bibliographic record."""

    id: str = Field(description="Stable SourceId.")
    kind: SourceKind = Field(default="other")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    bibtex: str | None = Field(default=None, description="Original BibTeX entry if available.")
    accessed: str | None = Field(default=None, description="ISO date the source was retrieved.")


class SectionDraft(_Base):
    section_id: str
    title: str
    markdown: str = Field(description="Section prose in Markdown with inline cite keys like [@id].")
    summary: str = Field(default="", description="2-3 sentence summary for dependents/synthesis.")
    cite_keys: list[str] = Field(default_factory=list)
    word_count: int = Field(default=0, ge=0)
    revision: int = Field(default=0, ge=0)


class SectionVerification(_Base):
    section_id: str
    resolved_cites: list[str] = Field(default_factory=list)
    unresolved_cites: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    passed: bool = True


class Issue(_Base):
    id: str
    severity: Severity = Field(default="minor")
    category: str = Field(default="other")
    target_section: str | None = None
    description: str
    evidence: str | None = None
    round: int = Field(default=0, ge=0)


class IssueList(_Base):
    """Wrapper so the critic can return a ranked list of issues."""

    issues: list[Issue] = Field(default_factory=list)


class ClaimSupport(_Base):
    """LLM judgment of which claims are not supported by their cited excerpts."""

    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str = Field(default="")


class SynthesisResult(_Base):
    markdown: str = Field(description="Cross-cutting synthesis in Markdown with inline cite keys.")
    cite_keys: list[str] = Field(default_factory=list)


class ToolCall(_Base):
    tool: str = Field(description="Exact MCP tool name to call.")
    arguments: dict = Field(default_factory=dict, description="Arguments for the tool.")


class ToolPlan(_Base):
    """One ReAct step: which read-only research tools to call next, if any."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    done: bool = Field(default=False, description="True when enough evidence has been gathered.")
    reasoning: str = Field(default="")




class NodeUsage(_Base):
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class BudgetState(_Base):
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    per_node: dict[str, NodeUsage] = Field(default_factory=dict)
    max_cost_usd: float | None = None
    exceeded: bool = False


LLM_SCHEMAS: dict[str, type[BaseModel]] = {
    "Brief": Brief,
    "ScopeProposal": ScopeProposal,
    "Outline": Outline,
    "ResearchBrief": ResearchBrief,
    "SectionEvidence": SectionEvidence,
    "SectionDraft": SectionDraft,
    "Issue": Issue,
}
