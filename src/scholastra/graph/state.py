"""Graph state definition.

Values are JSON-serialisable so the SQLite checkpointer can persist them.
Concurrent writes to ``evidence``/``drafts``/``sources``/``verification`` use the
key-merge reducer; ``issues`` and ``errors`` accumulate; ``budget`` sums.
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

from scholastra.graph.reducers import add_lists, merge_budget, merge_by_key


class DocState(TypedDict, total=False):
    project_id: str
    workspace_dir: str
    status: str

    # Raw intake inputs (must be declared or LangGraph drops them).
    topic: str
    doc_type: str
    preset: str
    audience: str | None
    constraints: list[str]
    max_cost_usd: float | None

    brief: dict[str, Any]
    scope: dict[str, Any]
    outline: dict[str, Any]
    contract: dict[str, Any]
    research_plan: dict[str, dict[str, Any]]

    evidence: Annotated[dict[str, dict[str, Any]], merge_by_key]
    sources: Annotated[dict[str, dict[str, Any]], merge_by_key]
    drafts: Annotated[dict[str, dict[str, Any]], merge_by_key]
    verification: Annotated[dict[str, dict[str, Any]], merge_by_key]
    issues: Annotated[list[dict[str, Any]], add_lists]
    editorial: Annotated[list[dict[str, Any]], add_lists]

    revision_round: int
    max_revision_rounds: int
    editor_round: int
    draft_wave: int

    synthesize: bool
    auto: bool
    export_formats: list[str]

    synthesis: str | None
    synthesis_verification: dict[str, Any]
    final_markdown: str
    quality: dict[str, Any]
    exports: Annotated[dict[str, str], merge_by_key]
    errors: Annotated[list[str], add_lists]
    budget: Annotated[dict[str, Any], merge_budget]
