"""Live progress reporting for a running project.

Streams the compiled graph with ``stream_mode="updates"`` and prints a
human-readable line per node/branch completion (research and drafting fan out,
so several branches report independently), plus phase headers and a final tally.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# node -> (phase key, phase label)
_NODE_PHASE: dict[str, tuple[str, str]] = {
    "intake": ("intake", "Intake"),
    "scope": ("scope", "Scope"),
    "outline": ("outline", "Outline"),
    "plan_research": ("plan", "Plan research"),
    "research_section": ("research", "Research"),
    "research_join": ("research", "Research"),
    "draft_section": ("draft", "Draft"),
    "draft_join": ("draft", "Draft"),
    "verify_citations": ("verify", "Verify citations"),
    "critique": ("critique", "Critique"),
    "revise_section": ("revise", "Revise"),
    "revise_join": ("revise", "Revise"),
    "synthesize": ("synthesize", "Synthesize"),
    "assemble": ("assemble", "Assemble"),
    "export": ("export", "Export"),
    "finalize": ("done", "Finalize"),
}


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _section_ids(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    return list(value.keys()) if isinstance(value, dict) else []


class ProgressReporter:
    """Prints phase headers, per-branch progress lines and a final tally."""

    def __init__(self, console: Console, *, enabled: bool = True) -> None:
        self.console = console
        self.enabled = enabled
        self._start = time.monotonic()
        self._last = self._start
        self._phases_seen: set[str] = set()
        self._phase_index = 0
        self._evidence_sections: set[str] = set()
        self._source_ids: set[str] = set()
        self._drafted_sections: set[str] = set()
        self._issues = 0
        self._research_total = 0
        self._draft_total = 0

    # -- helpers -----------------------------------------------------------
    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    def _delta(self) -> str:
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        return f"+{delta:.1f}s"

    def _line(self, markup: str) -> None:
        if self.enabled:
            self.console.print(markup)

    def _phase_header(self, phase_key: str, label: str, detail: str = "") -> None:
        if phase_key in self._phases_seen:
            if detail:
                self._line(f"      [dim]{detail}[/dim]")
            return
        self._phases_seen.add(phase_key)
        self._phase_index += 1
        suffix = f"  [dim]{detail}[/dim]" if detail else ""
        self._line(f"[bold cyan][{self._phase_index}][/bold cyan] [bold]{label}[/bold]{suffix}")

    # -- lifecycle ---------------------------------------------------------
    def banner(
        self,
        *,
        topic: str,
        doc_type: str,
        preset: str,
        project_dir: str,
        strong_model: str,
        cheap_model: str,
    ) -> None:
        if not self.enabled:
            return
        body = Text()
        body.append("topic:   ", style="dim")
        body.append(f"{topic}\n")
        body.append("type:    ", style="dim")
        body.append(f"{doc_type}   preset: {preset}\n")
        body.append("models:  ", style="dim")
        body.append(f"{strong_model}  |  {cheap_model}\n")
        body.append("project: ", style="dim")
        body.append(project_dir)
        self.console.print(Panel(body, title="Theoretical Exploration Bot", border_style="cyan"))

    def update(self, node: str, payload: dict[str, Any]) -> None:
        handler = _HANDLERS.get(node)
        if handler is not None:
            handler(self, payload)

    def interrupt(self, interrupts: Any) -> None:
        for item in interrupts or ():
            value = getattr(item, "value", item)
            kind = value.get("kind") if isinstance(value, dict) else None
            label = "scope" if kind == "scope_review" else "outline"
            self._line(f"[yellow]‖ paused for {label} review[/yellow]")

    def finished(self, values: dict[str, Any], *, exports: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        status = values.get("status", "unknown")
        colour = "green" if status == "completed" else "yellow"
        self.console.print(f"[{colour}]✓ {status}[/{colour}] in {self._elapsed():.1f}s")

    def note(self, markup: str) -> None:
        self._line(markup)


# -- per-node handlers ------------------------------------------------------
def _h_scope(r: ProgressReporter, payload: dict[str, Any]) -> None:
    question = (payload.get("scope") or {}).get("refined_question", "")
    r._phase_header("scope", "Scope", _short(question))


def _h_outline(r: ProgressReporter, payload: dict[str, Any]) -> None:
    sections = (payload.get("outline") or {}).get("sections", [])
    r._research_total = len(sections)
    r._draft_total = len(sections)
    ids = ", ".join(s.get("id", "?") for s in sections[:6])
    more = "" if len(sections) <= 6 else f" (+{len(sections) - 6} more)"
    r._phase_header("outline", "Outline", f"{len(sections)} sections: {ids}{more}")


def _h_plan(r: ProgressReporter, payload: dict[str, Any]) -> None:
    r._phase_header("plan", "Plan research", f"{len(payload.get('research_plan') or {})} briefs")


def _h_research(r: ProgressReporter, payload: dict[str, Any]) -> None:
    r._phase_header("research", "Research", f"{r._research_total} sections")
    evidence = payload.get("evidence") or {}
    for sid, ev in evidence.items():
        r._evidence_sections.add(sid)
        claims = len((ev or {}).get("claims") or [])
        coverage = (ev or {}).get("coverage", "?")
        r._line(
            f"      [green]•[/green] researched [bold]{sid}[/bold] "
            f"({claims} claims, {coverage}) [dim]{r._delta()}[/dim]"
        )
    for sid in _section_ids(payload, "sources"):
        r._source_ids.add(sid)


def _h_draft(r: ProgressReporter, payload: dict[str, Any]) -> None:
    r._phase_header("draft", "Draft", f"{r._draft_total} sections in dependency waves")
    drafts = payload.get("drafts") or {}
    for sid, draft in drafts.items():
        draft = draft or {}
        r._drafted_sections.add(sid)
        verb = "revised" if draft.get("revision") else "drafted"
        r._line(
            f"      [green]•[/green] {verb} [bold]{sid}[/bold] "
            f"({draft.get('word_count', 0)} words, {len(draft.get('cite_keys') or [])} cites) "
            f"[dim]{r._delta()}[/dim]"
        )


def _h_research_join(r: ProgressReporter, payload: dict[str, Any]) -> None:
    r._line(f"      [dim]{len(r._evidence_sections)} sections researched[/dim]")


def _h_verify(r: ProgressReporter, payload: dict[str, Any]) -> None:
    verification = payload.get("verification") or {}
    unresolved = sum(len((v or {}).get("unresolved_cites") or []) for v in verification.values())
    issues = payload.get("issues") or []
    r._issues += len(issues)
    blocking = sum(1 for i in issues if i.get("severity") in {"major", "blocking"})
    r._line(
        f"      verified {len(verification)} sections · {unresolved} unresolved cites · "
        f"{blocking} blocking issues [dim]{r._delta()}[/dim]"
    )


def _h_critique(r: ProgressReporter, payload: dict[str, Any]) -> None:
    issues = payload.get("issues") or []
    r._issues += len(issues)
    r._line(f"      critic raised {len(issues)} issues [dim]{r._delta()}[/dim]")


def _h_revise(r: ProgressReporter, payload: dict[str, Any]) -> None:
    if "drafts" in payload:
        _h_draft(r, payload)
    else:
        r._line(f"      revision round {payload.get('revision_round', '?')}")


def _h_synthesize(r: ProgressReporter, payload: dict[str, Any]) -> None:
    synthesis = payload.get("synthesis")
    if synthesis:
        r._line(f"      synthesis: {len(str(synthesis).split())} words [dim]{r._delta()}[/dim]")
    else:
        r._line("      [dim]synthesis skipped[/dim]")


def _h_assemble(r: ProgressReporter, payload: dict[str, Any]) -> None:
    words = len(str(payload.get("final_markdown") or "").split())
    r._line(f"      assembled {words} words from {len(r._drafted_sections)} sections")


def _h_export(r: ProgressReporter, payload: dict[str, Any]) -> None:
    outs = payload.get("exports") or {}
    r._line("      exports: " + ", ".join(sorted(outs)) if outs else "      [dim]no export[/dim]")
    for warning in payload.get("errors") or []:
        r._line(f"      [yellow]warning:[/yellow] {warning}")


_HANDLERS = {
    "scope": _h_scope,
    "outline": _h_outline,
    "plan_research": _h_plan,
    "research_section": _h_research,
    "research_join": _h_research_join,
    "draft_section": _h_draft,
    "verify_citations": _h_verify,
    "critique": _h_critique,
    "revise_section": _h_revise,
    "synthesize": _h_synthesize,
    "assemble": _h_assemble,
    "export": _h_export,
}


async def stream_graph(
    graph: Any, config: dict[str, Any], payload: Any, reporter: ProgressReporter
) -> Any:
    """Run/pause the graph while reporting progress; return the final snapshot."""
    async for chunk in graph.astream(payload, config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        if "__interrupt__" in chunk:
            reporter.interrupt(chunk["__interrupt__"])
            continue
        for node, update in chunk.items():
            if isinstance(update, dict):
                reporter.update(node, update)
    return await graph.aget_state(config)
