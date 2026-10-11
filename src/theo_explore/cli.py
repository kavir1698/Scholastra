"""Typer CLI: run / resume / status / export."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from theo_explore.config import Settings, get_settings, load_preset, repo_root
from theo_explore.io.artifacts import new_project_id, open_workspace, resolve_workspace_root
from theo_explore.io.export import detect_tools, export_document
from theo_explore.progress import ProgressReporter, stream_graph
from theo_explore.runtime import (
    Project,
    make_initial_state,
    open_project,
    pending_interrupt,
)

app = typer.Typer(
    add_completion=False,
    help="Theoretical Exploration Bot: research -> article/mini-book.",
)
console = Console()

DEFAULT_FORMATS = ["pdf", "docx"]


def _settings(max_concurrency: int | None = None, workspace: str | None = None) -> Settings:
    base = get_settings()
    updates: dict[str, Any] = {}
    if max_concurrency is not None:
        updates["max_concurrency"] = max_concurrency
    if workspace is not None:
        updates["workspace_root"] = workspace
    return base.model_copy(update=updates) if updates else base


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root() / ".env", override=False)
    except Exception:  # noqa: BLE001 - env is optional
        pass


async def _drive_interactive(
    project: Project, first_payload: Any, reporter: ProgressReporter
) -> Any:
    snapshot = await stream_graph(project.graph, project.config, first_payload, reporter)
    while True:
        review = pending_interrupt(snapshot)
        if review is None:
            break
        decision = _collect_decision(review)
        snapshot = await stream_graph(
            project.graph, project.config, Command(resume=decision), reporter
        )
    return snapshot


def _collect_decision(review: dict[str, Any]) -> dict[str, Any]:
    kind = review.get("kind")
    body = review.get("scope") or review.get("outline") or {}
    preview = json.dumps(body, indent=2, ensure_ascii=False)[:4000]
    console.print(Panel(preview, title=kind or "review"))
    if kind == "outline_review" and body.get("sections"):
        titles = ", ".join(s.get("id", "?") for s in body["sections"])
        console.print(f"[bold]Sections:[/bold] {titles}")
    approved = typer.confirm("Approve and continue?", default=True)
    if approved:
        return {"approve": True}
    edit_path = typer.prompt(
        "Provide a path to a JSON file with the edited object (or blank to abort)",
        default="",
        show_default=False,
    )
    if not edit_path:
        raise typer.Abort()
    edited = json.loads(Path(edit_path).read_text(encoding="utf-8"))
    key = "scope" if kind == "scope_review" else "outline"
    return {key: edited}


def _make_reporter(
    project: Project,
    workspace: Any,
    *,
    topic: str,
    doc_type: str,
    preset: str,
    quiet: bool,
) -> ProgressReporter:
    reporter = ProgressReporter(console, enabled=not quiet)
    reporter.banner(
        topic=topic or "(unknown)",
        doc_type=doc_type or "?",
        preset=preset or "?",
        project_dir=str(workspace.dir),
        strong_model=project.deps.llm.node_config("scope").model,
        cheap_model=project.deps.llm.node_config("research").model,
    )
    try:
        outline = workspace.load_outline()
        if outline is not None:
            reporter.seed_sections(sum(1 for s in outline.sections if s.target_words > 0))
    except Exception:  # noqa: BLE001 - seeding is best-effort
        pass
    return reporter


async def _run_project(
    project: Project,
    initial_state: dict[str, Any],
    *,
    auto: bool,
    reporter: ProgressReporter,
) -> Any:
    if auto:
        return await stream_graph(project.graph, project.config, initial_state, reporter)
    return await _drive_interactive(project, initial_state, reporter)


@app.command()
def run(
    topic: str = typer.Argument(..., help="The theoretical question to explore."),
    doc_type: str = typer.Option("article", "--type", help="article | mini-book"),
    preset: str = typer.Option("article-short", "--preset", help="Length/quality preset."),
    audience: str | None = typer.Option(None, "--audience"),
    constraint: list[str] = typer.Option(None, "--constraint", help="Repeatable constraint."),
    auto: bool = typer.Option(False, "--auto", help="Skip HITL approvals."),
    synthesize: bool = typer.Option(False, "--synthesize", help="Add whole-document synthesis."),
    max_concurrency: int = typer.Option(4, "--max-concurrency"),
    fake_llm: bool = typer.Option(False, "--fake-llm", help="Deterministic offline backend."),
    export: bool = typer.Option(True, "--export/--no-export", help="Render PDF/DOCX via Pandoc."),
    formats: str = typer.Option(",".join(DEFAULT_FORMATS), "--formats"),
    max_cost: float | None = typer.Option(None, "--max-cost", help="Abort budget in USD."),
    project_id: str | None = typer.Option(None, "--project-id"),
    workspace: str | None = typer.Option(None, "--workspace"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress live progress output."),
) -> None:
    """Start a new exploration."""
    _load_env()
    settings = _settings(max_concurrency, workspace)
    preset_obj = load_preset(preset, settings)
    quality = preset_obj.get("quality", {})
    effective_synthesize = synthesize or bool(quality.get("synthesize_default"))
    export_formats = [f.strip() for f in formats.split(",") if f.strip()] if export else []
    pid = project_id or new_project_id(topic, doc_type=doc_type)
    asyncio.run(
        _run(
            settings=settings,
            project_id=pid,
            state=make_initial_state(
                project_id=pid,
                topic=topic,
                doc_type=doc_type,
                preset=preset,
                audience=audience,
                constraints=list(constraint or []),
                auto=auto,
                synthesize=effective_synthesize,
                export_formats=export_formats,
                max_cost_usd=max_cost,
            ),
            fake=fake_llm,
            auto=auto,
            quiet=quiet,
        )
    )


async def _run(
    settings: Settings,
    project_id: str,
    state: dict[str, Any],
    *,
    fake: bool,
    auto: bool,
    quiet: bool,
) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    root = resolve_workspace_root(settings.workspace_root)
    workspace = open_workspace(root, project_id)
    tool_status = detect_tools()
    async with AsyncSqliteSaver.from_conn_string(str(workspace.checkpoint_path)) as saver:
        project = open_project(
            project_id, settings=settings, fake=fake, checkpointer=saver
        )
        reporter = _make_reporter(
            project,
            workspace,
            topic=state.get("topic", ""),
            doc_type=state.get("doc_type", ""),
            preset=state.get("preset", ""),
            quiet=quiet,
        )
        if not quiet:
            console.print(f"[dim]{tool_status.describe()}[/dim]")
        try:
            snapshot = await _run_project(project, state, auto=auto, reporter=reporter)
        finally:
            await project.deps.tools.aclose()
            if project.deps.logger is not None:
                project.deps.logger.close()
    reporter.finished(snapshot.values)
    _print_summary(workspace.project_id, settings)


@app.command()
def resume(
    project_id: str = typer.Argument(...),
    fake_llm: bool = typer.Option(False, "--fake-llm"),
    workspace: str | None = typer.Option(None, "--workspace"),
    auto: bool = typer.Option(True, "--auto/--interactive"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress live progress output."),
) -> None:
    """Resume a paused or crashed project (thread_id = project_id)."""
    _load_env()
    settings = _settings(None, workspace)
    asyncio.run(_resume(settings, project_id, fake=fake_llm, auto=auto, quiet=quiet))


async def _resume(
    settings: Settings, project_id: str, *, fake: bool, auto: bool, quiet: bool
) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    root = resolve_workspace_root(settings.workspace_root)
    workspace = open_workspace(root, project_id)
    brief = workspace.load_brief()
    async with AsyncSqliteSaver.from_conn_string(str(workspace.checkpoint_path)) as saver:
        project = open_project(project_id, settings=settings, fake=fake, checkpointer=saver)
        reporter = _make_reporter(
            project,
            workspace,
            topic=(brief.topic if brief else project_id),
            doc_type=(brief.doc_type if brief else ""),
            preset=(brief.preset if brief else ""),
            quiet=quiet,
        )
        try:
            snapshot = await project.snapshot()
            review = pending_interrupt(snapshot)
            if review is not None:
                decision = {"approve": True} if auto else _collect_decision(review)
                snapshot = await stream_graph(
                    project.graph, project.config, Command(resume=decision), reporter
                )
            else:
                snapshot = await stream_graph(project.graph, project.config, None, reporter)
            while (review := pending_interrupt(snapshot)) is not None:
                decision = {"approve": True} if auto else _collect_decision(review)
                snapshot = await stream_graph(
                    project.graph, project.config, Command(resume=decision), reporter
                )
        finally:
            await project.deps.tools.aclose()
            if project.deps.logger is not None:
                project.deps.logger.close()
    reporter.finished(snapshot.values)
    _print_summary(project_id, settings)


@app.command()
def status(
    project_id: str = typer.Argument(...),
    workspace: str | None = typer.Option(None, "--workspace"),
) -> None:
    """Show a project's manifest and artifacts."""
    _load_env()
    settings = _settings(None, workspace)
    _print_status(project_id, settings)


@app.command()
def export(
    project_id: str = typer.Argument(...),
    formats: str = typer.Option(",".join(DEFAULT_FORMATS), "--formats"),
    workspace: str | None = typer.Option(None, "--workspace"),
) -> None:
    """Re-export an existing project's Markdown to PDF/DOCX."""
    _load_env()
    settings = _settings(None, workspace)
    root = resolve_workspace_root(settings.workspace_root)
    ws = open_workspace(root, project_id)
    md_path = ws.path("artifacts/final.md")
    if not md_path.exists():
        console.print(f"[red]No final.md for project {project_id}[/red]")
        raise typer.Exit(code=1)
    result = export_document(
        md_path,
        ws.artifacts_dir,
        formats=[f.strip() for f in formats.split(",") if f.strip()],
        stem="final",
    )
    for name, path in result.outputs.items():
        console.print(f"[green]{name}[/green] -> {path}")
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _print_status(project_id: str, settings: Settings) -> None:
    root = resolve_workspace_root(settings.workspace_root)
    ws = open_workspace(root, project_id)
    manifest = ws.load_manifest()
    if not manifest:
        console.print(f"[red]No manifest for project {project_id} in {ws.dir}[/red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


def _print_summary(project_id: str, settings: Settings) -> None:
    root = resolve_workspace_root(settings.workspace_root)
    ws = open_workspace(root, project_id)
    manifest = ws.load_manifest()
    report = manifest.get("run_report", {})
    status = manifest.get("status", "unknown")
    console.print(f"[bold green]{status}[/bold green] {project_id}")
    console.print(
        f"  sections: {len(manifest.get('sections', []))}  "
        f"issues: {manifest.get('issues', 0)}  "
        f"revisions: {manifest.get('revision_rounds', 0)}"
    )
    outputs = manifest.get("exports", {})
    if outputs:
        console.print("  outputs: " + ", ".join(f"{k}={v}" for k, v in outputs.items()))
    console.print(
        f"  cost: ${report.get('cost_usd', 0)}  "
        f"tokens: {report.get('tokens_in', 0)}/{report.get('tokens_out', 0)}  "
        f"tool_calls: {report.get('tool_calls', 0)}"
    )
    console.print(f"  dir: {ws.dir}")
    for warning in manifest.get("warnings", []):
        console.print(f"  [yellow]warning:[/yellow] {warning}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
