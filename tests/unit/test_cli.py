"""End-to-end CLI run with the fake backend (offline)."""

from __future__ import annotations

from typer.testing import CliRunner

from scholastra.cli import app


def test_cli_run_offline(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "a tiny offline topic",
            "--auto",
            "--fake-llm",
            "--no-export",
            "--project-id",
            "cli-smoke",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = tmp_path / "cli-smoke" / "manifest.json"
    final_md = tmp_path / "cli-smoke" / "artifacts" / "final.md"
    report = tmp_path / "cli-smoke" / "run_report.json"
    assert manifest.exists()
    assert final_md.exists()
    assert report.exists()
    assert "completed" in manifest.read_text(encoding="utf-8")


def test_cli_status_unknown_project(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["status", "nope", "--workspace", str(tmp_path)])
    assert result.exit_code == 1
