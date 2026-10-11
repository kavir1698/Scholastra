"""Live progress reporter output."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from theo_explore.progress import ProgressReporter


def _reporter(*, enabled: bool = True) -> tuple[ProgressReporter, StringIO]:
    buf = StringIO()
    console = Console(file=buf, width=120, no_color=True)
    return ProgressReporter(console, enabled=enabled), buf


def test_reporter_prints_phases_and_branches():
    reporter, buf = _reporter()
    reporter.banner(
        topic="free will",
        doc_type="article",
        preset="article-short",
        project_dir="ws/proj",
        strong_model="openrouter/a",
        cheap_model="openrouter/b",
    )
    reporter.update("outline", {"outline": {"sections": [{"id": "intro"}, {"id": "body"}]}})
    reporter.update(
        "research_section",
        {"evidence": {"intro": {"claims": [{"id": "c0"}], "coverage": "high"}}},
    )
    reporter.update(
        "draft_section",
        {"drafts": {"intro": {"word_count": 120, "cite_keys": ["doi:1"], "revision": 0}}},
    )
    reporter.update(
        "verify_citations",
        {"verification": {"intro": {"unresolved_cites": []}}, "issues": []},
    )
    reporter.finished({"status": "completed"})

    out = buf.getvalue()
    assert "Outline" in out and "2 sections" in out
    assert "researched" in out and "intro" in out
    assert "drafted" in out and "120 words" in out
    assert "verified 1 sections" in out
    assert "completed" in out


def test_reporter_revision_line():
    reporter, buf = _reporter()
    reporter.update(
        "draft_section",
        {"drafts": {"intro": {"word_count": 10, "cite_keys": [], "revision": 1}}},
    )
    assert "revised" in buf.getvalue()


def test_disabled_reporter_is_silent():
    reporter, buf = _reporter(enabled=False)
    reporter.banner(
        topic="t",
        doc_type="article",
        preset="article-short",
        project_dir="ws/p",
        strong_model="o/a",
        cheap_model="o/b",
    )
    reporter.update("scope", {"scope": {"refined_question": "q"}})
    reporter.finished({"status": "completed"})
    assert buf.getvalue() == ""
