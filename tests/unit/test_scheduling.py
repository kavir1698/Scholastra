"""Dependency scheduling, topological waves and outline repair."""

from __future__ import annotations

import pytest

from scholastra.config import load_preset
from scholastra.graph.validation import ready_sections, sanitize_outline, topological_waves
from scholastra.models.schemas import Outline, Section


def test_topological_waves_orders_parallel_groups():
    waves = topological_waves({"a": [], "b": [], "c": ["a", "b"], "d": ["c"]})
    assert waves[0] == ["a", "b"]
    assert waves[1] == ["c"]
    assert waves[2] == ["d"]


def test_topological_waves_detects_cycle():
    with pytest.raises(ValueError, match="cycle"):
        topological_waves({"a": ["b"], "b": ["a"]})


def test_topological_waves_unknown_dep():
    with pytest.raises(ValueError, match="unknown"):
        topological_waves({"a": ["missing"]})


def test_ready_sections():
    sections = [
        Section(id="a", title="A", purpose="p"),
        Section(id="b", title="B", purpose="p"),
        Section(id="c", title="C", purpose="p", depends_on=["a", "b"]),
    ]
    assert sorted(ready_sections(sections, set())) == ["a", "b"]
    assert ready_sections(sections, {"a", "b"}) == ["c"]


def test_sanitize_outline_removes_unknown_and_breaks_cycle():
    outline = Outline(
        title="T",
        sections=[
            Section(id="a", title="A", purpose="p", depends_on=["ghost"]),
            Section(id="b", title="B", purpose="p", depends_on=["c"]),
            Section(id="c", title="C", purpose="p", depends_on=["b"]),
        ],
    )
    repaired, warnings = sanitize_outline(outline)
    deps = {s.id: s.depends_on for s in repaired.sections}
    assert "ghost" not in deps["a"]
    topological_waves(deps)  # does not raise
    assert warnings


def test_sanitize_outline_slugifies_ids_and_remaps_deps():
    outline = Outline(
        title="T",
        sections=[
            Section(id="Intro Section!!", title="Intro", purpose="p"),
            Section(id="../../evil", title="Evil", purpose="p", depends_on=["Intro Section!!"]),
        ],
    )
    repaired, warnings = sanitize_outline(outline)
    ids = [s.id for s in repaired.sections]
    assert all(".." not in i and "/" not in i and "\\" not in i for i in ids)
    assert repaired.sections[1].depends_on == [ids[0]]
    assert warnings


def test_preset_differences():
    short = load_preset("article-short")
    book = load_preset("mini-book")
    assert book["target_words"]["max"] > short["target_words"]["max"]
    assert book["section_budget"]["max_sections"] > short["section_budget"]["max_sections"]
