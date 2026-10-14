"""Workspace path safety: project-id validation and write containment."""

from __future__ import annotations

import pytest

from scholastra.io.artifacts import open_workspace, validate_project_id


@pytest.mark.parametrize(
    "bad",
    ["../escape", "..", ".", "a/b", "a\\b", "C:\\x", "with space", ""],
)
def test_validate_project_id_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        validate_project_id(bad)


def test_validate_project_id_accepts_slug():
    assert validate_project_id("my-proj_1.0") == "my-proj_1.0"


def test_open_workspace_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        open_workspace(tmp_path, "../escape")


def test_write_outside_workspace_rejected(tmp_path):
    ws = open_workspace(tmp_path, "p1")
    with pytest.raises(ValueError):
        ws.write_text("../escape.md", "x")
    with pytest.raises(ValueError):
        ws.write_json("artifacts/../../escape.json", {})
