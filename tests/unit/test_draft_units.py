"""Recursive drafting-unit traversal."""

from __future__ import annotations

from theo_explore.graph.subgraphs.draft_section import walk_units


def test_walk_units_depth_first_order_and_leaf_flags():
    units = [
        {
            "id": "a",
            "title": "A",
            "children": [
                {"id": "a1", "title": "A1"},
                {"id": "a2", "title": "A2"},
            ],
        },
        {"id": "b", "title": "B"},
    ]
    walked = list(walk_units(units))
    assert [(depth, unit["id"], has_children) for depth, unit, has_children in walked] == [
        (1, "a", True),
        (2, "a1", False),
        (2, "a2", False),
        (1, "b", False),
    ]
