"""Outline validation and topological scheduling helpers."""

from __future__ import annotations

from theo_explore.models.schemas import Outline, Section
from theo_explore.utils import slugify


def topological_waves(deps: dict[str, list[str]]) -> list[list[str]]:
    """Return dependency waves. Raises ValueError on cycles or unknown deps."""
    remaining = {node: set(d) for node, d in deps.items()}
    for node, node_deps in remaining.items():
        unknown = node_deps - remaining.keys()
        if unknown:
            raise ValueError(f"section '{node}' depends on unknown sections: {sorted(unknown)}")
    waves: list[list[str]] = []
    done: set[str] = set()
    while remaining:
        ready = sorted(n for n, d in remaining.items() if d <= done)
        if not ready:
            raise ValueError(f"dependency cycle among sections: {sorted(remaining)}")
        waves.append(ready)
        done.update(ready)
        for node in ready:
            remaining.pop(node)
    return waves


def sanitize_outline(outline: Outline) -> tuple[Outline, list[str]]:
    """Repair an outline: slugify ids, remap/clean deps, break cycles.

    Section ids come from the LLM and are used as file-path segments, so they
    are forced to a safe slug and dependencies are remapped to the new ids.
    """
    warnings: list[str] = []
    section_map: dict[str, Section] = {}
    ordered_ids: list[str] = []
    id_map: dict[str, str] = {}
    for index, section in enumerate(outline.sections):
        raw = section.id or f"section-{index + 1}"
        base = slugify(raw, max_len=48) or f"section-{index + 1}"
        sid = base
        if sid in section_map:
            sid = f"{base}-{index + 1}"
            warnings.append(f"duplicate section id renamed to '{sid}'")
        if sid != raw:
            warnings.append(f"section id sanitized: '{raw}' -> '{sid}'")
        id_map.setdefault(raw, sid)
        id_map.setdefault(sid, sid)
        section = section.model_copy(update={"id": sid})
        section_map[sid] = section
        ordered_ids.append(sid)

    known = set(section_map)
    for sid in ordered_ids:
        section = section_map[sid]
        clean_deps: list[str] = []
        for dep in section.depends_on:
            mapped = id_map.get(dep, slugify(dep, max_len=48) or dep)
            if mapped == sid:
                warnings.append(f"section '{sid}' depended on itself; removed")
            elif mapped not in known:
                warnings.append(f"section '{sid}' depended on unknown '{dep}'; removed")
            elif mapped not in clean_deps:
                clean_deps.append(mapped)
        section_map[sid] = section.model_copy(update={"depends_on": clean_deps})

    # Break any remaining cycle by dropping the offending edges.
    while True:
        try:
            topological_waves({sid: section_map[sid].depends_on for sid in ordered_ids})
            break
        except ValueError:
            # Drop one edge deterministically from the lexicographically smallest offender.
            for sid in ordered_ids:
                deps = section_map[sid].depends_on
                # A node that participates in a cycle still has deps that are not yet done.
                offending = [d for d in deps if d > sid or _reaches(section_map, d, sid)]
                if offending:
                    drop = offending[0]
                    warnings.append(f"broke dependency cycle: removed '{sid}' -> '{drop}'")
                    section_map[sid] = section_map[sid].model_copy(
                        update={"depends_on": [d for d in deps if d != drop]}
                    )
                    break
            else:  # pragma: no cover - defensive
                raise ValueError("unable to break dependency cycle") from None

    repaired = outline.model_copy(
        update={"sections": [section_map[sid] for sid in ordered_ids]}
    )
    return repaired, warnings


def _reaches(section_map: dict[str, Section], start: str, target: str) -> bool:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        section = section_map.get(node)
        if section:
            stack.extend(section.depends_on)
    return False


def ready_sections(
    sections: list[Section], drafted: set[str]
) -> list[str]:
    """Section ids whose dependencies are all drafted and that are not yet drafted."""
    known = {s.id for s in sections}
    ready: list[str] = []
    for section in sections:
        if section.id in drafted:
            continue
        unmet = [d for d in section.depends_on if d in known and d not in drafted]
        if not unmet:
            ready.append(section.id)
    return ready
