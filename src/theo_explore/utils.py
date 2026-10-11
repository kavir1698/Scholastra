"""Small shared helpers: stable ids, slugs, JSON extraction, word counting."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def slugify(text: str, *, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len] or "section"


def stable_id(*parts: str, prefix: str = "", length: int = 12) -> str:
    """Deterministic short id from arbitrary parts (used for SourceId / Issue ids)."""
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles: pure JSON, fenced ```json blocks, and text with a leading/trailing
    prose wrapper around the first balanced ``{...}`` / ``[...]`` span.
    """
    if text is None:
        raise ValueError("no content to parse")
    candidate = text.strip()

    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(candidate)):
            ch = candidate[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    span = candidate[start : idx + 1]
                    try:
                        return json.loads(span)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not extract JSON from response: {candidate[:200]!r}")
