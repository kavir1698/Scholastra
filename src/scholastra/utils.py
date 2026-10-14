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


_BRACKETS = {"{": "}", "[": "]"}


def _balanced_span(text: str, start: int) -> str | None:
    """Return the balanced ``{...}``/``[...]`` span at ``start``, or None if unterminated.

    Bracket characters inside JSON strings are ignored.
    """
    opener = text[start]
    closer = _BRACKETS[opener]
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
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
                return text[start : idx + 1]
    return None


def _incomplete(candidate: str) -> ValueError:
    return ValueError(
        "incomplete JSON response (unbalanced brackets — likely truncated): "
        f"{candidate[:200]!r}"
    )


def _loads_or_raise(span: str) -> Any:
    try:
        return json.loads(span)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in response ({exc}): {span[:200]!r}") from exc


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles: pure JSON, fenced ```json blocks, and text with a leading/trailing
    prose wrapper around the first balanced ``{...}`` / ``[...]`` span.

    An object opener is preferred so that bracketed prose before the real object
    (e.g. ``"[1] Here is the JSON: {...}"``) does not shadow it. The one thing we
    never do is fall back to an array nested inside an *unterminated* object:
    that is what previously turned a truncated object into a silently wrong-shaped
    list (which then failed schema validation with a misleading "input should be a
    dictionary" error).
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

    obj_start = candidate.find("{")
    arr_start = candidate.find("[")
    if obj_start == -1 and arr_start == -1:
        raise ValueError(f"no JSON object or array found: {candidate[:200]!r}")

    if obj_start != -1:
        obj_span = _balanced_span(candidate, obj_start)
        if obj_span is not None:
            return _loads_or_raise(obj_span)
        # The object is unterminated. Only a *top-level* array that begins before
        # the object is a valid alternative; an array nested inside the truncated
        # object must not be returned.
        if arr_start != -1 and arr_start < obj_start:
            arr_span = _balanced_span(candidate, arr_start)
            if arr_span is not None:
                return _loads_or_raise(arr_span)
        raise _incomplete(candidate)

    arr_span = _balanced_span(candidate, arr_start)
    if arr_span is None:
        raise _incomplete(candidate)
    return _loads_or_raise(arr_span)
