"""Draft sentence/claim parsing and factual-claim heuristics.

The citation verifier works at the sentence level: each factual claim unit is a
sentence (or list-item segment) carrying its inline citation group. Parsing is
deterministic and side-effect free so the same units can be re-derived for
diagnostics and tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scholastra.citations import iter_citation_keys
from scholastra.utils import word_count

_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "cf.",
    "vs.",
    "fig.",
    "figs.",
    "no.",
    "nos.",
    "dr.",
    "prof.",
    "mr.",
    "mrs.",
    "ms.",
    "ph.d.",
    "approx.",
    "esp.",
    "etc.",
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

# Abbreviations that almost never end a sentence: their dot is always protected.
_ALWAYS_CONTINUE = frozenset(
    {
        "e.g.",
        "i.e.",
        "cf.",
        "vs.",
        "dr.",
        "prof.",
        "mr.",
        "mrs.",
        "ms.",
        "ph.d.",
        "approx.",
        "esp.",
    }
)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_DOT_SENTINEL = "\x00DOT\x00"

_INTERPRETIVE = re.compile(
    r"\b(we|i|the author|this (?:essay|paper|document|review|section)|in (?:our|my) view|"
    r"arguably|it seems|may|might|could|suggests? that|we believe|interpretation|"
    r"limitations?|open questions?|future work)\b",
    re.IGNORECASE,
)
_FACTUAL = re.compile(
    r"(\d|%|\b(shows?|found|reports?|demonstrat\w*|proves?|estimat\w*|measures?|"
    r"increas\w*|decreas\w*|causes?|correlat\w*|is|are|was|were|has|have|had)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SentenceUnit:
    """A parsed sentence/claim unit and the citations attached to it."""

    id: str
    text: str
    cite_keys: tuple[str, ...] = ()
    has_citation: bool = False
    subsection: str | None = None


@dataclass
class _Builder:
    section_id: str
    subsection: str | None = None
    units: list[SentenceUnit] = field(default_factory=list)

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        keys = tuple(sorted(set(iter_citation_keys(text))))
        index = len(self.units) + 1
        self.units.append(
            SentenceUnit(
                id=f"{self.section_id}:s{index}",
                text=text,
                cite_keys=keys,
                has_citation=bool(keys),
                subsection=self.subsection,
            )
        )


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, tolerating decimals and common abbreviations.

    An abbreviation's dot is protected only when it does *not* end a sentence, so
    ``"… etc. The rate rose."`` still splits into two claim units.
    """
    if not text:
        return []
    protected = text
    for abbr in sorted(_ABBREVIATIONS, key=len, reverse=True):
        lookahead = "" if abbr in _ALWAYS_CONTINUE else r"(?!\s+[\"'(\[]?[A-Z])"
        pattern = re.compile(r"(?<![\w])(?i:" + re.escape(abbr) + r")" + lookahead)
        protected = pattern.sub(
            lambda match: match.group(0).replace(".", _DOT_SENTINEL), protected
        )
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [part.replace(_DOT_SENTINEL, ".").strip() for part in parts if part.strip()]


def _segments(markdown: str) -> list[str]:
    segments: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        segments.append(_LIST_MARKER.sub("", stripped))
    return segments


def parse_sentences(
    markdown: str, *, section_id: str = "", subsection: str | None = None
) -> list[SentenceUnit]:
    """Return the ordered claim units of ``markdown`` with citation locations."""
    builder = _Builder(section_id=section_id, subsection=subsection)
    for segment in _segments(markdown):
        for sentence in split_sentences(segment):
            builder.add(sentence)
    return builder.units


def looks_factual(text: str) -> bool:
    """Heuristic: does this sentence assert something verifiable?"""
    return bool(_FACTUAL.search(text or ""))


def is_interpretive(text: str) -> bool:
    """Heuristic: is this sentence explicitly marked as interpretation/synthesis?"""
    return bool(_INTERPRETIVE.search(text or ""))


def uncited_factual_units(units: list[SentenceUnit], *, min_words: int = 6) -> list[SentenceUnit]:
    """Claim units that assert a fact but carry no citation and are not interpretive."""
    return [
        unit
        for unit in units
        if not unit.has_citation
        and looks_factual(unit.text)
        and not is_interpretive(unit.text)
        and word_count(unit.text) >= min_words
    ]
