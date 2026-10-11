"""Tiered scholarly source policy.

Source server names do not reliably tell us whether a record is full text, an
abstract, book metadata or orientation-only web content. This module therefore
classifies :class:`Source` records *after* normalisation, using bibliographic
metadata and the retrieved domain, and decides which tiers may support
substantive claims.

Tiers (see ``config/source_policy.yaml``):

- Tier 1: peer-reviewed primary/review articles and canonical books/classics.
- Tier 2: reputable preprints/working papers and scholarly repositories.
- Tier 3: authoritative reference works (encyclopedias, handbooks, curated web).
- Tier 4: orientation-only web/Wikipedia; never sufficient for a substantive claim.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from theo_explore.models.schemas import AccessMode, Source, SourceRole, SourceTier

TIER_LABELS: dict[int, str] = {
    1: "peer-reviewed primary/review article or canonical book/classic",
    2: "reputable preprint/working paper or scholarly repository record",
    3: "authoritative reference work (encyclopedia, handbook, curated web)",
    4: "orientation-only web/Wikipedia content",
}

_CANONICAL_PUBLISHERS = (
    "university press",
    "oxford",
    "cambridge",
    "harvard",
    "mit press",
    "princeton",
    "yale",
    "routledge",
    "wiley",
    "springer",
    "elsevier",
    "palgrave",
    "sage",
    "academic press",
    "penguin",
    "clarendon",
)

_ORIENTATION_DOMAINS = (
    "wikipedia.org",
    "wikimedia.org",
    "wikihow.com",
    "medium.com",
    "reddit.com",
    "quora.com",
    "blogspot.com",
    "wordpress.com",
    "substack.com",
    "fandom.com",
)

_AUTHORITATIVE_WEB_DOMAINS = (
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "plato.stanford.edu",
    "iep.utm.edu",
    "britannica.com",
    "nobelprize.org",
)


def _host(url: str | None) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url if "://" in url else f"https://{url}")
    except ValueError:
        return ""
    return parts.netloc.lower().removeprefix("www.")


def _domain_matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_canonical_book(source: Source) -> bool:
    if source.peer_reviewed:
        return True
    venue = (source.venue or "").lower()
    if any(publisher in venue for publisher in _CANONICAL_PUBLISHERS):
        return True
    if source.year is not None and source.year <= 1950:
        return True
    return bool(source.doi)


def _base_tier(source: Source, policy: dict[str, Any] | None = None) -> SourceTier:
    policy = policy or {}
    host = _host(source.url or source.retrieved_url)
    kind = source.kind
    if kind == "paper":
        return 1
    if kind == "preprint":
        return 2
    if kind == "book":
        return 1 if is_canonical_book(source) else 3
    if kind == "encyclopedia":
        orientation = tuple(policy.get("orientation_domains") or _ORIENTATION_DOMAINS)
        return 4 if _domain_matches(host, orientation) else 3
    if kind == "web":
        orientation = tuple(policy.get("orientation_domains") or _ORIENTATION_DOMAINS)
        if _domain_matches(host, orientation):
            return 4
        authoritative = tuple(policy.get("authoritative_domains") or _AUTHORITATIVE_WEB_DOMAINS)
        if _domain_matches(host, authoritative) or host.endswith(".gov") or host.endswith(".edu"):
            return 3
        return 4
    return 4


def classify_tier(source: Source, policy: dict[str, Any] | None = None) -> SourceTier:
    """Return the tier (1-4) for a normalised source record.

    An explicit ``tier`` from (untrusted) tool metadata is only honoured when it
    is *not more authoritative* than the tier derived from bibliographic metadata:
    the effective tier is the weaker of the two. This prevents a source server from
    self-declaring Tier 1 to satisfy the source policy.
    """
    base = _base_tier(source, policy)
    if source.tier is None:
        return base
    return max(base, source.tier)


def infer_access_mode(source: Source) -> AccessMode:
    if source.access_mode != "unknown":
        return source.access_mode
    if source.full_text_available:
        return "full_text"
    if source.abstract_only:
        return "abstract"
    if source.kind == "web":
        return "web"
    if source.kind == "encyclopedia":
        return "encyclopedia"
    if source.kind == "book":
        return "book_metadata"
    return "metadata"


def classify_role(source: Source, tier: SourceTier) -> SourceRole:
    if source.role not in (None, "unknown"):
        return source.role
    mode = infer_access_mode(source)
    if mode in {"abstract", "metadata"}:
        return "bibliographic"
    if source.bibliographic_confidence is not None and source.bibliographic_confidence < 0.5:
        # Low-confidence bibliographic metadata may be used for discovery only.
        return "bibliographic"
    if tier == 4:
        return "orientation"
    if source.kind in {"book", "encyclopedia"}:
        return "reference"
    if tier == 1:
        return "primary"
    return "corroborating"


def apply_source_policy(source: Source, policy: dict[str, Any] | None = None) -> Source:
    """Fill in tier/role/access metadata without overwriting explicit values."""
    tier = classify_tier(source, policy)
    mode = infer_access_mode(source)
    role = classify_role(source.model_copy(update={"tier": tier, "access_mode": mode}), tier)
    return source.model_copy(
        update={
            "tier": tier,
            "role": role,
            "access_mode": mode,
            "abstract_only": source.abstract_only or mode == "abstract",
        }
    )


def tier_allows_claim_support(tier: int | None, policy: dict[str, Any] | None = None) -> bool:
    """Whether a source of ``tier`` may support a substantive factual claim."""
    if tier is None:
        return True
    policy = policy or {}
    allowed = set(policy.get("allowed_tiers_for_claims") or [1, 2, 3])
    if tier == 4 and policy.get("allow_orientation_support"):
        return True
    return tier in allowed


def policy_lines(policy: dict[str, Any] | None = None) -> list[str]:
    """Human/LLM-readable lines describing the active source policy."""
    policy = policy or {}
    lines: list[str] = []
    labels: dict = dict(policy.get("tier_labels") or TIER_LABELS)
    for tier in (1, 2, 3, 4):
        label = labels.get(tier) or labels.get(str(tier)) or TIER_LABELS[tier]
        lines.append(f"Tier {tier}: {label}")
    allowed = policy.get("allowed_tiers_for_claims") or [1, 2, 3]
    lines.append(
        "Claims of substance must be supported by "
        + ", ".join(f"Tier {t}" for t in allowed)
        + " sources."
    )
    if policy.get("allow_orientation_support"):
        lines.append("Orientation-only sources MAY be used for support in this run.")
    else:
        lines.append("Orientation-only (Tier 4) sources may NOT be the sole support for a claim.")
    if policy.get("prefer_full_text", True):
        lines.append("Prefer full-text extracts; mark abstract-only or metadata-only records.")
    if policy.get("require_corroborating", True):
        lines.append("Seek an independent corroborating source alongside the primary one.")
    return lines


def source_quality_summary(sources: dict[str, Source] | list[Source]) -> dict[str, int]:
    records = sources.values() if isinstance(sources, dict) else sources
    summary = {
        "total": 0,
        "tier_1": 0,
        "tier_2": 0,
        "tier_3": 0,
        "tier_4": 0,
        "full_text": 0,
        "abstract_only": 0,
        "bibliographic_only": 0,
        "orientation": 0,
        "low_confidence": 0,
    }
    for source in records:
        summary["total"] += 1
        tier = classify_tier(source)
        summary[f"tier_{tier}"] = summary.get(f"tier_{tier}", 0) + 1
        if source.full_text_available or infer_access_mode(source) == "full_text":
            summary["full_text"] += 1
        if source.abstract_only:
            summary["abstract_only"] += 1
        if classify_role(source, tier) == "bibliographic":
            summary["bibliographic_only"] += 1
        if tier == 4:
            summary["orientation"] += 1
        if source.bibliographic_confidence is not None and source.bibliographic_confidence < 0.5:
            summary["low_confidence"] += 1
    return summary
