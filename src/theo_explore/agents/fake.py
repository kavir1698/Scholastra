"""Deterministic, network-free LLM backend for tests and ``--fake-llm``.

The :class:`ScriptedResponder` returns schema-valid objects keyed by node name,
parsing the prompt for the small conventions used by the research/draft prompts
(``CANDIDATE SOURCES`` and ``AVAILABLE CITE KEYS``). This lets a full graph run
execute end-to-end without API keys.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from theo_explore.agents.llm import RawCompletion

Responder = Callable[[str, str, list[dict[str, str]]], dict[str, Any]]

_CANDIDATE_RE = re.compile(r"^\s*-\s*(\S+)\s*::", re.MULTILINE)
_CITE_RE = re.compile(r"AVAILABLE CITE KEYS:[ \t]*(.*)", re.IGNORECASE)


class FakeBackend:
    """An :class:`LLMBackend` that never touches the network."""

    def __init__(self, responder: Responder | None = None, *, cost_per_call: float = 0.0) -> None:
        self.responder = responder or ScriptedResponder()
        self.cost_per_call = cost_per_call
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def complete(
        self,
        *,
        node: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_object: bool,
        json_schema: dict[str, Any] | None,
        reasoning_effort: str | None = None,
    ) -> RawCompletion:
        self.calls.append((node, messages))
        system = messages[0]["content"] if messages else ""
        result = self.responder(node, system, messages)
        text = result if isinstance(result, str) else json.dumps(result)
        return RawCompletion(
            text=text, tokens_in=10, tokens_out=20, cost_usd=self.cost_per_call, model=model
        )


def _fake_subsections(sid: str, title: str, words: int, preset: str) -> list[dict[str, Any]]:
    core = {
        "id": f"{sid}-a",
        "title": f"{title}: core",
        "purpose": "Introduce the core idea.",
        "key_questions": ["What is the core idea?"],
        "target_words": max(words // 2, 50),
    }
    if preset != "mini-book":
        return [core]
    core["children"] = [
        {
            "id": f"{sid}-a-1",
            "title": f"{title}: core I",
            "purpose": "Definition and scope.",
            "key_questions": ["What is it?"],
            "target_words": max(words // 4, 40),
        },
        {
            "id": f"{sid}-a-2",
            "title": f"{title}: core II",
            "purpose": "Key distinctions.",
            "key_questions": ["How is it distinguished?"],
            "target_words": max(words // 4, 40),
        },
    ]
    context = {
        "id": f"{sid}-b",
        "title": f"{title}: context",
        "purpose": "Set context.",
        "key_questions": ["Why does it matter?"],
        "target_words": max(words // 3, 50),
    }
    return [core, context]


class ScriptedResponder:
    """Produces deterministic, schema-valid shapes for each node."""

    def __call__(self, node: str, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        combined = "\n".join(m.get("content", "") for m in messages)
        handler = getattr(self, f"_node_{node}", None)
        if handler is not None:
            return handler(combined)
        return self._node_default(combined)

    # -- node handlers -----------------------------------------------------
    def _node_default(self, prompt: str) -> dict[str, Any]:
        return {}

    def _node_scope(self, prompt: str) -> dict[str, Any]:
        return {
            "refined_question": "How does the topic behave under stated assumptions, and what "
            "does the current literature establish?",
            "sub_questions": [
                "What is the topic?",
                "What does prior work establish?",
                "What remains open?",
            ],
            "in_scope": ["Definitions", "Prior work", "Open problems"],
            "out_of_scope": ["Empirical replication", "Policy prescriptions"],
            "audience": "informed general reader",
            "assumptions": ["At least one search provider is configured."],
            "length_preset": "article-short",
        }

    def _node_outline(self, prompt: str) -> dict[str, Any]:
        def section(
            sid: str, title: str, deps: list[str], words: int, domains: list[str]
        ) -> dict[str, Any]:
            return {
                "id": sid,
                "title": title,
                "purpose": f"Establish {title.lower()}.",
                "key_questions": [f"What is {title.lower()}?"],
                "target_words": words,
                "depends_on": deps,
                "source_domains": domains,
                "required": True,
                "subsections": _fake_subsections(sid, title, words, preset),
            }

        preset = _preset(prompt)
        research = [
            section("introduction", "Introduction", [], 300, ["exa", "wikipedia"]),
            section("key-concepts", "Key Concepts", ["introduction"], 400, ["arxiv", "wikipedia"]),
            section("open-problems", "Open Problems", ["introduction"], 400, ["arxiv", "pubmed"]),
        ]
        if preset == "mini-book":
            research.insert(
                2,
                section(
                    "historical-background",
                    "Historical Background",
                    ["introduction"],
                    500,
                    ["wikipedia", "exa"],
                ),
            )
            research.append(
                section("prior-work", "Prior Work", ["introduction"], 500, ["arxiv"])
            )
            research.append(
                section(
                    "debates",
                    "Current Debates",
                    ["key-concepts", "prior-work"],
                    450,
                    ["arxiv", "exa"],
                )
            )
        research_ids = [s["id"] for s in research]

        sections = [
            *research,
            section("synthesis", "Synthesis", research_ids, 350, ["exa"]),
            section("conclusions", "Conclusions", ["synthesis"], 200, []),
        ]
        return {
            "title": "A Research-Grounded Exploration",
            "abstract_target_words": 120,
            "sections": sections,
        }

    def _node_contract(self, prompt: str) -> dict[str, Any]:
        section_ids = re.findall(r"^- (\S+):", prompt, re.MULTILINE)
        glossary = [
            {
                "term": "the topic",
                "definition": "the theoretical subject under investigation",
                "aliases": ["the subject"],
                "forbidden_variants": ["the thing"],
                "owner_section": section_ids[0] if section_ids else None,
                "defined_once": True,
            }
        ]
        contributions = [
            {"section_id": sid, "role": "develop", "expected_contribution": f"Develop {sid}."}
            for sid in section_ids
        ]
        return {
            "thesis": "The topic is best understood through its core concepts and open problems.",
            "argument_arc": ["Define the topic", "Survey prior work", "Identify open problems"],
            "glossary": glossary,
            "handoffs": [],
            "contributions": contributions,
            "repetition_budget": "Define each concept once; do not re-introduce it later.",
            "terminology_rules": ["Use the canonical glossary terms."],
            "source_policy": ["Prefer peer-reviewed and canonical sources."],
            "notes": "",
        }

    def _node_editor(self, prompt: str) -> dict[str, Any]:
        return {"findings": [], "edits": [], "notes": ""}

    def _node_plan_research(self, prompt: str) -> dict[str, Any]:
        return {
            "section_id": _section_id(prompt),
            "queries": ["overview", "recent advances", "survey"],
            "source_domains": ["arxiv", "exa"],
            "max_tool_calls": 4,
            "max_sources": 4,
            "wall_clock_seconds": 60,
        }

    def _node_research_plan(self, prompt: str) -> dict[str, Any]:
        step = _int_match(prompt, r"STEP:\s*(\d+)", 0)
        tools = _list_match(prompt, r"ALLOWED TOOLS:\s*(.+)", sep=",")
        queries = _list_match(prompt, r"QUERIES:\s*(.+)", sep="|")
        if step >= 1 or not tools or not queries:
            return {"tool_calls": [], "done": True, "reasoning": "coverage sufficient"}
        tool = next((t for t in tools if "search" in t), tools[0])
        return {
            "tool_calls": [{"tool": tool, "arguments": {"query": queries[0]}}],
            "done": False,
            "reasoning": "initial literature search",
        }

    def _node_research(self, prompt: str) -> dict[str, Any]:
        ids = [m.group(1) for m in _CANDIDATE_RE.finditer(prompt)]
        if not ids:
            ids = ["web:example.com/overview"]
        claims = [
            {
                "id": f"c{i}",
                "text": f"Finding {i + 1} about the topic, grounded in the retrieved source.",
                "source_ids": [ids[i % len(ids)]],
                "excerpt": "Verbatim excerpt supporting the finding.",
                "locator": "section 1",
            }
            for i in range(min(2, max(len(ids), 1)))
        ]
        return {
            "section_id": _section_id(prompt),
            "claims": claims,
            "notes": "Synthetic evidence produced by the fake backend.",
            "coverage": "medium",
            "tool_calls": len(ids),
        }

    def _node_draft(self, prompt: str) -> dict[str, Any]:
        match = _CITE_RE.search(prompt)
        keys = [k.strip() for k in match.group(1).split(",")] if match else []
        keys = [k for k in keys if k]
        body = "This section develops the topic with grounded prose. "
        if keys:
            body += " ".join(f"Supported claim [@{k}]." for k in keys)
        else:
            body += "No sources were available, so no citations are made."
        title = _subsection_title(prompt) or _section_title(prompt)
        return {
            "section_id": _section_id(prompt),
            "title": title,
            "markdown": f"## {title}\n\n{body}",
            "summary": "A grounded summary of the section.",
            "cite_keys": keys,
            "word_count": len(body.split()),
            "revision": _revision(prompt),
        }

    def _node_verify(self, prompt: str) -> dict[str, Any]:
        # The mechanical verifier does the real work; the LLM pass adds nothing here.
        return {"unsupported_claims": [], "notes": ""}

    def _node_critique(self, prompt: str) -> dict[str, Any]:
        return {"issues": []}

    def _node_synthesize(self, prompt: str) -> dict[str, Any]:
        return {
            "markdown": "## Synthesis\n\nAcross sections, the evidence converges on several "
            "themes while leaving open questions for future work.",
            "cite_keys": [],
        }


def _section_id(prompt: str) -> str:
    match = re.search(r"SECTION[_ ]ID:\s*(\S+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"section_id['\"]?\s*[:=]\s*['\"]?([\w-]+)", prompt)
    return match.group(1) if match else "section"


def _section_title(prompt: str) -> str:
    match = re.search(r"SECTION[_ ]TITLE:\s*(.+)", prompt)
    if match:
        return match.group(1).strip()
    return "Section"


def _subsection_title(prompt: str) -> str | None:
    match = re.search(r"SUBSECTION[_ ]TITLE:\s*(.+)", prompt)
    return match.group(1).strip() if match else None


def _revision(prompt: str) -> int:
    match = re.search(r"REVISION[_ ]ROUND:\s*(\d+)", prompt)
    return int(match.group(1)) if match else 0


def _int_match(prompt: str, pattern: str, default: int) -> int:
    match = re.search(pattern, prompt)
    if not match:
        return default
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return default


def _list_match(prompt: str, pattern: str, *, sep: str) -> list[str]:
    match = re.search(pattern, prompt)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(sep) if item.strip()]


def _preset(prompt: str) -> str:
    match = re.search(r"Length preset:\s*(\S+)", prompt)
    return match.group(1) if match else "article-short"
