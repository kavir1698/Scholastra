"""A trivial in-repo MCP stdio server used by tests and the offline smoke path.

Run with ``python -m theo_explore.tools.stub_server``. It exposes deterministic
read-only "research" tools so the MCP plumbing can be tested end-to-end without
network access or third-party servers.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stub-research")

_SAMPLE = [
    {
        "id": "arxiv:2401.00001",
        "kind": "preprint",
        "title": "A Survey of the Topic",
        "authors": ["A. Researcher", "B. Scholar"],
        "year": 2024,
        "url": "https://arxiv.org/abs/2401.00001",
        "arxiv_id": "2401.00001",
    },
    {
        "id": "web:example.com/overview",
        "kind": "web",
        "title": "An Overview of the Topic",
        "authors": [],
        "year": 2023,
        "url": "https://example.com/overview",
    },
    {
        "id": "pmid:12345678",
        "kind": "paper",
        "title": "Biomedical Perspectives on the Topic",
        "authors": ["C. Clinician"],
        "year": 2022,
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "pmid": "12345678",
    },
]


@mcp.tool()
def search(query: str) -> list[dict]:
    """Search the stub literature corpus for a query."""
    return [dict(item, query=query) for item in _SAMPLE]


@mcp.tool()
def get_paper(arxiv_id: str) -> dict:
    """Return metadata + an abstract for an arXiv paper id."""
    return {
        "id": f"arxiv:{arxiv_id}",
        "kind": "preprint",
        "title": f"Paper {arxiv_id}",
        "authors": ["A. Researcher"],
        "year": 2024,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_id": arxiv_id,
        "abstract": "This stub abstract states a finding about the topic.",
    }


@mcp.tool()
def fetch(url: str) -> str:
    """Fetch a URL and return Markdown (stub returns canned text)."""
    return f"# Stub page\n\nContent retrieved from {url} about the topic."


def main() -> None:
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run()


if __name__ == "__main__":
    main()
