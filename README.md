# Theoretical Exploration Bot

A Python, **LangGraph**-orchestrated agentic system that turns a theoretical question into a
research-grounded **article** or **mini-book**. It decomposes the question, plans a scholarly
structure, deep-researches each section through **MCP** tools, drafts in parallel where sections
are independent and sequentially where they depend on each other, runs citation verification +
adversarial critique + a bounded revision loop, optionally synthesises the whole, and exports
Markdown → PDF/DOCX with Pandoc.

> Output is a **survey/review-grade draft** requiring human review. It is not a guarantee of
> publishable novelty, and it never submits anywhere automatically.

---

## Requirements

| Tool | Why | Notes |
|---|---|---|
| Python 3.12 | runtime | managed by `uv` |
| [`uv`](https://docs.astral.sh/uv/) | env + deps | `uv sync --extra dev` |
| [Pandoc](https://pandoc.org/) | MD → PDF/DOCX | detected at startup; without it the run still writes Markdown |
| A LaTeX engine (e.g. `xelatex`) | PDF rendering | optional; DOCX needs no LaTeX |
| Node.js (`npx`) | some MCP servers | needed only for the Node-based servers you enable |

On Windows, Node-based MCP servers are launched as `cmd /c npx …` automatically.

## Install

```powershell
uv sync --extra dev
uv run theo-explore --help
```

## Configure keys

Copy `.env.example` to `.env` and fill in what you have. **The tool runs with any subset of
providers** — each MCP server is started only when its required key is present.

```powershell
Copy-Item .env.example .env
```

- **LLM** (required): set `THEO_MODEL_STRONG` / `THEO_MODEL_CHEAP` and the matching provider key.
  This project is configured for **OpenRouter** and **DeepSeek**:
  - OpenRouter: `OPENROUTER_API_KEY`, model ids like `openrouter/deepseek/deepseek-v3.2` or
    `openrouter/anthropic/claude-sonnet-4.5`.
  - DeepSeek native: `DEEPSEEK_API_KEY`, model ids like `deepseek/deepseek-chat` or
    `deepseek/deepseek-reasoner`.

  Model strings follow LiteLLM's `provider/model` convention; defaults live in
  [`config/models.yaml`](config/models.yaml). Model ids copied straight from the OpenRouter site
  (e.g. `anthropic/claude-haiku-5.5` or `~deepseek/deepseek-flash-latest`) can be used as-is —
  they are automatically routed through `THEO_DEFAULT_PROVIDER` (default `openrouter`).
- **Search** (recommended): `EXA_API_KEY` and/or `TAVILY_API_KEY`.
- **Optional**: `SEMANTIC_SCHOLAR_API_KEY`, `NCBI_API_KEY`, `NCBI_ADMIN_EMAIL`, `UNPAYWALL_EMAIL`.

MCP servers, their transports and per-agent tool allowlists live in
[`config/mcp_servers.json`](config/mcp_servers.json) — the single source of truth.

## Usage

```powershell
# Interactive run (pauses for approval at scope and outline).
uv run theo-explore run "Why does time have a direction?"

# Fully autonomous, short article.
uv run theo-explore run "Why does time have a direction?" --auto --preset article-short

# A mini-book with an explicit synthesis pass.
uv run theo-explore run "The measurement problem in quantum mechanics" `
    --type mini-book --preset mini-book --synthesize

# Cap spend; the run aborts remaining research/drafting cleanly when exceeded.
uv run theo-explore run "Open problem in category theory" --max-cost 5.0

# Resume a paused/crashed project.
uv run theo-explore resume <project-id>

# Inspect a project's manifest and artifacts.
uv run theo-explore status <project-id>

# Re-export an existing project to PDF/DOCX.
uv run theo-explore export <project-id> --formats pdf,docx
```

Artifacts land in `workspace/<project-id>/` (override the root with `THEO_WORKSPACE_ROOT`):

```
workspace/<project-id>/
  manifest.json          # project state + artifact index + run report
  run_report.json        # per-node tokens/cost, tool calls, budget status
  checkpoints.sqlite     # LangGraph checkpointer (thread_id = project-id)
  artifacts/
    brief.json scope.json outline.json sources.json refs.bib
    evidence/<section>.json
    drafts/<section>.md
    final.md final.pdf final.docx
  run.log.jsonl          # per-node tokens/cost/tool-calls
```

The canonical `final.md` carries inline citations and a rendered author-date reference list;
`refs.bib` and the shipped CSL style let you re-render with Pandoc/citeproc if you prefer.

### Offline smoke test (no keys, no network)

```powershell
uv run theo-explore run "smoke test topic" --auto --fake-llm --no-export
```

`--fake-llm` swaps in a deterministic backend so a full graph run completes offline.

## How it works

```
intake → scope →[HITL]→ outline →[HITL]→ plan_research
   → dispatch_research (Send fan-out; independent sections run concurrently)
   → research_section (per-section ReAct over read-only MCP tools)
   → research_join (dedupe sources, aggregate evidence)
   → draft_section waves (topological by depends_on)
   → verify_citations → critique → revise (bounded loop)
   → synthesize (optional) → assemble → export → finalize
```

- **Dependency semantics.** Each section carries `depends_on`. Independent sections research and
  draft concurrently; "Synthesis"/"Conclusions" wait for their dependencies.
- **HITL.** `interrupt()` pauses at scope and outline; resume with `Command(resume=...)` in
  interactive mode or `--auto` to skip.
- **Safety.** Retrieved paper/web text is treated as untrusted data. Research agents have no
  side-effecting tools, and every prompt instructs the model never to follow instructions found
  inside retrieved content.

## Presets

`config/presets/*.yaml` control length targets, section budgets, research budgets and revision
rounds: `article-short` (≈1.5k–3k words), `article-long` (≈4k–8k), `mini-book` (≈12k–25k).

## Troubleshooting

- **A server "fails to connect" on Windows.** Child processes may not inherit your shell `PATH`.
  Prefer `uvx` servers; for Node servers the project wraps `npx` as `cmd /c npx …`. If it still
  fails, set the server's `command` to an absolute path in `config/mcp_servers.json`.
- **Rate limits.** arXiv throttles to ~3 s/request (handled by the server); unauthenticated
  Semantic Scholar quota is small (set a key); NCBI is 3 req/s without a key, 10 with one.
  Retries use `tenacity` exponential backoff.
- **No PDF.** Install Pandoc + a LaTeX engine (`xelatex`). Without them the run still produces
  Markdown (and DOCX if Pandoc is present) and logs an actionable message.
- **Version drift.** Pin MCP packages at install time to the registry's `isLatest` version.

## Development

```powershell
uv run ruff check .
uv run mypy
uv run pytest -q
```

Core dependencies: `langgraph`, `langchain-core`, `langchain-mcp-adapters`, `litellm`,
`pydantic`, `typer`, `rich`, `httpx`, `trafilatura`, `bibtexparser`, `pyyaml`.
