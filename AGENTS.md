# Repository Guidelines

## Project Structure & Module Organization

The project is an evidence-first research pipeline. `src/deep_research/orchestrator.py` owns the state transitions from planning through citation validation. Agent roles live in `agents.py`; they return Pydantic models from `models.py` instead of free-form intermediate text. `tools.py` contains external boundaries: the separately installed `footnote-mcp` stdio client, optional SearXNG search, HTTP page retrieval, text extraction, URL normalization, and public-host protection. Persisted research state, sources, evidence, events, and reports are handled solely by `store.py` through SQLite. `llm.py` is the only Ollama adapter. Keep provider-specific protocol code out of agents and orchestration.

`ARCHITECTURE.md` is the product-level source of truth. Changes to roles, evidence status, citation behaviour, state transitions, budgets, or safety boundaries should update it. `config.example.yaml` documents the supported runtime configuration. The `data/` directory is generated local state and is intentionally ignored by Git.

## Build, Test, and Development Commands

Create the local environment and install dependencies with:

```bash
uv sync --all-groups
```

Run a research task after copying `config.example.yaml` to `config.yaml`, starting Ollama, and ensuring `search.footnote.command` points to an installed `footnote-mcp` executable:

```bash
uv run deep-research run "question" --config config.yaml
```

Inspect a run or render its persisted report with `uv run deep-research status <research-id>` and `uv run deep-research report <research-id>`.

Run checks before handing off changes:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Coding Style & Testing Guidelines

Use Python 3.12 and Ruff's configured 128-character line length. Keep schemas explicit and validate every LLM structured response. Tests use `pytest` and `pytest-asyncio`; place isolated tests in `tests/`. Use fake search, fetch, and LLM adapters for orchestration tests so they do not need network access, Ollama, an installed `footnote-mcp`, or a running SearXNG service. Cover citation validation, SQLite persistence, and any state-machine behaviour changed by a patch.
