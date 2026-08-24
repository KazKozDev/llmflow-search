# LLMFlow-Search — local web-search AI agent

A LangGraph agent that runs multi-step web research through MCP and returns a cited, verified answer.

```bash
git clone https://github.com/KazKozDev/footnote-mcp.git && git clone https://github.com/KazKozDev/llmflow-search.git && cd llmflow-search && uv sync --locked
```

![LLMFlow-Search running a source-grounded web research session in the terminal](https://raw.githubusercontent.com/KazKozDev/llmflow-search/main/assets/llmflow-search-demo.gif)

Local inference · Cited sources · Open source

<div align="center">

<img src="https://img.shields.io/badge/macOS-333?style=for-the-badge&amp;logo=apple&amp;logoColor=fff" height="40" alt="macOS"> <img src="https://img.shields.io/badge/Linux-333?style=for-the-badge&amp;logo=linux&amp;logoColor=fff" height="40" alt="Linux"> <img src="https://img.shields.io/badge/Windows-333?style=for-the-badge&amp;logo=windows&amp;logoColor=fff" height="40" alt="Windows">

</div>

## Quick start

```bash
git clone https://github.com/KazKozDev/footnote-mcp.git
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
uv sync --locked
uv pip install -e ../footnote-mcp "mcp<2"
uv run --no-sync python -m playwright install chromium
uv run --no-sync python -m llmflow_search
```

The two repositories must sit side by side. Ollama must already be running locally with at least one model pulled. A launcher script runs all of the steps above in one go and clones `footnote-mcp` automatically if it isn't there yet: `./agent.command` on macOS, `./agent.sh` on Linux, `.\agent.ps1` on Windows (PowerShell).

```
Ollama models (70 total):

   1. qwen3:8b                              4.9 GB
   2. ornith-1.5:9b                         6.1 GB
   ...

Pick model number [Enter = gemma4:26b-mlx] > 1

Using: qwen3:8b
Connecting to MCP server (footnote-mcp)... <!-- TODO(user): paste real output of a footnote-mcp connection; this session only verified the generic/stub profile below -->
```

## Research a question with live, cited web sources

Ask anything that needs current information. The agent turns the question into explicit completion criteria, plans searches, reads pages through `footnote-mcp`, and challenges its own evidence before answering.

```bash
uv run --no-sync python -m llmflow_search
>>> What changed in Python packaging this month?
```

If the sources it read do not support a full answer, it says so instead of guessing:

```
The found sources do not provide enough information for a reliable answer.
```

No PDF is written for that run — a fail-closed result, not a fabricated one.

## Point it at any other MCP tool server

The `generic` profile works with any stdio MCP server, not just `footnote-mcp`. Every non-empty tool result becomes a source for the same answer-and-verify loop. The bundled stub server exercises MCP discovery without a live web search:

```bash
LLMFLOW_SEARCH_MCP_CMD=".venv/bin/python scripts/stub_mcp_server.py" \
  .venv/bin/python -m llmflow_search
```

```
Connecting to MCP server (.venv/bin/python)... ✓ (1 tools)
  Profile: generic

==================================================
  Interactive mode. Type 'exit' to quit.
==================================================
```

## Tune request pacing and model cost

Scraped search engines get throttled by default; loosen or tighten the pacing per backend family, or route cheap bookkeeping decisions to a second, smaller model:

```bash
LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS=20 \
LLMFLOW_SEARCH_FAST_MODEL=qwen3:8b \
  uv run --no-sync python -m llmflow_search
```

Full list of variables in [Configuration](#configuration).

## How it works

The interactive app starts Ollama model selection, launches the configured MCP server as a subprocess, inspects its tools, and compiles a conditional LangGraph `StateGraph`. Each question becomes a registry of completion criteria before the first tool call; a single policy module (not a chain of downstream patches) decides whether a round continues, replans, or stops. Page fetches within a round run in parallel, and long pages are split into ranked passages instead of being read from the top down. A run is bounded by fixed step, round, and character limits — see [Configuration](#configuration).

```
question → requirements/conditions → plan → MCP tools (parallel fetch)
         → evidence ledger → challenge → answer → verification → PDF
```

## Configuration

### Environment variables

| Variable | Default | What it does |
|---|---|---|
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Command launched as the stdio MCP server |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Force `footnote` or `generic`; `auto` detects from the tool list |
| `LLMFLOW_SEARCH_FAST_MODEL` | Unset | A second, cheaper Ollama model for bookkeeping decisions |
| `LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS` | `12.0` | Minimum delay between calls to scraped search engines |
| `LLMFLOW_SEARCH_MAX_SEARCH_CALLS` | `12` | Hard ceiling on rate-limited calls per question |
| `LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES` | `5` | Pages fetched at once within a round |
| `LLMFLOW_SEARCH_REPORTS_DIR` | `reports` | Output directory for verified PDF reports |
| `LLMFLOW_SEARCH_REPORT_LOGO` | Packaged `assets/llmflow.png` | Logo used in PDF reports; empty disables it |
| `LLMFLOW_SEARCH_DEBUG_REPORTS` | `0` | Set to `1` to write a JSON debug report after a run |
| `LLMFLOW_SEARCH_FORCE_COLOR` | Unset | Force ANSI color; `NO_COLOR` still disables it |

10 most-used variables above; the remaining ones (rate-limit tuning, passage ranking, research-memory path) are in [docs/configuration.md](docs/configuration.md).

## Requirements

- Python 3.10 or newer
- Ollama running locally with at least one model installed
- `footnote-mcp` cloned as a sibling `../footnote-mcp` directory, for the default profile
- `uv`, for locked dependency installation
- Internet access for live web search and page fetching — Ollama inference itself stays local
- Chromium via Playwright, used by `footnote-mcp`'s browser-backed fetches

## Limitations

- CI (`.github/workflows/ci.yml`) lints, type-checks, and tests on `ubuntu-latest` only; the interactive Ollama+MCP flow itself is not exercised in CI on any OS
- `agent.sh` (Linux) and `agent.ps1` (Windows) mirror `agent.command`'s steps but have not been run end-to-end on a clean machine of either OS — CI only exercises the `uv`/`pytest` commands directly on Ubuntu, not the launcher scripts
- Fixed bounds (40 plan steps, 5 evidence rounds, 2 stagnant rounds) can end a run before coverage is complete — the agent returns an explicit insufficient-evidence message rather than a partial answer
- PDF export depends on `xhtml2pdf` and an available system Unicode font; terminal output still appears if PDF rendering fails
- Not published to PyPI — install is from source only

<details>
<summary>Manual installation, Docker, development setup</summary>

### From source

```bash
git clone https://github.com/KazKozDev/footnote-mcp.git
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
uv sync --locked
uv pip install -e ../footnote-mcp "mcp<2"
uv run --no-sync python -m playwright install chromium
```

### Docker

No Dockerfile or image is provided in this repository.

### Development

```bash
uv sync --locked
uv run --locked ruff check src tests scripts
uv run --locked pyright
uv run --locked pytest tests -q --cov=llmflow_search --cov-report=term-missing:skip-covered
```

The suite replaces Ollama and MCP calls with test doubles, so it needs no network access or running model server:

```
151 passed in 4.44s
Required test coverage of 65.0% reached. Total coverage: 76.70%
```

For a real Ollama and `footnote-mcp` check: `PYTHONPATH=src .venv/bin/python scripts/live_smoke.py`

</details>

<br><br>

<div align="center">

[![CI](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml/badge.svg)](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Issues](https://github.com/KazKozDev/llmflow-search/issues) · [Configuration](docs/configuration.md) · [License](LICENSE) · [LinkedIn](https://www.linkedin.com/in/kazkozdev/)

</div>
