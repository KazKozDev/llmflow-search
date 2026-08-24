# LLMFlow-Search — LangGraph web research agent for macOS with Ollama

Run cited, verified web research locally through Ollama and MCP tools.



```bash
uvx llmflow-search
```

![Running a cited web research session in LLMFlow-Search](https://raw.githubusercontent.com/KazKozDev/llmflow-search/main/assets/llmflow-search-demo.gif)

Runs on macOS · Fail-closed answers · Open source
![macOS](https://img.shields.io/badge/macOS-333?style=flat-square&logo=apple&logoColor=fff)
---

## Quick start

With Git and Ollama installed, clone the source, pull a model, and launch:

```bash
git clone https://github.com/KazKozDev/llmflow-search.git && cd llmflow-search
ollama pull qwen2.5:7b
./agent.command
```

The launcher installs the locked environment and opens the REPL:

```text
Connecting to MCP server (footnote-mcp)... ✓ (45 tools)
  Profile: footnote
Interactive mode. Type 'exit' to quit.
```

## Research current questions with cited web sources

Enter a question that needs current evidence from the web.

```text
>>> What changed in Python packaging this month?
```

The agent prints a source-grounded answer and writes a PDF only after verification.

## Connect a different MCP server over standard input

Point the agent at another stdio MCP server; the bundled stub selects the generic profile.

```bash
LLMFLOW_SEARCH_MCP_CMD=".venv/bin/python scripts/stub_mcp_server.py" \
  .venv/bin/python -m llmflow_search
```

```text
Connecting to MCP server (.venv/bin/python)... ✓ (1 tools)
  Profile: generic
```

## Continue web research with contextual follow-up questions

Ask a follow-up without repeating the full context.

```text
>>> What is the latest stable Python release? Use official sources.
>>> Which changes affect package maintainers?
```

The REPL gives each question a new evidence run and includes the last three exchanges as context.

## How it works

The app selects an installed Ollama model before accepting a question.
It launches the configured MCP server over stdio and discovers its tools.
LangGraph turns the question into conditions, a plan, and bounded tool steps.
Retrieved pages enter an evidence ledger before drafting and verification.
Only supported answers receive cited sources and a PDF report.

```text
question → conditions → plan → MCP tools → evidence → challenge → answer → verification → PDF
```

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Sets the stdio MCP server command |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Detects the `footnote` or `generic` profile |
| `LLMFLOW_SEARCH_FAST_MODEL` | Unset | Assigns bookkeeping roles to another Ollama model |
| `LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS` | `12.0` | Delays scraped search calls |
| `LLMFLOW_SEARCH_MAX_SEARCH_CALLS` | `12` | Caps rate-limited calls per question |
| `LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES` | `5` | Caps concurrent page fetches per round |
| `LLMFLOW_SEARCH_REPORTS_DIR` | `reports` | Sets the verified PDF output directory |
| `LLMFLOW_SEARCH_REPORT_LOGO` | Packaged logo | Sets the PDF logo; empty disables it |
| `LLMFLOW_SEARCH_DEBUG_REPORTS` | `0` | Writes a JSON debug report when set to `1` |
| `LLMFLOW_SEARCH_FORCE_COLOR` | Unset | Forces ANSI color unless `NO_COLOR` is set |

## Requirements

- Python 3.10 or newer
- Git
- Ollama with at least one installed model
- Internet access for search and page fetching
- Chromium, installed automatically by the launcher
- macOS for the CI-tested launcher path

## Limitations

- CI does not run live Ollama or MCP research.
- Linux and Windows launchers are not covered by CI.
- Passage relevance is approximate: pages use overlapping 700-character windows.
- Research stops after 40 plan steps, 5 evidence rounds, or 2 stagnant rounds.
- No Docker image or PyPI package is published.

<details>
<summary>Manual installation, Docker, development setup</summary>
### From source
Run `uv sync --locked && uv pip install --python .venv/bin/python -e ../footnote-mcp "mcp<2" && uv run --no-sync llmflow-search`.
### Docker
No Dockerfile or image is provided.
### Development
Run `uv run ruff check src tests scripts && uv run pyright && uv run pytest tests -q`.
</details>

---

<div align="center">

[![CI](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml/badge.svg)](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Issues](https://github.com/KazKozDev/llmflow-search/issues) · [License](LICENSE) · [Configuration](docs/configuration.md) · [LinkedIn](https://www.linkedin.com/in/kazkozdev/)

</div>
