# LLMFlow-Search — LangGraph web research agent for macOS with Ollama

Run cited, verified web research locally through Ollama and MCP tools.

<!-- Editor deep links omitted: LLMFlow-Search is an MCP client. -->

```bash
./agent.command
```

![LLMFlow-Search running a source-grounded web research session in the terminal](https://raw.githubusercontent.com/KazKozDev/llmflow-search/main/assets/llmflow-search-demo.gif)

macOS launcher · Cited answers · Fail-closed verification
<p align="center"><img src="https://img.shields.io/badge/macOS-333?style=for-the-badge&amp;logo=apple&amp;logoColor=fff" height="40" alt="macOS"></p>
---

## Quick start

Requires Git, Ollama with at least one local model, and internet access.

```bash
# 1. Clone the agent.
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
# 2. Confirm that Ollama has a model.
ollama list
# 3. Install and start the agent.
./agent.command
```

The launcher installs `uv` if needed, clones `footnote-mcp` beside this checkout, syncs the locked environment, installs Chromium, and starts the REPL. Verified startup:

```text
Connecting to MCP server (footnote-mcp)... ✓ (45 tools)
  Profile: footnote
Interactive mode. Type 'exit' to quit.
```

## Research a current question

Enter a query that needs current sources:

```text
>>> What changed in Python packaging this month?
```

The agent searches, reads pages, challenges its evidence, and prints cited sources. If the evidence is insufficient, it returns the verified fail-closed result and writes no PDF:

```text
The found sources do not provide enough information for a reliable answer.
```

## Connect another MCP server

Override the stdio command to use any MCP server with tools. This bundled example exercises the generic profile:

```bash
LLMFLOW_SEARCH_MCP_CMD=".venv/bin/python scripts/stub_mcp_server.py" \
  .venv/bin/python -m llmflow_search
```

```text
Connecting to MCP server (.venv/bin/python)... ✓ (1 tools)
  Profile: generic
```

## Continue with a follow-up

The REPL keeps the last three exchanges as conversation context:

```text
>>> What is the latest stable Python release? Use official sources.
>>> Which changes affect package maintainers?
```

Each follow-up starts a new bounded evidence run while retaining the recent question-and-answer context.

## How it works

The app selects an installed Ollama model before accepting a question.
It launches the configured MCP server over stdio and discovers its tools.
LangGraph turns the question into conditions, a plan, and bounded tool steps.
Retrieved pages enter an evidence ledger before drafting and verification.
Only supported answers receive cited sources and a PDF report.

```text
question → conditions → plan → MCP tools → evidence ledger
         → challenge → answer → verification → PDF
```

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Stdio MCP server command |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Detect `footnote` or `generic` from tools |
| `LLMFLOW_SEARCH_FAST_MODEL` | Unset | Optional model for bookkeeping roles |
| `LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS` | `12.0` | Delay between scraped search calls |
| `LLMFLOW_SEARCH_MAX_SEARCH_CALLS` | `12` | Search-call ceiling per question |
| `LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES` | `5` | Concurrent page fetches per round |
| `LLMFLOW_SEARCH_REPORTS_DIR` | `reports` | Verified PDF output directory |
| `LLMFLOW_SEARCH_REPORT_LOGO` | Packaged logo | PDF logo; empty disables it |
| `LLMFLOW_SEARCH_DEBUG_REPORTS` | `0` | Write a JSON debug report when `1` |
| `LLMFLOW_SEARCH_FORCE_COLOR` | Unset | Force ANSI color unless `NO_COLOR` is set |

## Requirements

- Python 3.10 or newer
- Git
- Ollama with at least one installed model
- Internet access for search and page fetching
- Chromium, installed automatically by the launcher
- macOS for the verified launcher and CI path

## Limitations

- CI runs linting, typing, and tests on `macos-latest`; it does not run live Ollama or MCP research.
- Linux and Windows launchers exist but have not been tested end to end on clean machines.
- Bounds of 40 plan steps, 5 evidence rounds, and 2 stagnant rounds can stop incomplete research.
- PDF export needs `xhtml2pdf` and an available Unicode font; terminal output remains available on failure.
- No Docker image or PyPI package is published; installation is from source.

<details>
<summary>More setup</summary>

### Docker
No Dockerfile or image is provided.
### From source
Run `uv sync --locked`; the default profile also needs sibling `../footnote-mcp`.
### Development
Run `uv run ruff check src tests scripts && uv run pyright && uv run pytest -q` — latest result: `151 passed`.
</details>
<br><br>
<div align="center">

[![CI](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml/badge.svg)](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Issues](https://github.com/KazKozDev/llmflow-search/issues) · [Configuration](docs/configuration.md) · [License](LICENSE) · [LinkedIn](https://www.linkedin.com/in/kazkozdev/)

</div>
