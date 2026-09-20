# LLMFlow-Search — LangGraph web research agent for macOS with Ollama

Ask a question; a local model searches the web and answers with citations.

```bash
git clone https://github.com/KazKozDev/llmflow-search.git && cd llmflow-search && ./agent.command
```

![Running a cited web research session in LLMFlow-Search](https://raw.githubusercontent.com/KazKozDev/llmflow-search/main/assets/llmflow-search-demo.gif)

Local Ollama models · Every claim cited · Refuses thin evidence · MIT

---

## Quick start

```bash
git clone https://github.com/KazKozDev/llmflow-search.git && cd llmflow-search && ./agent.command
```

The launcher clones the `footnote-mcp` search server next to this checkout, syncs the locked environment, installs Chromium, then asks which of your Ollama models to run.

```
Pick model number [Enter = gemma4:26b-mlx] > 2

Using: deepseek-v4.1-flash:cloud
Connecting to MCP server (footnote-mcp)... ✓ (45 tools)
  Profile: footnote

==================================================
  Interactive mode. Type 'exit' to quit.
==================================================

>>>
```

Type a question. The agent prints the answer, a numbered source list, and any requirement it could not cover.

## Answer a question with sources you can check

Every factual sentence carries an `[n]` marker, and a verification pass re-reads each one against the page it points at. A claim no source supports is cut, not softened.

```
>>> what changed in Python 3.13's free-threading build
```

The answer, a numbered list of the pages it came from, and a coverage note listing anything the sources did not settle. When nothing usable is found, the run says so instead of writing an answer.

## Export a cited PDF report of a research session

A verified answer is written to PDF automatically, with an evidence window naming the period its sources cover and separating background pages from current ones.

```bash
LLMFLOW_SEARCH_REPORTS_DIR=~/reports ./agent.command
```

The report lists only the sources the text actually cites, each with its publication date and source type.

## Measure the agent against a single-call baseline

The same questions, the same tools, one model call instead of twelve — so the graph has to earn its cost.

```bash
uv run python scripts/benchmark_browsecomp.py --sample-size 30 --offset 0 --trace reports/run.trace.jsonl
uv run python scripts/score_attribution.py reports/run.answers.jsonl
```

On the first 30 BrowseComp-Plus questions, claims supported by a cited source: 45% for the baseline, 52% for the agent. Full table and method in [docs/evaluation.md](docs/evaluation.md).

## How it works

A question first becomes explicit completion criteria, so "done" is decided before any searching. A planner emits one search or fetch step at a time against the live MCP tool catalog. Retrieved pages go into an evidence ledger that ties every admitted claim to a source id, and an adversarial pass re-reads the ledger looking for what it let through. Only then is an answer drafted, and a verifier rewrites it against the same bounded source set, deleting whatever the pages do not carry. Two deterministic repair passes follow: one attaches a citation to every claim sentence or removes it, the other fixes structure defects a check can prove.

```
question → requirements → plan → search/read → evidence ledger → challenge → answer → verify → cited report
```

## Configuration

Every setting is an environment variable. The full table is in [docs/configuration.md](docs/configuration.md).

| Variable | Default | What it does |
|---|---|---|
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Command launched as the stdio MCP server |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Force `footnote` or `generic`; `auto` detects from the tool list |
| `LLMFLOW_SEARCH_FAST_MODEL` | unset | A second, cheaper Ollama model for bookkeeping decisions |
| `LLMFLOW_SEARCH_AUTO_READ_TOP_K` | `5` | Pages opened unconditionally after the first search |
| `LLMFLOW_SEARCH_MAX_SEARCH_CALLS` | `12` | Ceiling on rate-limited calls spent on one question |
| `LLMFLOW_SEARCH_CITATION_REPAIR` | `1` | One pass that cites or removes every uncited claim sentence |
| `LLMFLOW_SEARCH_STRUCTURE_REPAIR` | `1` | One pass over report structure defects a check can prove |
| `LLMFLOW_SEARCH_REPORTS_DIR` | `reports` | Where the PDF report is written |
| `LLMFLOW_SEARCH_TODAY` | system date | Explicit `YYYY-MM-DD` anchor for relative dates |

## Requirements

- Python 3.10 or newer
- macOS — the only platform covered by CI
- [Ollama](https://ollama.com) running locally, with at least one model pulled
- [uv](https://docs.astral.sh/uv/) and `git` — the launcher needs both
- Chromium, installed automatically by the launcher for browser-backed tools

## Limitations

- Not published to PyPI and no Docker image; installation is from a clone.
- Linux (`agent.sh`) and Windows (`agent.ps1`) launchers exist but are not covered by CI.
- Slower and dearer than one model call: 21.6 s median per question against 5.8 s, and 89,640 tokens against 33,941.
- The report's editorial checks read English only; on other languages they stay silent.
- MCP tools that write, publish, execute code, or delete require interactive approval.
- Published numbers come from 30 questions on one pinned model — a direction, not a ranking.

<details>
<summary>Manual installation, Docker, development setup</summary>

### From source

```bash
git clone https://github.com/KazKozDev/footnote-mcp.git ../footnote-mcp
uv sync --locked
uv pip install --python .venv/bin/python -e ../footnote-mcp "mcp<2"
uv run --no-sync python -m llmflow_search
```

### Docker

No Dockerfile or image is provided.

### Development

```bash
uv run ruff check src tests scripts && uv run pyright && uv run pytest tests -q
```

</details>

---

<div align="center">

![macOS](https://img.shields.io/badge/macOS-333?style=flat-square&logo=apple&logoColor=fff) [![CI](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml/badge.svg)](https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Issues](https://github.com/KazKozDev/llmflow-search/issues) · [Evaluation](docs/evaluation.md) · [Configuration](docs/configuration.md) · [Decision brief](docs/decision-brief.md) · [License](LICENSE) · [LinkedIn](https://www.linkedin.com/in/kazkozdev/)

</div>
