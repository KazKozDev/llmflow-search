<p align="center">
  <a href="https://github.com/KazKozDev/llmflow-search"><img src="src/llmflow_search/assets/llmflow.png" alt="LLMFlow-Search" width="620"></a>
</p>

# LLMFlow-Search — Local LLM Web Search Agent Powered by Ollama

LLMFlow-Search is a local LLM web search agent powered by Ollama. It adds local AI web search and controlled internet access through MCP, runs multi-step web research, and returns a source-grounded answer instead of sending the question to a hosted research model.

Built on LangGraph, LLMFlow-Search works as a self-hosted AI search tool, an open-source AI research assistant, or a terminal-based local Perplexity alternative. The agent plans searches, reads pages through [`footnote-mcp`](https://github.com/KazKozDev/footnote-mcp), challenges weak evidence, verifies the final prose against the admitted sources, and writes a citation-backed PDF only when the requested coverage is complete.

```bash
# macOS
git clone https://github.com/KazKozDev/footnote-mcp.git
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
./agent.command

# Linux / Windows (PowerShell)
git clone https://github.com/KazKozDev/footnote-mcp.git
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
uv sync --locked
uv pip install -e ../footnote-mcp "mcp<2"
uv run --no-sync python -m playwright install chromium
uv run --no-sync python -m llmflow_search
```

<p align="center">
  <a href="agent.command"><img src="https://raw.githubusercontent.com/KazKozDev/book-translator/main/assets/badges/macos.png" alt="macOS" height="36"></a>
  <a href="#manual-installation"><img src="https://raw.githubusercontent.com/KazKozDev/book-translator/main/assets/badges/windows.png" alt="Windows" height="36"></a>
  <a href="#manual-installation"><img src="https://raw.githubusercontent.com/KazKozDev/book-translator/main/assets/badges/linux.png" alt="Linux" height="36"></a>
</p>

<p align="center">Double-click <code>agent.command</code> on macOS. Windows and Linux use the manual <code>uv</code> commands.</p>

---

## Quick start

1. Run the commands above. The two repositories must share the same parent directory because the default launcher installs `../footnote-mcp`. On macOS, `agent.command` installs `uv` when needed, synchronizes the locked environment, installs the MCP server, attempts to install Chromium, and starts LLMFlow-Search.

2. Keep Ollama running with at least one local model. At startup, choose a main model and a separate fast model from the models already installed in Ollama.

3. Wait for the MCP connection and enter a research question:

   ```text
   Connecting to MCP server (footnote-mcp)... ✓ (<tool count> tools)
     Profile: footnote

   ==================================================
     Interactive mode. Type 'exit' to quit.
   ==================================================

   >>> What changed in Python packaging this month?
   ```

   The exact tool count depends on the installed `footnote-mcp` version. The agent prints research stages while it searches, then returns the verified answer, a source list, coverage notes, report paths, and the number of completed steps.

## Give Ollama internet access through web search

Ollama does not browse by itself. LLMFlow-Search connects the selected local model to an MCP server over stdio, exposes that server's tools to the LangGraph workflow, and keeps tool results separate from model-generated prose.

With the default `footnote` profile, the agent can plan web searches, read the resulting pages, extract usable source material, and continue searching when the evidence does not cover the question. This is the path for **web search with Ollama**, **local LLM internet access**, and questions that need real-time information rather than the model's training data alone.

The local model still reasons over external web content. “Local” describes where Ollama and the agent run; live web research still sends search queries and page requests to the configured search tools and websites.

## Run local deep research with citation-backed sources

The main design priority is evidence coverage, not answer completion at any cost. Each question becomes explicit completion criteria before the first tool call. Candidate sources then pass through an evidence ledger and a separate challenge stage before the writer can use them.

```text
Question → Requirements → Plan → MCP tools → Evidence ledger
         → Evidence challenge → Answer → Verification → PDF
```

Weak support can send the graph back for more agentic web search. The run is bounded by 40 plan steps, five evidence rounds, and two consecutive rounds that add no supported claims. A single tool result is clipped to 20,000 characters, each normalized source to 25,000 characters, and the combined source context to 100,000 characters.

If the evidence remains insufficient, the agent says so directly:

```text
The found sources do not provide enough information for a reliable answer.
```

No PDF is created for that run. This is a fail-closed research result, not a fabricated fallback answer.

## Use the same AI research assistant with other MCP servers

LLMFlow-Search selects a server profile from the connected tool list:

- `footnote` requires both `web_search` and `web_read` and enables the full source-grounded web research path.
- `generic` works with any other stdio MCP tool set. The model uses native function calling, and every non-empty tool result becomes a source for the same answer-and-verify loop.

The included stub server demonstrates the generic path:

```bash
LLMFLOW_SEARCH_MCP_CMD=".venv/bin/python scripts/stub_mcp_server.py" \
  .venv/bin/python -m llmflow_search
```

Ask `what is the capital of France?`. The stub exposes one fixed `get_fact` tool, so this checks MCP discovery and orchestration without performing a live web search.

The generic profile is intentionally less specific: it cannot apply `footnote-mcp` source typing or search-strategy evolution to arbitrary tool output.

## Get a source-grounded PDF research report

When verification marks the task complete and at least one admitted source is present, LLMFlow-Search writes an A4 PDF to `reports/`. The filename contains the date, a query slug, and a generated research ID.

The report includes the verified answer and its source list. The renderer looks for an available Unicode font on macOS or Linux so Russian, Spanish, and other non-ASCII text is not silently dropped. Set `LLMFLOW_SEARCH_REPORT_LOGO` to replace the packaged logo or to an empty value to omit it.

The last three question-and-answer exchanges remain in the interactive session for follow-up questions. Completed runs also update the JSON research memory with the strategy, attempted queries, sources, outcome, and source domains that produced no admitted evidence.

## How it works

The interactive Python application starts Ollama model selection, launches the configured MCP server as a subprocess, inspects its tools, and compiles a conditional LangGraph `StateGraph`.<br>
**REQUIREMENTS** converts the question into completion criteria.<br>
**PLAN** chooses tool steps; **EXECUTE** calls the MCP server and normalizes returned sources.<br>
**LEDGER** maps proposed claims to sources; **CHALLENGE** looks for missing or weak support.<br>
**ANSWER** drafts only from the bounded admitted sources; **VERIFY** checks the prose and requested coverage again.<br>
**ASSIMILATE** records both successful and unsuccessful completed runs in local research memory.

```text
Terminal question
      ↓
Ollama main model + fast model
      ↓
LangGraph requirements / plan / execute loop
      ↓
stdio MCP server → web search / page reading / other tools
      ↓
Evidence ledger → challenge → verified answer
      ↓
Terminal sources + JSON memory + verified PDF
```

<details>
<summary>Technical architecture</summary>

### Research pipeline

1. **Model selection** — [`llm.py`](src/llmflow_search/llm.py) lists local Ollama models and selects a main and fast model. JSON-schema calls can route from an MLX-tagged model to its installed non-MLX sibling because Ollama's MLX runtime does not enforce schema-constrained decoding.
2. **MCP connection** — [`app.py`](src/llmflow_search/app.py) launches `LLMFLOW_SEARCH_MCP_CMD` over stdio, initializes the session, loads tool schemas, and selects the `footnote` or `generic` profile.
3. **Requirements and planning** — [`nodes.py`](src/llmflow_search/nodes.py) extracts blocking completion criteria, builds tool steps, and executes them through [`mcp_client.py`](src/llmflow_search/mcp_client.py). Search-backed calls are separated by a configurable delay.
4. **Evidence review** — candidate sources are normalized and audited. The ledger ties supported claims to source IDs; the challenge stage can request another tool call, re-extract evidence, replan, or stop with insufficient evidence.
5. **Draft and verification** — the writer receives only the bounded source set. Verification re-grounds the prose against the same sources and produces a compact coverage verdict.
6. **Output and memory** — [`reports.py`](src/llmflow_search/reports.py) writes optional JSON diagnostics, creates a PDF only for a verified source-backed answer, and records the run in the JSON memory store.

The compiled graph contains 11 nodes: `requirements`, `plan`, `execute`, `evidence_ledger`, `evidence_challenge`, `evidence_reextract`, `evaluate`, `answer`, `verify`, `strategy`, and `assimilate`. The interactive runner uses a recursion limit of 200.

### Important files

- `agent.command` — macOS launcher and locked environment bootstrap.
- `src/llmflow_search/app.py` — interactive entry point and MCP session lifecycle.
- `src/llmflow_search/nodes.py` — graph nodes, routers, evidence gates, and retry limits.
- `src/llmflow_search/prompts.py` — system contracts for every model stage.
- `src/llmflow_search/mcp_client.py` — MCP tool discovery and invocation.
- `src/llmflow_search/sources.py` — source extraction, normalization, deduplication, clipping, and auditing.
- `src/llmflow_search/search_memory.py` — per-run query, URL, observation, and strategy memory.
- `src/llmflow_search/memory.py` — persistent strategy, skill, and experience store.
- `src/llmflow_search/pdf_report.py` — Markdown-to-PDF rendering and Unicode font selection.
- `scripts/stub_mcp_server.py` — one-tool server for the generic MCP profile.
- `scripts/live_smoke.py` — manual live query against Ollama and the real MCP server.
- `tests/` — offline graph, entry-point, MCP, model, console, profile, and PDF tests.

</details>

<details>
<summary>Configuration</summary>

| Variable | Default | What it changes |
|---|---|---|
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Command launched as the stdio MCP server |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Automatic tool-based detection, `footnote`, or `generic` |
| `LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS` | `3.0` | Minimum delay between `web_search` and `web_deep_search` requests |
| `LLMFLOW_SEARCH_TODAY` | Current system date | Explicit `YYYY-MM-DD` date anchor; `CURRENT_DATE` is the lower-priority alias |
| `LLMFLOW_SEARCH_RESEARCH_MEMORY` | `~/.llmflow-search/research_memory.json` | Persistent strategy, skill, and experience store |
| `LLMFLOW_SEARCH_REPORTS_DIR` | `reports` | Output directory for verified PDF reports |
| `LLMFLOW_SEARCH_REPORT_LOGO` | Packaged `assets/llmflow.png` | Logo used in PDF reports; an empty value disables it |
| `LLMFLOW_SEARCH_DEBUG_REPORTS` | `0` | Set to `1` to write a JSON debug report after a completed run |
| `LLMFLOW_SEARCH_DEBUG_REPORT_DIR` | `~/.llmflow-search/debug_reports` | JSON debug-report directory |
| `LLMFLOW_SEARCH_FORCE_COLOR` | Unset | Forces ANSI color for `1`, `true`, `yes`, or `on`; `NO_COLOR` still disables automatic color |

Enable a JSON research trace without changing the normal PDF output:

```bash
LLMFLOW_SEARCH_DEBUG_REPORTS=1 uv run --no-sync python -m llmflow_search
```

</details>

<details>
<summary>Requirements</summary>

- **Python 3.10 or newer**, as declared by `pyproject.toml`.
- **Ollama** running locally with at least one installed model.
- **`footnote-mcp` in `../footnote-mcp`** for the default launcher and source-aware web research profile.
- **`uv`** for locked installation. The macOS launcher installs it automatically when missing; manual Windows and Linux setup expects it to be available.
- **Internet access** for dependency installation and live web search. The Ollama inference remains local, but search queries and page fetches leave the machine.
- **Chromium through Playwright** for browser-backed fetches used by the MCP server. The launcher attempts this installation but does not make a failed browser download fatal.

The packaged application has one macOS launcher. Windows and Linux use the manual Python/`uv` path; this checkout has not been verified through a clean-machine end-to-end run on those systems.

</details>

<details>
<summary>Limitations</summary>

- LLMFlow-Search is not an offline search engine. Ollama inference is local, while current information still comes from external search and page-reading tools.
- Evidence ledger, challenge, drafting, and final prose verification use the selected Ollama models. Deterministic source bounds and routing prevent several failure modes, but model quality still affects planning, extraction, and judgments.
- A weak model may return malformed JSON or poor tool plans. JSON responses are retried once; MLX models only gain schema-constrained decoding when a compatible non-MLX sibling is installed.
- Hard limits can end a difficult research task before coverage is complete. The correct result in that case is the explicit insufficient-evidence message, not a partial PDF.
- The default source-aware path depends on the separately installed `footnote-mcp` server. Its search engines, browser tiers, and website access have their own availability limits.
- The generic MCP profile accepts arbitrary tool output as source material and therefore has weaker provenance and source typing than the `footnote` profile.
- `agent.command` is a macOS/zsh launcher. There is no Windows launcher, Linux launcher, browser UI, official Docker image, or hosted service in this repository.
- PDF export depends on `xhtml2pdf` and an available system Unicode font. Research output still appears in the terminal if PDF rendering fails.

</details>

<details>
<summary>Manual installation, Docker, development setup</summary>

### Manual installation

Clone the agent and `footnote-mcp` beside each other:

```bash
git clone https://github.com/KazKozDev/footnote-mcp.git
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
uv sync --locked
uv pip install -e ../footnote-mcp "mcp<2"
uv run --no-sync python -m playwright install chromium
uv run --no-sync python -m llmflow_search
```

`uv sync` creates `.venv` from `uv.lock` and installs LLMFlow-Search. The editable sibling install supplies the `footnote-mcp` executable. Playwright installs the Chromium browser used by browser-backed fetches, and the final command starts model selection and the interactive prompt.

On macOS, double-click `agent.command` after cloning both repositories instead of entering these commands manually.

### Docker

This repository does not include a Dockerfile or published image. Run the Python application natively so it can reach local Ollama and launch the configured stdio MCP server.

### Development setup

```bash
uv sync --locked
uv run --locked ruff check src tests scripts
uv run --locked pyright
uv run --locked pytest tests -q --cov=llmflow_search --cov-report=term-missing:skip-covered
```

The suite replaces Ollama and MCP calls with controlled test doubles, so it does not need network access or a running model server. The current checkout passes lint, type checking, all 60 tests, and the configured 65% coverage gate:

```text
60 passed in 3.28s
Required test coverage of 65.0% reached. Total coverage: 68.91%
```

For a real Ollama and `footnote-mcp` check:

```bash
PYTHONPATH=src .venv/bin/python scripts/live_smoke.py
```

</details>

## License

LLMFlow-Search is free and open-source software licensed under the [MIT License](LICENSE).

<br><br>

<p align="center">
  <a href="https://github.com/KazKozDev/llmflow-search/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/KazKozDev/llmflow-search/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&amp;logoColor=white"></a>
  <a href="https://docs.langchain.com/oss/python/langgraph/overview"><img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C.svg"></a>
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/uv-locked-DE5FE9.svg"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/Ruff-passing-D7FF64.svg"></a>
</p>

<p align="center">
  <a href="https://github.com/KazKozDev/llmflow-search/issues">Issues</a> ·
  <a href="https://github.com/KazKozDev/llmflow-search/actions">CI</a> ·
  <a href="https://github.com/KazKozDev/footnote-mcp">footnote-mcp</a> ·
  <a href="https://github.com/KazKozDev/llmflow-search/blob/main/LICENSE">LICENSE</a> ·
  <a href="https://www.linkedin.com/in/kazkozdev/">LinkedIn</a>
</p>
