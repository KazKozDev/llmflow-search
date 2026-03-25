<div align="center">
  <img src="web/static/logo.png" alt="LLMFlow Search Logo" width="400"/>

  <h1>LLMFlow Search</h1>
  <p>
    Local-first research agent that orchestrates multi-source web search with iterative query refinement to produce citation-backed reports, all in one workflow, powered by local LLM.
  </p>

  <p>
    <img src="https://img.shields.io/badge/status-experimental-orange" alt="Status: Experimental"/>
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/>
  </p>
</div>

## Highlights

- Multi-source search across nine specialized tools
- CLI and FastAPI interfaces in one codebase
- Real-time progress streaming over WebSockets
- Background deep-search jobs with queue tracking
- Local-first Ollama default with provider flexibility

## Demo

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/5fdf2620-cd98-445d-8ceb-2f82cc245ea2" alt="LLMFlow Search preview" />
</p>

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/ac89e2ff-4bc3-401a-a514-e645fd093465" alt="LLMFlow Search report preview" />
</p>

## Overview

LLMFlow Search is a Python research assistant that turns a free-form question into a search plan, executes searches across web and domain-specific sources, parses selected links, and writes a markdown report with references. It is built for users who need local or self-hosted research workflows without being locked to a single search endpoint or a single LLM provider.

## Motivation

Research workflows often break down in one of three ways: generic web search lacks synthesis, hosted LLM workflows require external APIs, and single-provider assistants limit how results are gathered. LLMFlow Search addresses that gap by combining local-first Ollama support with multi-tool retrieval and report generation in one stack. In its current shape, it is positioned against OpenAI-backed research workflows, Anthropic-backed research workflows, and Gemini-backed research workflows by keeping provider choice in configuration instead of hardwiring a managed service.

## Features

- Query planning with iterative plan revision
- Search intent analysis for alternate query paths
- Search tools for DuckDuckGo, Wikipedia, SearXNG, ArXiv, PubMed, YouTube, Gutenberg, OpenStreetMap, and Wayback
- Cached tool execution with SQLite-backed storage
- Rate limiting per search source
- WebSocket progress events for live UI updates
- Background job queue for deep-search sessions
- Metrics endpoint for system and LLM telemetry
- Markdown report generation with linked sources

## Architecture

Components:
- `main.py` drives the interactive CLI workflow.
- `web_server.py` exposes the FastAPI API, WebSocket stream, and static UI.
- `core/agent_factory.py` initializes shared cache and LLM resources.
- `core/planning_module.py` builds and revises the search plan.
- `core/tools_module.py` dispatches search tools with caching and rate limits.
- `core/memory_module.py` stores gathered results for later synthesis.
- `core/report_generator.py` produces the final markdown report.

Flow: Query -> planning -> tool execution -> parsing and memory -> report generation -> CLI output or streamed web result

```mermaid
graph TD
    A[User Query] --> B[Planning Module]
    B --> C[Tools Module]
    C --> D[Search Providers]
    C --> E[Cache and Rate Limiter]
    D --> F[Memory Module]
    F --> G[Report Generator]
    G --> H[CLI Report or Web UI Result]
```

## Tech Stack

- Python 3.11 runtime in Docker, Python 3.9+ expected locally
- FastAPI and Uvicorn for the web server
- aiohttp and httpx for async network access
- Pydantic for config validation
- Ollama, OpenAI, Anthropic, and Gemini provider hooks
- SQLite caching through aiosqlite
- Selenium and Chromium for browser-assisted retrieval
- pytest for tests

## Quick Start

1. Clone the repository and install dependencies.

```bash
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
pip install -r requirements.txt
```

2. Review `config.json` and keep the default `ollama` provider, or switch to another provider and set the matching API key.

3. If you use Ollama, start an Ollama server locally or expose it through `OLLAMA_HOST`.

4. Run the web app or the CLI.

```bash
python web_server.py
```

```bash
python main.py --output reports/report.md --max-iterations 10
```

Detailed setup notes are in [docs/setup.md](docs/setup.md).

## Usage

Run the web interface:

```bash
python web_server.py
```

Run the CLI and save the generated markdown report:

```bash
python main.py --output reports/report.md --verbose --max-iterations 12
```

Start the containerized environment:

```bash
docker compose up --build
```

## API Overview

The web server in `web_server.py` exposes a small HTTP and WebSocket surface for the UI.

- `GET /` serves the static interface from `web/static/index.html`
- `POST /api/search` starts a standard session or queues a deep-search job
- `GET /api/sessions` lists in-memory standard sessions
- `GET /api/tools` returns the available search tools
- `GET /api/metrics` returns system and LLM metrics
- `GET /api/jobs` lists background jobs
- `GET /api/jobs/{job_id}` returns one background job
- `POST /api/jobs/{job_id}/cancel` cancels a running background job
- `WS /ws/search/{session_id}` streams `status`, `progress`, `result`, `complete`, and `error` messages for a standard session

## Project Structure

```text
core/
  caching/         # Cache backends and factory
  tools/           # Search tool implementations and parsers
  agent_factory.py # Shared resource lifecycle
  agent_core.py    # Agent execution loop
  report_generator.py
tests/
  test_agent_react_loop.py
  test_background_jobs.py
  test_tool_usage.py
  test_web_server.py
web/
  static/          # HTML, JS, CSS, and logo
main.py            # Interactive CLI entry point
web_server.py      # FastAPI + WebSocket server
config.json        # Runtime configuration
docker-compose.yml # Container orchestration
```

## Status

Stage: Experimental

Planned:
- TODO: define public roadmap items

## Testing

```bash
python -m pytest tests -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

MIT - see LICENSE

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[LinkedIn](https://www.linkedin.com/in/kazkozdev/)
[Email](mailto:kazkozdev@gmail.com)
