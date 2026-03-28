<div align="center">
  <img src="web/static/logo.png" alt="LLMFlow Search logo" width="320" />

  <h1>LLMFlow Search</h1>

  <p>
    Research agent that turns a query into a search plan, runs it across multiple sources,
    and returns a Markdown report with references.
  </p>

  <p>
    <img src="https://img.shields.io/badge/python-3.9%2B-3776AB" alt="Python 3.9+" />
    <img src="https://img.shields.io/badge/default%20LLM-Ollama-111111" alt="Default LLM: Ollama" />
    <img src="https://img.shields.io/badge/web-FastAPI-009688" alt="Web: FastAPI" />
    <img src="https://img.shields.io/badge/license-MIT-6F8F76" alt="License: MIT" />
  </p>
</div>

<p align="center">
  <img src="docs/assets/llmflow-search-demo.gif" alt="LLMFlow Search demo" width="1000" />
</p>

## Demo

Demo assets included in the repository:

- [Search demo GIF](docs/assets/llmflow-search-demo.gif)
- [Search demo video](docs/assets/llmflow-search-demo.mp4)
- [Search demo video (compat)](docs/assets/llmflow-search-demo-compat.mp4)
- [Product promo video](docs/assets/llmflow-product-promo-v2.mp4)
- [Real desktop capture](docs/assets/llmflow-real-screen.mp4)

## Search Result Preview

<p align="center">
  <img width="1000" src="https://github.com/user-attachments/assets/ac89e2ff-4bc3-401a-a514-e645fd093465" alt="LLMFlow Search report preview" />
</p>

## Highlights

- Ollama-first setup with `ollama` as the default provider in [`config.json`](config.json)
- Multiple search backends exposed through one pipeline, including DuckDuckGo, Wikipedia, SearXNG, ArXiv, PubMed, YouTube, Gutenberg, OpenStreetMap, and Wayback
- Two interfaces out of the box: an interactive CLI in [`main.py`](main.py) and a FastAPI web app in [`web_server.py`](web_server.py)
- Standard WebSocket sessions and background "deep" jobs in the web server
- SQLite caching, per-tool rate limits, and generated Markdown reports saved to `reports/`

## Overview

LLMFlow Search is a Python application for local-first research workflows. It accepts a natural-language query, creates a search plan, executes tool calls, stores retrieved context in memory, and generates a final report with sources.

The repository includes both a terminal workflow and a browser-based UI. By default the project is configured for Ollama, but the config and dependency set also support OpenAI, Anthropic, and Gemini-backed runs through environment variables.

## Quick Start

### Requirements

- Python 3.9 or newer for local runs
- Chromium or Google Chrome for Selenium-backed retrieval
- Ollama running locally if you keep the default `ollama` provider

### Install

```bash
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
pip install -r requirements.txt
```

### Configure

The app reads runtime settings from [`config.json`](config.json).

Default local configuration:

```json
{
  "llm": {
    "provider": "ollama",
    "model": "qwen3:8b",
    "temperature": 0.2,
    "max_tokens": 4096
  }
}
```

Environment variables used by the project:

- `OLLAMA_HOST` for a non-default Ollama host
- `OPENAI_API_KEY` when `llm.provider` is `openai`
- `ANTHROPIC_API_KEY` when `llm.provider` is `anthropic`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` when `llm.provider` is `gemini`

## Usage

### CLI

```bash
python main.py --output reports/report.md --max-iterations 10
```

The CLI prompts for the query interactively and writes the generated report to the output path.

Useful flags:

- `--verbose` enables debug logging
- `--config` points to a custom config file
- `--disable-intent-analyzer` disables the search intent analyzer

### Web App

```bash
python web_server.py
```

This starts the FastAPI server and serves the UI at [http://localhost:8000](http://localhost:8000).

Relevant HTTP endpoints exposed by the server:

- `GET /` serves the static UI
- `POST /api/search` starts a standard session or queues a deep-search job
- `GET /api/tools` lists available search tools
- `GET /api/metrics` returns system and LLM metrics
- `GET /api/jobs` and `GET /api/jobs/{job_id}` inspect background jobs
- `POST /api/jobs/{job_id}/cancel` cancels a running job

Standard search progress is streamed over `WS /ws/search/{session_id}`.

### Docker

```bash
docker compose up --build
```

The Docker setup uses `python:3.11-slim-bookworm`, installs Chromium and `chromium-driver`, mounts the repository into `/app`, and forwards `OLLAMA_HOST` to `host.docker.internal:11434`.

## How It Works

The main pipeline is organized around a small set of core modules:

- [`core/planning_module.py`](core/planning_module.py) builds and revises the search plan
- [`core/tools_module.py`](core/tools_module.py) dispatches search tools and parsing work
- [`core/memory_module.py`](core/memory_module.py) stores gathered context and links
- [`core/report_generator.py`](core/report_generator.py) produces the final Markdown report
- [`core/agent_core.py`](core/agent_core.py) orchestrates the search loop
- [`core/background_jobs.py`](core/background_jobs.py) manages deep-search jobs for the web app

## Project Structure

```text
core/
  caching/      Cache backends and factory
  tools/        Search tool implementations and parsers
tests/          Unit and route tests
web/static/     Browser UI assets
docs/           Setup notes and demo assets
main.py         Interactive CLI entry point
web_server.py   FastAPI app and WebSocket server
config.json     Runtime configuration
```

## Testing

```bash
python -m pytest tests -q
```

The repository includes tests for the agent loop, background jobs, planning, report generation, tool usage, and web routes.

## Additional Docs

- [Setup guide](docs/setup.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)

## License

MIT. See [LICENSE](LICENSE).
