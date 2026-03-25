# Setup

## Prerequisites

- Python 3.9 or newer for local runs
- Chromium or Google Chrome for Selenium-backed retrieval
- An LLM provider configured through `config.json`
- Ollama running locally if you keep the default provider

## Local Installation

```bash
git clone https://github.com/KazKozDev/llmflow-search.git
cd llmflow-search
pip install -r requirements.txt
```

## Configuration

The project reads runtime settings from `config.json`.

- `llm.provider`: provider name such as `ollama`, `openai`, `anthropic`, or `gemini`
- `llm.model`: model identifier used by the selected provider
- `search.max_results`: max search results per tool
- `search.parse_top_results`: number of top links to parse in depth
- `cache.sqlite_path`: SQLite cache file path
- `memory.path`: directory used for agent memory

If you switch away from `ollama`, set the matching API key in your environment.

## Environment Variables

Copy `.env.example` if you want a starting point for local environment setup.

- `OLLAMA_HOST`: Ollama server URL, default `http://localhost:11434`
- `OPENAI_API_KEY`: required when `llm.provider` is `openai`
- `ANTHROPIC_API_KEY`: required when `llm.provider` is `anthropic`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: used by Gemini or Google provider modes
- `CACHE_TTL_WIKI`: optional override for Wikipedia cache TTL

## Running the CLI

```bash
python main.py --output reports/report.md --max-iterations 10
```

The CLI prompts for the research query interactively.

## Running the Web App

```bash
python web_server.py
```

The app serves the UI at `http://localhost:8000` and exposes REST and WebSocket endpoints from the same process.

## Docker

```bash
docker compose up --build
```

The container uses Python 3.11, installs Chromium and `chromium-driver`, mounts the repository into `/app`, and forwards `OLLAMA_HOST` to `host.docker.internal:11434` by default.

## Troubleshooting

- Missing provider key: check that `config.json` matches the environment variable name expected by `main.py` and `core/llm_service.py`.
- Ollama connection issues: verify `OLLAMA_HOST` and confirm the Ollama server is reachable.
- Selenium browser errors: make sure Chromium or Chrome is installed locally, or use the Docker setup which installs Chromium automatically.
- Slow or repeated requests: inspect `data/cache.db` and the configured rate limits in `config.json`.