# Configuration reference

Every setting is an environment variable. Defaults come from `src/llmflow_search/config.py`
unless noted otherwise. The most commonly used ones are also listed in the main
[README](../README.md#configuration).

| Variable | Default | What it does | Source |
|---|---|---|---|
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Command launched as the stdio MCP server | `config.py` |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Force `footnote` or `generic`; `auto` detects from the connected tool list | `profiles.py` |
| `LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS` | `12.0` | Minimum delay between calls to scraped search engines (one call fans out to four of them) | `config.py` |
| `LLMFLOW_SEARCH_API_DELAY_SECONDS` | `2.0` | Minimum delay between calls to keyed/official search APIs | `config.py` |
| `LLMFLOW_SEARCH_ARCHIVE_DELAY_SECONDS` | `10.0` | Minimum delay between archive lookups | `config.py` |
| `LLMFLOW_SEARCH_SEARCH_JITTER` | `0.35` | Random fraction of the interval added on top of each delay | `config.py` |
| `LLMFLOW_SEARCH_MAX_SEARCH_BATCH` | `2` | Rate-limited steps run per round; the rest are deferred to the next one | `config.py` |
| `LLMFLOW_SEARCH_MAX_SEARCH_CALLS` | `12` | Hard ceiling on rate-limited calls spent answering one question | `config.py` |
| `LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES` | `5` | Pages fetched at once within a round | `config.py` |
| `LLMFLOW_SEARCH_READ_RANKING` | `search_rank` | Orders read candidates by search rank, or by `passage_bm25` (best-scoring passage per URL) | `config.py` |
| `LLMFLOW_SEARCH_RELEVANCE_EXCERPTS` | `1` (on) | Shows the question-relevant passages of a long page instead of just its first characters; `0` restores head-of-document behavior | `config.py` |
| `LLMFLOW_SEARCH_FAST_MODEL` | Unset | A second, cheaper Ollama model for bookkeeping decisions | `config.py` |
| `LLMFLOW_SEARCH_FAST_MODEL_ROLES` | `observation,requirements,evaluate,reflect` | Which roles route to the fast model when one is configured | `config.py` |
| `LLMFLOW_SEARCH_TODAY` | Current system date | Explicit `YYYY-MM-DD` date anchor; `CURRENT_DATE` is the lower-priority alias | `config.py` |
| `LLMFLOW_SEARCH_RESEARCH_MEMORY` | `~/.llmflow-search/research_memory.json` | Persistent strategy, skill, and experience store | `memory.py` |
| `LLMFLOW_SEARCH_REPORTS_DIR` | `reports` | Output directory for verified PDF reports | `config.py` |
| `LLMFLOW_SEARCH_REPORT_LOGO` | Packaged `assets/llmflow.png` | Logo used in PDF reports; an empty value disables it | `config.py` |
| `LLMFLOW_SEARCH_DEBUG_REPORTS` | `0` | Set to `1` to write a JSON debug report after a completed run | `reports.py` |
| `LLMFLOW_SEARCH_DEBUG_REPORT_DIR` | `~/.llmflow-search/debug_reports` | JSON debug-report directory | `reports.py` |
| `LLMFLOW_SEARCH_FORCE_COLOR` | Unset | Forces ANSI color for `1`/`true`/`yes`/`on`; `NO_COLOR` still disables automatic color | `console.py` |

## Fixed limits (not configurable via environment variables)

These are constants in `src/llmflow_search/config.py`:

| Constant | Value | What it bounds |
|---|---|---|
| `MAX_PLAN_STEPS` | 40 | Plan steps per question |
| `MAX_EVIDENCE_ROUNDS` | 5 | Evidence-gathering rounds per question |
| `MAX_STAGNANT_ROUNDS` | 2 | Consecutive rounds with no new supported claims before giving up |
| `TOOL_RESULT_MAX_CHARS` | 20,000 | Characters kept from a single tool result |
| `SOURCE_CONTENT_MAX_CHARS` | 25,000 | Characters kept per normalized source |
| `TOTAL_SOURCES_MAX_CHARS` | 100,000 | Combined source context size |
