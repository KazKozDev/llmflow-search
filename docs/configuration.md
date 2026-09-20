# Configuration reference

Every setting is an environment variable. Defaults come from `src/llmflow_search/config.py`
unless noted otherwise. The most commonly used ones are also listed in the main
[README](../README.md#configuration).

| Variable | Default | What it does | Source |
|---|---|---|---|
| `LLMFLOW_SEARCH_MODEL` | Unset | Preselects the Ollama model and skips the interactive picker; a scripted run, a recording or anything driven from a pipe needs this | `llm.py` |
| `LLMFLOW_SEARCH_MCP_CMD` | `footnote-mcp` | Command launched as the stdio MCP server | `config.py` |
| `LLMFLOW_SEARCH_PROFILE` | `auto` | Force `footnote` or `generic`; `auto` detects from the connected tool list | `profiles.py` |
| `LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS` | `12.0` | Minimum delay between calls to scraped search engines (one call fans out to four of them) | `config.py` |
| `LLMFLOW_SEARCH_API_DELAY_SECONDS` | `2.0` | Minimum delay between calls to keyed/official search APIs | `config.py` |
| `LLMFLOW_SEARCH_ARCHIVE_DELAY_SECONDS` | `10.0` | Minimum delay between archive lookups | `config.py` |
| `LLMFLOW_SEARCH_SEARCH_JITTER` | `0.35` | Random fraction of the interval added on top of each delay | `config.py` |
| `LLMFLOW_SEARCH_MAX_SEARCH_BATCH` | `2` | Rate-limited steps run per round; the rest are deferred to the next one | `config.py` |
| `LLMFLOW_SEARCH_MAX_SEARCH_CALLS` | `12` | Hard ceiling on rate-limited calls spent answering one question | `config.py` |
| `LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES` | `5` | Pages fetched at once within a round | `config.py` |
| `LLMFLOW_SEARCH_MCP_TIMEOUT_SECONDS` | `120` | Deadline for each MCP initialize, list, or tool call | `config.py` |
| `LLMFLOW_SEARCH_OLLAMA_TIMEOUT_SECONDS` | `300` | HTTP deadline for Ollama list/chat requests | `config.py` |
| `LLMFLOW_SEARCH_AUTO_READ_TOP_K` | `5` | Pages opened unconditionally after the first search of a question. `2` reads a third as many pages for the same measured quality, at 15% fewer tokens and 43% more latency | `config.py` |
| `LLMFLOW_SEARCH_AUTO_READ_FOLLOWUP_TOP_K` | `3` | The same, for later searches | `config.py` |
| `LLMFLOW_SEARCH_READ_SELECTION` | `auto` | `model` lets the post-batch controller pick pages from the catalog instead. Measured worse on every axis — kept so the comparison can be re-run | `config.py` |
| `LLMFLOW_SEARCH_DIAGNOSE_OBSERVATIONS` | `0` | Spends one model call diagnosing every tool result. Measured at 47% of all tokens, and turning it off improved every quality measure — kept only so the comparison can be re-run | `config.py` |
| `LLMFLOW_SEARCH_IDENTIFYING_CRITERIA` | `1` | Lets requirements mark criteria that only identify the subject; strict mode then requires only the rest, and unconfirmed clues are disclosed in the answer. `0` demands proof of every criterion | `config.py` |
| `LLMFLOW_SEARCH_CITATION_REPAIR` | `1` | One pass over the verified answer for claim sentences carrying no `[n]`: each is cited or removed. `0` restores the pre-measurement behaviour | `config.py` |
| `LLMFLOW_SEARCH_CITATION_TARGET` | `0.95` | Citation coverage at or above which that pass is skipped | `config.py` |
| `LLMFLOW_SEARCH_STRUCTURE_REPAIR` | `1` | One pass over the verified report for the structure defects a check can prove: an empty heading, a comparative with no figure, a point made twice, a finding resting only on an aggregator, a finding that names nobody and counts nothing, an appendix holding more than the body. Each is fixed from the sources or cut — never by moving it to the appendix, which is inspected too. `0` leaves them in place, still counted in the trace | `config.py` |
| `LLMFLOW_SEARCH_EVAL_MODEL` | `deepseek-v4.1-flash:cloud` | Model used by the benchmark runner and the single-call baseline | `config.py` |
| `LLMFLOW_SEARCH_JUDGE_MODEL` | `gemma4:31b-cloud` | Model that judges claim support; deliberately not the model under test | `config.py` |
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

## MCP tool authorization

Tool descriptions may declare `[effect: read_only]`, `[effect: local_write]`,
`[effect: external_write]`, or `[effect: destructive]`. Undeclared tools are classified
conservatively from their action name. Read-only calls run automatically; every other class
requires explicit confirmation in the interactive launcher and is denied when no authorizer
is available.

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
