from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelsConfig:
    analyzer: str = "ornith:9b"
    planner: str = "ornith:9b"
    researcher: str = "ornith:9b"
    fact_checker: str = "ornith:9b"
    writer: str = "ornith:9b"


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 600
    max_parallel_requests: int = 2
    keep_alive: str | int = -1
    max_retries: int = 3
    num_predict: int = 8192
    think: bool | None = None


@dataclass(frozen=True)
class SearchConfig:
    provider: str = "footnote_mcp"
    # Secondary provider used when the primary errors or returns no results.
    fallback_provider: str | None = None
    base_url: str = "http://localhost:8080"
    language: str = "all"
    max_results_per_query: int = 5
    footnote: FootnoteConfig = field(default_factory=lambda: FootnoteConfig())


@dataclass(frozen=True)
class FootnoteConfig:
    command: str = "/Users/artemk/projects/footnote-mcp/.venv/bin/footnote-mcp"
    args: list[str] = field(default_factory=list)
    provider: str = "auto"
    semantic_rerank: bool = False
    min_search_interval_seconds: float = 8.0


@dataclass(frozen=True)
class RuntimeConfig:
    max_research_workers: int = 3
    max_parallel_fetches: int = 6
    max_research_rounds: int = 2
    fact_check_batch_size: int = 3


@dataclass(frozen=True)
class BudgetConfig:
    max_total_searches: int = 30
    max_total_pages: int = 20
    max_wall_time_minutes: int = 20
    max_retries_per_operation: int = 2
    max_duplicate_queries: int = 2


@dataclass(frozen=True)
class OutputConfig:
    # Directory (relative to the working directory the CLI is launched from) that
    # completed runs' PDF reports are written into.
    reports_dir: str = "reports"
    # Logo placed top-left on the PDF's title page. Empty string disables it.
    logo_path: str = "assets/images/llmflow.png"


@dataclass(frozen=True)
class AppConfig:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: dict, name: str, cls: type):
    values = data.get(name, {})
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    valid = {key: value for key, value in values.items() if key in cls.__dataclass_fields__}
    return cls(**valid)


def _search_section(data: dict) -> SearchConfig:
    values = data.get("search", {})
    if not isinstance(values, dict):
        raise ValueError("Configuration section 'search' must be a mapping")
    footnote_values = values.get("footnote", {})
    if not isinstance(footnote_values, dict):
        raise ValueError("Configuration section 'search.footnote' must be a mapping")
    search_values = {
        key: value for key, value in values.items() if key in SearchConfig.__dataclass_fields__ and key != "footnote"
    }
    footnote = FootnoteConfig(
        **{key: value for key, value in footnote_values.items() if key in FootnoteConfig.__dataclass_fields__}
    )
    return SearchConfig(**search_values, footnote=footnote)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return AppConfig(
        models=_section(data, "models", ModelsConfig),
        ollama=_section(data, "ollama", OllamaConfig),
        search=_search_section(data),
        runtime=_section(data, "runtime", RuntimeConfig),
        budgets=_section(data, "budgets", BudgetConfig),
        output=_section(data, "output", OutputConfig),
    )
