from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import load_config
from .llm import OllamaClient
from .orchestrator import ResearchOrchestrator
from .progress import TerminalProgressReporter
from .store import EvidenceStore
from .tools import (
    ClassifyingFetcher,
    DomainClassifier,
    FallbackFetcher,
    FallbackSearchProvider,
    FootnoteMCPProvider,
    PageFetcher,
    SearxNGProvider,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evidence-first multi-agent deep research")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--database", default="data/research.sqlite3", help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a new research task")
    run.add_argument("query")
    run.add_argument("--quiet", action="store_true", help="Do not print progress to the terminal")
    run.add_argument("--no-report", action="store_true", help="Do not print the completed report")
    resume = subparsers.add_parser("resume", help="Continue an interrupted run from the last checkpoint")
    resume.add_argument("research_id")
    resume.add_argument("--quiet", action="store_true", help="Do not print progress to the terminal")
    resume.add_argument("--no-report", action="store_true", help="Do not print the completed report")
    subparsers.add_parser(
        "resumable",
        help="Print the id of the most recent interrupted run that can be resumed (empty if none)",
    )
    status = subparsers.add_parser("status", help="Show a research task status")
    status.add_argument("research_id")
    report = subparsers.add_parser("report", help="Print a completed report")
    report.add_argument("research_id")
    smoke = subparsers.add_parser("smoke", help="Check the live Ollama, SearXNG, and SQLite path")
    smoke.add_argument("--smoke-database", default="data/production-smoke.sqlite3")
    benchmark = subparsers.add_parser("benchmark", help="Run a JSONL benchmark suite")
    benchmark.add_argument("--cases", required=True, help="Path to benchmark JSONL cases")
    benchmark.add_argument("--output", default="benchmarks/results/latest.json")
    return parser


def _footnote_provider(config) -> FootnoteMCPProvider:
    return FootnoteMCPProvider(
        config.search.footnote.command,
        config.search.footnote.args,
        config.search.language,
        config.search.footnote.provider,
        config.search.footnote.semantic_rerank,
        config.search.footnote.min_search_interval_seconds,
    )


def _build_orchestrator(config, store: EvidenceStore, quiet: bool) -> ResearchOrchestrator:
    llm = OllamaClient(
        config.ollama.base_url,
        config.ollama.timeout_seconds,
        keep_alive=config.ollama.keep_alive,
        max_retries=config.ollama.max_retries,
        num_predict=config.ollama.num_predict,
        think=config.ollama.think,
    )
    if config.search.provider == "footnote_mcp":
        search = _footnote_provider(config)
        fetcher = search
    elif config.search.provider == "searxng":
        search = SearxNGProvider(
            config.search.base_url,
            config.search.language,
            timeout_seconds=min(config.ollama.timeout_seconds, 30),
        )
        fetcher = PageFetcher()
        if config.search.fallback_provider == "footnote_mcp":
            mcp = _footnote_provider(config)
            search = FallbackSearchProvider(search, mcp)
            fetcher = FallbackFetcher(fetcher, mcp)
        elif config.search.fallback_provider is not None:
            raise ValueError("search.fallback_provider must be 'footnote_mcp' or omitted")
    else:
        raise ValueError("search.provider must be 'footnote_mcp' or 'searxng'")
    # Unknown domains get one cached LLM classification; static rules stay the fast path.
    fetcher = ClassifyingFetcher(fetcher, DomainClassifier(llm, config.models.researcher, store))
    return ResearchOrchestrator(
        config=config,
        llm=llm,
        search=search,
        fetcher=fetcher,
        store=store,
        on_event=None if quiet else TerminalProgressReporter(),
    )


async def run_research(query: str, config_path: str, database_path: str, quiet: bool = False) -> str:
    config = load_config(config_path if Path(config_path).exists() else None)
    store = EvidenceStore(database_path)
    try:
        orchestrator = _build_orchestrator(config, store, quiet)
        return await orchestrator.run(query)
    finally:
        store.close()


async def resume_research(research_id: str, config_path: str, database_path: str, quiet: bool = False) -> str:
    config = load_config(config_path if Path(config_path).exists() else None)
    store = EvidenceStore(database_path)
    try:
        orchestrator = _build_orchestrator(config, store, quiet)
        return await orchestrator.resume(research_id)
    finally:
        store.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"run", "resume"}:
        if args.command == "run":
            research_id = asyncio.run(run_research(args.query, args.config, args.database, args.quiet))
        else:
            research_id = asyncio.run(resume_research(args.research_id, args.config, args.database, args.quiet))
        print(f"\nResearch ID: {research_id}")
        if not args.no_report:
            store = EvidenceStore(args.database)
            try:
                print("\n--- Report ---\n")
                print(store.get_report(research_id))
            finally:
                store.close()
        return
    if args.command == "smoke":
        from .production import run_production_smoke

        result = asyncio.run(run_production_smoke(args.config, args.smoke_database))
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "benchmark":
        from .evaluation import run_benchmark

        result = asyncio.run(run_benchmark(args.cases, args.config, args.database, args.output))
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
        if result["passed_count"] != result["case_count"]:
            raise SystemExit(1)
        return
    store = EvidenceStore(args.database)
    try:
        if args.command == "status":
            print(store.get_run(args.research_id).model_dump_json(indent=2))
        elif args.command == "report":
            print(store.get_report(args.research_id))
        elif args.command == "resumable":
            candidate = store.latest_resumable()
            if candidate is not None:
                print(candidate.research_id)
    finally:
        store.close()


if __name__ == "__main__":
    main()
