import asyncio

import pytest

from deep_research import cli
from deep_research.cli import build_parser
from deep_research.config import AppConfig, FootnoteConfig, SearchConfig
from deep_research.models import EvidenceItem, ResearchStatus, Source
from deep_research.store import EvidenceStore
from deep_research.tools import (
    ClassifyingFetcher,
    FallbackFetcher,
    FallbackSearchProvider,
    FootnoteMCPProvider,
    PageFetcher,
    SearxNGProvider,
)


def test_run_print_controls_are_available() -> None:
    args = build_parser().parse_args(["run", "research question"])
    assert args.quiet is False
    assert args.no_report is False

    quiet_args = build_parser().parse_args(["run", "research question", "--quiet", "--no-report"])
    assert quiet_args.quiet is True
    assert quiet_args.no_report is True


def _config(**search_overrides) -> AppConfig:
    return AppConfig(search=SearchConfig(**search_overrides))


def test_footnote_provider_builds_from_config() -> None:
    config = AppConfig(
        search=SearchConfig(
            footnote=FootnoteConfig(command="/bin/footnote-mcp", args=["--flag"], provider="duckduckgo")
        )
    )

    provider = cli._footnote_provider(config)

    assert isinstance(provider, FootnoteMCPProvider)
    assert provider.command == "/bin/footnote-mcp"
    assert provider.args == ["--flag"]
    assert provider.provider == "duckduckgo"


def test_build_orchestrator_wires_footnote_mcp_provider(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "db.sqlite3")
    try:
        orchestrator = cli._build_orchestrator(_config(provider="footnote_mcp"), store, quiet=True)

        assert isinstance(orchestrator.worker.search, FootnoteMCPProvider)
        assert isinstance(orchestrator.worker.fetcher, ClassifyingFetcher)
        assert orchestrator.worker.fetcher.inner is orchestrator.worker.search
        assert orchestrator.on_event is None
    finally:
        store.close()


def test_build_orchestrator_wires_searxng_without_fallback(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "db.sqlite3")
    try:
        orchestrator = cli._build_orchestrator(_config(provider="searxng"), store, quiet=False)

        assert isinstance(orchestrator.worker.search, SearxNGProvider)
        assert isinstance(orchestrator.worker.fetcher.inner, PageFetcher)
        assert orchestrator.on_event is not None
    finally:
        store.close()


def test_build_orchestrator_wires_searxng_with_footnote_fallback(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "db.sqlite3")
    try:
        orchestrator = cli._build_orchestrator(
            _config(provider="searxng", fallback_provider="footnote_mcp"), store, quiet=True
        )

        assert isinstance(orchestrator.worker.search, FallbackSearchProvider)
        assert isinstance(orchestrator.worker.fetcher.inner, FallbackFetcher)
    finally:
        store.close()


def test_build_orchestrator_rejects_unknown_fallback_provider(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "db.sqlite3")
    try:
        with pytest.raises(ValueError, match="fallback_provider"):
            cli._build_orchestrator(_config(provider="searxng", fallback_provider="bing"), store, quiet=True)
    finally:
        store.close()


def test_build_orchestrator_rejects_unknown_provider(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "db.sqlite3")
    try:
        with pytest.raises(ValueError, match="search.provider"):
            cli._build_orchestrator(_config(provider="bing"), store, quiet=True)
    finally:
        store.close()


class _StubOrchestrator:
    last_instance = None

    def __init__(self) -> None:
        self.ran_query: str | None = None
        self.resumed_id: str | None = None
        _StubOrchestrator.last_instance = self

    async def run(self, query: str) -> str:
        self.ran_query = query
        return "res_stub"

    async def resume(self, research_id: str) -> str:
        self.resumed_id = research_id
        return research_id


def test_run_research_builds_orchestrator_with_default_config_when_path_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_build_orchestrator", lambda config, store, quiet: _StubOrchestrator())
    database_path = tmp_path / "db.sqlite3"

    research_id = asyncio.run(cli.run_research("what is RAG", "no-such-config.yaml", str(database_path)))

    assert research_id == "res_stub"
    assert _StubOrchestrator.last_instance.ran_query == "what is RAG"
    assert database_path.exists()


def test_resume_research_delegates_to_orchestrator(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_build_orchestrator", lambda config, store, quiet: _StubOrchestrator())
    database_path = tmp_path / "db.sqlite3"

    research_id = asyncio.run(cli.resume_research("res_existing", "no-such-config.yaml", str(database_path)))

    assert research_id == "res_existing"
    assert _StubOrchestrator.last_instance.resumed_id == "res_existing"


def test_main_run_prints_research_id_and_report(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_build_orchestrator", lambda config, store, quiet: _StubOrchestrator())
    database_path = tmp_path / "db.sqlite3"
    store = EvidenceStore(database_path)
    store.create_run("res_stub", "q", "2026-01-01T00:00:00Z", {})
    store.save_report("res_stub", "# Report body", "2026-01-01T00:00:00Z")
    store.close()

    monkeypatch.setattr(
        "sys.argv", ["deep-research", "--database", str(database_path), "run", "what is RAG"]
    )
    cli.main()

    output = capsys.readouterr().out
    assert "Research ID: res_stub" in output
    assert "# Report body" in output


def test_main_run_with_no_report_skips_report_printing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_build_orchestrator", lambda config, store, quiet: _StubOrchestrator())
    database_path = tmp_path / "db.sqlite3"

    monkeypatch.setattr(
        "sys.argv",
        ["deep-research", "--database", str(database_path), "run", "what is RAG", "--no-report"],
    )
    cli.main()

    output = capsys.readouterr().out
    assert "Research ID: res_stub" in output
    assert "--- Report ---" not in output


def test_main_resume_dispatches_to_resume_research(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_build_orchestrator", lambda config, store, quiet: _StubOrchestrator())
    database_path = tmp_path / "db.sqlite3"
    store = EvidenceStore(database_path)
    store.create_run("res_existing", "q", "2026-01-01T00:00:00Z", {})
    store.save_report("res_existing", "resumed report", "2026-01-01T00:00:00Z")
    store.close()

    monkeypatch.setattr(
        "sys.argv", ["deep-research", "--database", str(database_path), "resume", "res_existing"]
    )
    cli.main()

    output = capsys.readouterr().out
    assert "Research ID: res_existing" in output
    assert "resumed report" in output


def test_main_status_prints_run_summary(tmp_path, monkeypatch, capsys) -> None:
    database_path = tmp_path / "db.sqlite3"
    store = EvidenceStore(database_path)
    store.create_run("res_1", "some query", "2026-01-01T00:00:00Z", {})
    store.close()

    monkeypatch.setattr("sys.argv", ["deep-research", "--database", str(database_path), "status", "res_1"])
    cli.main()

    output = capsys.readouterr().out
    assert '"research_id": "res_1"' in output


def test_main_report_prints_saved_report(tmp_path, monkeypatch, capsys) -> None:
    database_path = tmp_path / "db.sqlite3"
    store = EvidenceStore(database_path)
    store.create_run("res_1", "some query", "2026-01-01T00:00:00Z", {})
    store.save_report("res_1", "Final report text", "2026-01-01T00:00:00Z")
    store.close()

    monkeypatch.setattr("sys.argv", ["deep-research", "--database", str(database_path), "report", "res_1"])
    cli.main()

    assert "Final report text" in capsys.readouterr().out


def test_main_resumable_prints_nothing_when_none_pending(tmp_path, monkeypatch, capsys) -> None:
    database_path = tmp_path / "db.sqlite3"
    EvidenceStore(database_path).close()

    monkeypatch.setattr("sys.argv", ["deep-research", "--database", str(database_path), "resumable"])
    cli.main()

    assert capsys.readouterr().out == ""


def test_main_resumable_prints_id_of_interrupted_run_with_evidence(tmp_path, monkeypatch, capsys) -> None:
    database_path = tmp_path / "db.sqlite3"
    store = EvidenceStore(database_path)
    store.create_run("res_interrupted", "some query", "2026-01-01T00:00:00Z", {})
    store.update_status("res_interrupted", ResearchStatus.RESEARCHING, "2026-01-01T00:01:00Z")
    source_id = store.save_source(
        Source(
            url="https://example.org/a",
            canonical_url="https://example.org/a",
            title="Example",
            content_hash="sha256:x",
            quality_score=0.8,
        )
    )
    store.save_evidence(
        EvidenceItem(
            research_id="res_interrupted",
            task_id="task_1",
            claim="claim",
            quote="quote",
            source_id=source_id,
            relevance=0.8,
            source_quality=0.8,
            support_type="supports",
        )
    )
    store.close()

    monkeypatch.setattr("sys.argv", ["deep-research", "--database", str(database_path), "resumable"])
    cli.main()

    assert capsys.readouterr().out.strip() == "res_interrupted"


def test_main_smoke_success_prints_result(tmp_path, monkeypatch, capsys) -> None:
    async def fake_smoke(config_path, database_path):
        return {"passed": True, "checks": []}

    monkeypatch.setattr("deep_research.production.run_production_smoke", fake_smoke)
    monkeypatch.setattr("sys.argv", ["deep-research", "smoke"])

    cli.main()

    output = capsys.readouterr().out
    assert '"passed": true' in output


def test_main_smoke_failure_exits_nonzero(monkeypatch) -> None:
    async def fake_smoke(config_path, database_path):
        return {"passed": False, "checks": []}

    monkeypatch.setattr("deep_research.production.run_production_smoke", fake_smoke)
    monkeypatch.setattr("sys.argv", ["deep-research", "smoke"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1


def test_main_benchmark_success_prints_result(monkeypatch, capsys) -> None:
    async def fake_benchmark(cases_path, config_path, database_path, output_path):
        return {"passed_count": 2, "case_count": 2, "scores": []}

    monkeypatch.setattr("deep_research.evaluation.run_benchmark", fake_benchmark)
    monkeypatch.setattr("sys.argv", ["deep-research", "benchmark", "--cases", "cases.jsonl"])

    cli.main()

    output = capsys.readouterr().out
    assert '"case_count": 2' in output


def test_main_benchmark_partial_failure_exits_nonzero(monkeypatch) -> None:
    async def fake_benchmark(cases_path, config_path, database_path, output_path):
        return {"passed_count": 1, "case_count": 2, "scores": []}

    monkeypatch.setattr("deep_research.evaluation.run_benchmark", fake_benchmark)
    monkeypatch.setattr("sys.argv", ["deep-research", "benchmark", "--cases", "cases.jsonl"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
