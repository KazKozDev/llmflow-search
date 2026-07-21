import asyncio
import json

import pytest

from deep_research.evaluation import BenchmarkCase, load_cases, run_benchmark, score_report
from deep_research.store import EvidenceStore


def test_scores_cited_report_against_case() -> None:
    case = BenchmarkCase(
        case_id="example",
        query="Explain example behaviour",
        expected_keywords=["json"],
        required_source_domains=["example.org"],
        min_citations=1,
    )
    report = "The API returns JSON [Official documentation](https://docs.example.org/api)."

    score = score_report(case, report, "res_123")

    assert score.passed is True
    assert score.citation_count == 1


def test_reports_missing_criteria() -> None:
    case = BenchmarkCase(
        case_id="example",
        query="Explain example behaviour",
        expected_keywords=["json"],
        required_source_domains=["example.org"],
    )

    score = score_report(case, "No citations here.", "res_123")

    assert score.passed is False
    assert score.missing_keywords == ["json"]
    assert score.missing_required_domains == ["example.org"]


def test_loads_jsonl_cases(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id":"a","query":"A valid question"}\n', encoding="utf-8")

    cases = load_cases(path)

    assert cases[0].case_id == "a"


def test_load_cases_skips_blank_lines_and_comments(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        "# a leading comment\n"
        '{"case_id":"a","query":"A valid question"}\n'
        "   \n"
        '{"case_id":"b","query":"Another valid question"}\n',
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert [case.case_id for case in cases] == ["a", "b"]


def test_load_cases_raises_on_invalid_json_line(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id": "a", "query": "a valid query"}\nnot valid json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid benchmark case at line 2"):
        load_cases(path)


def test_load_cases_raises_when_file_has_no_cases(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n# only comments\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no cases"):
        load_cases(path)


def test_run_benchmark_scores_each_case_and_writes_output(tmp_path, monkeypatch) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"case_id":"a","query":"Explain example behaviour","expected_keywords":["json"],"min_citations":1}\n',
        encoding="utf-8",
    )
    database_path = tmp_path / "db.sqlite3"
    output_path = tmp_path / "results" / "latest.json"

    async def fake_run_research(query, config_path, database_path, quiet=False):
        store = EvidenceStore(database_path)
        try:
            store.create_run("res_a", query, "2026-01-01T00:00:00Z", {})
            store.save_report("res_a", "The API returns JSON [Docs](https://docs.example.org/api).", "2026-01-01T00:00:00Z")
        finally:
            store.close()
        return "res_a"

    monkeypatch.setattr("deep_research.evaluation.run_research", fake_run_research)

    result = asyncio.run(run_benchmark(cases_path, "config.yaml", database_path, output_path))

    assert result["case_count"] == 1
    assert result["passed_count"] == 1
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["scores"][0]["case_id"] == "a"
