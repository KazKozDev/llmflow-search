from deep_research.evaluation import BenchmarkCase, load_cases, score_report


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
