from pathlib import Path

from pypdf import PdfReader

from llmflow_search import reports as reports_module
from llmflow_search.pdf_report import (
    _MARKDOWN_HR_LINE,
    _extract_title,
    _truncate_at_word_boundary,
    render_report_pdf,
    report_pdf_filename,
)


def test_extract_title_pulls_leading_h1():
    title, remainder = _extract_title("# My Report Title\n\nBody text here.", fallback="fallback")
    assert title == "My Report Title"
    assert remainder == "Body text here."


def test_extract_title_falls_back_without_h1():
    title, remainder = _extract_title("Just a paragraph.", fallback="Fallback Title")
    assert title == "Fallback Title"
    assert remainder == "Just a paragraph."


def test_report_pdf_filename_is_safe_and_traceable():
    name = report_pdf_filename("res_abc123", "Spain's AI Strategy: 2030 & Beyond!")
    assert name.endswith("res_abc123.pdf")
    assert " " not in name
    assert "'" not in name
    assert "&" not in name


def test_render_report_pdf_writes_unicode_text(tmp_path):
    output_path = tmp_path / "nested" / "report.pdf"
    markdown_report = (
        "# España y стратегия ИИ\n\n"
        "## Inversión\n\n"
        "El gobierno invierte **€2.400 millones**. "
        "Правительство инвестирует 200 миллиардов рублей.\n\n"
        "- Punto uno\n- Пункт два\n"
    )

    result = render_report_pdf(
        research_id="res_test",
        query="Spain AI strategy",
        report_markdown=markdown_report,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.read_bytes().startswith(b"%PDF-")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output_path).pages)
    assert "España" in text
    assert "Правительство" in text


def test_render_report_pdf_without_logo_still_succeeds(tmp_path):
    output_path = tmp_path / "report.pdf"
    render_report_pdf(
        research_id="res_test",
        query="Query",
        report_markdown="# Title\n\nBody.",
        output_path=output_path,
        logo_path=Path("/nonexistent/logo.png"),
    )
    assert output_path.is_file()


def test_truncate_at_word_boundary():
    short = "A short title"
    assert _truncate_at_word_boundary(short, 80) == short

    title = "Venture Capital and Public-Market Investment in AI Companies: A Comparative Report"
    truncated = _truncate_at_word_boundary(title, 80)
    assert len(truncated) <= 80
    assert truncated.endswith("...")
    assert "Repo..." not in truncated


def test_markdown_rule_lines_are_recognized():
    for line in ("---", "***", "___", "  ----  ", "-----------"):
        assert _MARKDOWN_HR_LINE.search(f"before\n{line}\nafter"), line
    for line in ("- a list item", "-- not quite", "note: --- inline"):
        assert not _MARKDOWN_HR_LINE.search(f"\n{line}\n"), line


def _successful_state():
    return {
        "task": "Проверенный тестовый отчёт",
        "final_answer": "Подтверждённый ответ со ссылкой на источник [1].",
        "sources": [
            {
                "title": "Официальный источник",
                "url": "https://example.org/source",
                "content": "Подтверждённые данные.",
            }
        ],
        "verification_result": {
            "task_complete": True,
            "insufficient_evidence": False,
        },
    }


def test_successful_run_writes_pdf_with_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(reports_module, "PDF_REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(reports_module, "PDF_LOGO_PATH", "")

    result = reports_module._write_pdf_report(_successful_state())

    assert result is not None
    output_path = Path(result)
    assert output_path.parent == tmp_path
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output_path).pages)
    assert "Проверенный тестовый отчёт" in text
    assert "Подтверждённый ответ" in text
    assert "Официальный источник" in text
    assert "https://example.org/source" in text


def test_incomplete_run_does_not_write_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(reports_module, "PDF_REPORTS_DIR", str(tmp_path))
    state = _successful_state()
    state["verification_result"]["task_complete"] = False
    state["verification_result"]["insufficient_evidence"] = True

    assert reports_module._write_pdf_report(state) is None
    assert list(tmp_path.glob("*.pdf")) == []
