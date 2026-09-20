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
    title, remainder = _extract_title(
        "# My Report Title\n\nBody text here.", fallback="fallback"
    )
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
        "# España y la estrategia de IA\n\n"
        "## Inversión\n\n"
        "El gobierno invierte **€2.400 millones**. "
        "Investigación conjunta con Ελλάδα por 200 millones.\n\n"
        "- Punto uno\n- Sección dos\n"
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
    assert "Investigación" in text


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
        "task": "Informe de prueba verificado",
        "final_answer": "Respuesta verificada con una cita [1].",
        "sources": [
            {
                "title": "Fuente oficial",
                "url": "https://example.org/source",
                "content": "Datos verificados.",
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
    assert "Informe de prueba verificado" in text
    assert "Respuesta verificada" in text
    assert "Fuente oficial" in text
    assert "https://example.org/source" in text


def test_incomplete_run_does_not_write_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(reports_module, "PDF_REPORTS_DIR", str(tmp_path))
    state = _successful_state()
    state["verification_result"]["task_complete"] = False
    state["verification_result"]["insufficient_evidence"] = True

    assert reports_module._write_pdf_report(state) is None
    assert list(tmp_path.glob("*.pdf")) == []


def _dated_state():
    state = _successful_state()
    state["final_answer"] = (
        "Respuesta verificada con citas [1][2][3]."
    )
    state["sources"] = [
        {
            "title": "Vendor announcement",
            "url": "https://vendor.example/blog",
            "published": "2026-03-04",
            "source_quality": {"source_type": "vendor blog"},
            "content": "Vendor claim.",
        },
        {
            "title": "Regulator filing",
            "url": "https://regulator.example/filing",
            "published": "2026-01-09T10:00:00Z",
            "kind": "fetched_page",
            "content": "Filed figures.",
        },
        {
            "title": "Undated wiki page",
            "url": "https://wiki.example/page",
            "content": "Background.",
        },
    ]
    return state


def test_report_header_separates_fresh_evidence_from_background():
    """A first-to-last range hides the mixing it was added to expose: one stale page cited
    beside today's news prints as a seven-month window that describes neither."""
    report = reports_module._report_markdown(_dated_state())

    assert "Evidence window: 2026-03-04 (1 of 2 dated sources)" in report
    assert "1 older, 2026-01-09" in report
    assert "3 sources" in report
    assert "1 undated" in report
    # The window belongs above the body, or it cannot frame what follows.
    assert report.index("Evidence window") < report.index("Respuesta verificada")


def test_single_dated_source_reports_one_date_not_a_range():
    state = _successful_state()
    state["sources"] = [{"url": "https://example.org/a", "published": "2026-02-02"}]

    report = reports_module._report_markdown(state)

    assert "Evidence window: 2026-02-02 ·" in report
    assert " to " not in report.split("\n")[2]
    assert "undated" not in report.split("\n")[2]
    assert "older" not in report.split("\n")[2]


def test_sources_carry_date_and_type_for_checking_claims():
    report = reports_module._report_markdown(_dated_state())

    assert "(2026-03-04, vendor blog)" in report
    assert "(2026-01-09, fetched_page)" in report
    assert "(undated)" in report


def test_report_without_sources_omits_the_window_line():
    state = _successful_state()
    state["sources"] = []

    report = reports_module._report_markdown(state)

    assert "Evidence window" not in report
    assert "## Sources" not in report


def test_only_the_sources_the_text_cites_are_listed():
    """The admitted list is not the evidence: the prompt's size cuts it, and the writer
    cites a subset of what it was shown. Listing the rest credits the report with pages
    that could not have informed a word of it."""
    state = _dated_state()
    state["final_answer"] = "One finding, from the filing alone [2]."

    report = reports_module._report_markdown(state)

    assert "regulator.example" in report
    assert "vendor.example" not in report
    assert "wiki.example" not in report


def test_citation_numbers_survive_the_filtering():
    """Renumbering the bibliography would point every [n] in the text at the wrong entry."""
    state = _dated_state()
    state["final_answer"] = "One finding, from the filing alone [2]."

    report = reports_module._report_markdown(state)

    assert "2. [Regulator filing]" in report
    assert "1. [Regulator filing]" not in report


def test_the_window_covers_the_cited_sources_only():
    """An uncited page cannot widen the window of a report it is not in."""
    state = _dated_state()
    state["final_answer"] = "One finding, from the filing alone [2]."

    report = reports_module._report_markdown(state)

    assert "Evidence window: 2026-01-09 ·" in report
    assert "1 source ·" in report
    assert "undated" not in report


def test_a_report_with_no_citations_still_lists_its_sources():
    """A missing marker is a failure of the citation pass, not proof of no evidence."""
    state = _dated_state()
    state["final_answer"] = "A finding with no marker at all."

    report = reports_module._report_markdown(state)

    assert "3 sources" in report
    assert "vendor.example" in report


def test_a_date_shaped_string_that_is_not_a_date_is_treated_as_undated():
    """`published` is whatever the page declared. "2026-02-30" has the shape and not the
    day, and parsing it took a finished run down while it was writing its report."""
    assert reports_module._source_date({"published": "2026-02-30"}) == ""
    assert reports_module._source_date({"published": "2026-13-01"}) == ""
    assert reports_module._source_date({"published": "2026-03-04"}) == "2026-03-04"


def test_an_impossible_date_does_not_break_the_window():
    window = reports_module._evidence_window(
        [{"published": "2026-02-30"}, {"published": "2026-03-04"}]
    )
    assert "2026-03-04" in window
    assert "1 undated" in window
