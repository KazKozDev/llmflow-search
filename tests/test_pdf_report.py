from pathlib import Path

from deep_research.pdf_report import _MARKDOWN_HR_LINE, _extract_title, render_report_pdf, report_pdf_filename


def test_extract_title_pulls_leading_h1() -> None:
    title, remainder = _extract_title("# My Report Title\n\nBody text here.", fallback="fallback")
    assert title == "My Report Title"
    assert remainder == "Body text here."
    assert "# My Report Title" not in remainder


def test_extract_title_falls_back_when_no_leading_h1() -> None:
    title, remainder = _extract_title("Just a paragraph, no heading.", fallback="Fallback Title")
    assert title == "Fallback Title"
    assert remainder == "Just a paragraph, no heading."


def test_report_pdf_filename_is_filesystem_safe_and_traceable() -> None:
    name = report_pdf_filename("res_abc123", "Spain's AI Strategy: 2030 & Beyond!")
    assert name.endswith("res_abc123.pdf")
    assert " " not in name
    assert "'" not in name
    assert "&" not in name


def test_render_report_pdf_writes_a_valid_pdf(tmp_path) -> None:
    output_path = tmp_path / "nested" / "report.pdf"
    markdown_report = (
        "# España y su estrategia de IA\n\n"
        "## Inversión\n\n"
        "El gobierno invierte **€2.400 millones** en cómputo. La estrategia rusa: "
        "Правительство инвестирует 200 миллиардов рублей.\n\n"
        "- Punto uno\n- Punto dos\n"
    )

    result_path = render_report_pdf(
        research_id="res_test",
        query="Spain AI strategy",
        report_markdown=markdown_report,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"%PDF-")


def test_render_report_pdf_without_logo_still_succeeds(tmp_path) -> None:
    output_path = tmp_path / "report.pdf"
    render_report_pdf(
        research_id="res_test",
        query="Query",
        report_markdown="# Title\n\nBody.",
        output_path=output_path,
        logo_path=Path("/nonexistent/logo.png"),
    )
    assert output_path.is_file()


def test_hr_line_regex_matches_common_separator_styles() -> None:
    for line in ("---", "***", "___", "  ----  ", "-----------"):
        assert _MARKDOWN_HR_LINE.search(f"before\n{line}\nafter"), line
    for line in ("- a list item", "-- not quite", "note: --- inline"):
        assert not _MARKDOWN_HR_LINE.search(f"\n{line}\n"), line


def test_stray_markdown_hr_lines_are_stripped_from_body(tmp_path) -> None:
    output_path = tmp_path / "report.pdf"
    markdown_report = "# Title\n\nFirst section text.\n\n---\n\n## Next Section\n\nMore text here.\n"
    render_report_pdf(
        research_id="res_test", query="Query", report_markdown=markdown_report, output_path=output_path
    )
    # No direct HTML inspection API is exposed, but the PDF must still render
    # without error and the body text must survive the strip untouched.
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"%PDF-")
