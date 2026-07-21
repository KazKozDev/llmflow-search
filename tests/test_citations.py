import pytest

from deep_research.citations import CitationError, validate_and_render_citations
from deep_research.models import Source


def source(source_id: str = "src_abc123", title: str = "Example source", url: str = "https://example.org/article") -> Source:
    return Source(
        source_id=source_id,
        url=url,
        canonical_url=url,
        title=title,
        content_hash="sha256:test",
        quality_score=0.9,
    )


def test_renders_known_citation_as_a_numbered_bracket() -> None:
    rendered = validate_and_render_citations("Fact <cite src_abc123>", {"src_abc123": source()})
    assert "Fact [1]" in rendered
    assert "<cite" not in rendered


def test_appends_a_numbered_reference_list() -> None:
    rendered = validate_and_render_citations("Fact <cite src_abc123>", {"src_abc123": source()})
    assert "## References" in rendered
    assert "[1] [Example source](https://example.org/article)" in rendered


def test_numbers_sources_by_order_of_first_appearance() -> None:
    src_a = source("src_a", "First source", "https://example.org/a")
    src_b = source("src_b", "Second source", "https://example.org/b")
    rendered = validate_and_render_citations(
        "Second fact <cite src_b>. First fact <cite src_a>.",
        {"src_a": src_a, "src_b": src_b},
    )
    assert "Second fact [1]" in rendered
    assert "First fact [2]" in rendered
    assert "[1] [Second source](https://example.org/b)" in rendered
    assert "[2] [First source](https://example.org/a)" in rendered


def test_repeated_citation_reuses_the_same_number() -> None:
    rendered = validate_and_render_citations(
        "Fact one <cite src_abc123>. Fact two <cite src_abc123>.",
        {"src_abc123": source()},
    )
    body, _, references = rendered.partition("## References")
    assert body.count("[1]") == 2
    assert references.count("[1] [Example source]") == 1


def test_reference_heading_matches_report_language() -> None:
    rendered_en = validate_and_render_citations("The fact is true <cite src_abc123>.", {"src_abc123": source()})
    assert "## References" in rendered_en

    rendered_ru = validate_and_render_citations(
        "Некоторый факт подтверждён источником <cite src_abc123>.", {"src_abc123": source()}
    )
    assert "## Источники" in rendered_ru


def test_rejects_malformed_citation_left_after_substitution() -> None:
    with pytest.raises(CitationError, match="malformed citations"):
        validate_and_render_citations(
            "Fact one <cite src_abc123>. Bad tag <cite foo>.", {"src_abc123": source()}
        )


def test_rejects_unknown_citation() -> None:
    with pytest.raises(CitationError, match="unknown source"):
        validate_and_render_citations("Fact <cite src_missing>", {"src_abc123": source()})


def test_allows_no_citation_when_no_sources_exist() -> None:
    assert validate_and_render_citations("No evidence found.", {}) == "No evidence found."


def test_normalizes_attribute_style_citations() -> None:
    rendered = validate_and_render_citations('Fact <cite source_id="src_abc123">', {"src_abc123": source()})
    assert "Fact [1]" in rendered


def test_normalizes_quoted_src_and_closing_tags() -> None:
    rendered = validate_and_render_citations(
        'Fact <cite src="src_abc123">quoted</cite> and more.', {"src_abc123": source()}
    )
    assert "Fact [1]quoted and more." in rendered


def test_rejects_report_without_any_citations_when_sources_exist() -> None:
    with pytest.raises(CitationError, match="no evidence citations"):
        validate_and_render_citations("A fact with no tag.", {"src_abc123": source()})


def test_escapes_square_brackets_in_reference_titles() -> None:
    rendered = validate_and_render_citations(
        "Fact <cite src_abc123>", {"src_abc123": source(title="Report [Draft]")}
    )
    assert "[1] [Report \\[Draft\\]](https://example.org/article)" in rendered


def test_references_are_rendered_as_a_markdown_list_not_a_run_on_paragraph() -> None:
    src_a = source("src_a", "First source", "https://example.org/a")
    src_b = source("src_b", "Second source", "https://example.org/b")
    rendered = validate_and_render_citations(
        "Fact one <cite src_a>. Fact two <cite src_b>.", {"src_a": src_a, "src_b": src_b}
    )
    lines = [line for line in rendered.splitlines() if line.startswith("- [")]
    assert lines == [
        "- [1] [First source](https://example.org/a)",
        "- [2] [Second source](https://example.org/b)",
    ]
