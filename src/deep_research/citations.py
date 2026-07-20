from __future__ import annotations

import re

from .models import Source


class CitationError(ValueError):
    pass


_CITATION_PATTERN = re.compile(r"<cite\s+(src_[a-zA-Z0-9]+)>")
# Small models drift on the tag syntax: <cite source_id="src_x">, <cite src="src_x">,
# a closing </cite>, etc. Any cite tag that names a source id is normalized to the
# canonical <cite src_x> before validation instead of failing the whole report.
_CITE_TAG_VARIANT = re.compile(r"<cite\b[^>]*?(src_[a-zA-Z0-9]+)[^>]*>")
_CITE_CLOSING = re.compile(r"\s*</cite>")
_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
_LATIN = re.compile(r"[a-zA-Z]")


def normalize_citation_syntax(markdown: str) -> str:
    markdown = _CITE_TAG_VARIANT.sub(lambda match: f"<cite {match.group(1)}>", markdown)
    return _CITE_CLOSING.sub("", markdown)


def _escape_markdown_link_text(title: str) -> str:
    return title.replace("[", "\\[").replace("]", "\\]")


def _references_heading(text: str) -> str:
    # The report body is written in the user's language by the model; the generated
    # heading should match rather than sticking out as an unrelated English label.
    return "Источники" if len(_CYRILLIC.findall(text)) > len(_LATIN.findall(text)) else "References"


def validate_and_render_citations(markdown: str, sources: dict[str, Source]) -> str:
    """Renders <cite src_x> tags as numbered academic citations: [1], [2] inline,
    matched to a numbered reference list appended after the body. Sources are
    numbered by order of first appearance in the text, the standard numeric/IEEE
    citation convention."""
    markdown = normalize_citation_syntax(markdown)
    cited = _CITATION_PATTERN.findall(markdown)
    if not cited:
        if sources:
            raise CitationError("Report contains no evidence citations")
        return markdown
    unknown = sorted(set(cited) - set(sources))
    if unknown:
        raise CitationError(f"Report refers to unknown source IDs: {', '.join(unknown)}")

    numbers: dict[str, int] = {}
    for source_id in cited:
        if source_id not in numbers:
            numbers[source_id] = len(numbers) + 1

    def replace(match: re.Match[str]) -> str:
        return f"[{numbers[match.group(1)]}]"

    body = _CITATION_PATTERN.sub(replace, markdown)
    if "<cite" in body:
        raise CitationError("Report contains malformed citations")

    reference_lines = [
        f"[{number}] [{_escape_markdown_link_text(sources[source_id].title)}]({sources[source_id].url})"
        for source_id, number in sorted(numbers.items(), key=lambda item: item[1])
    ]
    heading = _references_heading(body)
    # Markdown collapses lines separated by a single newline into one run-on
    # paragraph — a "- " list-item prefix is what actually forces one line per
    # reference, since list syntax (unlike plain paragraphs) doesn't need blank
    # lines between items.
    reference_list = "\n".join(f"- {line}" for line in reference_lines)
    return f"{body.rstrip()}\n\n## {heading}\n\n{reference_list}\n"
