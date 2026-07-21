from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path

import markdown
from xhtml2pdf import pisa

_LEADING_H1 = re.compile(r"^#\s+(.+?)\s*\n", re.MULTILINE)
# The writer sometimes drops a bare "---" between sections as a visual separator;
# markdown turns that into an <hr>, which renders as an unstyled line cutting
# across the page. Section headings already provide the visual break, so any
# standalone rule line is stripped rather than styled.
_MARKDOWN_HR_LINE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)

# Reports are frequently non-English (Russian, Spanish, ...); the PDF base-14 fonts
# (Helvetica etc.) only cover WinAnsi Latin and silently drop Cyrillic, ñ/ü/€, and
# similar glyphs. Embed a broad-coverage system TTF via @font-face (xhtml2pdf's
# supported path for custom fonts — pre-registering directly with reportlab's
# pdfmetrics does not reach xhtml2pdf's own CSS font resolver) and fall back to
# Helvetica (Latin-only) if none of the candidates are present. Each candidate is
# paired with its true bold companion (when one exists) so headings can request
# real bold glyphs instead of a faux-bold synthesis, which can look muddy for
# scripts like Cyrillic.
_UNICODE_FONT_CANDIDATES: tuple[tuple[Path, Path | None], ...] = (
    (Path("/System/Library/Fonts/Supplemental/Tahoma.ttf"), Path("/System/Library/Fonts/Supplemental/Tahoma Bold.ttf")),
    (Path("/System/Library/Fonts/Supplemental/Verdana.ttf"), Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf")),
    (Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), None),
    (
        Path("/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/NotoSans-Bold.ttf"),
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
)


def _find_unicode_font() -> tuple[Path, Path | None] | None:
    for regular, bold in _UNICODE_FONT_CANDIDATES:
        if regular.is_file():
            return regular, bold if bold and bold.is_file() else None
    return None


def _build_style() -> str:
    found = _find_unicode_font()
    unicode_font, bold_font = found if found else (None, None)
    body_font = "ReportBody" if unicode_font else "Helvetica"
    # xhtml2pdf's url() resolver wants a plain filesystem path — a file:// scheme
    # prefix is silently un-parseable and makes it fall back to Helvetica with no
    # error, which is why this must NOT be a URI.
    font_face_rules = []
    if unicode_font:
        font_face_rules.append(f'@font-face {{ font-family: "ReportBody"; src: url("{unicode_font}"); }}')
        if bold_font:
            font_face_rules.append(
                f'@font-face {{ font-family: "ReportBody"; src: url("{bold_font}"); font-weight: bold; }}'
            )
    font_face_rule = "\n".join(font_face_rules)
    # Bold is only safe to request when a true bold face is registered (base-14
    # Helvetica always has one built in); otherwise force normal weight everywhere
    # to prevent a faux-bold render.
    has_true_bold = bold_font is not None or unicode_font is None
    weight_rule = "" if has_true_bold else "font-weight: normal;"
    heading_weight = "font-weight: bold;" if has_true_bold else ""
    return f"""
{font_face_rule}
@page {{
    size: a4 portrait;
    margin: 2.1cm 2cm 2.6cm 2cm;
    @frame header_frame {{
        -pdf-frame-content: header_content;
        top: 0.9cm; left: 2cm; right: 2cm; height: 0.9cm;
    }}
    @frame footer_frame {{
        -pdf-frame-content: footer_content;
        bottom: 1cm; left: 2cm; right: 2cm; height: 1.2cm;
    }}
}}
body {{ font-family: "{body_font}"; font-size: 8.5pt; color: #333333; line-height: 1.5; }}
#header_content img {{ height: 16px; }}
.doc-title {{ font-family: "{body_font}"; font-size: 15pt; color: #333333; margin: 0 0 2px 0; {weight_rule} }}
.doc-meta {{ font-size: 7.5pt; color: #6b6b6b; margin-bottom: 4px; }}
hr.top-rule {{ border: none; border-top: 1.2pt solid #333333; margin: 10px 0 20px 0; }}
h1 {{ font-family: "{body_font}"; font-size: 11pt; color: #333333; border-bottom: 0.8pt solid #c7d0d8;
    padding-bottom: 4px; margin-top: 22px; {heading_weight} }}
h2 {{ font-family: "{body_font}"; font-size: 10pt; color: #333333; margin-top: 18px; {heading_weight} }}
h3 {{ font-family: "{body_font}"; font-size: 9.5pt; color: #4d4d4d; margin-top: 14px; {heading_weight} }}
p {{ margin: 6px 0; text-align: left; }}
ul, ol {{ margin: 6px 0 6px 18px; padding: 0; }}
li {{ margin: 3px 0; }}
a {{ color: #1a5276; text-decoration: underline; }}
strong {{ color: #333333; {weight_rule} }}
blockquote {{ border-left: 2pt solid #c7d0d8; margin: 8px 0; padding-left: 10px; color: #5c5c5c; }}
#footer_content {{ font-size: 7.5pt; color: #8a8a8a; border-top: 0.5pt solid #c7d0d8; padding-top: 4px; }}
"""


def _extract_title(report_markdown: str, fallback: str) -> tuple[str, str]:
    """Pulls a leading '# Title' line out so it renders as a styled cover heading
    instead of a plain markdown H1, returning (title, remaining_markdown)."""
    match = _LEADING_H1.match(report_markdown.lstrip("\n"))
    if match:
        stripped = report_markdown.lstrip("\n")
        return match.group(1).strip(), stripped[match.end() :].lstrip("\n")
    return fallback, report_markdown


def _truncate_at_word_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",.;:—-") + "…"


def _logo_data_uri(logo_path: Path) -> str | None:
    if not logo_path.is_file():
        return None
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_report_pdf(
    *,
    research_id: str,
    query: str,
    report_markdown: str,
    output_path: Path,
    logo_path: Path | None = None,
) -> Path:
    """Renders a completed report's markdown into a letterhead-style business PDF."""
    title, body_markdown = _extract_title(report_markdown, fallback=query)
    body_markdown = _MARKDOWN_HR_LINE.sub("", body_markdown)
    body_html = markdown.markdown(body_markdown, extensions=["extra", "sane_lists"])
    generated_at = datetime.now().strftime("%d %B %Y, %H:%M")

    logo_html = ""
    if logo_path is not None:
        data_uri = _logo_data_uri(logo_path)
        if data_uri:
            logo_html = f'<img src="{data_uri}" alt="logo" />'

    document = f"""<html>
<head><meta charset="utf-8" /><style>{_build_style()}</style></head>
<body>
<div id="header_content">{logo_html}</div>
<div class="doc-title">{html.escape(title)}</div>
<div class="doc-meta">Deep Research Report &middot; Generated {generated_at} &middot; ID: {html.escape(research_id)}</div>
<hr class="top-rule" />
{body_html}
<div id="footer_content">
  <span>{html.escape(_truncate_at_word_boundary(title, 80))}</span> &mdash;
  <span>Page <pdf:pagenumber /> of <pdf:pagecount /></span>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        result = pisa.CreatePDF(document, dest=handle)
    err_count = getattr(result, "err", 0)
    if err_count:
        raise RuntimeError(f"PDF generation failed with {err_count} error(s)")
    return output_path


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def report_pdf_filename(research_id: str, query: str) -> str:
    slug = _SLUG_STRIP.sub("-", query.strip().lower()).strip("-")[:60] or "report"
    date = datetime.now().strftime("%Y-%m-%d")
    return f"{date}_{slug}_{research_id}.pdf"
