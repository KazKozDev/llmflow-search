"""Is each sentence of the answer actually supported by the source it cites?

The pipeline already refuses to answer without an evidence ledger, and the ledger ties
every admitted claim to a source id. That is a claim about the *ledger*, not about the
prose: the answer is written by a model that is shown the sources and asked to cite them
inline, and nothing downstream re-reads the cited page to check that the sentence it is
attached to is in there. "Cited" and "supported" are different properties, and only the
first one was ever measured — a run that cites five real URLs and states a sixth fact
they do not contain scores exactly like a run that does not.

This module is the deterministic half of measuring the second property. It cuts the
answer into the units that carry claims, reads the ``[n]`` markers that bind each unit to
a source, and — given a quote a judge says supports a unit — checks that the quote is
genuinely present in that source's text. The judging itself is a model call and lives in
``scripts/score_attribution.py``; everything here runs offline, is unit-testable, and
does not depend on which judge was used.

The quote check is what makes the metric worth trusting. A judge asked only for a verdict
can return "supported" for a sentence the page never makes, and nothing in the transcript
distinguishes that from a correct verdict. A judge required to copy out the supporting
span can still be wrong about entailment, but it can no longer be wrong about *existence*
without that being visible here.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .config import INSUFFICIENT_EVIDENCE_MESSAGE

# A citation marker: [3], [3, 4] or [3][4]. Ranges ([2-4]) are not emitted by the answer
# prompts and are deliberately not parsed — silently expanding one would invent citations
# the model did not make.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Where a sentence ends: terminal punctuation followed by space and something that starts
# a new sentence. A citation marker counts as a start, since answers routinely open a
# sentence with one.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")

# The trailing bibliography. It repeats URLs that are already bound to claims by their
# markers, so counting its lines as claims would inflate both the numerator and the
# denominator with text that asserts nothing.
_SOURCES_HEADING = re.compile(
    r"^\s*(#{1,6}\s*)?(\*\*)?\s*(sources|references|citations)\b\s*:?\s*(\*\*)?\s*$",
    re.IGNORECASE,
)

_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# A unit shorter than this is a label, a heading or a fragment, not an assertion anyone
# could check against a page.
MIN_CLAIM_WORDS = 4

# Sentences that report the run's own state rather than assert anything about the world.
# A refusal makes no claims, so it must contribute none: counted as an unsupported claim
# it would punish the fail-closed behaviour the pipeline exists to provide, and counted as
# a supported one it would reward saying nothing. Either way the number would stop meaning
# "of what this answer asserts".
_STATUS_SENTENCES = (INSUFFICIENT_EVIDENCE_MESSAGE,)

# A judge that is told to copy a span often copies two and joins them with an ellipsis —
# the sentence that states the claim and the clause elsewhere on the page that qualifies
# it. Those are real spans, so each part is located separately rather than the whole
# string being looked for and missed. Parts shorter than this are punctuation debris from
# the split, not evidence, and are ignored.
_ELLIPSIS = re.compile(r"\s*(?:\.\s*\.\s*\.+|…|\[\s*\.\.\.\s*\])\s*")
MIN_QUOTE_FRAGMENT_CHARS = 20

# How close a judge's quote must come to text that is really in the source before it
# counts as found. Below 1.0 because models normalize away soft hyphens, ellipses and
# non-breaking spaces while copying; far enough above chance that a paraphrase fails.
QUOTE_MATCH_RATIO = 0.92

_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have in into is it its of on
    or that the their there these this to was were which who will with within would
    about after also been over under between during more most other such than then
    they them when where while you your we our not no can could should may might""".split()
)


@dataclass
class Claim:
    """One checkable unit of the answer, with the sources it points at."""

    index: int
    text: str
    citations: list[int] = field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return bool(self.citations)

    @property
    def bare_text(self) -> str:
        """The unit with its citation markers removed, for showing to a judge."""
        stripped = _collapse(_CITATION.sub(" ", self.text))
        # A marker sits before the period it belongs to; removing it must not leave the
        # punctuation floating, or every claim reaches the judge looking mistyped.
        return re.sub(r"\s+([.,;:!?)])", r"\1", stripped)


def split_claims(answer: str) -> list[Claim]:
    """Cut an answer into the units a citation can be attached to.

    Sentences, except that a list item or a table row is one unit whether or not it ends
    in a period — answers put one fact per bullet, and a bullet with no terminal
    punctuation is still an assertion.
    """
    claims: list[Claim] = []
    for line in _body_lines(answer):
        for unit in _units_in_line(line):
            text = _collapse(unit)
            if not _is_claim(text):
                continue
            claims.append(
                Claim(index=len(claims), text=text, citations=citations_in(text))
            )
    return claims


def citations_in(text: str) -> list[int]:
    """Every source id the text points at, deduplicated, in order of first appearance."""
    found: list[int] = []
    for match in _CITATION.finditer(text or ""):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            source_id = int(part)
            if source_id not in found:
                found.append(source_id)
    return found


def _body_lines(answer: str) -> list[str]:
    lines: list[str] = []
    for line in (answer or "").splitlines():
        if _SOURCES_HEADING.match(line):
            break  # everything below is the bibliography
        lines.append(line)
    return lines


def _units_in_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or _MARKDOWN_HEADING.match(stripped):
        return []
    if _TABLE_ROW.match(stripped):
        # A separator row (|---|---|) asserts nothing; a data row is one unit.
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            return []
        return [" ".join(cell for cell in cells if cell)]
    stripped = _LIST_MARKER.sub("", stripped)
    return [part for part in _SENTENCE_BREAK.split(stripped) if part.strip()]


def is_refusal(answer: str) -> bool:
    """Whether the whole answer is the pipeline declining to answer."""
    return normalize(answer).startswith(normalize(INSUFFICIENT_EVIDENCE_MESSAGE))


def _is_claim(text: str) -> bool:
    """Whether a unit asserts something, as opposed to labelling or introducing it."""
    bare = _collapse(_CITATION.sub(" ", text))
    if any(normalize(bare) == normalize(status) for status in _STATUS_SENTENCES):
        return False
    words = [word for word in re.findall(r"[^\W\d_]+", bare, re.UNICODE)]
    if len(words) < MIN_CLAIM_WORDS:
        return False
    # "Key findings:" and friends: a colon-terminated lead-in whose content is the
    # bullets under it, which are counted on their own.
    return not bare.rstrip().endswith(":")


def normalize(text: str) -> str:
    """Fold the differences that are not differences in what a page says."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "")  # soft hyphen
    for dash in "‐‑‒–—―":
        text = text.replace(dash, "-")
    for quote in "‘’‚‛′":
        text = text.replace(quote, "'")
    for quote in "“”„‟″":
        text = text.replace(quote, '"')
    return _collapse(text.lower())


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip()


@dataclass
class QuoteCheck:
    """Whether a judge's supporting quote is really in the source it came from."""

    found: bool
    exact: bool
    ratio: float
    window: str = ""

    @property
    def verdict(self) -> str:
        if self.exact:
            return "exact"
        return "near" if self.found else "absent"


def find_quote(quote: str, source_text: str) -> QuoteCheck:
    """Locate ``quote`` in ``source_text``, allowing only cosmetic drift.

    An exact hit on the normalized text is the common case. The near-match path exists
    because models re-type a span instead of copying it and lose a hyphen or a unicode
    space on the way; it slides a window of the quote's length across the source and
    takes the best similarity, so a paraphrase — which differs in words, not in
    punctuation — stays below the threshold and is reported absent.

    A quote joined by an ellipsis is treated as several quotes, each of which must be
    found. That is not a relaxation: every part still has to be on the page, and a
    paraphrase split into two paraphrases fails twice instead of once.
    """
    fragments = [
        part
        for part in _ELLIPSIS.split(quote or "")
        if len(part.strip()) >= MIN_QUOTE_FRAGMENT_CHARS
    ]
    if len(fragments) > 1:
        checks = [_find_span(part, source_text) for part in fragments]
        return QuoteCheck(
            found=all(check.found for check in checks),
            exact=all(check.exact for check in checks),
            ratio=round(min(check.ratio for check in checks), 3),
            window=next(
                (check.window for check in checks if check.window), ""
            ),
        )
    return _find_span(quote, source_text)


def _find_span(quote: str, source_text: str) -> QuoteCheck:
    needle, haystack = normalize(quote), normalize(source_text)
    if not needle or not haystack:
        return QuoteCheck(found=False, exact=False, ratio=0.0)
    position = haystack.find(needle)
    if position != -1:
        return QuoteCheck(
            found=True,
            exact=True,
            ratio=1.0,
            window=haystack[position : position + len(needle)],
        )

    best_ratio, best_window = 0.0, ""
    width = len(needle)
    step = max(1, width // 4)
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)
    for start in range(0, max(1, len(haystack) - width + 1), step):
        window = haystack[start : start + width]
        matcher.set_seq1(window)
        # real_quick_ratio/quick_ratio are cheap upper bounds; skip windows that cannot win.
        if matcher.real_quick_ratio() < best_ratio or matcher.quick_ratio() < best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_window = ratio, window
    return QuoteCheck(
        found=best_ratio >= QUOTE_MATCH_RATIO,
        exact=False,
        ratio=round(best_ratio, 3),
        window=best_window,
    )


def content_terms(text: str) -> list[str]:
    """The words of a claim that carry its content, in order, deduplicated."""
    terms: list[str] = []
    for token in re.findall(r"[^\W_]+", normalize(text), re.UNICODE):
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def lexical_overlap(claim_text: str, source_text: str) -> float:
    """Share of a claim's content words that appear anywhere in the source.

    Not a support metric — a page can contain every word of a sentence it contradicts —
    but a floor that needs no model. When the judge calls a claim supported and this is
    near zero, one of the two is wrong, and the pair is worth reading by hand.
    """
    terms = content_terms(claim_text)
    if not terms:
        return 0.0
    haystack = normalize(source_text)
    present = sum(1 for term in terms if term in haystack)
    return round(present / len(terms), 3)
