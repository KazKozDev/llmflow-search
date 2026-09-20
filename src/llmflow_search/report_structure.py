"""Deterministic structure defects in a finished report.

The editorial contract in ``prompts.REPORT_EDITORIAL_RULES`` is an instruction, and an
instruction is a tendency: the drafter follows it on an easy answer and drops it on a hard
one, and nothing downstream notices. This module is the half of the contract that does not
depend on the model agreeing — the three defects that can be found by reading the text
alone, named precisely enough that a repair call can act on each one.

Only four, deliberately. "Is the frame present", "is the tone level", "is this one idea
per block" cannot be decided by a regular expression, and a detector that guesses at them
would spend a model call arguing about a report that is fine. What is left is checkable
without judgement:

* a heading with nothing under it — the reader is promised a section and given a title;
* a comparative with no figure — "materially lower" asserts a magnitude and supplies none;
* the same point made twice — a restatement that reads, to a skimming reader, like a
  second piece of evidence;
* a finding resting only on coverage of an event — an outlet's retelling of an
  announcement standing in for the announcement;
* a finding that names nobody and counts nothing — the shape of a report with the facts
  taken out;
* an appendix holding more than the body — a report sorted rather than edited.

The appendix is inspected like everything else, and deliberately so. While it was exempt,
every per-item check had a trivial answer: move the item down. A report then passed as a
body of headings above a feed of everything that could not defend itself below.

Each defect names the exact text that carries it, so the repair prompt can quote it back
rather than asking the model to re-read the whole report looking for trouble.

These checks read English and only English. The magnitude words, the written-out
quantities and the appendix heading are all English literals, so on a report in another
language the comparative check goes quiet and the appendix check misfires — material
correctly filed under a translated appendix heading is read as sitting in the body. The
rest of the pipeline is language-neutral (the PDF embeds a Unicode font, the claim splitter
knows a translated "Sources" heading), so this is a limit of this module, deliberately
left rather than half-covered: a word list in a language nobody measured would be a
detector nobody can trust.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from . import attribution
from .attribution import normalize

# A heading line. Markdown ATX only: the answer prompts ask for Markdown, and a
# Setext-underlined heading has never appeared in a produced answer.
_HEADING = re.compile(r"^\s*(#{1,6})\s+(\S.*?)\s*$")

# The bibliography heading ends the body. Everything below it is links, and a link list
# is not a section with a missing body.
_SOURCES_HEADING = attribution._SOURCES_HEADING

_DIGIT = re.compile(r"\d")

# Digits that are not a magnitude. A date tells the reader when, never how much, so
# "latency is far lower since March 2026" carries a number and states no size — and the
# plain digit test cleared it, along with every table row whose only figure was its date
# column. Dates are removed before the test asks whether a figure is present.
_DATE_LIKE = re.compile(
    r"\d{4}-\d{2}-\d{2}"  # 2026-03-04
    r"|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"  # 4/3/2026, 4.3.26
    r"|\b(?:19|20)\d{2}\b"  # a year on its own
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    re.IGNORECASE,
)

# Citation markers are digits, and a sentence's citation is not its figure: checking
# "[1]" as a number would clear every uncountable comparative in a cited report, which is
# every comparative the detector exists to find.
_CITATION_MARKER = attribution._CITATION

# Comparatives and magnitude words that assert a quantity. A unit containing one of these
# and no digit at all states a size without stating it.
#
# The list is deliberately short and one-sided: every entry here is a word whose whole job
# is to report a magnitude. Words that merely *often* accompany one — "increased",
# "improved", "reduced" — are left out, because a sentence can legitimately report that a
# thing happened without quantifying it, and flagging those would turn the detector into
# a style complaint the writer cannot always satisfy from the sources.
_MAGNITUDE_WORDS = (
    r"lower than",
    r"higher than",
    r"greater than",
    r"less than",
    r"more than",
    r"fewer than",
    r"faster",
    r"slower",
    r"cheaper",
    r"more expensive",
    r"significantly",
    r"substantially",
    r"dramatically",
    r"considerably",
    r"markedly",
    r"sharply",
    r"vastly",
    r"far more",
    r"far fewer",
    r"far less",
    # "far lower", "far cheaper", "far slower" — an intensifier in front of any
    # comparative. Listing the pairs by hand always misses one.
    r"far \w+er",
    r"much larger",
    r"much smaller",
    r"the majority",
    r"a majority",
    r"vast majority",
    r"most users",
    r"most customers",
    r"most providers",
    r"orders of magnitude",
)
_MAGNITUDE = re.compile(
    r"\b(" + "|".join(_MAGNITUDE_WORDS) + r")\b",
    re.IGNORECASE,
)

# Written-out quantities are quantities. A sentence saying "fewer than half" or "doubled"
# has supplied its figure in words, and demanding digits from it would be pedantry.
_WRITTEN_QUANTITY = re.compile(
    r"\b(half|third|quarter|double[ds]?|doubling|triple[ds]?|tripled|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|thousand|"
    r"million|billion|trillion)\b",
    re.IGNORECASE,
)

# Two units count as the same point at or above this similarity. High, because the cost of
# a false positive is a repair call that deletes a real distinction.
DUPLICATE_RATIO = 0.93

# Similarity alone is not enough, and the reason is arithmetic rather than stylistic: two
# rows of the same series — the EU figure and the US figure, March and April — differ by a
# region name and a number inside an otherwise identical sentence, which scores about 0.94.
# That is a table, not a repetition. So units whose figures differ are never duplicates,
# however similar they read.
_NUMBER = re.compile(r"\d[\d,.]*")

# The same holds for names, and it is what turns a thin report into a set of templates.
# Once the excerpting has stripped the detail, three different items read as "The team won
# its opening match" three times over, and a similarity test merges them into one. They
# are not one item; they are three whose names were lost upstream. So two units naming
# different people, teams or works are never duplicates.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _proper_names(text: str) -> set[str]:
    """Capitalized words that are not simply starting a sentence, plus acronyms."""
    names: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        words = _WORD.findall(sentence)
        # The first word of a sentence is capitalized by grammar, not by being a name.
        for word in words[1:]:
            if word[:1].isupper():
                names.add(word.casefold())
        if words and words[0].isupper() and len(words[0]) > 1:
            names.add(words[0].casefold())  # an acronym leading the sentence
    return names

# A unit this short is a label or a fragment; two of them matching says nothing.
_MIN_DUPLICATE_WORDS = 6

# Where the body ends. Material under an appendix heading is already where secondary
# material belongs, so the body-only checks stop here.
_APPENDIX_HEADING = re.compile(r"^\s*#{1,6}\s*appendix\b", re.IGNORECASE)

# Source types whose page is coverage of an event rather than the event. Matched as
# substrings of the server's Source type string, which is free text ("news aggregator",
# "content farm") rather than a fixed enum.
#
# Deliberately narrow, after a first version of this list demoted whole reports. Two
# entries had to come out:
#
# "secondary" — a match report, a review, a court reporter's account. In sport, culture
# and most of what is worth reporting there is no lab page behind the event; the outlet IS
# the reporting institution, and treating secondary as weak moved those sections wholesale
# into the appendix and left the body a set of empty headings.
#
# "blog" — a company's own blog is the company page. It is the primary record of its own
# announcement, which is exactly what the body is supposed to carry. What is weak is
# somebody else's retelling of it, and that arrives typed as an aggregator.
#
# What is left is unambiguous: a page whose whole purpose is to restate what another page
# said. Unknown or missing types never count as weak, because the type comes from the MCP
# server and is often absent.
_WEAK_SOURCE_MARKERS = (
    "aggregator",
    "listicle",
    "roundup",
    "content farm",
    "link farm",
)


@dataclass(frozen=True)
class Defect:
    """One structural defect, quoted exactly as it appears in the report."""

    kind: str
    text: str
    detail: str

    def describe(self) -> str:
        return f"[{self.kind}] {self.detail}\n    {self.text}"


def _body(report: str) -> str:
    """The report without its trailing bibliography."""
    lines: list[str] = []
    for line in (report or "").splitlines():
        if _SOURCES_HEADING.match(line):
            break
        lines.append(line)
    return "\n".join(lines)


def _before_appendix(body: str) -> str:
    """The part of the body that is not the appendix."""
    lines: list[str] = []
    for line in (body or "").splitlines():
        if _APPENDIX_HEADING.match(line):
            break
        lines.append(line)
    return "\n".join(lines)


def orphan_headings(report: str) -> list[Defect]:
    """Headings with no content between them and the next heading or the end."""
    lines = _body(report).splitlines()
    heading_positions: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match:
            heading_positions.append((index, match.group(2)))

    defects: list[Defect] = []
    for position, (index, title) in enumerate(heading_positions):
        end = (
            heading_positions[position + 1][0]
            if position + 1 < len(heading_positions)
            else len(lines)
        )
        has_body = any(line.strip() for line in lines[index + 1 : end])
        if not has_body:
            defects.append(
                Defect(
                    kind="orphan_heading",
                    text=title,
                    detail="Heading with no body. Write the section or drop the heading.",
                )
            )
    return defects


def numberless_comparatives(report: str) -> list[Defect]:
    """Claim units that assert a magnitude and give no figure for it."""
    defects: list[Defect] = []
    for claim in attribution.split_claims(report):
        match = _MAGNITUDE.search(claim.text)
        if not match:
            continue
        bare = _DATE_LIKE.sub(" ", _CITATION_MARKER.sub(" ", claim.text))
        if _DIGIT.search(bare) or _WRITTEN_QUANTITY.search(bare):
            continue
        defects.append(
            Defect(
                kind="numberless_comparative",
                text=claim.text,
                detail=(
                    f'"{match.group(0)}" states a magnitude with no figure. '
                    "Give the number and its unit from the sources, or cut the sentence."
                ),
            )
        )
    return defects


def duplicate_blocks(report: str) -> list[Defect]:
    """Units that make a point an earlier unit already made."""
    seen: list[tuple[str, str, tuple[set[str], set[str]]]] = []  # (normalized, original, (figures, names))
    defects: list[Defect] = []
    for claim in attribution.split_claims(report):
        bare = _CITATION_MARKER.sub(" ", claim.text)
        normalized = normalize(claim.text)
        marks = (set(_NUMBER.findall(bare)), _proper_names(bare))
        if len(normalized.split()) < _MIN_DUPLICATE_WORDS:
            continue
        for earlier_normalized, earlier_text, earlier_marks in seen:
            if marks != earlier_marks:
                continue
            ratio = SequenceMatcher(None, earlier_normalized, normalized).ratio()
            if ratio >= DUPLICATE_RATIO:
                defects.append(
                    Defect(
                        kind="duplicate_block",
                        text=claim.text,
                        detail=(
                            "Repeats a point already made above "
                            f'("{earlier_text[:80]}"). Keep one, delete the other.'
                        ),
                    )
                )
                break
        else:
            seen.append((normalized, claim.text, marks))
    return defects


def _is_weak(source_type: str) -> bool:
    lowered = (source_type or "").lower()
    return any(marker in lowered for marker in _WEAK_SOURCE_MARKERS)


def source_types(sources: list[dict] | None) -> dict[int, str]:
    """Map each source's citation number to the type line the answer prompt showed.

    The numbering is the one ``sources._format_sources_for_llm`` gives the model: the
    position in the admitted source list, 1-based. Sources it skips for having no content
    keep their position, so the mapping cannot drift against the [n] markers in the text.
    """
    types: dict[int, str] = {}
    for index, source in enumerate(sources or [], 1):
        if not isinstance(source, dict):
            continue
        declared = str(
            (source.get("source_quality") or {}).get("source_type")
            or source.get("kind")
            or ""
        ).strip()
        if declared:
            types[index] = declared
    return types


def body_claims_on_weak_sources(
    report: str, sources: list[dict] | None
) -> list[Defect]:
    """Body findings whose only support is coverage of an event rather than the event.

    An outlet's retelling of a lab's announcement is not a second source for it, and three
    outlets retelling the same announcement are not three findings. The rule the editorial
    contract states — such material belongs in the appendix, labelled as a report of a
    claim, or nowhere — is enforced here only where the evidence is unambiguous: every
    source the claim cites has a type, and every one of those types is explicitly weak.

    A claim with no citations is not this defect (the citation pass owns that one), and a
    claim citing anything whose type is unknown is left alone. The check is built to miss
    rather than to misfire: demoting a correctly sourced finding costs the report more
    than leaving one weak finding in the body.
    """
    types = source_types(sources)
    if not types:
        return []
    defects: list[Defect] = []
    for claim in attribution.split_claims(_body(report)):
        if not claim.citations:
            continue
        cited_types = [types.get(source_id) for source_id in claim.citations]
        if any(not declared for declared in cited_types):
            continue
        if all(_is_weak(str(declared)) for declared in cited_types):
            defects.append(
                Defect(
                    kind="weak_source_in_body",
                    text=claim.text,
                    detail=(
                        "Supported only by "
                        + ", ".join(sorted({str(d) for d in cited_types}))
                        + " — coverage of an event, not the event. Move it to the "
                        "appendix as a reported claim, attribute it to whoever made it, "
                        "or cut it."
                    ),
                )
            )
    return defects


# Capitalized words that identify nothing. A nationality is the classic substitute for a
# name — "an Argentine conductor", "a Brazilian forward" is exactly the shape a report
# collapses into — and a month or a weekday is a date wearing a capital letter.
_NOT_A_NAME = frozenset(
    """january february march april may june july august september october november
    december monday tuesday wednesday thursday friday saturday sunday
    african american argentine argentinian australian austrian basque belgian brazilian
    british canadian catalan chilean chinese colombian croatian czech danish dutch
    egyptian english european finnish french german greek indian indonesian iranian
    iraqi irish israeli italian japanese korean mexican moroccan nigerian norwegian
    polish portuguese russian saudi scottish serbian slovak spanish swedish swiss
    syrian taiwanese thai turkish ukrainian venezuelan vietnamese welsh""".split()
)


def _identifying_names(text: str) -> set[str]:
    """Capitalized words that actually single something out.

    Stricter than ``_proper_names`` in one direction and looser in the other, because the
    two callers need opposite things. Duplicate detection wants every capitalized word
    that could tell two items apart, demonyms included: "Brazilian forward" and "Argentine
    forward" are two items. This one is asking whether the reader was told *who*, and
    "Argentine" does not tell them.

    Looser in that a sentence may begin with its name. ``_proper_names`` drops the first
    word because grammar capitalizes it, which is right when comparing two sentences and
    wrong here — it would read "Barenboim led the orchestra" as naming nobody. So the
    first word counts unless it is an ordinary word of the language.
    """
    names: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        words = _WORD.findall(sentence)
        for position, word in enumerate(words):
            if not word[:1].isupper():
                continue
            folded = word.casefold()
            if folded in _NOT_A_NAME:
                continue
            if position == 0 and folded in attribution._STOPWORDS:
                continue
            names.add(folded)
    return names


def _appendix_split(report: str) -> tuple[str, str]:
    """The report's body and its appendix, both without the bibliography."""
    body = _body(report)
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if _APPENDIX_HEADING.match(line):
            return "\n".join(lines[:index]), "\n".join(lines[index + 1 :])
    return body, ""


def nameless_body_findings(report: str) -> list[Defect]:
    """Body findings that name nothing and count nothing.

    "A Brazilian forward scored twice", "a 22-year-old rider took the stage" — these are
    the shapes of findings with the fact removed, and they are what a report degrades into
    when the excerpting upstream hands the writer headlines instead of paragraphs. They
    read as content and survive every other check: they are grounded, cited, unique, and
    say nothing a reader can check or follow up.

    A claim carrying neither a proper name nor a figure is the whole of the rule. One or
    the other is enough — "prices fell 12%" names nothing and is still a finding — so the
    check fires only where both are missing, which is where nothing is left.
    """
    body, _ = _appendix_split(report)
    defects: list[Defect] = []
    for claim in attribution.split_claims(body):
        bare = _CITATION_MARKER.sub(" ", claim.text)
        if _identifying_names(bare) or _DIGIT.search(_DATE_LIKE.sub(" ", bare)):
            continue
        defects.append(
            Defect(
                kind="nameless_body_finding",
                text=claim.text,
                detail=(
                    "No name and no figure — a finding with its fact removed. Name who or "
                    "what it is about from the sources, or delete the sentence."
                ),
            )
        )
    return defects


def appendix_outweighs_body(report: str) -> list[Defect]:
    """An appendix holding more than the body it is supposed to support.

    The appendix earns its place as reference material the body points at. When more of
    the report sits below that heading than above it, the heading has been used as a
    destination for everything that could not defend itself upstairs — which is what a
    per-item check invites, since moving an item is the cheapest way to stop it failing.
    """
    body, appendix = _appendix_split(report)
    if not appendix.strip():
        return []
    body_claims = len(attribution.split_claims(body))
    appendix_claims = len(attribution.split_claims(appendix))
    if appendix_claims <= body_claims:
        return []
    return [
        Defect(
            kind="appendix_outweighs_body",
            text=f"{appendix_claims} items in the appendix, {body_claims} in the body",
            detail=(
                "The report has been sorted, not edited. Move back every appendix item "
                "something follows from, with its name, and delete the rest."
            ),
        )
    ]


def find_defects(report: str, sources: list[dict] | None = None) -> list[Defect]:
    """Every structural defect in the report, in the order a reader would meet them."""
    if not report or attribution.is_refusal(report):
        return []
    defects = [
        *orphan_headings(report),
        *numberless_comparatives(report),
        *duplicate_blocks(report),
        *body_claims_on_weak_sources(report, sources),
    ]
    # One sentence, one defect. A nameless sentence is very often also a numberless one,
    # and naming it twice tells the repair nothing extra while making the report look
    # worse than it is — and the acceptance test counts defects.
    already = {defect.text for defect in defects}
    defects.extend(
        defect
        for defect in nameless_body_findings(report)
        if defect.text not in already
    )
    defects.extend(appendix_outweighs_body(report))
    return defects
