"""Everything the run retrieved, cut into passages, with one place that ranks them.

Two decisions used to be made without any notion of relevance at all.

*Which page to open next* was decided by a model looking at a flat list of up to forty
bare addresses, ordered by whichever query happened to return them first — so the tail of
a weak opening query outranked the best hit of the query that finally worked.

*Which part of a page to show* was decided by taking the first ``TOOL_RESULT_MAX_CHARS``
characters. On a long page that is the masthead and the navigation, and the sentence the
question turns on sits somewhere past the cut.

This module is where both get an answer. Passages carry their provenance — the address
they came from, the offset inside it, whether they came from a search snippet or from a
fetched body, and the rank the search gave them — and ``PassageScorer`` is the single
seam through which they are ordered. ``Bm25PassageScorer`` is the one implementation
here; swapping in a dense one is ``set_scorer(...)`` and touches no calling code. That
substitution is the point of the seam, not something to do in this package: no embedding
model is loaded, called, or depended on anywhere below.

The store is derived from the run's existing memory rather than persisted beside it. A
second copy of "what we have retrieved" is a second thing that can disagree with the
first, and disagreement between two records of the same fetch is exactly the class of bug
this codebase has been paying for.
"""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .config import (
    PASSAGE_CHARS,
    PASSAGE_STRIDE,
)

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that appear in almost every question and carry no discriminating power. BM25's own
# IDF handles this over a large corpus; over the handful of passages one page yields, a
# stopword can still dominate, so it is dropped from the query outright.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who whom whose why will with""".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def query_terms(query: str) -> list[str]:
    return [token for token in tokenize(query) if token not in _STOPWORDS]


@dataclass(frozen=True)
class Passage:
    """One window of retrieved text, and where it came from."""

    url: str
    text: str
    # Character offset inside the document the passage was cut from. A search snippet has
    # no offset of its own; it is recorded at 0 because that is where the server put it.
    offset: int = 0
    # "search_snippet" — what a search engine showed us about a page we have not opened.
    # "page" — text from a body we actually fetched.
    origin: str = "page"
    # The best rank the search result carrying this passage ever held, when it came from
    # a search. None for passages cut from a fetched body.
    rank: int | None = None
    title: str = ""
    # Arrival order within the run. Wall-clock time for the same events is in the trace;
    # what ranking needs from time is only "which of these did we learn later".
    seq: int = 0


class PassageScorer(Protocol):
    """Order passages by how well they answer a query.

    One method, and it takes the whole batch: a dense implementation wants to embed the
    passages in one call rather than one at a time, and a lexical one wants corpus
    statistics over the batch. Returns one score per passage, positionally aligned.
    """

    def score(self, query: str, passages: Sequence[Passage]) -> list[float]: ...


class Bm25PassageScorer:
    """Okapi BM25 over the passage batch it is handed.

    Deliberately dependency-free and deliberately lexical. It is the baseline the seam
    exists to be measured against, not the answer.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, query: str, passages: Sequence[Passage]) -> list[float]:
        terms = query_terms(query)
        if not terms or not passages:
            return [0.0] * len(passages)

        documents = [tokenize(passage.text) for passage in passages]
        lengths = [len(document) for document in documents]
        average_length = (sum(lengths) / len(lengths)) or 1.0
        counts = [
            {term: document.count(term) for term in set(terms) if term in document}
            for document in documents
        ]
        document_frequency = {
            term: sum(1 for count in counts if term in count) for term in set(terms)
        }

        scores = []
        for index, count in enumerate(counts):
            total = 0.0
            for term in terms:
                frequency = count.get(term, 0)
                if not frequency:
                    continue
                # The +1 inside the log keeps the IDF of a term present in every passage
                # at a small positive value rather than zero or negative, which matters
                # when the batch is one page's worth of passages rather than a corpus.
                idf = math.log(
                    1
                    + (len(documents) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                norm = 1 - self.b + self.b * (lengths[index] / average_length)
                total += idf * (frequency * (self.k1 + 1)) / (
                    frequency + self.k1 * norm
                )
            scores.append(total)
        return scores


class SearchRankScorer:
    """Order by the rank the search engine gave the result, ignoring the query.

    This is not a relevance model — it is the ordering the run already used, expressed
    through the same seam so that a real one can be compared against it without either
    side of the comparison changing any calling code.
    """

    def score(self, query: str, passages: Sequence[Passage]) -> list[float]:
        del query  # rank is what the engine decided; the query does not enter into it
        return [
            0.0 if passage.rank is None else 1.0 / (1.0 + passage.rank)
            for passage in passages
        ]


_scorer: PassageScorer = Bm25PassageScorer()


def set_scorer(scorer: PassageScorer) -> None:
    """Install the implementation every store uses. The whole point of the seam."""
    global _scorer
    _scorer = scorer


def get_scorer() -> PassageScorer:
    return _scorer


def split_into_passages(
    text: str,
    url: str = "",
    title: str = "",
    seq: int = 0,
    chars: int = PASSAGE_CHARS,
    stride: int = PASSAGE_STRIDE,
) -> list[Passage]:
    """Cut a document into overlapping windows.

    Overlapping, because a fact that straddles a hard boundary is invisible to every
    window that does not contain the whole of it.
    """
    body = (text or "").strip()
    if not body:
        return []
    if len(body) <= chars:
        return [Passage(url=url, text=body, offset=0, title=title, seq=seq)]

    passages = []
    for offset in range(0, len(body), stride):
        window = body[offset : offset + chars]
        if not window.strip():
            continue
        passages.append(
            Passage(url=url, text=window, offset=offset, title=title, seq=seq)
        )
        if offset + chars >= len(body):
            break
    return passages


class PassageStore:
    """Every passage the run has, and the ranking that orders them."""

    def __init__(
        self, passages: Iterable[Passage] | None = None, scorer: PassageScorer | None = None
    ):
        self._passages: list[Passage] = list(passages or [])
        self._scorer = scorer

    @property
    def scorer(self) -> PassageScorer:
        return self._scorer or get_scorer()

    def __len__(self) -> int:
        return len(self._passages)

    def add(self, passage: Passage) -> None:
        self._passages.append(passage)

    def add_search_snippets(self, search_memory: dict) -> "PassageStore":
        """Take in what search told us about pages we have not opened."""
        snippets = search_memory.get("discovered_snippets") or {}
        titles = search_memory.get("discovered_titles") or {}
        ranks = search_memory.get("discovered_ranks") or {}
        for seq, url in enumerate(search_memory.get("discovered_urls") or []):
            snippet = str(snippets.get(url) or "")
            title = str(titles.get(url) or "")
            self.add(
                Passage(
                    url=url,
                    # The title is part of what a search result says about a page, and on
                    # a thin snippet it is most of it. With neither, the address itself is
                    # still evidence — and a URL with no snippet must stay in the running,
                    # or a top-ranked result vanishes for want of a description.
                    text=f"{title}. {snippet}".strip(". ") or url,
                    origin="search_snippet",
                    rank=ranks.get(url),
                    title=title,
                    seq=seq,
                )
            )
        return self

    def add_sources(self, sources: Iterable[dict]) -> "PassageStore":
        """Take in the bodies of pages the run actually fetched."""
        for seq, source in enumerate(sources or []):
            if not isinstance(source, dict):
                continue
            url = str(source.get("url") or "")
            content = str(source.get("content") or "")
            if not url or not content:
                continue
            for passage in split_into_passages(
                content, url=url, title=str(source.get("title") or ""), seq=seq
            ):
                self.add(passage)
        return self

    @classmethod
    def from_run(
        cls, search_memory: dict | None = None, sources: Iterable[dict] | None = None
    ) -> "PassageStore":
        store = cls()
        if search_memory:
            store.add_search_snippets(search_memory)
        if sources:
            store.add_sources(sources)
        return store

    def ranked(
        self, query: str, urls: Sequence[str] | None = None, origin: str | None = None
    ) -> list[tuple[float, Passage]]:
        """Passages matching the filters, best first."""
        wanted = set(urls) if urls is not None else None
        subset = [
            passage
            for passage in self._passages
            if (wanted is None or passage.url in wanted)
            and (origin is None or passage.origin == origin)
        ]
        if not subset:
            return []
        scores = self.scorer.score(query, subset)
        pairs = list(zip(scores, subset, strict=True))
        pairs.sort(key=lambda pair: (-pair[0], pair[1].seq, pair[1].offset))
        return pairs

    def rank_urls(
        self, query: str, urls: Sequence[str], origin: str | None = None
    ) -> list[str]:
        """Order candidate addresses by the best passage each one offers.

        A URL the store knows nothing about scores nothing and lands at the end in the
        order it was given, which is the caller's own fallback ordering.
        """
        best: dict[str, float] = {}
        for score, passage in self.ranked(query, urls=urls, origin=origin):
            if score > best.get(passage.url, float("-inf")):
                best[passage.url] = score
        given = {url: position for position, url in enumerate(urls)}
        return sorted(
            urls,
            key=lambda url: (-best.get(url, 0.0), given[url]),
        )


def relevant_excerpt(text: str, query: str, budget: int, store: PassageStore | None = None) -> str:
    """The ``budget`` characters of ``text`` most likely to answer ``query``.

    Windows are chosen by score and then re-joined in document order, so the excerpt still
    reads as the page reads; an ellipsis marks each place where something was skipped. The
    alternative — and what this replaces — is the first ``budget`` characters, which on a
    long page is the part every page has in common.
    """
    body = (text or "").strip()
    if len(body) <= budget:
        return body
    if not query_terms(query):
        return body[:budget]

    passages = split_into_passages(body)
    if not passages:
        return body[:budget]

    scorer = (store or PassageStore()).scorer
    scores = scorer.score(query, passages)
    if not any(score > 0 for score in scores):
        return body[:budget]

    ordered = sorted(zip(scores, passages, strict=True), key=lambda pair: (-pair[0], pair[1].offset))
    chosen: list[Passage] = []
    used = 0
    for score, passage in ordered:
        if score <= 0:
            continue
        cost = len(passage.text)
        if used + cost > budget and chosen:
            continue
        chosen.append(passage)
        used += cost
        if used >= budget:
            break

    if not chosen:
        return body[:budget]

    chosen.sort(key=lambda passage: passage.offset)
    parts: list[str] = []
    previous_end = 0
    for passage in chosen:
        if passage.offset > previous_end:
            parts.append("…")
        # Overlapping windows would repeat their shared text verbatim; keep only the part
        # of this window that the previous one did not already contribute.
        start = max(0, previous_end - passage.offset)
        parts.append(passage.text[start:])
        previous_end = max(previous_end, passage.offset + len(passage.text))
    if previous_end < len(body):
        parts.append("…")
    return "".join(parts)[: budget + 2].strip()
