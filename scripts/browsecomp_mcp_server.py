"""Stdio MCP server for the BrowseComp-Plus document corpus.

Exposes `web_search` and `web_read` tools over stdio to emulate web search
over the benchmark's gold and negative evidence documents.
"""

import json
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from rank_bm25 import BM25Okapi

# The agent's memory stores every address through this normalizer, so a step it writes
# names the normalized form. Importing the agent's own function rather than restating
# the rule here is what keeps the corpus index and the agent's addresses from drifting:
# they disagreed on one trailing slash, and every read of such a page returned
# "Document not found" while the corpus held the document.
from llmflow_search.sources import _normalize_source_url

mcp = FastMCP("browsecomp-server")

# Shared state loaded at startup
DOCUMENTS: list[dict[str, Any]] = []
URL_MAP: dict[str, dict[str, Any]] = {}
BM25_INDEX: BM25Okapi | None = None

# Snippet window: long enough to carry a fact, short enough that ten of them fit in
# one search result without crowding out the rest of the context.
PASSAGE_CHARS = 400
PASSAGE_STRIDE = 200


def load_corpus() -> None:
    """Load documents from JSON file path given in BROWSECOMP_DOCS_JSON."""
    global DOCUMENTS, URL_MAP, BM25_INDEX

    docs_file = os.getenv("BROWSECOMP_DOCS_JSON")
    if not docs_file or not os.path.exists(docs_file):
        DOCUMENTS = []
        URL_MAP = {}
        BM25_INDEX = None
        return

    try:
        with open(docs_file, encoding="utf-8") as f:
            data = json.load(f)
            DOCUMENTS = data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[browsecomp_mcp] Failed to load corpus: {exc}", file=sys.stderr)
        DOCUMENTS = []

    URL_MAP = {}
    corpus_tokens = []
    for idx, doc in enumerate(DOCUMENTS):
        doc_id = str(doc.get("docid") or f"doc_{idx}")
        url = str(doc.get("url") or f"https://browsecomp.local/{doc_id}")
        doc["url"] = url
        doc["docid"] = doc_id
        URL_MAP[_normalize_source_url(url)] = doc
        text = str(doc.get("text") or "")
        corpus_tokens.append(text.lower().split())

    if corpus_tokens:
        BM25_INDEX = BM25Okapi(corpus_tokens)


def _best_passage(text: str, query_tokens: list[str]) -> str:
    """Return the window of `text` that best matches the query.

    A real search engine shows the passage it matched on. Returning the head of the
    document instead tells the agent nothing about why the document was retrieved,
    so it picks pages to read blind.
    """
    if not text:
        return ""
    if len(text) <= PASSAGE_CHARS:
        return text

    wanted = set(query_tokens)
    if not wanted:
        return text[:PASSAGE_CHARS] + "..."

    best_start, best_hits = 0, -1
    for start in range(0, len(text) - PASSAGE_CHARS + 1, PASSAGE_STRIDE):
        window = text[start : start + PASSAGE_CHARS].lower()
        hits = sum(1 for token in wanted if token in window)
        if hits > best_hits:
            best_start, best_hits = start, hits

    passage = text[best_start : best_start + PASSAGE_CHARS]
    prefix = "..." if best_start else ""
    suffix = "..." if best_start + PASSAGE_CHARS < len(text) else ""
    return f"{prefix}{passage}{suffix}"


@mcp.tool()
def web_search(query: str, count: int = 10) -> str:
    """Perform BM25 search over the BrowseComp-Plus document corpus. [throttle: local]

    Args:
        query: Search query string.
        count: Number of search results to return (default: 10).

    Returns:
        JSON string containing the query and candidate sources.
    """
    if not BM25_INDEX or not DOCUMENTS:
        return json.dumps({"query": query, "sources": []}, ensure_ascii=False)

    tokenized_query = (query or "").lower().split()
    top_docs = BM25_INDEX.get_top_n(tokenized_query, DOCUMENTS, n=count)

    sources = []
    for doc in top_docs:
        url = doc["url"]
        title = doc.get("title") or doc.get("docid") or url
        text = doc.get("text", "")
        snippet = _best_passage(text, tokenized_query)
        sources.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
                "pub_date": doc.get("pub_date"),
                "source_type": "article",
            }
        )

    return json.dumps({"query": query, "sources": sources}, ensure_ascii=False)


@mcp.tool()
def web_read(url: str) -> str:
    """Read the full text content of a document by URL.

    Args:
        url: Document URL to read.

    Returns:
        JSON string with page text and metadata.
    """
    doc = URL_MAP.get(_normalize_source_url(url))
    if not doc:
        for d in DOCUMENTS:
            if d.get("docid") == url or d.get("url") == url:
                doc = d
                break

    if not doc:
        return json.dumps(
            {"url": url, "error": "Document not found", "text": ""}, ensure_ascii=False
        )

    text = doc.get("text", "")
    title = doc.get("title") or doc.get("docid") or url

    return json.dumps(
        {
            "url": doc["url"],
            "title": title,
            "text": text,
            "pub_date": doc.get("pub_date"),
            "source_type": {"source_type": "browsecomp_document"},
            "links": [],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    load_corpus()
    mcp.run(transport="stdio")
