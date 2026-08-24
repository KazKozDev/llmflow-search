"""Runtime configuration constants."""

import os
import re
from datetime import date
from pathlib import Path


def current_date_iso() -> str:
    """Return the agent's explicit current date anchor."""
    for name in ("LLMFLOW_SEARCH_TODAY", "CURRENT_DATE"):
        value = os.getenv(name, "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return value
    return date.today().strftime("%Y-%m-%d")


SERVER_CMD = os.getenv("LLMFLOW_SEARCH_MCP_CMD", "footnote-mcp").split()


def _non_negative_float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


SEARCH_REQUEST_DELAY_SECONDS = _non_negative_float_env(
    "LLMFLOW_SEARCH_SEARCH_DELAY_SECONDS",
    12.0,
)


# Minimum spacing between calls to one throttled backend family. Scraped engines are
# the ones that actually block: a single web_search call already fans out to four of
# them inside the server, so the interval here governs bursts of four, not of one.
# Official APIs tolerate a far tighter cadence and get their own slot.
#
# Which family a tool belongs to is the connected server's business, not the execution
# loop's — see mcp_client._throttle_group. A server that answers from local data says so
# and lands in "local", where the only thing still bounding it is the per-question call
# budget: pausing twelve seconds before reading a file on disk buys nothing.
SEARCH_GROUP_DELAY_SECONDS = {
    "scraper": SEARCH_REQUEST_DELAY_SECONDS,
    "api": _non_negative_float_env("LLMFLOW_SEARCH_API_DELAY_SECONDS", 2.0),
    "archive": _non_negative_float_env("LLMFLOW_SEARCH_ARCHIVE_DELAY_SECONDS", 10.0),
    "local": 0.0,
}


# Fraction of the interval added at random on top of it. A metronome-exact cadence
# spreads no better than a burst does when several runs overlap.
SEARCH_DELAY_JITTER = _non_negative_float_env("LLMFLOW_SEARCH_SEARCH_JITTER", 0.35)


# Throttled steps executed in one batch before the rest are deferred to the next
# round. A plan that queues six searches should not empty the whole queue at once.
MAX_THROTTLED_STEPS_PER_BATCH = int(
    os.getenv("LLMFLOW_SEARCH_MAX_SEARCH_BATCH", "2") or 2
)


# Hard ceiling on rate-limited calls spent answering one question. Keyed providers bill
# per call and free tiers are small, so a question that keeps re-searching without
# finding sources must stop rather than drain the month's quota.
MAX_SEARCH_CALLS_PER_QUESTION = int(
    os.getenv("LLMFLOW_SEARCH_MAX_SEARCH_CALLS", "12") or 12
)


# Pages fetched at once. Reading is not metered, but twelve sites in one second is a
# burst that individual hosts notice even though no single one is being hammered.
MAX_PARALLEL_FETCHES = int(os.getenv("LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES", "5") or 5)


def group_batch_cap(group: str) -> int:
    """Steps of one backend family executed in a single batch.

    Trickling steps out across rounds exists to spread the *pauses*; a family that
    does not pause has nothing to spread, so it gets the ordinary fetch width.
    """
    if SEARCH_GROUP_DELAY_SECONDS.get(group, 0.0) <= 0:
        return MAX_PARALLEL_FETCHES
    return MAX_THROTTLED_STEPS_PER_BATCH


PDF_REPORTS_DIR = os.getenv("LLMFLOW_SEARCH_REPORTS_DIR", "reports")


PDF_LOGO_PATH = os.getenv(
    "LLMFLOW_SEARCH_REPORT_LOGO",
    str(Path(__file__).with_name("assets") / "llmflow.png"),
)


MAX_PLAN_STEPS = 40


MAX_EVIDENCE_ROUNDS = 5


MAX_STAGNANT_ROUNDS = (
    2  # give up once this many consecutive evidence rounds add no new supported claims
)


ROUNDUP_MIN_SOURCES = (
    3  # answer_mode=roundup: minimum admissible sources with usable on-topic content
)


ROUNDUP_MIN_CLAIMS = (
    3  # allow one strong listing/source when it supports several distinct roundup items
)


LISTING_DRILLDOWN_TOP_K = 5  # max article links auto-enqueued from a listing/index page
DISCOVERED_URL_CATALOG_TOP_K = (
    40  # max search-result URLs offered to the post-batch model
)


TOOL_RESULT_MAX_CHARS = 20000


# Per-URL search snippet kept in memory, and the slice of it shown in the readable
# catalog the post-batch model picks its next web_read from.
SNIPPET_MEMORY_MAX_CHARS = 400
SNIPPET_CATALOG_MAX_CHARS = 180


# A second, cheaper model for the decisions that do not need the main one. Latency is the
# number of decisions times the latency of each, and several of these decisions are
# bookkeeping: classifying what a tool result was, restating the question's requirements,
# judging whether a round made progress. Empty means every role runs on the main model,
# which is the behaviour measured so far.
FAST_MODEL = os.getenv("LLMFLOW_SEARCH_FAST_MODEL", "").strip()

# Roles routed to FAST_MODEL when one is configured. Deliberately excludes every decision
# that reads evidence and every one that writes an answer — the ledger, the challenge, the
# post-batch controller, the draft and its verification stay on the main model.
FAST_MODEL_ROLES = frozenset(
    (os.getenv("LLMFLOW_SEARCH_FAST_MODEL_ROLES", "").strip() or "observation,requirements,evaluate,reflect").split(",")
)


def _flag_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


# Window a document is cut into before anything ranks it (see passages.py). Long enough
# to hold a claim and the sentence that qualifies it; overlapping, so a fact straddling a
# boundary is not invisible to every window.
PASSAGE_CHARS = 700
PASSAGE_STRIDE = 500


# Which excerpt of an over-long page the answer model is shown. Position took the first
# N characters of every page — on a long one, the masthead. Relevance takes the windows
# that mention the question. Off restores the old head-of-document behaviour.
PASSAGE_RELEVANCE_EXCERPTS = _flag_env("LLMFLOW_SEARCH_RELEVANCE_EXCERPTS", True)


# Whether a round whose page reads are already decided still pays for the post-batch
# controller's opinion. Its only remaining effect there is to queue further searching
# behind pages the run has not opened yet, and the next round re-derives that from a
# state which includes them. Off restores the call on every batch.
SKIP_CONTROLLER_WHEN_READS_FORCED = _flag_env(
    "LLMFLOW_SEARCH_SKIP_FORCED_CONTROLLER", True
)


# How read candidates are ordered once the passage layer has assembled them:
#   "search_rank"  — best rank the search gave the URL (the behaviour measured so far)
#   "passage_bm25" — the score of the best passage the URL offers for this question
# The selection goes through the passage store either way; this only chooses the
# comparison it uses, so a new scorer can be measured against the shipped default.
READ_RANKING = os.getenv("LLMFLOW_SEARCH_READ_RANKING", "search_rank").strip().lower()


SOURCE_CONTENT_MAX_CHARS = 25000


TOTAL_SOURCES_MAX_CHARS = 100000


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The found sources do not provide enough information for a reliable answer."
)
