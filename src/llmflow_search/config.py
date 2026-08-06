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
SEARCH_GROUP_DELAY_SECONDS = {
    "scraper": SEARCH_REQUEST_DELAY_SECONDS,
    "api": _non_negative_float_env("LLMFLOW_SEARCH_API_DELAY_SECONDS", 2.0),
    "archive": _non_negative_float_env("LLMFLOW_SEARCH_ARCHIVE_DELAY_SECONDS", 10.0),
}


# Fraction of the interval added at random on top of it. A metronome-exact cadence
# spreads no better than a burst does when several runs overlap.
SEARCH_DELAY_JITTER = _non_negative_float_env("LLMFLOW_SEARCH_SEARCH_JITTER", 0.35)


# Throttled steps executed in one batch before the rest are deferred to the next
# round. A plan that queues six searches should not empty the whole queue at once.
MAX_THROTTLED_STEPS_PER_BATCH = int(os.getenv("LLMFLOW_SEARCH_MAX_SEARCH_BATCH", "2") or 2)


# Hard ceiling on rate-limited calls spent answering one question. Keyed providers bill
# per call and free tiers are small, so a question that keeps re-searching without
# finding sources must stop rather than drain the month's quota.
MAX_SEARCH_CALLS_PER_QUESTION = int(os.getenv("LLMFLOW_SEARCH_MAX_SEARCH_CALLS", "12") or 12)


# Pages fetched at once. Reading is not metered, but twelve sites in one second is a
# burst that individual hosts notice even though no single one is being hammered.
MAX_PARALLEL_FETCHES = int(os.getenv("LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES", "5") or 5)


PDF_REPORTS_DIR = os.getenv("LLMFLOW_SEARCH_REPORTS_DIR", "reports")


PDF_LOGO_PATH = os.getenv(
    "LLMFLOW_SEARCH_REPORT_LOGO",
    str(Path(__file__).with_name("assets") / "llmflow.png"),
)


MAX_PLAN_STEPS = 40


MAX_EVIDENCE_ROUNDS = 5


MAX_STAGNANT_ROUNDS = 2  # give up once this many consecutive evidence rounds add no new supported claims


ROUNDUP_MIN_SOURCES = 3  # answer_mode=roundup: minimum admissible sources with usable on-topic content


ROUNDUP_MIN_CLAIMS = 3  # allow one strong listing/source when it supports several distinct roundup items


LISTING_DRILLDOWN_TOP_K = 5  # max article links auto-enqueued from a listing/index page
DISCOVERED_URL_CATALOG_TOP_K = 40  # max search-result URLs offered to the post-batch model


TOOL_RESULT_MAX_CHARS = 20000


SOURCE_CONTENT_MAX_CHARS = 25000


TOTAL_SOURCES_MAX_CHARS = 100000


INSUFFICIENT_EVIDENCE_MESSAGE = "The found sources do not provide enough information for a reliable answer."
