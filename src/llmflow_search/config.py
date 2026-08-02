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
    3.0,
)


PDF_REPORTS_DIR = os.getenv("LLMFLOW_SEARCH_REPORTS_DIR", "reports")


PDF_LOGO_PATH = os.getenv(
    "LLMFLOW_SEARCH_REPORT_LOGO",
    str(Path(__file__).with_name("assets") / "llmflow.png"),
)


MAX_PLAN_STEPS = 40


MAX_EVIDENCE_ROUNDS = 5


MAX_STAGNANT_ROUNDS = 2  # give up once this many consecutive evidence rounds add no new supported claims


ROUNDUP_MIN_SOURCES = 3  # answer_mode=roundup: minimum admissible sources with usable on-topic content


LISTING_DRILLDOWN_TOP_K = 5  # max article links auto-enqueued from a listing/index page


TOOL_RESULT_MAX_CHARS = 20000


SOURCE_CONTENT_MAX_CHARS = 25000


TOTAL_SOURCES_MAX_CHARS = 100000


INSUFFICIENT_EVIDENCE_MESSAGE = "The found sources do not provide enough information for a reliable answer."
