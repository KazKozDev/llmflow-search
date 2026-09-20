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


def _positive_float_env(name: str, default: float) -> float:
    value = _non_negative_float_env(name, default)
    return value if value > 0 else default


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)) or default))
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
MAX_THROTTLED_STEPS_PER_BATCH = _positive_int_env(
    "LLMFLOW_SEARCH_MAX_SEARCH_BATCH", 2
)


# Hard ceiling on rate-limited calls spent answering one question. Keyed providers bill
# per call and free tiers are small, so a question that keeps re-searching without
# finding sources must stop rather than drain the month's quota.
MAX_SEARCH_CALLS_PER_QUESTION = _positive_int_env(
    "LLMFLOW_SEARCH_MAX_SEARCH_CALLS", 12
)


# Pages fetched at once. Reading is not metered, but twelve sites in one second is a
# burst that individual hosts notice even though no single one is being hammered.
MAX_PARALLEL_FETCHES = _positive_int_env("LLMFLOW_SEARCH_MAX_PARALLEL_FETCHES", 5)

# Network/model deadlines are deliberately finite. Graph recursion limits only bound
# completed node transitions; they cannot interrupt a hung subprocess, HTTP request, or
# MCP ``call_tool`` await.
MCP_TIMEOUT_SECONDS = _positive_float_env("LLMFLOW_SEARCH_MCP_TIMEOUT_SECONDS", 120.0)
OLLAMA_TIMEOUT_SECONDS = _positive_float_env(
    "LLMFLOW_SEARCH_OLLAMA_TIMEOUT_SECONDS", 300.0
)


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


# The model every number in docs/evaluation.md was measured on, and the default for the
# benchmark runner, the baseline and the attribution judge. Pinned rather than "whatever
# the picker offers", because a benchmark whose model drifts cannot be compared with its
# own earlier results.
EVAL_MODEL = os.getenv("LLMFLOW_SEARCH_EVAL_MODEL", "deepseek-v4.1-flash:cloud")


# The model that judges whether a cited source supports a claim. Deliberately not
# EVAL_MODEL: a model grading its own output shares its blind spots, and "the writer
# agrees with the writer" is not evidence. Keeping it separate also means the judge can be
# held fixed while the system under test changes, which is what makes two scored runs
# comparable at all.
JUDGE_MODEL = os.getenv("LLMFLOW_SEARCH_JUDGE_MODEL", "gemma4:31b-cloud")


# Pages opened without anyone being asked, after a discovery round that queued no read.
# The rule exists because runs used to search repeatedly and answer from result snippets
# alone, having opened nothing. Measured on the benchmark corpus its cost is visible: the
# gold document is the top hit for 83% of questions, yet 5 pages are opened after the
# first search and 3 after each later one — 14 per question, of which 12% mattered.
AUTO_READ_TOP_K = _positive_int_env("LLMFLOW_SEARCH_AUTO_READ_TOP_K", 5)
AUTO_READ_FOLLOWUP_TOP_K = _positive_int_env(
    "LLMFLOW_SEARCH_AUTO_READ_FOLLOWUP_TOP_K", 3
)


# Who picks the pages to open after a search:
#   "auto"  — the top AUTO_READ_TOP_K of the ranked catalog, unconditionally
#   "model" — the post-batch controller, which is already shown that catalog with each
#             URL's title and search snippet and is already allowed to propose reads. In
#             "auto" it is not even asked, because its answer would be discarded.
# The safety net that made "auto" necessary survives either way: a round that would end
# having opened nothing still opens the top of the catalog.
READ_SELECTION = os.getenv("LLMFLOW_SEARCH_READ_SELECTION", "auto").strip().lower()


# Whether every tool result is handed to a model to be diagnosed. One call per search and
# per page opened, which measured 26 calls and 47% of all tokens on a 30-question run —
# more than the evidence ledger, the challenge pass and the post-batch controller put
# together. What it produces is a per-page summary, a source-quality guess and a
# do-not-revisit list; the evidence ledger judges sources by their text and never reads
# any of it. Off, observations._fallback_observation supplies the objective facts — title,
# publication date, whether content came back — directly from the tool's own payload.
#
# Default off, because measuring it settled the question the other way round from the one
# being asked. It was not merely expensive: removing it improved every quality measure at
# once — more right answers, more attributable claims, fewer refusals — while halving the
# tokens and cutting median latency by 59%. Its per-page verdicts were reaching the
# post-batch controller as advice about what not to read, and that advice was worse than
# no advice.
DIAGNOSE_OBSERVATIONS = _flag_env("LLMFLOW_SEARCH_DIAGNOSE_OBSERVATIONS", False)


# Whether a completion criterion may be marked as identifying the subject rather than as
# something the answer must state with a citation. Off restores the behaviour measured
# before the distinction existed: every criterion is a proof obligation, which on a
# ten-clue puzzle question makes strict mode unreachable and refuses runs that already
# hold the answer. See requirements._identifying_indices.
SEPARATE_IDENTIFYING_CRITERIA = _flag_env(
    "LLMFLOW_SEARCH_IDENTIFYING_CRITERIA", True
)


# Whether the verifier's prose gets one deterministic pass for sentences that assert
# something and carry no [n] marker. The answer prompt has always asked for a citation on
# every factual sentence and has never got one: measured over ten BrowseComp-Plus
# questions, half the claim sentences in a verified answer had no marker at all, which put
# a ceiling of about one half on the share of an answer that can be attributed at all.
# Asking again in the prompt is what was already being done; this re-reads the produced
# text and asks once about the specific sentences that failed.
CITATION_REPAIR = _flag_env("LLMFLOW_SEARCH_CITATION_REPAIR", True)


# The repair is skipped when coverage is already at least this high: the last uncited
# sentence in a well-cited answer is usually a framing line, and a model call to hunt it
# costs more than it returns.
CITATION_COVERAGE_TARGET = _non_negative_float_env(
    "LLMFLOW_SEARCH_CITATION_TARGET", 0.95
)


# A repair that shortens the answer past this fraction of its original length has not
# attached citations, it has deleted the answer. Such a result is discarded and the
# verified text is kept as it was.
CITATION_REPAIR_MIN_KEPT = 0.6


# The same treatment for the editorial contract: the drafter and the verifier are asked
# for a frame, one cutoff, figures behind comparatives and no empty headings, and asking
# is all that happened until the text was re-read. This re-reads it, names the defects a
# regular expression can prove, and spends one call on them.
REPORT_STRUCTURE_REPAIR = _flag_env("LLMFLOW_SEARCH_STRUCTURE_REPAIR", True)


# A structure repair is allowed to cut — deleting an unsupported comparative is the
# correct fix. It is not allowed to cut this far: a result shorter than this fraction of
# the report has not repaired it, and the unrepaired report is the better of the two.
STRUCTURE_REPAIR_MIN_KEPT = 0.6


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The found sources do not provide enough information for a reliable answer."
)
