from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, TextIO


class _Ansi:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    BRIGHT_RED = "\x1b[91m"
    BRIGHT_GREEN = "\x1b[92m"
    BRIGHT_YELLOW = "\x1b[93m"
    BRIGHT_CYAN = "\x1b[96m"
    WHITE = "\x1b[37m"


# One color per pipeline component, held constant across its whole lifecycle (started,
# progress, done) — the LangChain-verbose convention of tagging each "agent" so a
# scrolling log stays readable at a glance instead of being a wall of uniform text.
_COMPONENT_COLOR = {
    "STAGE": _Ansi.BRIGHT_CYAN,
    "PLAN": _Ansi.CYAN,
    "SEARCH": _Ansi.BLUE,
    "FETCH": _Ansi.YELLOW,
    "EVIDENCE": _Ansi.GREEN,
    "TASK": _Ansi.WHITE,
    "FACT CHECK": _Ansi.MAGENTA,
    "GAP AUDIT": _Ansi.BRIGHT_YELLOW,
    "REPORT": _Ansi.BRIGHT_GREEN,
    "RESUME": _Ansi.CYAN,
    "FAILED": _Ansi.BRIGHT_RED,
}

# Leading glyph: the quick-scan cue for "is this component actively working right now".
# ● = just started (an agent call is in flight), ✓ = finished, ○ = skipped/filtered out
# (not an error — a normal integrity check declined something), ✗ = failed.
RUNNING, DONE, SKIPPED, FAILED = "●", "✓", "○", "✗"
_GLYPH_COLOR = {RUNNING: None, DONE: _Ansi.GREEN, SKIPPED: _Ansi.DIM, FAILED: _Ansi.BRIGHT_RED}


class TerminalProgressReporter:
    """Human-readable, colorized progress for a research run; writes to stderr.

    Each pipeline component (planner, search, fetch, fact-checker, ...) keeps one
    color through its lifecycle; a leading glyph shows whether a step just started
    (an agent call is actively running), finished, was skipped, or failed. Colors are
    auto-disabled when the stream is not a live terminal (e.g. piped to a file, as
    happens when output is captured for later inspection) or when NO_COLOR is set
    (https://no-color.org), so plain-text logs stay grep-friendly.
    """

    def __init__(self, stream: TextIO | None = None, color: bool | None = None) -> None:
        self.stream = stream or sys.stderr
        if color is None:
            color = bool(getattr(self.stream, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")
        self.color = color

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        component, glyph, message = self._render(event_type, payload)
        if self.color:
            component_color = _COMPONENT_COLOR.get(component, "")
            glyph_color = _GLYPH_COLOR.get(glyph) or component_color
            line = (
                f"{_Ansi.DIM}[{timestamp}]{_Ansi.RESET} "
                f"{glyph_color}{glyph}{_Ansi.RESET} "
                f"{component_color}{_Ansi.BOLD}{component}{_Ansi.RESET} {message}"
            )
        else:
            line = f"[{timestamp}] {glyph} {component} {message}"
        print(line, file=self.stream, flush=True)

    @staticmethod
    def _render(event_type: str, payload: dict[str, Any]) -> tuple[str, str, str]:
        """Maps a pipeline event to (component, glyph, message)."""
        if event_type == "status_changed":
            status = payload["status"]
            return "STAGE", DONE if status == "completed" else RUNNING, status

        if event_type == "plan_created":
            return "PLAN", DONE, f"tasks created — {len(payload.get('tasks', []))}"

        if event_type == "search_started":
            return "SEARCH", RUNNING, f"[{payload['task_id']}] query queued — {payload['query']}"
        if event_type == "search_completed":
            dropped = payload.get("off_topic_dropped", 0)
            extra = f" ({dropped} off-topic dropped)" if dropped else ""
            return "SEARCH", DONE, f"[{payload['task_id']}] results found — {payload['result_count']}{extra}"
        if event_type == "search_failed":
            return "SEARCH", FAILED, f"[{payload['task_id']}] error — {payload['error']}"
        if event_type == "search_selection":
            return "SEARCH", DONE, f"[{payload['task_id']}] selected sources — {len(payload['selected'])}"

        if event_type == "page_fetch_started":
            return "FETCH", RUNNING, f"[{payload['task_id']}] {payload['title']}"
        if event_type == "page_fetched":
            return "FETCH", DONE, f"[{payload['task_id']}] {payload['title']} ({payload['characters']} characters)"
        if event_type == "page_reused":
            return "FETCH", DONE, f"[{payload['task_id']}] {payload['title']} (already fetched, reused)"
        if event_type == "page_skipped":
            return "FETCH", SKIPPED, f"[{payload['task_id']}] skipped — {payload['reason']}"

        if event_type == "evidence_saved":
            return "EVIDENCE", DONE, f"[{payload['task_id']}] saved — {payload['evidence_id']}"
        if event_type == "evidence_rejected":
            return "EVIDENCE", SKIPPED, f"[{payload['task_id']}] rejected — {payload['reason']}"
        if event_type == "task_completed":
            return "TASK", DONE, f"[{payload['task_id']}] evidence — {payload['evidence_count']}"

        if event_type == "fact_check_batch_started":
            return (
                "FACT CHECK",
                RUNNING,
                f"batch {payload['batch']}/{payload['total_batches']} started ({payload['claim_groups']} claim groups)",
            )
        if event_type == "fact_check_batch_completed":
            return (
                "FACT CHECK",
                DONE,
                f"batch {payload['batch']}/{payload['total_batches']} completed ({payload['reviews']} reviews)",
            )
        if event_type == "verification_completed":
            return "FACT CHECK", DONE, f"claim groups reviewed — {payload['reviews']}"

        if event_type == "coverage_gaps_identified":
            gaps = payload.get("gaps", [])
            return "GAP AUDIT", RUNNING, f"{len(gaps)} gap(s) found — " + "; ".join(gaps)
        if event_type == "gap_followup_started":
            return (
                "GAP AUDIT",
                RUNNING,
                f"round {payload['round']} — {payload['tasks']} follow-up task(s) queued",
            )
        if event_type == "gap_followup_exhausted":
            return "GAP AUDIT", SKIPPED, f"round {payload['round']} — no new evidence found, stopping"

        if event_type == "resume_started":
            return (
                "RESUME",
                RUNNING,
                f"{payload['evidence']} evidence loaded, {payload['already_reviewed']} already reviewed",
            )

        if event_type == "report_admission":
            return (
                "REPORT",
                DONE,
                f"admitted evidence — {payload['admitted']}/{payload['verified']} (excluded {payload['excluded']})",
            )
        if event_type == "report_rewrite":
            return "REPORT", RUNNING, f"citation check failed, rewriting — {payload['reason']}"
        if event_type == "report_ready":
            return "REPORT", DONE, "citations validated, research completed"
        if event_type == "pdf_report_saved":
            return "REPORT", DONE, f"PDF saved — {payload['path']}"
        if event_type == "pdf_report_failed":
            return "REPORT", SKIPPED, f"PDF export failed — {payload['error']}"

        if event_type == "run_failed":
            return "FAILED", FAILED, f"[{payload['error_type']}] {payload['error']}"

        return event_type.upper(), RUNNING, str(payload)
