"""Machine-readable run trace: one JSON object per line, one line per event.

The only thing a run used to leave behind was its final score, so every question about
*why* a run scored what it did was answered by running regular expressions over a
human-readable log. That log is written for a person watching a terminal: its wording
changes whenever a print statement is reworded, and it does not record the things the
diagnosis actually needs — which URLs a search returned and at what rank, which of them
the run chose to read and on whose authority, what the post-batch controller was shown
before it decided, and where the wall clock went.

This module writes those events instead. It is deliberately a side channel: nothing in
the graph reads it back, emitting is a no-op when no trace is open, and a broken trace
writer must never take a run down with it.
"""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# The post-batch input runs to tens of thousands of characters. The trace keeps its size
# exactly and its opening as evidence of what the model was actually shown.
INPUT_EXCERPT_MAX_CHARS = 2000


class RunTrace:
    """An open JSONL trace file. One instance per process run, shared by every query."""

    def __init__(self, path: str | os.PathLike, run_id: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._fields: dict[str, Any] = {"query_id": None, "round": 0}

    def bind(self, **fields: Any) -> None:
        """Set the fields every later event carries (query_id, round)."""
        self._fields.update(fields)

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            **self._fields,
            "event": event,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


_ACTIVE: RunTrace | None = None


def start_run(path: str | os.PathLike | None, run_id: str = "") -> RunTrace | None:
    """Open a trace at ``path``, or return None when tracing is off."""
    global _ACTIVE
    if not path:
        return None
    _ACTIVE = RunTrace(path, run_id)
    return _ACTIVE


def close_run() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close()
        _ACTIVE = None


def active() -> RunTrace | None:
    return _ACTIVE


def emit(event: str, **payload: Any) -> None:
    """Record one event, or do nothing when no trace is open."""
    trace = _ACTIVE
    if trace is None:
        return
    try:
        trace.emit(event, **payload)
    except Exception:  # a diagnostic channel must not be able to fail a run
        pass


def bind(**fields: Any) -> None:
    if _ACTIVE is not None:
        _ACTIVE.bind(**fields)


def begin_query(query_id: str, **payload: Any) -> None:
    bind(query_id=query_id, round=0)
    emit("query_start", **payload)


def set_round(number: int) -> None:
    bind(round=int(number))


@contextmanager
def model_call(role: str, prompt: str = "", model: str = ""):
    """Time one model call and record which decision it was serving.

    Wrapped at the call site rather than inside the Ollama helper: the role is a property
    of the node that asked, not of the transport, and latency work needs to know which
    decisions are worth making without a model at all.
    """
    with span("llm_call", role=role, model=model, prompt_chars=len(prompt or "")):
        yield


@contextmanager
def span(event: str, **payload: Any):
    """Time a block and emit one event carrying its duration.

    The yielded dict is merged into the event, so a caller can record what it learned
    during the block (a result size, an error) alongside how long the block took.
    """
    extra: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        yield extra
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        emit(event, duration_ms=duration_ms, **payload, **extra)


def excerpt(text: str, limit: int = INPUT_EXCERPT_MAX_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"
