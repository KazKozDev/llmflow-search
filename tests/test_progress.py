from io import StringIO

from deep_research.progress import DONE, FAILED, RUNNING, SKIPPED, TerminalProgressReporter


def test_prints_human_readable_search_progress() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("search_started", {"task_id": "task_1", "query": "local RAG"})
    reporter("page_fetched", {"task_id": "task_1", "title": "Source", "characters": 321})

    output = stream.getvalue()
    assert f"{RUNNING} SEARCH [task_1] query queued — local RAG" in output
    assert f"{DONE} FETCH [task_1] Source (321 characters)" in output


def test_same_component_tracks_a_step_from_start_to_finish() -> None:
    """The whole point of the glyph is telling apart in-progress vs done for one agent step."""
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("page_fetch_started", {"task_id": "task_1", "title": "Source"})
    reporter("page_fetched", {"task_id": "task_1", "title": "Source", "characters": 100})
    reporter("page_skipped", {"task_id": "task_1", "reason": "too small"})

    lines = stream.getvalue().splitlines()
    assert all(" FETCH " in line for line in lines)
    assert lines[0].split()[1] == RUNNING
    assert lines[1].split()[1] == DONE
    assert lines[2].split()[1] == SKIPPED


def test_failure_glyph_on_run_failed() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("run_failed", {"error": "boom", "error_type": "ValueError"})

    output = stream.getvalue()
    assert f"{FAILED} FAILED [ValueError] boom" in output


def test_gap_followup_events_are_readable() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("coverage_gaps_identified", {"gaps": ["missing investment figures"]})
    reporter("gap_followup_started", {"round": 2, "gaps": ["missing investment figures"], "tasks": 2})
    reporter("gap_followup_exhausted", {"round": 2})

    output = stream.getvalue()
    assert "GAP AUDIT" in output
    assert "missing investment figures" in output
    assert "round 2" in output


def test_color_is_disabled_for_non_tty_streams() -> None:
    stream = StringIO()  # StringIO has no isatty()==True, mirrors a redirected/piped log
    reporter = TerminalProgressReporter(stream)

    reporter("plan_created", {"tasks": [1, 2]})

    assert "\x1b[" not in stream.getvalue()


def test_color_can_be_forced_on() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream, color=True)

    reporter("plan_created", {"tasks": [1, 2]})

    output = stream.getvalue()
    assert "\x1b[" in output
    assert "tasks created — 2" in output


def test_status_changed_event() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("status_changed", {"status": "researching"})
    reporter("status_changed", {"status": "completed"})

    lines = stream.getvalue().splitlines()
    assert f"{RUNNING} STAGE researching" in lines[0]
    assert f"{DONE} STAGE completed" in lines[1]


def test_search_completed_and_failed_and_selection_events() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("search_completed", {"task_id": "task_1", "result_count": 3, "off_topic_dropped": 2})
    reporter("search_completed", {"task_id": "task_1", "result_count": 1})
    reporter("search_failed", {"task_id": "task_1", "error": "timed out"})
    reporter("search_selection", {"task_id": "task_1", "selected": ["a", "b"]})

    output = stream.getvalue()
    assert "results found — 3 (2 off-topic dropped)" in output
    assert "results found — 1" in output
    assert f"{FAILED} SEARCH [task_1] error — timed out" in output
    assert "selected sources — 2" in output


def test_page_reused_and_evidence_and_task_events() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("page_reused", {"task_id": "task_1", "title": "Source"})
    reporter("evidence_saved", {"task_id": "task_1", "evidence_id": "ev_1"})
    reporter("evidence_rejected", {"task_id": "task_1", "reason": "low quality"})
    reporter("task_completed", {"task_id": "task_1", "evidence_count": 4})

    output = stream.getvalue()
    assert "Source (already fetched, reused)" in output
    assert f"{DONE} EVIDENCE [task_1] saved — ev_1" in output
    assert f"{SKIPPED} EVIDENCE [task_1] rejected — low quality" in output
    assert f"{DONE} TASK [task_1] evidence — 4" in output


def test_fact_check_batch_and_verification_events() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("fact_check_batch_started", {"batch": 1, "total_batches": 2, "claim_groups": 5})
    reporter("fact_check_batch_completed", {"batch": 1, "total_batches": 2, "reviews": 5})
    reporter("verification_completed", {"reviews": 10})

    output = stream.getvalue()
    assert "batch 1/2 started (5 claim groups)" in output
    assert "batch 1/2 completed (5 reviews)" in output
    assert "claim groups reviewed — 10" in output


def test_resume_started_event() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("resume_started", {"evidence": 7, "already_reviewed": 3})

    assert "7 evidence loaded, 3 already reviewed" in stream.getvalue()


def test_report_lifecycle_events() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("report_admission", {"admitted": 5, "verified": 8, "excluded": 3})
    reporter("report_rewrite", {"reason": "unresolved citation"})
    reporter("report_ready", {})
    reporter("pdf_report_saved", {"path": "reports/out.pdf"})
    reporter("pdf_report_failed", {"error": "renderer crashed"})

    output = stream.getvalue()
    assert "admitted evidence — 5/8 (excluded 3)" in output
    assert "citation check failed, rewriting — unresolved citation" in output
    assert "citations validated, research completed" in output
    assert f"{DONE} REPORT PDF saved — reports/out.pdf" in output
    assert f"{SKIPPED} REPORT PDF export failed — renderer crashed" in output


def test_unknown_event_falls_back_to_generic_rendering() -> None:
    stream = StringIO()
    reporter = TerminalProgressReporter(stream)

    reporter("something_new", {"detail": "x"})

    output = stream.getvalue()
    assert f"{RUNNING} SOMETHING_NEW {{'detail': 'x'}}" in output


def test_no_color_env_var_overrides_tty_autodetection(monkeypatch) -> None:
    class _FakeTty(StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    stream = _FakeTty()
    reporter = TerminalProgressReporter(stream)

    reporter("plan_created", {"tasks": []})

    assert "\x1b[" not in stream.getvalue()
