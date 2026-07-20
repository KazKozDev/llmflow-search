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


def test_no_color_env_var_overrides_tty_autodetection(monkeypatch) -> None:
    class _FakeTty(StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    stream = _FakeTty()
    reporter = TerminalProgressReporter(stream)

    reporter("plan_created", {"tasks": []})

    assert "\x1b[" not in stream.getvalue()
