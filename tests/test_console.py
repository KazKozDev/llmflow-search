from io import StringIO

from llmflow_search import console


def test_stage_log_uses_muted_truecolor_when_forced(monkeypatch):
    stream = StringIO()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("LLMFLOW_SEARCH_FORCE_COLOR", "1")

    console.print("  [PLAN] Building a grounded plan...", file=stream)

    output = stream.getvalue()
    assert "\033[38;2;112;143;181m" in output
    assert "[PLAN]" in output
    assert "\033[0m Building a grounded plan...\n" in output


def test_no_color_disables_color_without_app_override(monkeypatch):
    stream = StringIO()
    monkeypatch.delenv("LLMFLOW_SEARCH_FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    console.print("  [VERIFY] Checking claims...", file=stream)

    assert stream.getvalue() == "  [VERIFY] Checking claims...\n"


def test_app_override_can_force_color_despite_global_no_color(monkeypatch):
    stream = StringIO()
    monkeypatch.setenv("LLMFLOW_SEARCH_FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    console.print("  [VERIFY] Checking claims...", file=stream)

    assert "\033[38;2;113;157;130m" in stream.getvalue()


def test_redirected_output_is_plain_without_force(monkeypatch):
    stream = StringIO()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("LLMFLOW_SEARCH_FORCE_COLOR", raising=False)

    console.print("    [web_search] ", end="", file=stream)

    assert stream.getvalue() == "    [web_search] "


def test_success_and_error_have_distinct_tones(monkeypatch):
    stream = StringIO()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("LLMFLOW_SEARCH_FORCE_COLOR", "1")

    console.print("✓ connected", file=stream)
    console.print("[!] failed", file=stream)

    output = stream.getvalue()
    assert "\033[38;2;113;157;130m" in output
    assert "\033[38;2;184;112;121m" in output
