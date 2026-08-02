from llmflow_search.config import current_date_iso


def test_current_date_prefers_explicit_runtime_anchor(monkeypatch):
    monkeypatch.setenv("CURRENT_DATE", "2026-07-01")
    monkeypatch.delenv("LLMFLOW_SEARCH_TODAY", raising=False)

    assert current_date_iso() == "2026-07-01"


def test_current_date_prefers_llmflow_search_today(monkeypatch):
    monkeypatch.setenv("CURRENT_DATE", "2026-07-01")
    monkeypatch.setenv("LLMFLOW_SEARCH_TODAY", "2026-07-02")

    assert current_date_iso() == "2026-07-02"
