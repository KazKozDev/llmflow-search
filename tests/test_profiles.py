from __future__ import annotations

from llmflow_search.profiles import (
    FOOTNOTE_PROFILE,
    GENERIC_PROFILE,
    select_profile,
)
from llmflow_search.sources import generic_sources_from_tool_result


def test_select_profile_auto_detects_footnote_by_signature_tools():
    footnote_tools = {"web_search", "web_read", "web_extract_tables", "classify_source"}
    assert select_profile(footnote_tools, env="auto") is FOOTNOTE_PROFILE


def test_select_profile_auto_falls_back_to_generic_for_other_servers():
    other_tools = {"get_fact", "list_files", "read_file"}
    assert select_profile(other_tools, env="auto") is GENERIC_PROFILE
    # a partial footnote signature is not enough
    assert select_profile({"web_search"}, env="auto") is GENERIC_PROFILE


def test_select_profile_env_override_forces_choice():
    assert select_profile({"get_fact"}, env="footnote") is FOOTNOTE_PROFILE
    assert select_profile({"web_search", "web_read"}, env="generic") is GENERIC_PROFILE


def test_generic_profile_forces_native_tool_calling():
    # Returning None for every step makes _execute_single_step use native tool-calling.
    assert GENERIC_PROFILE.tool_call_from_step("anything: at all") is None
    assert GENERIC_PROFILE.uses_search_memory is False
    assert FOOTNOTE_PROFILE.uses_search_memory is True


def test_generic_sources_wrap_tool_output_as_one_source():
    sources = generic_sources_from_tool_result("get_fact", "The capital of France is Paris.")
    assert len(sources) == 1
    src = sources[0]
    assert src["content"] == "The capital of France is Paris."
    assert src["title"] == "get_fact"
    assert src["kind"] == "tool_output"
    # A non-empty, unique URL is required so _merge_sources keeps it.
    assert src["url"].startswith("tool://get_fact/")


def test_generic_sources_dedup_identical_outputs_but_keep_distinct():
    a = generic_sources_from_tool_result("t", "same output")
    b = generic_sources_from_tool_result("t", "same output")
    c = generic_sources_from_tool_result("t", "different output")
    assert a[0]["url"] == b[0]["url"]
    assert a[0]["url"] != c[0]["url"]


def test_generic_sources_skip_empty_output():
    assert generic_sources_from_tool_result("t", "") == []
    assert generic_sources_from_tool_result("t", "   \n  ") == []
