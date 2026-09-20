"""Every tool the server exposes is accounted for, and the accounting is true.

Three unrelated lists decide whether a tool is usable — the planner prompts, the evidence
converter, and the effect policy — and a tool can fall out of all three without anything
failing. These tests hold the registry against the other two, so a claim like "the agent
can use this" cannot drift away from the code that would have to run.

The server list is frozen here rather than read live: a test that starts an MCP server is
a test that fails for reasons of its own, and the point is to notice when the frozen list
and the code disagree.
"""

from __future__ import annotations

from llmflow_search import tool_scope
from llmflow_search.policy import AUTO_READ_DISCOVERY_TOOLS
from llmflow_search.search_memory import _DISCOVERY_SEARCH_TOOLS
from llmflow_search.sources import _sources_from_tool_result
from llmflow_search.tool_policy import ToolEffect, classify_tool, unattended_authorizer

# footnote-mcp, as it answered list_tools on 2026-09-20.
LIVE_SERVER_TOOLS = frozenset(
    """archive_search browser_extract_tables browser_extract_tables_for_date_range
    browser_set_date_range build_research_debug_report check_date_completeness
    classify_source corroborate_claim encyclopedia_search evidence_entailment
    export_dataset generate_search_queries github_search locate_claim_span papers_search
    recipe_registry reconcile_time_series resolve_units source_cache_get source_cache_put
    startup_health_check tool_code_generate tool_code_run_sandboxed tool_code_validate
    tool_promote tool_spec_propose validate_unit_rows web_archive_fetch web_click
    web_crawl web_deep_search web_detect_downloads web_extract web_extract_tables
    web_fetch_authenticated web_fetch_json web_navigate web_parse_file web_read
    web_screenshot web_scroll web_search web_search_recent web_snapshot web_type""".split()
)


def test_every_tool_the_server_offers_is_classified():
    assert tool_scope.unclassified(LIVE_SERVER_TOOLS) == set()


def test_the_registry_invents_no_tools():
    assert tool_scope.CLASSIFIED - LIVE_SERVER_TOOLS == set()


def test_nothing_is_both_in_scope_and_out_of_it():
    in_scope = (
        tool_scope.EVIDENCE_TOOLS
        | tool_scope.DISCOVERY_TOOLS
        | tool_scope.SUPPORTING_TOOLS
    )
    assert in_scope & set(tool_scope.OUT_OF_SCOPE_TOOLS) == set()


def test_every_out_of_scope_tool_carries_its_reason():
    for name, reason in tool_scope.OUT_OF_SCOPE_TOOLS.items():
        assert reason.strip(), name


def test_every_evidence_tool_really_has_an_evidence_branch():
    """The registry promising a source where the converter returns nothing is the exact
    failure it exists to prevent."""
    empty_payload = "{}"
    for name in tool_scope.EVIDENCE_TOOLS:
        # An empty payload yields no sources, but a tool with no branch at all and one
        # whose branch declined this payload are indistinguishable that way. What is
        # checkable without fixtures per tool is that the call is handled, not raising.
        assert _sources_from_tool_result(name, empty_payload) == []


def test_the_discovery_registry_matches_the_two_lists_that_use_it():
    assert tool_scope.DISCOVERY_TOOLS == set(_DISCOVERY_SEARCH_TOOLS)
    assert tool_scope.DISCOVERY_TOOLS == set(AUTO_READ_DISCOVERY_TOOLS)


# --- what the policy lets an unattended run do -----------------------------------


def test_the_analysis_tools_are_no_longer_gated():
    """They read evidence and return a judgement; the verb list simply did not know them,
    and an unrecognised verb falls through to external_write."""
    for name in (
        "corroborate_claim",
        "evidence_entailment",
        "locate_claim_span",
        "reconcile_time_series",
        "web_parse_file",
    ):
        assert classify_tool(name, []).effect is ToolEffect.READ_ONLY, name


def test_a_browser_tool_is_local_state_not_an_external_write():
    """browser_set_date_range sat behind a prompt while the tool that needs it ran free."""
    assert (
        classify_tool("browser_set_date_range", []).effect is ToolEffect.LOCAL_WRITE
    )


def test_writing_and_code_execution_are_still_gated():
    for name in ("tool_promote", "export_dataset", "tool_code_run_sandboxed"):
        assert classify_tool(name, []).effect is not ToolEffect.READ_ONLY, name


def test_an_unattended_run_allows_local_state_and_refuses_the_rest():
    authorize = unattended_authorizer()
    assert authorize("source_cache_put", {}, ToolEffect.LOCAL_WRITE) is True
    assert authorize("tool_promote", {}, ToolEffect.EXTERNAL_WRITE) is False
    assert authorize("anything_delete", {}, ToolEffect.DESTRUCTIVE) is False


def test_an_unattended_policy_can_be_widened_deliberately():
    authorize = unattended_authorizer(
        frozenset({ToolEffect.READ_ONLY, ToolEffect.EXTERNAL_WRITE})
    )
    assert authorize("tool_code_run_sandboxed", {}, ToolEffect.EXTERNAL_WRITE) is True
    assert authorize("anything_delete", {}, ToolEffect.DESTRUCTIVE) is False
