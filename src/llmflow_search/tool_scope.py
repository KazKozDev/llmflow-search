"""What the agent does with each tool the footnote server exposes — stated, not implied.

A tool can be present on the server and absent from the agent in three different ways, and
none of them raises anything. The planner prompts name some tools and not others, so the
model never thinks of the rest. ``sources._sources_from_tool_result`` turns some results
into citable evidence and drops the rest on the floor. ``tool_policy`` gates whatever its
verb list does not recognise. Measured against the live server, 45 tools were exposed, 19
were named in a prompt, 15 could become evidence, and 19 were reachable only by accident
and useless if reached.

None of that was decided. It was what three unrelated lists happened to add up to. This
module is the decision written down: every tool is in exactly one bucket, and
``test_tool_scope`` fails when the server grows one that is in none — so the next gap is
an argument someone has, rather than a silence.
"""

# Tools whose result becomes a citable source. The evidence path runs through
# ``sources._sources_from_tool_result``; adding a name here without a branch there is a
# promise the pipeline does not keep.
EVIDENCE_TOOLS = frozenset(
    {
        "archive_search",
        "browser_extract_tables",
        "browser_extract_tables_for_date_range",
        "encyclopedia_search",
        "github_search",
        "papers_search",
        "tool_code_run_sandboxed",
        "web_archive_fetch",
        "web_crawl",
        "web_deep_search",
        "web_extract",
        "web_extract_tables",
        "web_fetch_json",
        "web_parse_file",
        "web_read",
    }
)

# Discovery tools. They return a catalog of addresses rather than evidence, and the
# addresses drive the automatic read pass (``policy.AUTO_READ_DISCOVERY_TOOLS``), which is
# where the evidence actually comes from. No evidence branch is missing here.
DISCOVERY_TOOLS = frozenset(
    {
        "web_search",
        "web_search_recent",
        "web_deep_search",
        "papers_search",
        "encyclopedia_search",
        "github_search",
        "archive_search",
    }
)

# Tools that support a step without producing evidence of their own: they shape a query,
# set up a browser, or judge something the run already holds. A result from one of these
# is meant to inform the controller, not to be cited.
SUPPORTING_TOOLS = frozenset(
    {
        "browser_set_date_range",
        "check_date_completeness",
        "classify_source",
        "corroborate_claim",
        "evidence_entailment",
        "generate_search_queries",
        "locate_claim_span",
        "reconcile_time_series",
        "resolve_units",
        "validate_unit_rows",
        "web_detect_downloads",
        "web_fetch_authenticated",
        "web_navigate",
    }
)

# Deliberately unused, with the reason. Not a backlog: a text research agent that cites
# pages has no use for a pointing device or a picture of a page, and the authoring tools
# belong to whoever extends the server, not to a run answering a question.
OUT_OF_SCOPE_TOOLS = {
    "web_click": "interaction: the agent reads pages, it does not operate them",
    "web_type": "interaction: nothing in a research run fills in a form",
    "web_scroll": "interaction: web_read and web_extract already return the whole text",
    "web_screenshot": "visual: an image cannot be cited or checked against a claim",
    "web_snapshot": "visual: same, and the accessibility tree duplicates web_read",
    "source_cache_get": "server-side plumbing, invisible to the answer",
    "source_cache_put": "server-side plumbing, invisible to the answer",
    "export_dataset": "writes a file for a human, after the run has answered",
    "startup_health_check": "server diagnostics, not research",
    "build_research_debug_report": (
        "used as a Python import in reports.py, never as an MCP call"
    ),
    "recipe_registry": "authoring: extends the server, not this run",
    "tool_promote": "authoring: extends the server, not this run",
    "tool_spec_propose": "authoring: extends the server, not this run",
    "tool_code_generate": "authoring: extends the server, not this run",
    "tool_code_validate": "authoring: extends the server, not this run",
}

CLASSIFIED = (
    EVIDENCE_TOOLS | DISCOVERY_TOOLS | SUPPORTING_TOOLS | set(OUT_OF_SCOPE_TOOLS)
)


def unclassified(tool_names) -> set[str]:
    """Tools the server offers that nobody has decided about."""
    return {str(name) for name in tool_names} - CLASSIFIED
