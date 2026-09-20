"""Claim segmentation, citation parsing and quote verification.

The attribution number is only as trustworthy as the denominator under it. These pin the
decisions that set that denominator — what counts as a claim, what a marker binds to, and
when a judge's quote counts as really being on the page.
"""

import json

from llmflow_search import attribution, eval_records, schema_guard


def test_sentences_carry_their_own_citations():
    claims = attribution.split_claims(
        "Python 3.13 added a free-threaded build [1]. The JIT stays experimental [2][3]."
    )
    assert [claim.citations for claim in claims] == [[1], [2, 3]]


def test_comma_separated_marker_is_several_citations():
    assert attribution.citations_in("supported by evidence [2, 5,9]") == [2, 5, 9]


def test_repeated_marker_counts_once():
    assert attribution.citations_in("as reported [4] and again [4]") == [4]


def test_bibliography_is_not_a_claim():
    claims = attribution.split_claims(
        "The release shipped in October [1].\n\n"
        "Sources:\n"
        "[1] https://example.com/release-notes\n"
        "[2] https://example.com/changelog\n"
    )
    assert len(claims) == 1


def test_list_item_without_a_period_is_still_a_claim():
    claims = attribution.split_claims("- The REPL was rewritten in Python [2]")
    assert len(claims) == 1
    assert claims[0].citations == [2]


def test_headings_and_lead_ins_are_not_claims():
    claims = attribution.split_claims(
        "## Key findings\n\nThe main changes are as follows:\n"
        "- Free threading became available as a build option [1]\n"
    )
    assert [claim.bare_text for claim in claims] == [
        "Free threading became available as a build option"
    ]


def test_uncited_sentence_is_still_counted_as_a_claim():
    """The headline metric must not improve when the model simply stops citing."""
    claims = attribution.split_claims(
        "The build is experimental [1]. It will be stable in a future release."
    )
    assert len(claims) == 2
    assert [claim.is_cited for claim in claims] == [True, False]


def test_table_row_is_one_claim_and_separator_is_none():
    claims = attribution.split_claims(
        "| Feature | Status |\n"
        "| --- | --- |\n"
        "| Free-threaded build | shipped as experimental in 3.13 [1] |\n"
    )
    assert len(claims) == 1
    assert claims[0].citations == [1]


def test_bare_text_drops_markers_without_stranding_punctuation():
    claim = attribution.split_claims("The JIT is experimental [1][2].")[0]
    assert claim.bare_text == "The JIT is experimental."


SOURCE = (
    "In Python 3.13 the interpreter gained an experimental free-threaded build, "
    "disabled by default. The JIT remains off in released binaries."
)


def test_exact_quote_is_found():
    check = attribution.find_quote("gained an experimental free-threaded build", SOURCE)
    assert check.found and check.exact


def test_cosmetic_drift_still_counts_as_found():
    """Models retype spans and lose punctuation; that is not a fabricated quote."""
    check = attribution.find_quote(
        "gained an  experimental free—threaded build", SOURCE
    )
    assert check.found


def test_paraphrase_is_not_a_quote():
    check = attribution.find_quote(
        "the interpreter now supports running without the GIL", SOURCE
    )
    assert not check.found


def test_invented_quote_is_not_found():
    check = attribution.find_quote("the GIL was removed entirely in 3.13", SOURCE)
    assert not check.found


def test_empty_quote_is_not_found():
    assert not attribution.find_quote("", SOURCE).found


def test_lexical_overlap_ignores_stopwords():
    assert attribution.lexical_overlap("the and of it is", SOURCE) == 0.0
    assert attribution.lexical_overlap("experimental free-threaded build", SOURCE) == 1.0


LEDGER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
        "quote": {"type": "string"},
        "index": {"type": "integer", "minimum": 0, "maximum": 3},
    },
    "required": ["verdict", "quote"],
    "additionalProperties": False,
}


def test_conforming_object_has_no_violations():
    assert schema_guard.conforms(
        {"verdict": "supported", "quote": "x", "index": 2}, LEDGER_SCHEMA
    )


def test_missing_key_unexpected_key_and_bad_enum_are_all_caught():
    violations = schema_guard.schema_violations(
        {"capital": "Paris", "quote": "x"}, LEDGER_SCHEMA
    )
    assert any("missing required key 'verdict'" in v for v in violations)
    assert any("unexpected key 'capital'" in v for v in violations)


def test_out_of_range_integer_is_caught():
    assert schema_guard.schema_violations(
        {"verdict": "supported", "quote": "", "index": 9}, LEDGER_SCHEMA
    )


def test_booleans_are_not_integers():
    assert schema_guard.schema_violations(
        {"verdict": "supported", "quote": "", "index": True}, LEDGER_SCHEMA
    )


def test_schema_description_names_keys_types_and_constraints():
    described = schema_guard.describe_schema(LEDGER_SCHEMA)
    assert "verdict: string" in described
    assert "supported|unsupported" in described
    assert "optional" in described  # index is not required


def test_sidecar_path_sits_next_to_its_trace():
    assert (
        eval_records.sidecar_path("reports/agent.trace.jsonl").name
        == "agent.answers.jsonl"
    )


def test_source_records_number_sources_the_way_the_prompt_did():
    records = eval_records.source_records(
        [
            {"url": "https://a.test", "title": "A", "content": "alpha " * 50},
            {"url": "https://b.test", "title": "B", "content": "beta " * 50},
        ],
        "alpha",
    )
    assert [record["id"] for record in records] == [1, 2]
    assert records[0]["url"] == "https://a.test"
    assert "alpha" in records[0]["shown"]


def test_model_call_bills_every_round_trip_to_one_decision(tmp_path):
    """A repair retry is part of the decision's cost, not a second decision."""
    from llmflow_search import trace

    path = tmp_path / "run.trace.jsonl"
    trace.start_run(path)
    trace.begin_query("q1")
    try:
        with trace.model_call("evidence_ledger", "prompt", "m"):
            trace.record_usage("m", 1000, 50)
            trace.record_schema(False)
            trace.record_usage("m", 1100, 60)  # the repair attempt
            trace.record_schema(True)
    finally:
        trace.close_run()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    call = next(e for e in events if e["event"] == "llm_call")
    assert call["prompt_tokens"] == 2100
    assert call["completion_tokens"] == 110
    assert call["round_trips"] == 2
    assert call["schema_conforms"] is True
    assert call["schema_conforms_first_try"] is False
    assert call["query_id"] == "q1"


def test_usage_recorded_outside_a_decision_is_dropped_not_crashed():
    from llmflow_search import trace

    trace.record_usage("m", 10, 10)  # no open trace, no open model_call
    trace.record_schema(True)


def test_a_quote_stitched_from_two_real_spans_is_found():
    """Judges join the sentence and its qualifier with an ellipsis; both are real."""
    source = (
        "Alpha beta gamma delta epsilon zeta. Many unrelated words sit here. "
        "The final clause matters greatly indeed."
    )
    check = attribution.find_quote(
        "Alpha beta gamma delta epsilon zeta... The final clause matters greatly indeed",
        source,
    )
    assert check.found


def test_a_stitched_quote_with_one_invented_half_is_not_found():
    source = (
        "Alpha beta gamma delta epsilon zeta. Many unrelated words sit here. "
        "The final clause matters greatly indeed."
    )
    check = attribution.find_quote(
        "Alpha beta gamma delta epsilon zeta... and the committee rejected the proposal",
        source,
    )
    assert not check.found
