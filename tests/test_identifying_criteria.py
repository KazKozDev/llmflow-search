"""Clues that identify the subject versus requirements the answer must prove.

A puzzle question decomposes into a conjunction, and every part of it used to be a proof
obligation — so strict mode could not be satisfied on the questions it exists for, and
runs refused while already holding the answer. These pin the new rule and, more
importantly, the parts of the old one that must not move: an answer with nothing
supported is still refused, and a marking that would make the gate vacuous is ignored.
"""

import pytest

from llmflow_search.evidence import _normalize_evidence_ledger_result
from llmflow_search.requirements import _normalize_requirements, reportable_indices

CRITERIA = [
    "The subject was born in 1886.",
    "The subject was mistaken for a shaman on a 1915 trip.",
    "Name the publication that resulted.",
]
SOURCES = [{"url": "https://a.test", "content": "..."}]


def _ledger(rows, identifying=None):
    return _normalize_evidence_ledger_result(
        {"ledger": rows, "global_missing": [], "next_steps": [], "reason": ""},
        SOURCES,
        CRITERIA,
        "strict",
        None,
        identifying,
    )


def _row(index, status="supported"):
    return {
        "claim_id": f"c{index}",
        "requirement_index": index,
        "support_status": status,
        "support_level": status,
        "can_use_in_answer": status == "supported",
        "source_ids": [1],
    }


def test_marking_is_parsed_and_the_rest_stays_reportable():
    requirements = _normalize_requirements(
        {"completion_criteria": CRITERIA, "identifying_criteria": [0, 1]}, "q"
    )
    assert sorted(reportable_indices(requirements)) == [2]


def test_out_of_range_and_duplicate_markings_are_dropped():
    requirements = _normalize_requirements(
        {"completion_criteria": CRITERIA, "identifying_criteria": [0, 0, 7, "x"]}, "q"
    )
    assert requirements["identifying_criteria"] == [0]


def test_marking_every_criterion_is_ignored():
    """Otherwise the gate proves nothing and any answer passes."""
    requirements = _normalize_requirements(
        {"completion_criteria": CRITERIA, "identifying_criteria": [0, 1, 2]}, "q"
    )
    assert requirements["identifying_criteria"] == []


def test_unproven_clues_no_longer_block_a_proven_answer():
    result = _ledger([_row(2)], identifying=[0, 1])
    assert result["answer_ready"] is True


def test_the_same_ledger_is_blocked_without_the_marking():
    """The old behaviour, still reachable with LLMFLOW_SEARCH_IDENTIFYING_CRITERIA=0."""
    assert _ledger([_row(2)])["answer_ready"] is False


def test_an_unproven_reportable_criterion_still_blocks():
    """Relaxing the clues must not relax what the user actually asked to be told."""
    result = _ledger([_row(0), _row(1)], identifying=[0, 1])
    assert result["answer_ready"] is False


def test_nothing_supported_is_still_a_refusal():
    result = _ledger([_row(2, "missing")], identifying=[0, 1])
    assert result["answer_ready"] is False
    assert result["admissible_count"] == 0


def test_unconfirmed_clues_are_reported_for_the_answer_to_disclose():
    result = _ledger([_row(2)], identifying=[0, 1])
    assert result["unconfirmed_identifying"] == [CRITERIA[0], CRITERIA[1]]


def test_a_confirmed_clue_is_not_reported_as_unconfirmed():
    result = _ledger([_row(0), _row(2)], identifying=[0, 1])
    assert result["unconfirmed_identifying"] == [CRITERIA[1]]


def test_a_blocking_gap_still_blocks_whatever_is_marked():
    result = _normalize_evidence_ledger_result(
        {
            "ledger": [_row(2)],
            "global_missing": [{"requirement_index": 2, "text": "no source names it"}],
            "next_steps": [],
            "reason": "",
        },
        SOURCES,
        CRITERIA,
        "strict",
        None,
        [0, 1],
    )
    assert result["answer_ready"] is False


@pytest.mark.parametrize("marking", [None, [], [0, 1]])
def test_roundup_mode_is_untouched(marking):
    result = _normalize_evidence_ledger_result(
        {"ledger": [_row(2)], "global_missing": [], "next_steps": [], "reason": ""},
        SOURCES,
        CRITERIA,
        "roundup",
        None,
        marking,
    )
    assert result["answer_mode"] == "roundup"
    assert result["unconfirmed_identifying"] == []


def test_a_gap_reported_against_a_clue_does_not_block():
    """The relaxation is worthless if the gaps list closes the gate again."""
    result = _normalize_evidence_ledger_result(
        {
            "ledger": [_row(2)],
            "global_missing": [
                {"requirement_index": 0, "text": "no source gives the birth year"}
            ],
            "next_steps": [],
            "reason": "",
        },
        SOURCES,
        CRITERIA,
        "strict",
        None,
        [0, 1],
    )
    assert result["answer_ready"] is True
    assert "no source gives the birth year" in result["unconfirmed_identifying"]


def test_the_challenge_pass_cannot_re_block_on_a_clue():
    from llmflow_search.evidence import _normalize_evidence_challenge_result

    ready = _ledger([_row(2)], identifying=[0, 1])
    challenge = _normalize_evidence_challenge_result(
        {
            "blocking_gaps": [
                {"requirement_index": 1, "text": "the shaman incident is unproven"}
            ],
            "next_steps": [],
            "reason": "",
        },
        ready,
        len(CRITERIA),
        [0, 1],
    )
    assert challenge["answer_permitted"] is True
    assert challenge["unconfirmed_identifying"] == ["the shaman incident is unproven"]


def test_the_challenge_pass_still_blocks_on_a_reportable_gap():
    from llmflow_search.evidence import _normalize_evidence_challenge_result

    ready = _ledger([_row(2)], identifying=[0, 1])
    challenge = _normalize_evidence_challenge_result(
        {
            "blocking_gaps": [
                {"requirement_index": 2, "text": "no source names the publication"}
            ],
            "next_steps": [],
            "reason": "",
        },
        ready,
        len(CRITERIA),
        [0, 1],
    )
    assert challenge["answer_permitted"] is False
