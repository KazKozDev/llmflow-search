"""An objection that blocked once may not vanish by losing its index.

The gate discards an objection it cannot tie to a numbered requirement, so that a
requirement the model invented cannot block an answer forever. That is right for an
objection nobody has seen. It is wrong for one that already blocked this run under a valid
index and came back without it: the requirement is real, the index is a formatting
failure, and discarding it turns a correctly blocked run into a complete one with nothing
found.

Both halves are tested here, because the fix is only safe if the original protection still
holds: a novel unindexed objection must still be discarded.
"""

from llmflow_search.evidence import (
    _normalize_evidence_challenge_result,
    _normalize_evidence_ledger_result,
    _recurring_gap,
)

CRITERIA = ["The figure is stated.", "The figure comes from a primary source."]
SOURCES = [{"url": "https://a.test", "title": "A", "content": "x"}]

PRIMARY = "No primary source confirms the 1913 production figure."
REWORDED = "The 1913 production figure is confirmed by no primary source."
UNRELATED = "The report should also cover the 1925 merger."


def _ledger(global_missing, enforced=None):
    return _normalize_evidence_ledger_result(
        {"ledger": [], "global_missing": global_missing},
        SOURCES,
        CRITERIA,
        "strict",
        {1},
        None,
        enforced,
    )


def _challenge(blocking_gaps, enforced=None, ready=True):
    return _normalize_evidence_challenge_result(
        {"blocking_gaps": blocking_gaps},
        {"answer_ready": ready},
        len(CRITERIA),
        None,
        enforced,
    )


# --- the matcher ---------------------------------------------------------------


def test_the_same_objection_reworded_is_recognised():
    assert _recurring_gap(REWORDED, [PRIMARY])


def test_an_unrelated_objection_is_not_a_recurrence():
    assert not _recurring_gap(UNRELATED, [PRIMARY])


def test_nothing_recurs_when_nothing_was_ever_enforced():
    assert not _recurring_gap(PRIMARY, [])
    assert not _recurring_gap(PRIMARY, None)


# --- the ledger gate -----------------------------------------------------------


def test_an_indexed_objection_blocks_and_is_worth_remembering():
    result = _ledger([{"requirement_index": 1, "text": PRIMARY}])
    assert result["global_missing"] == [PRIMARY]
    assert result["answer_ready"] is False
    assert result["dropped_gaps"] == []


def test_a_novel_unindexed_objection_is_still_discarded():
    """The original protection: an invented requirement cannot block an answer."""
    result = _ledger([{"requirement_index": None, "text": UNRELATED}])
    assert result["dropped_gaps"] == [UNRELATED]
    assert result["global_missing"] == []
    assert result["recovered_gaps"] == []


def test_an_objection_that_blocked_before_survives_losing_its_index():
    result = _ledger(
        [{"requirement_index": None, "text": REWORDED}], enforced=[PRIMARY]
    )
    assert result["global_missing"] == [REWORDED]
    assert result["recovered_gaps"] == [REWORDED]
    assert result["dropped_gaps"] == []


def test_recovering_the_objection_keeps_the_answer_blocked():
    """The whole point: readiness must not flip because an index went missing."""
    blocked = _ledger([{"requirement_index": 1, "text": PRIMARY}])
    then = _ledger([{"requirement_index": None, "text": REWORDED}], enforced=[PRIMARY])
    assert blocked["answer_ready"] is False
    assert then["answer_ready"] is False


def test_memory_of_one_objection_does_not_rescue_a_different_one():
    result = _ledger(
        [{"requirement_index": None, "text": UNRELATED}], enforced=[PRIMARY]
    )
    assert result["dropped_gaps"] == [UNRELATED]
    assert result["global_missing"] == []


# --- the challenge gate --------------------------------------------------------


def test_the_challenge_gate_discards_a_novel_unindexed_objection():
    result = _challenge([{"requirement_index": None, "text": UNRELATED}])
    assert result["dropped_gaps"] == [UNRELATED]
    assert result["answer_permitted"] is True


def test_the_challenge_gate_re_blocks_a_returning_objection():
    result = _challenge(
        [{"requirement_index": None, "text": REWORDED}], enforced=[PRIMARY]
    )
    assert result["blocking_gaps"] == [REWORDED]
    assert result["recovered_gaps"] == [REWORDED]
    assert result["answer_permitted"] is False


def test_a_gap_with_no_content_words_never_matches():
    """Bare punctuation or stopwords would otherwise match everything at 0/0."""
    assert not _recurring_gap("the of and", [PRIMARY])
    assert not _recurring_gap(PRIMARY, ["the of and"])
