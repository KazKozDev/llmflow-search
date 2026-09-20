"""The deterministic gate on uncited claim sentences.

The repair spends a model call, so what matters is when it is skipped, and — more
importantly — when its result is thrown away. A repair that returns a stub has deleted the
answer rather than cited it, and shipping that would trade a measurable metric for the
thing the metric is supposed to stand for.
"""

import pytest

from llmflow_search import answering, llm
from llmflow_search.profiles import FOOTNOTE_PROFILE

SOURCES = "[1]\nTitle: A\nURL: https://a.test\nContent:\nThe release shipped in October.\n"


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the transport with a scripted reply, and record what it was asked."""
    calls = []

    def install(reply: str):
        def _chat(model, messages, tools=None, system="", **kwargs):
            calls.append({"system": system, "prompt": messages[0]["content"]})
            return {"role": "assistant", "content": reply}

        monkeypatch.setattr(llm, "_ollama_chat", _chat)
        return calls

    return install


def test_coverage_counts_claim_sentences_only():
    coverage, uncited = answering._citation_coverage(
        "## Findings\n\nIt shipped in October [1]. The next release is unscheduled.\n"
    )
    assert coverage == 0.5
    assert uncited == ["The next release is unscheduled."]


def test_a_fully_cited_answer_is_not_sent_for_repair(fake_model):
    calls = fake_model("should never be used")
    answer = "It shipped in October [1]. The build is experimental [1]."
    assert (
        answering._repair_citations(answer, SOURCES, "m", FOOTNOTE_PROFILE) == answer
    )
    assert calls == []


def test_uncited_sentences_are_named_in_the_repair_prompt(fake_model):
    calls = fake_model(
        "It shipped in October [1]. The next release is unscheduled [1]."
    )
    answering._repair_citations(
        "It shipped in October [1]. The next release is unscheduled.",
        SOURCES,
        "m",
        FOOTNOTE_PROFILE,
    )
    assert "The next release is unscheduled." in calls[0]["prompt"]


def test_a_repair_that_improves_coverage_is_taken(fake_model):
    fake_model("It shipped in October [1]. The next release is unscheduled [1].")
    repaired = answering._repair_citations(
        "It shipped in October [1]. The next release is unscheduled.",
        SOURCES,
        "m",
        FOOTNOTE_PROFILE,
    )
    assert repaired.endswith("unscheduled [1].")


def test_a_repair_that_guts_the_answer_is_discarded(fake_model):
    """Deleting everything scores perfectly on coverage and answers nothing."""
    original = (
        "It shipped in October [1]. The next release is unscheduled. "
        "Maintainers have not commented. The changelog lists nine entries."
    )
    fake_model("It shipped in October [1].")
    assert (
        answering._repair_citations(original, SOURCES, "m", FOOTNOTE_PROFILE)
        == original
    )


def test_a_repair_that_does_not_improve_coverage_is_discarded(fake_model):
    original = "It shipped in October [1]. The next release is unscheduled."
    fake_model("It shipped in October [1]. The next release is still unscheduled.")
    assert (
        answering._repair_citations(original, SOURCES, "m", FOOTNOTE_PROFILE)
        == original
    )


def test_an_empty_reply_leaves_the_verified_answer_alone(fake_model):
    original = "It shipped in October [1]. The next release is unscheduled."
    fake_model("")
    assert (
        answering._repair_citations(original, SOURCES, "m", FOOTNOTE_PROFILE)
        == original
    )


def test_the_repair_can_be_turned_off(monkeypatch, fake_model):
    calls = fake_model("anything")
    monkeypatch.setattr(answering, "CITATION_REPAIR", False)
    original = "It shipped in October [1]. The next release is unscheduled."
    assert (
        answering._repair_citations(original, SOURCES, "m", FOOTNOTE_PROFILE)
        == original
    )
    assert calls == []
