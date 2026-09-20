"""The deterministic gate on report structure.

Two things are worth testing here and they pull in opposite directions. The detector has
to fire on the defects the editorial contract names, and it has to stay silent on prose
that merely resembles them — a false defect costs a model call and invites a rewrite of a
report that was fine. The repair, like the citation repair beside it, is judged by what it
refuses: a result that loses defects by deleting the report has not repaired anything.
"""

import pytest

from llmflow_search import answering, llm, report_structure
from llmflow_search.profiles import FOOTNOTE_PROFILE

SOURCES = (
    "[1]\nTitle: A\nURL: https://a.test\nPublished: 2026-02-01\n"
    "Content:\nLatency fell to 40 ms from 90 ms.\n"
)


@pytest.fixture
def fake_model(monkeypatch):
    calls = []

    def install(reply: str):
        def _chat(model, messages, tools=None, system="", **kwargs):
            calls.append({"system": system, "prompt": messages[0]["content"]})
            return {"role": "assistant", "content": reply}

        monkeypatch.setattr(llm, "_ollama_chat", _chat)
        return calls

    return install


# --- detection -----------------------------------------------------------------


def test_heading_with_no_body_is_a_defect():
    defects = report_structure.orphan_headings(
        "## Pricing\n\n## Availability\n\nShipped in the EU on 1 March [1].\n"
    )
    assert [d.kind for d in defects] == ["orphan_heading"]
    assert defects[0].text == "Pricing"


def test_last_heading_with_no_body_is_a_defect():
    defects = report_structure.orphan_headings("## Findings\n\nBody [1].\n\n## Outlook\n")
    assert [d.text for d in defects] == ["Outlook"]


def test_a_heading_followed_by_its_body_is_not_a_defect():
    assert (
        report_structure.orphan_headings("## Findings\n\nLatency fell to 40 ms [1].\n")
        == []
    )


def test_comparative_without_a_figure_is_a_defect():
    defects = report_structure.numberless_comparatives(
        "Latency is significantly lower than the prior release [1]."
    )
    assert [d.kind for d in defects] == ["numberless_comparative"]
    assert "significantly" in defects[0].detail


def test_a_citation_marker_is_not_the_sentence_figure():
    """`[1]` contains a digit. Counting it would clear every comparative in a cited report."""
    assert report_structure.numberless_comparatives("Costs are far lower this year [1].")


def test_a_comparative_with_its_number_passes():
    assert (
        report_structure.numberless_comparatives(
            "Latency is lower than 90 ms, at 40 ms [1]."
        )
        == []
    )


def test_a_quantity_written_in_words_counts_as_a_figure():
    assert (
        report_structure.numberless_comparatives(
            "Throughput more than doubled after the change [1]."
        )
        == []
    )


def test_a_plain_report_of_an_event_is_not_a_magnitude_claim():
    """"Increased" and "reduced" are left out of the word list on purpose."""
    assert (
        report_structure.numberless_comparatives(
            "The vendor reduced its published list price in March [1]."
        )
        == []
    )


def test_the_same_point_twice_is_a_defect():
    defects = report_structure.duplicate_blocks(
        "Costs fell by 12 percent year over year [1].\n"
        "Costs fell by 12 percent year over year [1].\n"
    )
    assert [d.kind for d in defects] == ["duplicate_block"]


def test_two_findings_differing_by_their_figure_are_not_duplicates():
    assert (
        report_structure.duplicate_blocks(
            "Latency in the EU region fell to 40 ms in March [1].\n"
            "Latency in the US region fell to 71 ms in March [1].\n"
        )
        == []
    )


def test_the_bibliography_is_not_searched_for_defects():
    report = (
        "## Findings\n\nLatency fell to 40 ms [1].\n\n"
        "## Sources\n\n1. [A](https://a.test) - https://a.test\n"
    )
    assert report_structure.find_defects(report) == []


def test_a_refusal_has_no_structure_to_repair():
    from llmflow_search.config import INSUFFICIENT_EVIDENCE_MESSAGE

    assert report_structure.find_defects(INSUFFICIENT_EVIDENCE_MESSAGE) == []


# --- repair --------------------------------------------------------------------

CLEAN = "## Findings\n\nLatency fell to 40 ms from 90 ms [1].\n"
DEFECTIVE = "## Pricing\n\n## Findings\n\nLatency is significantly lower now [1].\n"


def test_a_clean_report_is_not_sent_for_repair(fake_model):
    calls = fake_model("should never be used")
    answer, defects = answering._repair_structure(
        CLEAN, SOURCES, "m", FOOTNOTE_PROFILE
    )
    assert answer == CLEAN
    assert defects == []
    assert calls == []


def test_defects_are_quoted_back_in_the_repair_prompt(fake_model):
    calls = fake_model(CLEAN)
    answering._repair_structure(DEFECTIVE, SOURCES, "m", FOOTNOTE_PROFILE)
    prompt = calls[0]["prompt"]
    assert "orphan_heading" in prompt
    assert "numberless_comparative" in prompt
    assert "Latency is significantly lower now [1]." in prompt


def test_a_repair_that_fixes_the_defects_is_accepted(fake_model):
    fixed = "## Findings\n\nLatency fell to 40 ms from 90 ms [1]."
    fake_model(fixed)
    answer, defects = answering._repair_structure(
        DEFECTIVE, SOURCES, "m", FOOTNOTE_PROFILE
    )
    assert answer == fixed
    assert defects == []


def test_a_repair_that_guts_the_report_is_rejected(fake_model):
    """Losing every defect by deleting the body is not a repair."""
    fake_model("## Findings\n\nOK [1].")
    answer, _ = answering._repair_structure(
        DEFECTIVE * 3, SOURCES, "m", FOOTNOTE_PROFILE
    )
    assert answer == DEFECTIVE * 3


def test_a_repair_that_loses_citation_coverage_is_rejected(fake_model):
    """A rewrite that drops a marker to lose a defect trades attribution for tidiness."""
    fake_model("## Findings\n\nLatency fell to 40 ms from 90 ms yesterday everywhere.\n")
    answer, _ = answering._repair_structure(
        DEFECTIVE, SOURCES, "m", FOOTNOTE_PROFILE
    )
    assert answer == DEFECTIVE


def test_a_repair_that_returns_a_refusal_is_rejected(fake_model):
    from llmflow_search.config import INSUFFICIENT_EVIDENCE_MESSAGE

    fake_model(INSUFFICIENT_EVIDENCE_MESSAGE)
    answer, _ = answering._repair_structure(
        DEFECTIVE, SOURCES, "m", FOOTNOTE_PROFILE
    )
    assert answer == DEFECTIVE


def test_the_repair_can_be_switched_off(fake_model, monkeypatch):
    monkeypatch.setattr(answering, "REPORT_STRUCTURE_REPAIR", False)
    calls = fake_model("should never be used")
    answer, defects = answering._repair_structure(
        DEFECTIVE, SOURCES, "m", FOOTNOTE_PROFILE
    )
    assert answer == DEFECTIVE
    assert calls == []
    # Switched off means unrepaired, not unmeasured: the defects are still reported.
    assert sorted(set(defects)) == ["numberless_comparative", "orphan_heading"]
    # and not a second entry for the same sentence being nameless as well


# --- source tier ---------------------------------------------------------------

LAB = {"url": "https://lab.test/post", "source_quality": {"source_type": "primary"}}
OUTLET = {"url": "https://news.test/a", "source_quality": {"source_type": "aggregator"}}
UNTYPED = {"url": "https://x.test/a"}


def test_a_body_claim_resting_only_on_coverage_is_a_defect():
    defects = report_structure.body_claims_on_weak_sources(
        "The lab released the model on 4 March [2].", [LAB, OUTLET]
    )
    assert [d.kind for d in defects] == ["weak_source_in_body"]
    assert "aggregator" in defects[0].detail


def test_the_same_claim_cited_to_the_lab_page_is_fine():
    assert (
        report_structure.body_claims_on_weak_sources(
            "The lab released the model on 4 March [1].", [LAB, OUTLET]
        )
        == []
    )


def test_one_primary_citation_among_several_is_enough():
    assert (
        report_structure.body_claims_on_weak_sources(
            "The lab released the model on 4 March [1][2].", [LAB, OUTLET]
        )
        == []
    )


def test_an_untyped_source_never_makes_a_claim_a_defect():
    """Most backends supply no type. Silence must not be read as weakness."""
    assert (
        report_structure.body_claims_on_weak_sources(
            "The lab released the model on 4 March [1].", [UNTYPED]
        )
        == []
    )
    assert (
        report_structure.body_claims_on_weak_sources(
            "A claim with no source list at all [1].", None
        )
        == []
    )


def test_moving_coverage_to_the_appendix_does_not_clear_it():
    """The appendix was an escape hatch: a per-item check that a move satisfies is not a
    check. Relocation stopped being a repair, so the appendix is inspected too."""
    report = (
        "## Findings\n\nThe lab released the model on 4 March [1].\n\n"
        "## Appendix\n\nAn outlet reported a March release date [2].\n"
    )
    assert report_structure.body_claims_on_weak_sources(report, [LAB, OUTLET])


def test_an_uncited_sentence_is_not_this_defect():
    """The citation pass owns uncited sentences; flagging them twice helps nobody."""
    assert (
        report_structure.body_claims_on_weak_sources(
            "The lab released the model in March.", [LAB, OUTLET]
        )
        == []
    )


def test_source_numbering_follows_the_answer_prompt():
    types = report_structure.source_types([LAB, OUTLET, UNTYPED])
    assert types[1] == "primary"
    assert types[2] == "aggregator"
    assert 3 not in types


# --- what a thin report must not be flattened into --------------------------------

NEWS = {"url": "https://news.test", "source_quality": {"source_type": "secondary"}}
FARM = {"url": "https://farm.test", "source_quality": {"source_type": "news aggregator"}}


def test_a_match_report_is_not_a_weak_source():
    """In sport and culture the outlet is the reporting institution; there is no lab page.

    An earlier version of this list counted "secondary" as weak and moved whole sections
    into the appendix, leaving the body a set of empty headings.
    """
    assert (
        report_structure.body_claims_on_weak_sources(
            "The home side won the final on 12 May [1].", [NEWS]
        )
        == []
    )


def test_a_companys_own_blog_is_its_own_page():
    vendor = {"url": "https://vendor.test", "source_quality": {"source_type": "blog"}}
    assert (
        report_structure.body_claims_on_weak_sources(
            "The company announced the price change on 4 March [1].", [vendor]
        )
        == []
    )


def test_a_restating_aggregator_is_still_weak():
    assert report_structure.body_claims_on_weak_sources(
        "A launch was reported this week [1].", [FARM]
    )


def test_items_differing_only_by_name_are_not_duplicates():
    """Once the excerpting has stripped the detail, distinct items read alike.

    Merging them would delete two of three real findings and call the report tidy.
    """
    assert (
        report_structure.duplicate_blocks(
            "Chelsea won its opening match of the season [1].\n"
            "Arsenal won its opening match of the season [1].\n"
        )
        == []
    )


def test_a_genuine_restatement_is_still_a_duplicate():
    assert report_structure.duplicate_blocks(
        "Chelsea won its opening match of the season [1].\n"
        "Chelsea won its opening match of the season [2].\n"
    )


def test_sentence_initial_capitals_are_not_names():
    """Otherwise every pair of sentences would differ by its own first word."""
    assert report_structure.duplicate_blocks(
        "The council approved the budget without amendment [1].\n"
        "The council approved the budget without amendment [1].\n"
    )


def test_a_date_is_not_a_magnitude():
    """The plain digit test cleared every comparative that happened to mention a year."""
    for sentence in (
        "Latency is far lower since March 2026 [1].",
        "Costs are significantly higher than in 2024 [1].",
        "The service is faster as of 2026-03-04 [1].",
    ):
        assert report_structure.numberless_comparatives(sentence), sentence


def test_a_figure_beside_a_date_still_counts():
    assert (
        report_structure.numberless_comparatives(
            "Latency is far lower, at 40 ms, since March 2026 [1]."
        )
        == []
    )


# --- the skeleton-and-dump failure ------------------------------------------------


def test_a_finding_with_no_name_and_no_figure_is_not_a_finding():
    """"A Brazilian forward scored twice" is the shape of a finding with the fact gone.

    It passes every other check — grounded, cited, unique — and tells a reader nothing
    they can check or follow up.
    """
    defects = report_structure.nameless_body_findings(
        "An Argentine conductor led the orchestra on the closing night [1]."
    )
    assert [d.kind for d in defects] == ["nameless_body_finding"]


def test_a_named_finding_passes():
    assert (
        report_structure.nameless_body_findings(
            "Barenboim led the orchestra on the closing night [1]."
        )
        == []
    )


def test_a_figure_is_enough_on_its_own():
    """"Prices fell 12%" names nobody and is still a finding."""
    assert (
        report_structure.nameless_body_findings(
            "Consumer prices fell 12 percent over the quarter [1]."
        )
        == []
    )


def test_a_date_alone_does_not_rescue_a_nameless_sentence():
    assert report_structure.nameless_body_findings(
        "A rider took the stage win on 4 March [1]."
    )


def test_an_appendix_larger_than_the_body_is_a_defect():
    """Moving an item was the cheapest way to stop it failing, so the sorting is named."""
    report = (
        "## Findings\n\nSanchez opened the session on 12 May [1].\n\n"
        "## Appendix\n\n"
        "Cushions were reviewed by the paper [1].\n"
        "K-beauty was covered in the weekend edition [1].\n"
        "Tehran was discussed in an opinion piece [1].\n"
    )
    defects = report_structure.appendix_outweighs_body(report)
    assert [d.kind for d in defects] == ["appendix_outweighs_body"]
    assert "3 items in the appendix, 1 in the body" in defects[0].text


def test_an_appendix_supporting_a_full_body_is_fine():
    report = (
        "## Findings\n\n"
        "Sanchez opened the session on 12 May [1].\n"
        "Feijoo replied the same afternoon [1].\n"
        "The vote carried 176 to 171 [1].\n\n"
        "## Appendix\n\nFull roll call of the 176 votes [1].\n"
    )
    assert report_structure.appendix_outweighs_body(report) == []


def test_a_report_with_no_appendix_is_never_flagged_for_one():
    assert (
        report_structure.appendix_outweighs_body(
            "## Findings\n\nSanchez opened the session on 12 May [1].\n"
        )
        == []
    )


def test_one_sentence_is_named_once():
    """A nameless sentence is usually numberless too; counting it twice misstates the
    report and the repair's acceptance test counts defects."""
    defects = report_structure.find_defects(
        "## Findings\n\nLatency is significantly lower now [1].\n"
    )
    assert [d.kind for d in defects] == ["numberless_comparative"]


def test_a_nationality_is_not_a_name():
    """The reviewer's exact case: "Argentine director" where the report should say who."""
    assert report_structure.nameless_body_findings(
        "An Argentine conductor led the orchestra on the closing night [1]."
    )


def test_a_sentence_may_begin_with_its_name():
    """Grammar capitalizes the first word, so duplicate detection ignores it. Here that
    rule would read "Barenboim led the orchestra" as naming nobody."""
    assert (
        report_structure.nameless_body_findings(
            "Barenboim led the orchestra on the closing night [1]."
        )
        == []
    )


def test_nationalities_still_tell_two_items_apart():
    """Opposite requirement, same words: for duplicates a demonym is a distinguisher."""
    assert (
        report_structure.duplicate_blocks(
            "A Brazilian forward scored twice in the second half [1].\n"
            "An Argentine forward scored twice in the second half [1].\n"
        )
        == []
    )
