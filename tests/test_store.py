from deep_research.models import EvidenceItem, ResearchStatus, Source
from deep_research.store import EvidenceStore


def test_store_round_trip(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        store.create_run("res_one", "Question", "2026-07-16T00:00:00+00:00", {})
        store.update_status("res_one", ResearchStatus.RESEARCHING, "2026-07-16T00:00:01+00:00")
        source = Source(
            source_id="src_one",
            url="https://example.org/a",
            canonical_url="https://example.org/a",
            title="A",
            content_hash="sha256:a",
            quality_score=0.9,
        )
        store.save_source(source)
        evidence = EvidenceItem(
            evidence_id="ev_one",
            research_id="res_one",
            task_id="task_one",
            source_id="src_one",
            claim="A supported claim",
            quote="A direct supporting quote.",
            relevance=0.9,
            source_quality=0.9,
            support_type="supports",
        )
        store.save_evidence(evidence)
        store.apply_reviews("res_one", {"ev_one": "verified"})

        assert store.get_run("res_one").status == ResearchStatus.RESEARCHING
        assert store.list_evidence("res_one")[0].verification_status == "verified"
        assert store.list_sources({"src_one"})["src_one"].title == "A"
        assert store.list_sources(set()) == {}
        assert store.list_sources()["src_one"].title == "A"
    finally:
        store.close()


def test_latest_resumable_picks_interrupted_run_with_evidence(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        assert store.latest_resumable() is None

        # Completed run with evidence — not resumable.
        store.create_run("res_done", "q1", "2026-07-19T00:00:00+00:00", {})
        store.save_source(
            Source(
                source_id="src_a",
                url="https://example.org/a",
                canonical_url="https://example.org/a",
                title="A",
                content_hash="sha256:a",
                quality_score=0.9,
            )
        )
        store.save_evidence(
            EvidenceItem(
                evidence_id="ev_a",
                research_id="res_done",
                task_id="t",
                source_id="src_a",
                claim="c",
                quote="q",
                relevance=0.9,
                source_quality=0.9,
                support_type="supports",
            )
        )
        store.update_status("res_done", ResearchStatus.COMPLETED, "2026-07-19T00:05:00+00:00")

        # Failed run without evidence — nothing to resume from.
        store.create_run("res_empty", "q2", "2026-07-19T00:10:00+00:00", {})
        store.fail_run("res_empty", "boom", "2026-07-19T00:11:00+00:00")
        assert store.latest_resumable() is None

        # Failed run with evidence — the one to resume.
        store.create_run("res_pick", "q3", "2026-07-19T00:20:00+00:00", {})
        store.save_evidence(
            EvidenceItem(
                evidence_id="ev_b",
                research_id="res_pick",
                task_id="t",
                source_id="src_a",
                claim="c",
                quote="q",
                relevance=0.9,
                source_quality=0.9,
                support_type="supports",
            )
        )
        store.fail_run("res_pick", "timeout", "2026-07-19T00:21:00+00:00")

        candidate = store.latest_resumable()
        assert candidate is not None
        assert candidate.research_id == "res_pick"
    finally:
        store.close()
