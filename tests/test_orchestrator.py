import ast
import asyncio

import pytest

from deep_research.config import AppConfig, OutputConfig, RuntimeConfig
from deep_research.models import (
    ClaimReview,
    EvidenceItem,
    ExtractionResult,
    FactCheckResult,
    GapAssessment,
    ResearchPlan,
    SearchResult,
    Source,
    VerificationStatus,
)
from deep_research.orchestrator import ResearchOrchestrator, ResearchQualityError
from deep_research.store import EvidenceStore
from deep_research.tools import SearchProvider


class FakeSearch(SearchProvider):
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [SearchResult(title=f"Evidence: {query}", url="https://example.org/evidence", rank=1)]


class FakeFetcher:
    async def fetch(self, result: SearchResult) -> Source:
        text = "The project supports local execution through a documented HTTP API for all users."
        return Source(
            source_id="src_official",
            url=result.url,
            canonical_url=result.url,
            title=result.title,
            content_hash="sha256:evidence",
            quality_score=0.9,
            source_type="official_documentation",
            text=text,
        )


class PipelineLLM:
    def __init__(self, plan: dict, extraction: dict) -> None:
        self.plan = plan
        self.extraction = extraction

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        if schema is GapAssessment:
            return GapAssessment()
        if schema is ResearchPlan:
            return ResearchPlan.model_validate(self.plan)
        if schema is ExtractionResult:
            return ExtractionResult.model_validate(self.extraction)
        if schema is FactCheckResult:
            groups = ast.literal_eval(user.removeprefix("Evidence groups: "))
            return FactCheckResult(
                reviews=[
                    ClaimReview(
                        claim=group["claim"],
                        evidence_ids=[item["evidence_id"] for item in group["evidence"]],
                        status=VerificationStatus.VERIFIED,
                        confidence=0.95,
                    )
                    for group in groups
                ]
            )
        raise AssertionError(f"Unexpected schema: {schema}")

    async def complete(self, *, model: str, system: str, user: str) -> str:
        return "# Answer\n\nThe project supports local execution. <cite src_official>"


@pytest.mark.asyncio
async def test_runs_evidence_pipeline_without_network(tmp_path) -> None:
    plan = {
        "research_goal": "Does it support local execution?",
        "tasks": [
            {
                "task_id": "task_local",
                "objective": "Find support statement",
                    "questions": ["Does it support local execution?"],
                    "search_queries": ["project local execution"],
                    "preferred_source_types": ["official_documentation"],
                "max_searches": 1,
                "max_pages": 1,
            }
        ],
        "success_criteria": ["Find direct evidence"],
    }
    extraction = {
        "items": [
            {
                "claim": "The project supports local execution through an HTTP API.",
                "quote": "The project supports local execution through a documented HTTP API for all users.",
                "support_type": "supports",
                "relevance": 0.9,
            }
        ],
        "open_questions": [],
    }
    llm = PipelineLLM(plan, extraction)
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        config = AppConfig(runtime=RuntimeConfig(max_research_workers=1))
        orchestrator = ResearchOrchestrator(
            config=config,
            llm=llm,
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("Does it support local execution?")
        report = store.get_report(research_id)

        assert store.get_run(research_id).status == "completed"
        assert "supports local execution" in report
        assert "https://example.org/evidence" in report
    finally:
        store.close()


@pytest.mark.asyncio
async def test_run_fails_when_no_evidence_is_extracted(tmp_path) -> None:
    plan = {
        "research_goal": "Find evidence",
        "tasks": [
            {
                "task_id": "task_empty",
                "objective": "Find evidence",
                "questions": ["Is there evidence?"],
                "search_queries": ["evidence"],
                "preferred_source_types": ["web_page"],
                "max_searches": 1,
                "max_pages": 1,
            }
        ],
        "success_criteria": ["Find direct evidence"],
    }
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1)),
            llm=PipelineLLM(plan, {"items": [], "open_questions": ["No direct support found"]}),
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )

        with pytest.raises(ResearchQualityError, match="no source-backed evidence"):
            await orchestrator.run("Find evidence")

        row = store.connection.execute("SELECT status, error, report_markdown FROM research_runs").fetchone()
        assert row["status"] == "failed"
        assert "no source-backed evidence" in row["error"]
        assert row["report_markdown"] is None
    finally:
        store.close()


class ResumeLLM:
    """Fact-checks whatever groups it is handed and records them, then writes a report."""

    def __init__(self) -> None:
        self.checked_evidence_ids: list[str] = []

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        if schema is FactCheckResult:
            groups = ast.literal_eval(user.removeprefix("Evidence groups: "))
            for group in groups:
                self.checked_evidence_ids.extend(item["evidence_id"] for item in group["evidence"])
            return FactCheckResult(
                reviews=[
                    ClaimReview(
                        claim=group["claim"],
                        evidence_ids=[item["evidence_id"] for item in group["evidence"]],
                        status=VerificationStatus.VERIFIED,
                        confidence=0.95,
                    )
                    for group in groups
                ]
            )
        raise AssertionError(f"Unexpected schema: {schema}")

    async def complete(self, *, model: str, system: str, user: str) -> str:
        return "# Answer\n\nThe project supports local execution. <cite src_official>"


@pytest.mark.asyncio
async def test_resume_skips_already_reviewed_evidence_and_completes(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "research.sqlite3")
    research_id = "res_resume"
    try:
        store.create_run(research_id, "Does it support local execution?", "2026-07-19T00:00:00+00:00", {})
        store.save_source(
            Source(
                source_id="src_official",
                url="https://example.org/evidence",
                canonical_url="https://example.org/evidence",
                title="Evidence",
                content_hash="sha256:evidence",
                quality_score=0.9,
                source_type="official_documentation",
                text="The project supports local execution.",
            )
        )

        def evidence(evidence_id: str, claim: str, status: str) -> EvidenceItem:
            return EvidenceItem(
                evidence_id=evidence_id,
                research_id=research_id,
                task_id="task_local",
                claim=claim,
                quote="The project supports local execution.",
                source_id="src_official",
                relevance=0.9,
                source_quality=0.9,
                support_type="supports",
                verification_status=status,
            )

        # ev_done is a checkpoint persisted before the crash; ev_todo never got reviewed.
        store.save_evidence(evidence("ev_done", "It runs locally.", "verified"))
        store.save_evidence(evidence("ev_todo", "It exposes an HTTP API.", "pending"))

        llm = ResumeLLM()
        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1)),
            llm=llm,
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )
        returned = await orchestrator.resume(research_id)

        assert returned == research_id
        assert store.get_run(research_id).status == "completed"
        # The already-verified checkpoint was not re-sent to the fact checker.
        assert llm.checked_evidence_ids == ["ev_todo"]
        assert store.get_report(research_id)
    finally:
        store.close()


class GapSearch(SearchProvider):
    """Returns a URL derived from the query so each round discovers a new page."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        slug = "gap" if "investment" in query else "initial"
        return [SearchResult(title=f"Evidence {slug}: {query}", url=f"https://example.org/{slug}", rank=1)]


class GapFetcher:
    async def fetch(self, result: SearchResult) -> Source:
        slug = result.url.rsplit("/", 1)[-1]
        text = f"The {slug} page states the programme invests 200 billion euro in AI."
        return Source(
            source_id=f"src_{slug}",
            url=result.url,
            canonical_url=result.url,
            title=result.title,
            content_hash=f"sha256:{slug}",
            quality_score=0.9,
            source_type="official_documentation",
            text=text,
        )


class GapLLM:
    """A coverage gap (from the fact checker or the audit) triggers a follow-up round."""

    def __init__(self, gap_from_audit: bool = False) -> None:
        self.gap_from_audit = gap_from_audit
        self.plan_calls = 0
        self.fact_calls = 0
        self.audit_calls = 0
        self.followup_user = ""

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        if schema is GapAssessment:
            self.audit_calls += 1
            if self.gap_from_audit and self.audit_calls == 1:
                return GapAssessment(unanswered_questions=["No investment figures were found"])
            return GapAssessment()
        if schema is ResearchPlan:
            self.plan_calls += 1
            if self.plan_calls > 1:
                self.followup_user = user
            slug = "initial" if self.plan_calls == 1 else "gap"
            return ResearchPlan.model_validate(
                {
                    "research_goal": "goal",
                    "tasks": [
                        {
                            "task_id": f"task_{slug}",
                            "objective": f"Find {slug} facts",
                            "questions": ["What are the facts?"],
                            "search_queries": ["investment figures" if slug == "gap" else "initial facts"],
                            "preferred_source_types": ["official_documentation"],
                            "max_searches": 1,
                            "max_pages": 1,
                        }
                    ],
                    "success_criteria": ["facts found"],
                }
            )
        if schema is ExtractionResult:
            slug = "gap" if "example.org/gap" in user else "initial"
            return ExtractionResult.model_validate(
                {
                    "items": [
                        {
                            "claim": f"The {slug} claim about the programme.",
                            "quote": f"The {slug} page states the programme invests 200 billion euro in AI.",
                            "support_type": "supports",
                            "relevance": 0.9,
                        }
                    ],
                    "open_questions": [],
                }
            )
        if schema is FactCheckResult:
            self.fact_calls += 1
            groups = ast.literal_eval(user.removeprefix("Evidence groups: "))
            return FactCheckResult(
                reviews=[
                    ClaimReview(
                        claim=group["claim"],
                        evidence_ids=[item["evidence_id"] for item in group["evidence"]],
                        status=VerificationStatus.VERIFIED,
                        confidence=0.95,
                    )
                    for group in groups
                ],
                coverage_gaps=(
                    ["No investment figures were found"]
                    if self.fact_calls == 1 and not self.gap_from_audit
                    else []
                ),
            )
        raise AssertionError(f"Unexpected schema: {schema}")

    async def complete(self, *, model: str, system: str, user: str) -> str:
        if self.plan_calls > 1:
            return "# Answer\n\nThe programme invests. <cite src_initial> <cite src_gap>"
        return "# Answer\n\nThe programme invests. <cite src_initial>"


@pytest.mark.asyncio
async def test_coverage_gaps_trigger_followup_research_round(tmp_path) -> None:
    llm = GapLLM()
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1, max_research_rounds=2)),
            llm=llm,
            search=GapSearch(),
            fetcher=GapFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("How much does the programme invest?")

        assert store.get_run(research_id).status == "completed"
        # A follow-up plan was requested with the gap text, and its evidence was verified.
        assert llm.plan_calls == 2
        assert "No investment figures were found" in llm.followup_user
        assert llm.fact_calls >= 2
        evidence = store.list_evidence(research_id)
        assert {item.task_id for item in evidence} == {"task_initial", "task_gap"}
        assert all(item.verification_status == "verified" for item in evidence)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_audit_identified_gap_triggers_followup_round(tmp_path) -> None:
    llm = GapLLM(gap_from_audit=True)
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1, max_research_rounds=2)),
            llm=llm,
            search=GapSearch(),
            fetcher=GapFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("How much does the programme invest?")

        assert store.get_run(research_id).status == "completed"
        assert llm.plan_calls == 2
        assert "No investment figures were found" in llm.followup_user
    finally:
        store.close()


@pytest.mark.asyncio
async def test_followup_round_is_skipped_when_rounds_budget_is_one(tmp_path) -> None:
    llm = GapLLM()
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1, max_research_rounds=1)),
            llm=llm,
            search=GapSearch(),
            fetcher=GapFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("How much does the programme invest?")

        assert store.get_run(research_id).status == "completed"
        assert llm.plan_calls == 1
    finally:
        store.close()


class RewriteLLM(PipelineLLM):
    """First draft has no citation tags; the corrective rewrite fixes it."""

    def __init__(self, plan: dict, extraction: dict) -> None:
        super().__init__(plan, extraction)
        self.write_calls = 0
        self.rewrite_user = ""

    async def complete(self, *, model: str, system: str, user: str) -> str:
        self.write_calls += 1
        if self.write_calls == 1:
            return "# Answer\n\nA fact with no citation tag."
        self.rewrite_user = user
        return "# Answer\n\nThe project supports local execution. <cite src_official>"


@pytest.mark.asyncio
async def test_uncited_draft_triggers_one_corrective_rewrite(tmp_path) -> None:
    plan = {
        "research_goal": "Does it support local execution?",
        "tasks": [
            {
                "task_id": "task_local",
                "objective": "Find support statement",
                "questions": ["Does it support local execution?"],
                "search_queries": ["project local execution"],
                "preferred_source_types": ["official_documentation"],
                "max_searches": 1,
                "max_pages": 1,
            }
        ],
        "success_criteria": ["Find direct evidence"],
    }
    extraction = {
        "items": [
            {
                "claim": "The project supports local execution through an HTTP API.",
                "quote": "The project supports local execution through a documented HTTP API for all users.",
                "support_type": "supports",
                "relevance": 0.9,
            }
        ],
        "open_questions": [],
    }
    llm = RewriteLLM(plan, extraction)
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1)),
            llm=llm,
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("Does it support local execution?")

        assert store.get_run(research_id).status == "completed"
        assert llm.write_calls == 2
        assert "was rejected" in llm.rewrite_user
        assert "https://example.org/evidence" in store.get_report(research_id)
    finally:
        store.close()


def test_report_admission_requires_quality_or_independent_corroboration() -> None:
    def evidence(source_id: str, quality: float) -> EvidenceItem:
        return EvidenceItem(
            research_id="res_test",
            task_id="task_test",
            claim="The model was released.",
            quote="The model was released today.",
            source_id=source_id,
            relevance=0.9,
            source_quality=quality,
            support_type="supports",
            verification_status="verified",
        )

    weak_single = evidence("src_weak", 0.55)
    trusted = evidence("src_trusted", 0.82)
    corroborated = [evidence("src_one", 0.55), evidence("src_two", 0.55)]

    assert ResearchOrchestrator._admit_verified_evidence([weak_single]) == []
    assert ResearchOrchestrator._admit_verified_evidence([trusted]) == [trusted]
    assert ResearchOrchestrator._admit_verified_evidence(corroborated) == corroborated


class _GapAuditCapture:
    """Records the prompt sent for the coverage-gap audit and returns no gaps."""

    def __init__(self) -> None:
        self.captured_user = ""

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        if schema is GapAssessment:
            self.captured_user = user
            return GapAssessment()
        raise AssertionError(f"Unexpected schema: {schema}")

    async def complete(self, *, model: str, system: str, user: str) -> str:
        raise AssertionError("not used")


def test_identify_gaps_prioritizes_most_recently_gathered_evidence(tmp_path) -> None:
    """A gap-closing round's whole point is evidence the audit hasn't seen yet.

    Regression test: _identify_gaps used to head-slice the first 40 claims, so any
    evidence a follow-up round added was permanently invisible and the same "gap"
    would be reported forever. It must now see the newest claims instead.
    """
    llm = _GapAuditCapture()
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        research_id = "res_gap_audit"
        store.create_run(research_id, "How much did X invest?", "2026-07-19T00:00:00+00:00", {})
        store.save_source(
            Source(
                source_id="src_x",
                url="https://example.org/x",
                canonical_url="https://example.org/x",
                title="X",
                content_hash="sha256:x",
                quality_score=0.9,
                text="text",
            )
        )
        # 160 old claims (round 1), saved first, then one new claim (a gap-round finding).
        for index in range(160):
            store.save_evidence(
                EvidenceItem(
                    research_id=research_id,
                    task_id="task_old",
                    claim=f"old claim number {index}",
                    quote="q",
                    source_id="src_x",
                    relevance=0.9,
                    source_quality=0.9,
                    support_type="supports",
                    verification_status="verified",
                )
            )
        store.save_evidence(
            EvidenceItem(
                research_id=research_id,
                task_id="task_gap",
                claim="X invested 2 billion dollars in compute",
                quote="q",
                source_id="src_x",
                relevance=0.9,
                source_quality=0.9,
                support_type="supports",
                verification_status="verified",
            )
        )

        orchestrator = ResearchOrchestrator(
            config=AppConfig(runtime=RuntimeConfig(max_research_workers=1)),
            llm=llm,
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )
        asyncio.run(orchestrator._identify_gaps(research_id, "How much did X invest?"))

        assert "2 billion dollars" in llm.captured_user
        assert "old claim number 0" not in llm.captured_user
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_completed_run_writes_a_pdf_report_into_the_configured_directory(tmp_path) -> None:
    plan = {
        "research_goal": "Does it support local execution?",
        "tasks": [
            {
                "task_id": "task_local",
                "objective": "Find support statement",
                "questions": ["Does it support local execution?"],
                "search_queries": ["project local execution"],
                "preferred_source_types": ["official_documentation"],
                "max_searches": 1,
                "max_pages": 1,
            }
        ],
        "success_criteria": ["Find direct evidence"],
    }
    extraction = {
        "items": [
            {
                "claim": "The project supports local execution through an HTTP API.",
                "quote": "The project supports local execution through a documented HTTP API for all users.",
                "support_type": "supports",
                "relevance": 0.9,
            }
        ],
        "open_questions": [],
    }
    store = EvidenceStore(tmp_path / "research.sqlite3")
    reports_dir = tmp_path / "pdf_out"
    try:
        config = AppConfig(
            runtime=RuntimeConfig(max_research_workers=1),
            output=OutputConfig(reports_dir=str(reports_dir), logo_path=""),
        )
        orchestrator = ResearchOrchestrator(
            config=config,
            llm=PipelineLLM(plan, extraction),
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("Does it support local execution?")

        pdf_files = list(reports_dir.glob("*.pdf"))
        assert len(pdf_files) == 1
        assert research_id in pdf_files[0].name
        assert pdf_files[0].read_bytes().startswith(b"%PDF-")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_pdf_export_failure_does_not_fail_an_otherwise_successful_run(tmp_path) -> None:
    """PDF export is best-effort — a broken reports_dir must not sink the research run."""
    plan = {
        "research_goal": "Does it support local execution?",
        "tasks": [
            {
                "task_id": "task_local",
                "objective": "Find support statement",
                "questions": ["Does it support local execution?"],
                "search_queries": ["project local execution"],
                "preferred_source_types": ["official_documentation"],
                "max_searches": 1,
                "max_pages": 1,
            }
        ],
        "success_criteria": ["Find direct evidence"],
    }
    extraction = {
        "items": [
            {
                "claim": "The project supports local execution through an HTTP API.",
                "quote": "The project supports local execution through a documented HTTP API for all users.",
                "support_type": "supports",
                "relevance": 0.9,
            }
        ],
        "open_questions": [],
    }
    store = EvidenceStore(tmp_path / "research.sqlite3")
    # A file where a directory is expected makes report_dir creation fail.
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked")
    try:
        config = AppConfig(
            runtime=RuntimeConfig(max_research_workers=1),
            output=OutputConfig(reports_dir=str(blocking_file / "reports"), logo_path=""),
        )
        orchestrator = ResearchOrchestrator(
            config=config,
            llm=PipelineLLM(plan, extraction),
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            store=store,
        )
        research_id = await orchestrator.run("Does it support local execution?")

        assert store.get_run(research_id).status == "completed"
    finally:
        store.close()
