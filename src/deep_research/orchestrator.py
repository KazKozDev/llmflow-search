from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agents import FactChecker, Planner, ResearchWorker
from .citations import CitationError, validate_and_render_citations
from .config import AppConfig
from .llm import LLM, LLMError
from .models import EvidenceItem, GapAssessment, ResearchStatus, Source, VerificationStatus, now_iso
from .pdf_report import render_report_pdf, report_pdf_filename
from .store import EvidenceStore
from .tools import Fetcher, SearchProvider


class ResearchQualityError(RuntimeError):
    """The pipeline completed an operation but did not produce usable research evidence."""


MIN_REPORT_SOURCE_QUALITY = 0.65
PENDING_STATUS = "pending"  # EvidenceItem.verification_status default before fact-checking


class ResearchOrchestrator:
    def __init__(
        self,
        *,
        config: AppConfig,
        llm: LLM,
        search: SearchProvider,
        fetcher: Fetcher,
        store: EvidenceStore,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.planner = Planner(llm, config)
        self.worker = ResearchWorker(llm, search, fetcher, store, config)
        self.fact_checker = FactChecker(llm, config)
        self.llm = llm
        self.on_event = on_event

    async def run(self, query: str) -> str:
        research_id = f"res_{uuid4().hex[:12]}"
        created_at = now_iso()
        self.store.create_run(research_id, query, created_at, asdict(self.config))
        try:
            await self._transition(research_id, ResearchStatus.PLANNING)
            plan = await self.planner.plan(query)
            self._emit(research_id, "plan_created", plan.model_dump())

            await self._transition(research_id, ResearchStatus.RESEARCHING)
            evidence = self._deduplicate(await self._execute_tasks(research_id, plan.tasks))
            if not evidence:
                raise ResearchQualityError("Research produced no source-backed evidence")

            await self._transition(research_id, ResearchStatus.CONSOLIDATING)
            await self._transition(research_id, ResearchStatus.VERIFYING)
            coverage_gaps = await self._verify(research_id, evidence)
            # The fact checker rarely volunteers gaps, so audit coverage explicitly:
            # compare the question against the verified claims and add what is missing.
            coverage_gaps = list(dict.fromkeys([*coverage_gaps, *await self._identify_gaps(research_id, query)]))
            coverage_gaps = await self._close_coverage_gaps(research_id, query, coverage_gaps)
            return await self._finalize_report(research_id, query, coverage_gaps)
        except asyncio.CancelledError:
            self._fail(research_id, "Research was cancelled", "CancelledError")
            raise
        except Exception as exc:
            self._fail(research_id, str(exc), type(exc).__name__)
            raise

    async def resume(self, research_id: str) -> str:
        """Continue an interrupted run from where fact-checking stopped.

        Evidence and any reviews persisted before the crash are already in the store, so this
        re-runs only the batches that never completed and then writes the report.
        """
        run = self.store.get_run(research_id)
        if run.status == ResearchStatus.COMPLETED:
            return research_id
        try:
            evidence = self._deduplicate(self.store.list_evidence(research_id))
            if not evidence:
                raise ResearchQualityError("Cannot resume: no source-backed evidence was stored for this run")
            await self._transition(research_id, ResearchStatus.VERIFYING)
            # Evidence starts life as the raw string "pending"; anything else was already scored.
            already_reviewed = {item.evidence_id for item in evidence if item.verification_status != PENDING_STATUS}
            self._emit(
                research_id,
                "resume_started",
                {"evidence": len(evidence), "already_reviewed": len(already_reviewed)},
            )
            coverage_gaps = await self._verify(research_id, evidence, skip_evidence_ids=already_reviewed)
            return await self._finalize_report(research_id, run.query, coverage_gaps)
        except asyncio.CancelledError:
            self._fail(research_id, "Research was cancelled", "CancelledError")
            raise
        except Exception as exc:
            self._fail(research_id, str(exc), type(exc).__name__)
            raise

    async def _execute_tasks(self, research_id: str, tasks) -> list[EvidenceItem]:
        semaphore = asyncio.Semaphore(self.config.runtime.max_research_workers)

        async def execute_task(task):
            async with semaphore:
                found = await self.worker.research(
                    research_id,
                    task,
                    lambda event_type, payload: self._emit(research_id, event_type, payload),
                )
                self._emit(
                    research_id,
                    "task_completed",
                    {"task_id": task.task_id, "evidence_count": len(found)},
                )
                return found

        groups = await asyncio.gather(*(execute_task(task) for task in tasks))
        return [item for group in groups for item in group]

    async def _identify_gaps(self, research_id: str, query: str) -> list[str]:
        """Ask the model which parts of the question the verified claims leave unanswered."""
        verified = [
            item
            for item in self.store.list_evidence(research_id)
            if item.verification_status == VerificationStatus.VERIFIED
        ]
        # Take the most recently gathered claims, not the first ones: list_evidence returns
        # insertion order, and a gap-closing round's whole point is to add evidence the
        # audit hasn't seen yet. A head-slice would permanently hide it behind round-1
        # evidence, making every later round re-report the same "gap" forever.
        claims = [item.claim for item in verified][-150:]
        system = (
            "You audit research coverage. Compare the user's question with the verified claims "
            "and list up to 3 concrete sub-questions the claims do not answer. Only list gaps "
            "that further web research could plausibly close, phrased as searchable questions "
            "with named entities. If the claims already cover the question, return an empty list."
        )
        try:
            assessment = await self.llm.complete_json(
                model=self.config.models.planner,
                system=system,
                user=f"Question: {query}\nVerified claims: {claims}",
                schema=GapAssessment,
            )
        except LLMError:
            return []
        gaps = [gap.strip() for gap in assessment.unanswered_questions if gap.strip()]
        if gaps:
            self._emit(research_id, "coverage_gaps_identified", {"gaps": gaps})
        return gaps

    async def _close_coverage_gaps(self, research_id: str, query: str, gaps: list[str]) -> list[str]:
        """Run extra research rounds that target the fact checker's coverage gaps.

        Each round plans tasks from the current gap list, gathers and verifies new evidence,
        and stops when the gaps are gone, the round budget is spent, or a round produces
        nothing new. Follow-up is best-effort: a failure here never fails the run.
        """
        gaps = [gap for gap in gaps if gap.strip()]
        completed_rounds = 1
        while gaps and completed_rounds < self.config.runtime.max_research_rounds:
            completed_rounds += 1
            try:
                plan = await self.planner.plan_gap_followup(query, gaps)
            except LLMError:
                break
            if not plan.tasks:
                break
            self._emit(
                research_id,
                "gap_followup_started",
                {"round": completed_rounds, "gaps": gaps, "tasks": len(plan.tasks)},
            )
            before_ids = {item.evidence_id for item in self.store.list_evidence(research_id)}
            await self._execute_tasks(research_id, plan.tasks)
            all_evidence = self._deduplicate(self.store.list_evidence(research_id))
            new_items = [item for item in all_evidence if item.evidence_id not in before_ids]
            if not new_items:
                self._emit(research_id, "gap_followup_exhausted", {"round": completed_rounds})
                break
            already_reviewed = {
                item.evidence_id for item in all_evidence if item.verification_status != PENDING_STATUS
            }
            await self._verify(research_id, all_evidence, skip_evidence_ids=already_reviewed)
            # Re-run the audit rather than accumulating the fact checker's rarely-populated
            # coverage_gaps: without re-checking, a gap this round closed would keep
            # triggering follow-up rounds forever (it never gets removed from the list).
            gaps = await self._identify_gaps(research_id, query)
        return gaps

    async def _verify(
        self,
        research_id: str,
        evidence: list[EvidenceItem],
        *,
        skip_evidence_ids: set[str] | None = None,
    ) -> list[str]:
        def persist(reviews) -> None:
            statuses = {
                evidence_id: review.status.value for review in reviews for evidence_id in review.evidence_ids
            }
            if statuses:
                self.store.apply_reviews(research_id, statuses)

        verification = await self.fact_checker.check(
            evidence,
            lambda event_type, payload: self._emit(research_id, event_type, payload),
            skip_evidence_ids=skip_evidence_ids,
            on_batch_reviewed=persist,
        )
        self._emit(
            research_id,
            "verification_completed",
            {"reviews": len(verification.reviews), "coverage_gaps": verification.coverage_gaps},
        )
        return verification.coverage_gaps

    async def _finalize_report(self, research_id: str, query: str, coverage_gaps: list[str]) -> str:
        await self._transition(research_id, ResearchStatus.WRITING)
        refreshed = self.store.list_evidence(research_id)
        verified = [item for item in refreshed if item.verification_status == VerificationStatus.VERIFIED]
        if not verified:
            raise ResearchQualityError("Fact checking produced no verified evidence")
        admitted = self._admit_verified_evidence(verified)
        self._emit(
            research_id,
            "report_admission",
            {"verified": len(verified), "admitted": len(admitted), "excluded": len(verified) - len(admitted)},
        )
        if not admitted:
            raise ResearchQualityError(
                "No verified evidence met the report source-quality or independent-corroboration requirements"
            )
        sources = self.store.list_sources({item.source_id for item in admitted})
        draft = await self._write_report(query, admitted, sources, coverage_gaps)

        await self._transition(research_id, ResearchStatus.VALIDATING)
        try:
            report = validate_and_render_citations(draft, sources)
        except CitationError as exc:
            # One corrective rewrite: tell the writer exactly why the draft was rejected.
            self._emit(research_id, "report_rewrite", {"reason": str(exc)})
            draft = await self._write_report(query, admitted, sources, coverage_gaps, correction=str(exc))
            report = validate_and_render_citations(draft, sources)
        self.store.save_report(research_id, report, now_iso())
        self._emit(research_id, "report_ready", {})
        self._write_pdf_report(research_id, query, report)
        await self._transition(research_id, ResearchStatus.COMPLETED)
        return research_id

    def _write_pdf_report(self, research_id: str, query: str, report_markdown: str) -> None:
        """Best-effort: a PDF export failure (e.g. missing optional font/deps) must
        never fail an otherwise-successful research run."""
        try:
            logo_path = Path(self.config.output.logo_path) if self.config.output.logo_path else None
            output_path = Path(self.config.output.reports_dir) / report_pdf_filename(research_id, query)
            render_report_pdf(
                research_id=research_id,
                query=query,
                report_markdown=report_markdown,
                output_path=output_path,
                logo_path=logo_path,
            )
            self._emit(research_id, "pdf_report_saved", {"path": str(output_path)})
        except Exception as exc:
            self._emit(research_id, "pdf_report_failed", {"error": str(exc)})

    def _fail(self, research_id: str, error: str, error_type: str) -> None:
        self._emit(research_id, "run_failed", {"error": error, "error_type": error_type})
        self.store.fail_run(research_id, error, now_iso())

    async def _transition(self, research_id: str, status: ResearchStatus) -> None:
        self.store.update_status(research_id, status, now_iso())
        self._emit(research_id, "status_changed", {"status": status.value})

    def _emit(self, research_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.store.add_event(research_id, now_iso(), event_type, payload)
        if self.on_event:
            self.on_event(event_type, payload)

    @staticmethod
    def _deduplicate(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        best_by_key: dict[tuple[str, str], EvidenceItem] = {}
        for item in evidence:
            key = (item.claim.strip().lower(), item.source_id)
            current = best_by_key.get(key)
            if current is None or item.relevance > current.relevance:
                best_by_key[key] = item
        return list(best_by_key.values())

    @staticmethod
    def _admit_verified_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        sources_by_claim: dict[str, set[str]] = {}
        for item in evidence:
            key = item.claim.strip().casefold()
            sources_by_claim.setdefault(key, set()).add(item.source_id)
        return [
            item
            for item in evidence
            if item.source_quality >= MIN_REPORT_SOURCE_QUALITY
            or len(sources_by_claim[item.claim.strip().casefold()]) >= 2
        ]

    async def _write_report(
        self,
        query: str,
        evidence: list[EvidenceItem],
        sources: dict[str, Source],
        coverage_gaps: list[str],
        correction: str | None = None,
    ) -> str:
        verified = [item for item in evidence if item.verification_status == VerificationStatus.VERIFIED]
        if not verified:
            raise ResearchQualityError("Report writer received no verified evidence")
        facts = [
            {
                "claim": item.claim,
                "quote": item.quote,
                "source_id": item.source_id,
                "source_title": sources[item.source_id].title,
            }
            for item in verified
        ]
        # A qualitative "use most of the facts" instruction under-delivers on a small
        # model — it reliably picks a convenient handful and stops. A concrete number
        # computed from the actual fact count steers it far more effectively.
        distinct_claims = len({item.claim.strip().casefold() for item in verified})
        min_citations = max(1, round(distinct_claims * 0.7))
        system = (
            "You write thorough research reports in the user's language. Use only the supplied "
            "verified facts. The verified facts list is your complete evidentiary base, not a "
            "sample to pick from. Do not stop early because the report feels 'complete': add "
            "more sections or subsections to fit in facts that do not belong in the ones you "
            "already wrote, and only omit a fact if it is genuinely redundant with one already "
            "cited. After every factual sentence add a citation tag containing the bare "
            "source_id of the supporting fact, exactly in this form: <cite src_ab12cd34ef56>. "
            "No attributes, no quotes, no closing tag. Do not write URLs, do not invent "
            "citations, and list uncertainty in 'Limitations'. Do not add your own list of "
            "sources, links, or a References/Bibliography section — a numbered reference list "
            "is generated separately from your citation tags after you finish writing. Do not "
            "insert '---' horizontal-rule separators between sections; headings alone provide "
            "the visual break."
        )
        user = (
            f"Question: {query}\nVerified facts ({distinct_claims} distinct claims): {facts}\n"
            f"Coverage gaps: {coverage_gaps}\nYour report must cite at least {min_citations} of "
            f"the distinct claims above (different source_ids or clearly different claim text "
            "count separately) — do not settle for a small convenient subset. Write Markdown."
        )
        if correction:
            user += (
                f"\n\nYour previous draft was rejected: {correction}. Every factual sentence "
                "must end with a citation tag like <cite src_ab12cd34ef56> using a source_id "
                "that appears in the facts above."
            )
        return await self.llm.complete(model=self.config.models.writer, system=system, user=user)
