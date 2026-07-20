from datetime import date

import pytest

from deep_research.agents import Planner, ResearchWorker
from deep_research.config import AppConfig
from deep_research.llm import LLMError
from deep_research.models import ExtractionResult, ResearchPlan, ResearchTask, SearchResult, Source
from deep_research.store import EvidenceStore
from deep_research.tools import SearchProvider


class PlannerLLM:
    def __init__(self, response: ResearchPlan | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response

    async def complete(self, *, model: str, system: str, user: str) -> str:
        raise AssertionError("Planner does not use unstructured completion")


@pytest.mark.asyncio
async def test_planner_failure_is_not_hidden() -> None:
    planner = Planner(PlannerLLM(error=LLMError("planner unavailable")), AppConfig())

    with pytest.raises(LLMError, match="planner unavailable"):
        await planner.plan("latest LLM news")


@pytest.mark.asyncio
async def test_current_queries_are_scoped_to_the_current_period() -> None:
    plan = ResearchPlan.model_validate(
        {
            "research_goal": "Latest LLM news",
            "tasks": [
                {
                    "objective": "Find primary announcements",
                    "questions": ["What was announced?"],
                    "search_queries": ["LLM model releases"],
                    "preferred_source_types": ["official_documentation"],
                }
            ],
            "success_criteria": ["Find a dated primary announcement"],
        }
    )
    planner = Planner(PlannerLLM(response=plan), AppConfig())

    scoped = await planner.plan("latest LLM news")

    assert str(date.today().year) in scoped.tasks[0].search_queries[0]


def test_quote_matching_normalizes_unicode_and_whitespace() -> None:
    assert ResearchWorker._quote_is_present("A  direct\nquote", "Before: A direct quote. After.")
    assert not ResearchWorker._quote_is_present("A rewritten quote", "A direct quote")


class _RepeatingSearch(SearchProvider):
    """Both calls resolve to the same URL, simulating a follow-up round re-surfacing
    a page an earlier round already fetched."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [SearchResult(title="Official strategy page", url="https://example.gov/strategy", rank=1)]


class _CountingFetcher:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, result: SearchResult) -> Source:
        self.calls += 1
        return Source(
            url=result.url,
            canonical_url=result.url,
            title=result.title,
            content_hash="sha256:strategy",
            quality_score=0.85,
            source_type="official_documentation",
            text="The strategy commits 2 billion dollars to compute infrastructure by 2030.",
        )


class _ExtractionLLM:
    async def complete_json(self, *, model: str, system: str, user: str, schema):
        assert schema is ExtractionResult
        return ExtractionResult(items=[])

    async def complete(self, *, model: str, system: str, user: str) -> str:
        raise AssertionError("not used")


def _task(task_id: str) -> ResearchTask:
    return ResearchTask.model_validate(
        {
            "task_id": task_id,
            "objective": "Find investment figures",
            "questions": ["How much was invested?"],
            "search_queries": ["strategy investment"],
            "preferred_source_types": ["official_documentation"],
            "max_searches": 1,
            "max_pages": 1,
        }
    )


@pytest.mark.asyncio
async def test_worker_reuses_a_previously_fetched_page_instead_of_refetching(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "research.sqlite3")
    try:
        fetcher = _CountingFetcher()
        worker = ResearchWorker(_ExtractionLLM(), _RepeatingSearch(), fetcher, store, AppConfig())

        events: list[tuple[str, dict]] = []
        await worker.research("res_1", _task("task_round1"), lambda t, p: events.append((t, p)))
        await worker.research("res_1", _task("task_round2"), lambda t, p: events.append((t, p)))

        assert fetcher.calls == 1
        assert any(event_type == "page_reused" for event_type, _ in events)
    finally:
        store.close()
