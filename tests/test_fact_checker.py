import ast
import asyncio

import pytest

from deep_research.agents import FactChecker
from deep_research.config import AppConfig, RuntimeConfig
from deep_research.llm import LLM, LLMError, LLMResponseError
from deep_research.models import ClaimReview, EvidenceItem, FactCheckResult, VerificationStatus
from deep_research.tools import BASELINE_QUALITY


class FailingLLM:
    """Stands in for a fact checker model that always times out."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, model: str, system: str, user: str) -> str:
        raise LLMError("timed out")

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        self.calls += 1
        raise LLMError("timed out")


class CountingLLM:
    """Records how many requests the checker makes and verifies every claim it sees."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def complete(self, *, model: str, system: str, user: str) -> str:
        raise LLMError("not used")

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        payload = ast.literal_eval(user.removeprefix("Evidence groups: "))
        self.batch_sizes.append(len(payload))
        return FactCheckResult(
            reviews=[
                ClaimReview(
                    claim=group["claim"],
                    evidence_ids=[item["evidence_id"] for item in group["evidence"]],
                    status=VerificationStatus.VERIFIED,
                    confidence=0.9,
                )
                for group in payload
            ]
        )


def _evidence(claim: str, support_type: str = "supports", quality: float = BASELINE_QUALITY) -> EvidenceItem:
    return EvidenceItem(
        research_id="res_test",
        task_id="task_test",
        claim=claim,
        quote=f"quote for {claim}",
        source_id="src_test",
        relevance=0.9,
        source_quality=quality,
        support_type=support_type,
    )


def _checker(llm: LLM, batch_size: int = 8) -> FactChecker:
    config = AppConfig(runtime=RuntimeConfig(fact_check_batch_size=batch_size))
    return FactChecker(llm, config)


def test_fact_checker_failure_is_not_hidden() -> None:
    checker = _checker(FailingLLM())

    with pytest.raises(LLMError, match="timed out"):
        asyncio.run(checker.check([_evidence("LLM release happened")]))


def test_claims_are_split_into_batched_requests() -> None:
    llm = CountingLLM()
    checker = _checker(llm, batch_size=2)
    evidence = [_evidence(f"claim number {index}") for index in range(5)]

    asyncio.run(checker.check(evidence))

    assert llm.batch_sizes == [2, 2, 1]


class OmittingLLM(CountingLLM):
    """Drops the first evidence group's reviews; fixes it on retry only if told to."""

    def __init__(self, corrects_on_retry: bool) -> None:
        super().__init__()
        self.corrects_on_retry = corrects_on_retry
        self.retry_prompts: list[str] = []

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        is_retry = "Your previous answer was invalid" in user
        if is_retry:
            self.retry_prompts.append(user)
        payload = ast.literal_eval(user.split("Evidence groups: ", 1)[1].split("\n\nYour previous", 1)[0])
        complete = is_retry and self.corrects_on_retry
        groups = payload if complete else payload[1:]  # omit the first group until corrected
        return FactCheckResult(
            reviews=[
                ClaimReview(
                    claim=group["claim"],
                    evidence_ids=[item["evidence_id"] for item in group["evidence"]],
                    status=VerificationStatus.VERIFIED,
                    confidence=0.9,
                )
                for group in groups
            ]
        )


def test_invalid_reviews_are_fixed_by_corrective_retry() -> None:
    llm = OmittingLLM(corrects_on_retry=True)
    checker = _checker(llm, batch_size=3)
    evidence = [_evidence(f"claim number {index}") for index in range(3)]

    result = asyncio.run(checker.check(evidence))

    assert len(llm.retry_prompts) == 1
    assert "omitted evidence ids" in llm.retry_prompts[0]
    assert len(result.reviews) == 3
    assert all(review.status == VerificationStatus.VERIFIED for review in result.reviews)


def test_persistently_invalid_reviews_degrade_to_insufficient() -> None:
    llm = OmittingLLM(corrects_on_retry=False)
    checker = _checker(llm, batch_size=3)
    evidence = [_evidence(f"claim number {index}") for index in range(3)]

    result = asyncio.run(checker.check(evidence))

    # The two groups the model did review stay verified; the omitted one is not lost —
    # it is conservatively marked insufficient and surfaced as a coverage gap.
    statuses = {review.claim: review.status for review in result.reviews}
    assert statuses["claim number 0"] == VerificationStatus.INSUFFICIENT
    assert statuses["claim number 1"] == VerificationStatus.VERIFIED
    assert statuses["claim number 2"] == VerificationStatus.VERIFIED
    reviewed_ids = [eid for review in result.reviews for eid in review.evidence_ids]
    assert len(reviewed_ids) == len(set(reviewed_ids)) == 3
    assert any("could not evaluate" in gap for gap in result.coverage_gaps)


class LoopingLLM(CountingLLM):
    """Returns unusable output for multi-group requests; answers single-group ones,
    except for a designated poison claim that always fails."""

    def __init__(self, poison_claim: str | None = None) -> None:
        super().__init__()
        self.poison_claim = poison_claim

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        payload = ast.literal_eval(user.split("Evidence groups: ", 1)[1].split("\n\nYour previous", 1)[0])
        self.batch_sizes.append(len(payload))
        if len(payload) > 1:
            raise LLMResponseError("Ollama JSON did not match FactCheckResult: truncated")
        if self.poison_claim is not None and payload[0]["claim"] == self.poison_claim:
            raise LLMResponseError("Ollama JSON did not match FactCheckResult: truncated")
        return FactCheckResult(
            reviews=[
                ClaimReview(
                    claim=group["claim"],
                    evidence_ids=[item["evidence_id"] for item in group["evidence"]],
                    status=VerificationStatus.VERIFIED,
                    confidence=0.9,
                )
                for group in payload
            ]
        )


def test_unusable_batch_response_falls_back_to_single_groups() -> None:
    llm = LoopingLLM()
    checker = _checker(llm, batch_size=3)
    evidence = [_evidence(f"claim number {index}") for index in range(3)]

    result = asyncio.run(checker.check(evidence))

    # One failed 3-group request, then three successful single-group requests.
    assert llm.batch_sizes == [3, 1, 1, 1]
    assert len(result.reviews) == 3
    assert all(review.status == VerificationStatus.VERIFIED for review in result.reviews)


def test_group_that_always_fails_degrades_to_insufficient() -> None:
    llm = LoopingLLM(poison_claim="claim number 1")
    checker = _checker(llm, batch_size=3)
    evidence = [_evidence(f"claim number {index}") for index in range(3)]

    result = asyncio.run(checker.check(evidence))

    statuses = {review.claim: review.status for review in result.reviews}
    assert statuses["claim number 0"] == VerificationStatus.VERIFIED
    assert statuses["claim number 1"] == VerificationStatus.INSUFFICIENT
    assert statuses["claim number 2"] == VerificationStatus.VERIFIED
    assert any("could not evaluate" in gap for gap in result.coverage_gaps)


def test_one_failing_batch_fails_the_fact_check() -> None:
    class HalfFailingLLM(CountingLLM):
        async def complete_json(self, *, model: str, system: str, user: str, schema):
            payload = ast.literal_eval(user.removeprefix("Evidence groups: "))
            self.batch_sizes.append(len(payload))
            if len(self.batch_sizes) == 1:
                raise LLMError("timed out")
            return await super().complete_json(model=model, system=system, user=user, schema=schema)

    checker = _checker(HalfFailingLLM(), batch_size=2)
    evidence = [_evidence(f"claim number {index}") for index in range(4)]

    with pytest.raises(LLMError, match="timed out"):
        asyncio.run(checker.check(evidence))
    assert checker.llm.batch_sizes == [2]
