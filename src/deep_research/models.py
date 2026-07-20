from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ResearchStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RESEARCHING = "researching"
    CONSOLIDATING = "consolidating"
    VERIFYING = "verifying"
    WRITING = "writing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    REJECTED = "rejected"


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    provider: str = "searxng"
    rank: int = 0


PreferredSourceType = Literal["official_documentation", "institutional", "repository", "news", "web_page"]


class ResearchTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:8]}")
    objective: str
    questions: list[str] = Field(min_length=1, max_length=5)
    search_queries: list[str] = Field(min_length=1, max_length=5)
    preferred_source_types: list[PreferredSourceType] = Field(min_length=1, max_length=5)
    max_searches: int = Field(default=5, ge=1, le=20)
    max_pages: int = Field(default=4, ge=1, le=15)


class ResearchPlan(BaseModel):
    research_goal: str
    tasks: list[ResearchTask] = Field(min_length=1, max_length=8)
    success_criteria: list[str] = Field(min_length=1, max_length=8)


class Source(BaseModel):
    source_id: str = Field(default_factory=lambda: f"src_{uuid4().hex[:12]}")
    url: str
    canonical_url: str
    title: str
    author: str | None = None
    published_at: str | None = None
    retrieved_at: str = Field(default_factory=now_iso)
    content_hash: str
    source_type: str = "web_page"
    quality_score: float = Field(ge=0, le=1)
    text: str = ""


class ExtractedClaim(BaseModel):
    claim: str = Field(min_length=5, max_length=600)
    quote: str = Field(min_length=5, max_length=1200)
    support_type: str = Field(default="supports", pattern="^(supports|contradicts|context|unclear)$")
    relevance: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Decimal score from 0.0 to 1.0. Never use a 0-100 percentage.",
    )


class ExtractionResult(BaseModel):
    items: list[ExtractedClaim] = Field(default_factory=list, max_length=3)
    open_questions: list[str] = Field(default_factory=list, max_length=5)


class EvidenceItem(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    research_id: str
    task_id: str
    claim: str
    quote: str
    source_id: str
    relevance: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    support_type: str
    verification_status: str = "pending"


class ClaimReview(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(min_length=1)
    status: VerificationStatus
    confidence: float = Field(ge=0, le=1, description="Decimal score from 0.0 to 1.0, never a percentage")
    notes: str = ""


class FactCheckResult(BaseModel):
    reviews: list[ClaimReview] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)


class GapAssessment(BaseModel):
    unanswered_questions: list[str] = Field(default_factory=list, max_length=3)


class ReportDraft(BaseModel):
    markdown: str


class RunSummary(BaseModel):
    research_id: str
    query: str
    status: ResearchStatus
    created_at: str
    updated_at: str
    error: str | None = None
