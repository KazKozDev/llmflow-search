from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .cli import run_research
from .store import EvidenceStore

_MARKDOWN_LINK = re.compile(r"\[[^]]+\]\((https?://[^)]+)\)")


class BenchmarkCase(BaseModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    query: str = Field(min_length=5)
    expected_keywords: list[str] = Field(default_factory=list)
    required_source_domains: list[str] = Field(default_factory=list)
    min_citations: int = Field(default=1, ge=0)


@dataclass(frozen=True)
class BenchmarkScore:
    case_id: str
    research_id: str
    passed: bool
    citation_count: int
    unique_domains: list[str]
    missing_keywords: list[str]
    missing_required_domains: list[str]


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(BenchmarkCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid benchmark case at line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Benchmark file contains no cases")
    return cases


def score_report(case: BenchmarkCase, report: str, research_id: str) -> BenchmarkScore:
    urls = _MARKDOWN_LINK.findall(report)
    domains = sorted({urlparse(url).hostname for url in urls if urlparse(url).hostname})
    report_lower = report.lower()
    missing_keywords = [keyword for keyword in case.expected_keywords if keyword.lower() not in report_lower]
    missing_domains = [
        domain
        for domain in case.required_source_domains
        if not any(host == domain or host.endswith(f".{domain}") for host in domains)
    ]
    passed = len(urls) >= case.min_citations and not missing_keywords and not missing_domains
    return BenchmarkScore(
        case_id=case.case_id,
        research_id=research_id,
        passed=passed,
        citation_count=len(urls),
        unique_domains=domains,
        missing_keywords=missing_keywords,
        missing_required_domains=missing_domains,
    )


async def run_benchmark(
    cases_path: str | Path,
    config_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
) -> dict:
    cases = load_cases(cases_path)
    scores: list[BenchmarkScore] = []
    for case in cases:
        research_id = await run_research(case.query, str(config_path), str(database_path), quiet=True)
        store = EvidenceStore(database_path)
        try:
            report = store.get_report(research_id)
        finally:
            store.close()
        scores.append(score_report(case, report, research_id))

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_count": len(scores),
        "passed_count": sum(score.passed for score in scores),
        "scores": [asdict(score) for score in scores],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
