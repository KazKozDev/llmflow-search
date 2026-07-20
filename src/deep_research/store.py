from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import EvidenceItem, ResearchStatus, RunSummary, Source


class EvidenceStore:
    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS research_runs (
                research_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                report_markdown TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                canonical_url TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL REFERENCES research_runs(research_id),
                task_id TEXT NOT NULL,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                verification_status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_research ON evidence(research_id);
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                research_id TEXT NOT NULL REFERENCES research_runs(research_id),
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS domain_classifications (
                domain TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                quality_score REAL NOT NULL,
                classified_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def create_run(self, research_id: str, query: str, created_at: str, config: dict) -> None:
        self.connection.execute(
            "INSERT INTO research_runs VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            (research_id, query, ResearchStatus.CREATED, created_at, created_at, json.dumps(config)),
        )
        self.connection.commit()

    def update_status(self, research_id: str, status: ResearchStatus, updated_at: str) -> None:
        self.connection.execute(
            "UPDATE research_runs SET status = ?, updated_at = ? WHERE research_id = ?",
            (status, updated_at, research_id),
        )
        self.connection.commit()

    def fail_run(self, research_id: str, error: str, updated_at: str) -> None:
        self.connection.execute(
            "UPDATE research_runs SET status = ?, error = ?, updated_at = ? WHERE research_id = ?",
            (ResearchStatus.FAILED, error, updated_at, research_id),
        )
        self.connection.commit()

    def add_event(self, research_id: str, created_at: str, event_type: str, payload: dict) -> None:
        self.connection.execute(
            "INSERT INTO events (research_id, created_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
            (research_id, created_at, event_type, json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.commit()

    def get_source_by_url(self, canonical_url: str) -> Source | None:
        row = self.connection.execute(
            "SELECT payload_json FROM sources WHERE canonical_url = ?", (canonical_url,)
        ).fetchone()
        return Source.model_validate_json(row["payload_json"]) if row else None

    def save_source(self, source: Source) -> str:
        existing = self.connection.execute(
            "SELECT source_id FROM sources WHERE canonical_url = ?", (source.canonical_url,)
        ).fetchone()
        if existing:
            return str(existing["source_id"])
        self.connection.execute(
            "INSERT INTO sources VALUES (?, ?, ?)",
            (source.source_id, source.canonical_url, source.model_dump_json()),
        )
        self.connection.commit()
        return source.source_id

    def save_evidence(self, item: EvidenceItem) -> None:
        self.connection.execute(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
            (
                item.evidence_id,
                item.research_id,
                item.task_id,
                item.source_id,
                item.verification_status,
                item.model_dump_json(),
            ),
        )
        self.connection.commit()

    def list_evidence(self, research_id: str) -> list[EvidenceItem]:
        rows = self.connection.execute("SELECT payload_json FROM evidence WHERE research_id = ?", (research_id,)).fetchall()
        return [EvidenceItem.model_validate_json(row["payload_json"]) for row in rows]

    def list_sources(self, source_ids: set[str] | None = None) -> dict[str, Source]:
        if source_ids is not None:
            if not source_ids:
                return {}
            placeholders = ",".join("?" for _ in source_ids)
            rows = self.connection.execute(
                f"SELECT payload_json FROM sources WHERE source_id IN ({placeholders})", tuple(source_ids)
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT payload_json FROM sources").fetchall()
        sources = [Source.model_validate_json(row["payload_json"]) for row in rows]
        return {source.source_id: source for source in sources}

    def apply_reviews(self, research_id: str, reviews: dict[str, str]) -> None:
        for evidence_id, status in reviews.items():
            row = self.connection.execute(
                "SELECT payload_json FROM evidence WHERE evidence_id = ? AND research_id = ?",
                (evidence_id, research_id),
            ).fetchone()
            if not row:
                continue
            item = EvidenceItem.model_validate_json(row["payload_json"])
            item.verification_status = status
            self.connection.execute(
                "UPDATE evidence SET verification_status = ?, payload_json = ? WHERE evidence_id = ?",
                (status, item.model_dump_json(), evidence_id),
            )
        self.connection.commit()

    def save_report(self, research_id: str, report_markdown: str, updated_at: str) -> None:
        self.connection.execute(
            "UPDATE research_runs SET report_markdown = ?, updated_at = ? WHERE research_id = ?",
            (report_markdown, updated_at, research_id),
        )
        self.connection.commit()

    def get_run(self, research_id: str) -> RunSummary:
        row = self.connection.execute(
            "SELECT research_id, query, status, created_at, updated_at, error FROM research_runs WHERE research_id = ?",
            (research_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown research id: {research_id}")
        return RunSummary.model_validate(dict(row))

    def latest_resumable(self) -> RunSummary | None:
        """Most recently updated run that did not complete but has stored evidence to resume from."""
        row = self.connection.execute(
            """
            SELECT r.research_id, r.query, r.status, r.created_at, r.updated_at, r.error
            FROM research_runs r
            WHERE r.status NOT IN (?, ?)
              AND EXISTS (SELECT 1 FROM evidence e WHERE e.research_id = r.research_id)
            ORDER BY r.updated_at DESC
            LIMIT 1
            """,
            (ResearchStatus.COMPLETED, ResearchStatus.CREATED),
        ).fetchone()
        if not row:
            return None
        return RunSummary.model_validate(dict(row))

    def get_domain_classification(self, domain: str) -> tuple[str, float] | None:
        row = self.connection.execute(
            "SELECT source_type, quality_score FROM domain_classifications WHERE domain = ?",
            (domain,),
        ).fetchone()
        if not row:
            return None
        return str(row["source_type"]), float(row["quality_score"])

    def save_domain_classification(self, domain: str, source_type: str, quality_score: float, classified_at: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO domain_classifications VALUES (?, ?, ?, ?)",
            (domain, source_type, quality_score, classified_at),
        )
        self.connection.commit()

    def get_report(self, research_id: str) -> str:
        row = self.connection.execute(
            "SELECT report_markdown FROM research_runs WHERE research_id = ?", (research_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown research id: {research_id}")
        if row["report_markdown"] is None:
            raise ValueError(f"Research '{research_id}' has no completed report")
        return str(row["report_markdown"])
