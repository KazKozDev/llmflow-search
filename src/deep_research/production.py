from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from .config import load_config
from .llm import OllamaClient
from .store import EvidenceStore
from .tools import Fetcher, FootnoteMCPProvider, PageFetcher, SearchProvider, SearxNGProvider


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    passed: bool
    detail: str


class SmokeInference(BaseModel):
    ok: bool


async def run_production_smoke(config_path: str | Path, database_path: str | Path) -> dict:
    config = load_config(config_path)
    checks: list[SmokeCheck] = []
    models_available = False

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
            response = await client.get(f"{config.ollama.base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            installed_models = {model["name"] for model in response.json().get("models", [])}
        required_models = set(asdict(config.models).values())
        missing = sorted(required_models - installed_models)
        models_available = not missing
        checks.append(
            SmokeCheck(
                "ollama_models",
                not missing,
                "all configured models are installed" if not missing else f"missing: {', '.join(missing)}",
            )
        )
    except Exception as exc:
        checks.append(SmokeCheck("ollama_models", False, str(exc)))

    if models_available:
        try:
            llm = OllamaClient(
                config.ollama.base_url,
                config.ollama.timeout_seconds,
                keep_alive=config.ollama.keep_alive,
                max_retries=config.ollama.max_retries,
                num_predict=config.ollama.num_predict,
                think=config.ollama.think,
            )
            inference = await llm.complete_json(
                model=config.models.planner,
                system="Return the requested structured health-check response.",
                user="Set ok to true.",
                schema=SmokeInference,
            )
            if not inference.ok:
                raise RuntimeError("Ollama returned ok=false")
            checks.append(SmokeCheck("ollama_structured_inference", True, "structured inference succeeded"))
        except Exception as exc:
            checks.append(SmokeCheck("ollama_structured_inference", False, str(exc)))
    else:
        checks.append(SmokeCheck("ollama_structured_inference", False, "configured models are unavailable"))

    async def _search_and_read(search: SearchProvider, fetcher: Fetcher) -> str:
        results = await search.search("SearXNG", 1)
        if not results:
            raise RuntimeError("search returned no results")
        source = await fetcher.fetch(results[0])
        return f"read {len(source.text)} characters from {source.url}"

    def _footnote() -> FootnoteMCPProvider:
        return FootnoteMCPProvider(
            config.search.footnote.command,
            config.search.footnote.args,
            config.search.language,
            config.search.footnote.provider,
            config.search.footnote.semantic_rerank,
            config.search.footnote.min_search_interval_seconds,
        )

    if config.search.provider == "searxng":
        try:
            detail = await _search_and_read(
                SearxNGProvider(config.search.base_url, config.search.language), PageFetcher()
            )
            checks.append(SmokeCheck("searxng_search", True, detail))
        except Exception as exc:
            if config.search.fallback_provider == "footnote_mcp":
                # The run can still proceed on the fallback provider, so the check
                # passes as long as the fallback path works end to end.
                try:
                    mcp = _footnote()
                    detail = await _search_and_read(mcp, mcp)
                    checks.append(
                        SmokeCheck("searxng_search", True, f"SearXNG failed ({exc}); footnote-mcp fallback works: {detail}")
                    )
                except Exception as fallback_exc:
                    checks.append(
                        SmokeCheck(
                            "searxng_search",
                            False,
                            f"SearXNG failed ({exc}); footnote-mcp fallback also failed ({fallback_exc})",
                        )
                    )
            else:
                checks.append(SmokeCheck("searxng_search", False, str(exc)))
    elif config.search.provider == "footnote_mcp":
        try:
            mcp = _footnote()
            checks.append(SmokeCheck("searxng_search", True, await _search_and_read(mcp, mcp)))
        except Exception as exc:
            checks.append(SmokeCheck("searxng_search", False, str(exc)))
    else:
        checks.append(SmokeCheck("searxng_search", False, f"unknown search provider '{config.search.provider}'"))

    try:
        smoke_database = Path(database_path)
        store = EvidenceStore(smoke_database)
        try:
            research_id = "smoke"
            timestamp = datetime.now(UTC).isoformat()
            try:
                store.create_run(research_id, "production smoke", timestamp, {})
            except Exception:
                pass
            store.add_event(research_id, timestamp, "smoke_completed", {})
            store.get_run(research_id)
        finally:
            store.close()
        checks.append(SmokeCheck("sqlite_write", True, f"wrote {smoke_database}"))
    except Exception as exc:
        checks.append(SmokeCheck("sqlite_write", False, str(exc)))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }
