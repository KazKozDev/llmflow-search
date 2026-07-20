from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel


class LLMError(RuntimeError):
    pass


class LLMResponseError(LLMError):
    """The model responded, but the content was unusable (empty, truncated, or invalid JSON).

    Distinct from transport failures so callers can degrade gracefully instead of retrying
    a request the model has already shown it cannot answer.
    """


class LLM(Protocol):
    async def complete(self, *, model: str, system: str, user: str) -> str: ...

    async def complete_json(self, *, model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int,
        *,
        keep_alive: str | int = -1,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        num_predict: int = 4096,
        think: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds, connect=10)
        # keep_alive=-1 keeps the model resident between requests so Ollama does not
        # reload weights each call — the reloads are what produce sporadic ReadTimeouts.
        self.keep_alive = keep_alive
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = retry_backoff_seconds
        # Hard cap on generated tokens. Small models can fall into schema-constrained
        # generation loops that would otherwise run until the HTTP timeout.
        self.num_predict = num_predict
        # Thinking-model toggle. Thinking tokens are not constrained by the JSON schema and
        # count against num_predict, so leaving thinking on lets the model reason itself
        # into loops or starve the actual answer. None omits the field (Ollama rejects the
        # parameter for models that do not support thinking).
        self.think = think

    async def complete(self, *, model: str, system: str, user: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1, "num_predict": self.num_predict},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.think is not None:
            payload["think"] = self.think
        response = await self._post(payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("Ollama returned an empty message")
        return content.strip()

    async def complete_json(self, *, model: str, system: str, user: str, schema: type[T]) -> T:
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "num_predict": self.num_predict},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (f"{user}\n\nReturn only JSON matching the required schema. Do not include markdown fences."),
                },
            ],
        }
        if self.think is not None:
            payload["think"] = self.think
        response = await self._post(payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMResponseError("Ollama did not return JSON content")
        try:
            return schema.model_validate_json(content)
        except Exception as exc:
            raise LLMResponseError(f"Ollama JSON did not match {schema.__name__}: {exc}") from exc

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Retry only transient failures (read/connect timeouts, dropped connections). A 4xx
        # HTTPStatusError is a deterministic bad request, so it is re-raised without retrying.
        last_exc: httpx.HTTPError | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as exc:
                raise LLMError(f"Ollama request failed: {exc}") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                raise LLMError(f"Ollama request failed after {self.max_retries} attempts: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Ollama request failed: {exc}") from exc
            if not isinstance(data, dict):
                raise LLMError("Ollama returned an unexpected response")
            return data
        raise LLMError(f"Ollama request failed: {last_exc}")


class StaticLLM:
    """Small deterministic test double that returns queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def _next(self) -> str:
        if not self.responses:
            raise LLMError("StaticLLM has no remaining response")
        return self.responses.pop(0)

    async def complete(self, *, model: str, system: str, user: str) -> str:
        return self._next()

    async def complete_json(self, *, model: str, system: str, user: str, schema: type[T]) -> T:
        return schema.model_validate(json.loads(self._next()))
