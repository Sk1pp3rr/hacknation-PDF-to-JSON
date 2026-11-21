"""OpenAI-compatible async HTTP client for EU-hosted open-source LLM APIs."""

import asyncio
import time
from typing import Any

import httpx
from pydantic import BaseModel

from backend.config import settings
from backend.services.schema_utils import build_response_format


class LLMClientError(RuntimeError):
    """Raised when communication with the LLM API fails."""


class TokenUsage(BaseModel):
    """Token consumption summary for an extraction run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ChatCompletionResult(BaseModel):
    """Structured completion response and usage telemetry."""

    content: str
    usage: TokenUsage
    model: str
    latency_ms: float
    used_schema_fallback: bool = False


class LLMClient:
    """Thin async client for OpenAI-compatible EU endpoints (Mistral, Scaleway, OVHcloud)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        use_strict_schema: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout_seconds = settings.llm_timeout_seconds if timeout_seconds is None else timeout_seconds
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.use_strict_schema = settings.llm_use_strict_schema if use_strict_schema is None else use_strict_schema
        self._transport = transport

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        json_schema: dict[str, Any],
        max_tokens: int = 800,
        temperature: float | None = None,
    ) -> ChatCompletionResult:
        """Send a chat completion request enforcing JSON Schema (with json_object fallback)."""
        if not self.api_key and self._transport is None:
            raise LLMClientError("LLM_API_KEY is not configured")

        temp = self.temperature if temperature is None else temperature
        response_format: dict[str, Any] = (
            build_response_format(schema_name, json_schema, strict=True)
            if self.use_strict_schema
            else {"type": "json_object"}
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }

        used_fallback = False
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._transport) as client:
            resp = await self._post_with_retry(client, payload)

            if resp.status_code in (400, 422) and self.use_strict_schema:
                payload = {**payload, "response_format": {"type": "json_object"}}
                used_fallback = True
                resp = await self._post_with_retry(client, payload)

        if resp.status_code >= 400:
            raise LLMClientError(f"LLM API error {resp.status_code}: {resp.text[:300]}")

        body = resp.json()
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMClientError("LLM API returned no choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMClientError("LLM API returned empty message content")

        raw_usage = body.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        return ChatCompletionResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            model=str(body.get("model") or self.model),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            used_schema_fallback=used_fallback,
        )

    async def _post_with_retry(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"

        delays = (0.25, 0.75)
        last_exc: Exception | None = None

        for attempt in range(len(delays) + 1):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < len(delays):
                    await asyncio.sleep(delays[attempt])
                    continue
                return resp
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < len(delays):
                    await asyncio.sleep(delays[attempt])
                    continue
                raise LLMClientError(f"LLM transport error: {exc}") from exc

        raise LLMClientError(f"LLM request failed: {last_exc}")
