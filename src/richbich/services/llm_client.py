"""Wrapper around pluggable chat models."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

from richbich.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - optional dependency path
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class LLMResponse:
    text: str
    model: Optional[str]


class _BackendBase:
    model: str | None = None

    @property
    def available(self) -> bool:  # pragma: no cover - base implementation
        return False

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[LLMResponse]:  # pragma: no cover - base
        return None


class _NoopBackend(_BackendBase):
    def __init__(self, reason: str) -> None:
        self._reason = reason
        LOGGER.warning(reason)


class _OpenAIBackend(_BackendBase):
    def __init__(self, *, api_key: str, model: str, temperature: float, base_url: Optional[str]) -> None:
        self._client = None
        self.model = model
        self._temperature = temperature
        if OpenAI is None:
            LOGGER.warning("openai package is not installed; LLM explanations unavailable")
            return
        try:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime configuration issue
            LOGGER.error("Failed to initialise OpenAI client: %s", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[LLMResponse]:
        if not self.available:
            return None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=700,
            )
        except Exception as exc:  # pragma: no cover - network/runtime failure
            LOGGER.error("LLM generation failed: %s", exc)
            return None
        if not response.choices:
            return None
        message = response.choices[0].message
        if not message or not message.content:
            return None
        return LLMResponse(text=message.content.strip(), model=response.model)


class _OllamaBackend(_BackendBase):
    def __init__(self, *, model: str, temperature: float, base_url: Optional[str]) -> None:
        self.model = model
        self._temperature = temperature
        self._base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._session = requests.Session()

    @property
    def available(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[LLMResponse]:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self._temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = self._session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.error("Ollama request failed: %s", exc)
            return None
        data = response.json()
        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = message.get("content")
        if not content:
            LOGGER.warning("Ollama response did not include text content")
            return None
        return LLMResponse(text=str(content).strip(), model=data.get("model", self.model))


class LLMClient:
    """Lightweight helper to call an LLM if credentials are configured."""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.6,
    ) -> None:
        self.temperature = temperature
        env_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        env_model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        env_base_url = os.getenv("LLM_BASE_URL")
        env_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        if env_provider in {"openai", "openai-compatible", "azure"}:
            if not env_api_key:
                self._backend = _NoopBackend("OPENAI_API_KEY or LLM_API_KEY is not set; using heuristics.")
            else:
                os.environ.setdefault("OPENAI_API_KEY", env_api_key)
                self._backend = _OpenAIBackend(
                    api_key=env_api_key,
                    model=env_model,
                    temperature=self.temperature,
                    base_url=env_base_url,
                )
        elif env_provider == "ollama":
            self._backend = _OllamaBackend(
                model=env_model or "llama3",
                temperature=self.temperature,
                base_url=env_base_url,
            )
        elif env_provider in {"disabled", "none"}:
            self._backend = _NoopBackend("LLM provider disabled; summaries will be heuristic only.")
        else:
            self._backend = _NoopBackend(
                f"Unsupported LLM provider '{env_provider}'. Falling back to heuristics."
            )

        self.provider = env_provider
        self.model = env_model

    @property
    def available(self) -> bool:
        return self._backend.available

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[LLMResponse]:
        return self._backend.generate(system_prompt, user_prompt)


__all__ = ["LLMClient", "LLMResponse"]
