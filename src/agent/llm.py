from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from src.utils.config import AppConfig, load_config


class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 700) -> str:
        """Return a completion for a prompt."""


class MockLLMProvider(BaseLLMProvider):
    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 700) -> str:
        del temperature, max_tokens
        return (
            "Mock provider response: use the structured risk score, cited evidence, and recommended actions. "
            "No external LLM was called, so this demo remains reproducible without an API key."
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, config: AppConfig) -> None:
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAICompatibleProvider")
        self.config = config

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 700) -> str:
        payload: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful customer-success analyst. Ground every claim in supplied evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.config.openai_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible provider request failed: {exc}") from exc
        return str(body["choices"][0]["message"]["content"]).strip()


def get_llm_provider(config: AppConfig | None = None) -> BaseLLMProvider:
    resolved = config or load_config()
    if resolved.llm_provider == "openai" and resolved.openai_api_key:
        try:
            return OpenAICompatibleProvider(resolved)
        except ValueError:
            return MockLLMProvider()
    return MockLLMProvider()

