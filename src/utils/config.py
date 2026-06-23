from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "mock"


def load_config() -> AppConfig:
    api_key = os.getenv("OPENAI_API_KEY") or None
    provider = os.getenv("LLM_PROVIDER", "openai" if api_key else "mock").lower()
    return AppConfig(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        llm_provider=provider,
    )

