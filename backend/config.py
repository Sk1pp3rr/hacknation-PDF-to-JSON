"""Application configuration loaded from environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the backend and EU LLM provider."""

    app_name: str = "Odwołania2json API"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.mistral.ai/v1"
    llm_model: str = "ministral-3b-latest"
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.0
    llm_max_retries: int = 1
    llm_use_strict_schema: bool = True
    llm_strict_checksums: bool = False
    llm_extraction_mode: Literal["single", "section_swarm"] = "section_swarm"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
