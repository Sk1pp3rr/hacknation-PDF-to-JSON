"""Application configuration loaded from environment variables."""

from pathlib import Path
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
    llm_max_concurrency: int = 8
    ocr_enabled: bool = True
    ocr_engine: str = "tesseract"
    ocr_languages: str = "pol"
    ocr_tesseract_cmd: str | None = None
    ocr_dpi: int = 200
    ocr_min_chars: int = 30
    ocr_max_pages: int = 0
    ocr_workers: int = 4
    documents_config_dir: Path = Path(__file__).resolve().parent / "documents"
    prompts_config_path: Path = Path(__file__).resolve().parent / "prompts.yaml"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
