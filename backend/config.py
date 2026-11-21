"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the backend and EU LLM provider."""

    app_name: str = "Odwołania2json API"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.mistral.ai/v1"
    llm_model: str = "ministral-8b-latest"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
