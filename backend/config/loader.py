"""Dynamic YAML loader and Pydantic validator for document class profiles and prompts."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml

from backend.config.settings import settings
from backend.core.models import DocumentType


class ClassificationConfig(BaseModel):
    """Classifier keywords, filename filters, and threshold for a document class."""

    filtry_nazw_plikow: list[str] = Field(default_factory=list)
    frazy_kluczowe: list[str] = Field(default_factory=list)
    prog_pewnosci: float = 0.62


class SectionSwarmConfig(BaseModel):
    """Optional two-slice micro-swarm prompt and budget configuration."""

    header_max_tokens: int = 650
    action_max_tokens: int = 160
    header_system_prompt: str = ""
    action_system_prompt: str = ""


class ExtractionConfig(BaseModel):
    """LLM extraction schema, prompt, hints, and token budget configuration."""

    schema_name: str = ""
    json_schema: dict[str, Any] | None = None
    max_tokens: int = 600
    max_chars: int = 6000
    exclude_llm_fields: list[str] = Field(default_factory=list)
    force_string_fields: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    wskazowki: list[str] = Field(default_factory=list)
    section_swarm: SectionSwarmConfig | None = None

    def render_system_prompt(self, variables: dict[str, str] | None = None) -> str:
        """Combine base system_prompt with bullet hints and substitute template placeholders."""
        parts = [self.system_prompt.strip()]
        if self.wskazowki:
            parts.append("Zasady:\n" + "\n".join(f"- {tip.strip()}" for tip in self.wskazowki if tip.strip()))
        rendered = "\n".join(part for part in parts if part)
        for key, value in (variables or {}).items():
            rendered = rendered.replace(f"{{{key}}}", value)
        return rendered


class NormalizationConfig(BaseModel):
    """Post-extraction normalization and required-field configuration."""

    wymagane_pola: list[str] = Field(default_factory=list)
    formaty: dict[str, str] = Field(default_factory=dict)


class DocumentConfig(BaseModel):
    """Complete config-driven profile loaded from a `documents/<class>.yaml` file."""

    id: str
    nazwa: str = ""
    opis: str = ""
    klasyfikacja: ClassificationConfig = Field(default_factory=ClassificationConfig)
    ekstrakcja: ExtractionConfig = Field(default_factory=ExtractionConfig)
    normalizacja: NormalizationConfig = Field(default_factory=NormalizationConfig)


class RepairPromptsConfig(BaseModel):
    """Shared validation-repair prompt templates loaded from `prompts.yaml`."""

    surgical_repair_system: str
    surgical_repair_user_template: str
    full_repair_user_template: str


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML file must contain a mapping object: {path}")
    return raw


def get_all_document_configs(config_dir: Path | None = None) -> dict[str, DocumentConfig]:
    """Scan `config_dir` (default `settings.documents_config_dir`) for `*.yaml` document profiles."""
    target_dir = config_dir or settings.documents_config_dir
    if not target_dir.is_dir():
        return {}

    configs: dict[str, DocumentConfig] = {}
    for path in sorted(target_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        cfg = DocumentConfig.model_validate(_read_yaml(path))
        configs[cfg.id] = cfg
    return configs


def get_document_config(
    doc_type: str | DocumentType,
    config_dir: Path | None = None,
) -> DocumentConfig:
    """Return the validated `DocumentConfig` for a given document class ID."""
    key = doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type)
    configs = get_all_document_configs(config_dir=config_dir)
    if key not in configs:
        raise KeyError(f"Brak konfiguracji YAML dla klasy dokumentu: '{key}'")
    return configs[key]


def get_repair_prompts(prompts_path: Path | None = None) -> RepairPromptsConfig:
    """Load shared validation-repair prompt templates from `prompts.yaml`."""
    target_path = prompts_path or settings.prompts_config_path
    return RepairPromptsConfig.model_validate(_read_yaml(target_path))
