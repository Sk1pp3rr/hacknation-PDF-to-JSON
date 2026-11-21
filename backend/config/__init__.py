"""Configuration module."""

from backend.config.loader import (
    ClassificationConfig,
    DocumentConfig,
    ExtractionConfig,
    NormalizationConfig,
    RepairPromptsConfig,
    SectionSwarmConfig,
    get_all_document_configs,
    get_document_config,
    get_repair_prompts,
)
from backend.config.settings import Settings, settings

__all__ = [
    "ClassificationConfig",
    "DocumentConfig",
    "ExtractionConfig",
    "NormalizationConfig",
    "RepairPromptsConfig",
    "SectionSwarmConfig",
    "Settings",
    "get_all_document_configs",
    "get_document_config",
    "get_repair_prompts",
    "settings",
]
