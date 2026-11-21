"""Document classification and segmentation service."""

from backend.services.classifier.classifier import (
    CLASS_FILENAMES,
    DEFAULT_MODEL_PATH,
    FALLBACK_PHRASES,
    classify_and_segment,
    classify_document,
    fallback_classify,
    flatten_json,
    load_classifier,
    normalize_text,
    save_classifier,
    softmax,
    train_classifier,
)

__all__ = [
    "CLASS_FILENAMES",
    "DEFAULT_MODEL_PATH",
    "FALLBACK_PHRASES",
    "classify_and_segment",
    "classify_document",
    "fallback_classify",
    "flatten_json",
    "load_classifier",
    "normalize_text",
    "save_classifier",
    "softmax",
    "train_classifier",
]
