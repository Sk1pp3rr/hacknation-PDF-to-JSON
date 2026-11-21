"""Document classification and segmentation module."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from backend.core.models import DocumentSegment, DocumentType


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "document_classifier.joblib"

_model: dict[str, Any] | None = None

CLASS_FILENAMES = {
    "odwolanie": DocumentType.ODWOLANIE,
    "odwolanie.json": DocumentType.ODWOLANIE,
    "pelnomocnictwo": DocumentType.PELNOMOCNICTWO,
    "pelnomocnictwo.json": DocumentType.PELNOMOCNICTWO,
    "przelew": DocumentType.PRZELEW,
    "przelew.json": DocumentType.PRZELEW,
}

FALLBACK_PHRASES = {
    DocumentType.ODWOLANIE: (
        "odwołanie", "odwołuje", "wnoszę o uchylenie", "krajowa izba odwoławcza",
        "czynność zamawiającego", "odwołujący",
    ),
    DocumentType.PELNOMOCNICTWO: (
        "pełnomocnictwo", "pełnomocnika", "mocodawca", "pełnomocnik",
        "udzielam pełnomocnictwa", "pełnomocnictwo ogólne",
    ),
    DocumentType.PRZELEW: (
        "przelew", "tytuł przelewu", "kwota", "rachunek", "opłata",
        "dowód wpłaty", "numer transakcji",
    ),
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("\ufeff", " ").split())


def flatten_json(value: Any, prefix: str = "") -> str:
    if isinstance(value, dict):
        return " ".join(
            f"{prefix} {flatten_json(child, prefix=f'{prefix} {key}' if prefix else key)}"
            for key, child in value.items()
        )
    if isinstance(value, list):
        return " ".join(flatten_json(item, prefix) for item in value)
    if value is None:
        return prefix
    return f"{prefix} {value}".strip()


def train_classifier(texts: list[str], labels: list[str]) -> dict[str, Any]:
    if len(texts) < 3:
        raise ValueError("Potrzeba co najmniej trzech przykładowych dokumentów.")
    if len(set(labels)) < 2:
        raise ValueError("Zbiór treningowy musi zawierać co najmniej dwie klasy.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.98,
    )
    matrix = vectorizer.fit_transform(texts)
    classifier = LinearSVC(class_weight="balanced", random_state=42)
    classifier.fit(matrix, labels)
    return {"vectorizer": vectorizer, "classifier": classifier, "threshold": 0.62}


def save_classifier(model: dict[str, Any], path: Path = DEFAULT_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_classifier(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    global _model
    if _model is None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Brak modelu klasyfikatora: {path}. Uruchom: "
                "python scripts/classify_documents.py --train"
            )
        _model = joblib.load(path)
    return _model


def classify_document(
    text: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
) -> tuple[DocumentType, float]:
    text = normalize_text(text)
    if not text:
        return DocumentType.INNY, 1.0

    try:
        model = load_classifier(model_path)
        matrix = model["vectorizer"].transform([text])
        scores = model["classifier"].decision_function(matrix)[0]
        labels = model["classifier"].classes_

        if hasattr(scores, "__len__"):
            scores = list(scores)
            confidence = softmax(scores)
            position = max(range(len(scores)), key=scores.__getitem__)
            label = str(labels[position])
            score = confidence[position]
        else:
            label = str(labels[1] if scores > 0 else labels[0])
            score = 1.0 / (1.0 + math.exp(-abs(float(scores))))

        cutoff = threshold if threshold is not None else float(model.get("threshold", 0.62))
        if score < cutoff:
            fallback_type, fallback_confidence = fallback_classify(text)
            if fallback_confidence >= 0.60:
                return fallback_type, fallback_confidence
            return DocumentType.INNY, score
        mapping = {
            "odwolanie": DocumentType.ODWOLANIE,
            "pelnomocnictwo": DocumentType.PELNOMOCNICTWO,
            "przelew": DocumentType.PRZELEW,
            "dowod_wplaty": DocumentType.PRZELEW,
        }
        return mapping.get(label, DocumentType.INNY), score
    except FileNotFoundError:
        return fallback_classify(text)


def softmax(scores: list[float]) -> list[float]:
    largest = max(scores)
    exponents = [math.exp(score - largest) for score in scores]
    total = sum(exponents)
    return [value / total for value in exponents]


def fallback_classify(text: str) -> tuple[DocumentType, float]:
    normalized = normalize_text(text)
    matches: dict[DocumentType, int] = {}
    for doc_type, phrases in FALLBACK_PHRASES.items():
        matches[doc_type] = sum(1 for phrase in phrases if phrase in normalized)
    best_type = max(matches, key=matches.__getitem__)
    count = matches[best_type]
    if count == 0:
        return DocumentType.INNY, 1.0
    confidence = min(0.95, 0.50 + 0.10 * count)
    return best_type, confidence


def classify_and_segment(raw_text: str) -> list[DocumentSegment]:
    """Split raw text on form feeds/headings and classify each segment."""
    if not raw_text.strip():
        return []

    chunks = re.split(r"(?:\n\s*\n|\f)", raw_text)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 20]
    if not chunks:
        chunks = [raw_text.strip()]

    segments: list[DocumentSegment] = []
    for chunk in chunks:
        doc_type, _ = classify_document(chunk)
        if doc_type != DocumentType.INNY:
            segments.append(DocumentSegment(doc_type=doc_type, text=chunk))
    return segments
