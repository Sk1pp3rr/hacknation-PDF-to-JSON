"""Document classification and segmentation module."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from backend.config import get_all_document_configs
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
    "potwierdzenie_przelewu": DocumentType.PRZELEW,
}

_NEGATIVE_HEADER_MARKERS = (
    "dziennik urzedowy unii europejskiej",
    "dziennik urzędowy unii europejskiej",
    "biuletyn zamowien publicznych",
    "biuletyn zamówień publicznych",
    "krajowa izba odwolawcza w skladzie",
    "krajowa izba odwoławcza w składzie",
    "w imieniu rzeczypospolitej polskiej",
)

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


def _is_out_of_scope_header(normalized_text: str) -> bool:
    head = normalized_text[:800]
    return any(marker in head for marker in _NEGATIVE_HEADER_MARKERS)


def _looks_like_appeal_over_proxy(normalized_text: str) -> bool:
    """Detect when a KIO appeal mentions a representative/proxy in its header without being a standalone proxy."""
    head = normalized_text[:120]
    if head.startswith("pełnomocnictwo") or head.startswith("pelnomocnictwo"):
        return False
    has_parties = ("odwołując" in normalized_text or "odwolujac" in normalized_text) and (
        "zamawiając" in normalized_text or "zamawiajac" in normalized_text
    )
    has_appeal_body = any(
        marker in normalized_text
        for marker in ("zarzut", "zarzuca", "wnoszę o", "wnosze o", "odwołanie od", "odwolanie od")
    )
    return has_parties and has_appeal_body


def _phrases_by_doc_type() -> dict[DocumentType, tuple[str, ...]]:
    """Load classification phrases from YAML configs, falling back to built-in defaults."""
    phrases: dict[DocumentType, tuple[str, ...]] = {}
    try:
        configs = get_all_document_configs()
    except Exception:
        configs = {}
    for doc_id, cfg in configs.items():
        if cfg.klasyfikacja.frazy_kluczowe:
            try:
                doc_type = DocumentType(doc_id)
            except ValueError:
                continue
            phrases[doc_type] = tuple(p.lower() for p in cfg.klasyfikacja.frazy_kluczowe)
    for doc_type, default_phrases in FALLBACK_PHRASES.items():
        phrases.setdefault(doc_type, default_phrases)
    return phrases


def classify_document(
    text: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
) -> tuple[DocumentType, float]:
    text = normalize_text(text)
    if not text:
        return DocumentType.INNY, 1.0
    if _is_out_of_scope_header(text):
        return DocumentType.INNY, 0.95

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

        if label == DocumentType.INNY.value:
            return DocumentType.INNY, score

        cutoff = threshold if threshold is not None else float(model.get("threshold", 0.62))
        if score < cutoff:
            fallback_type, fallback_confidence = fallback_classify(text)
            if fallback_confidence >= 0.60 and fallback_type != DocumentType.INNY:
                return fallback_type, fallback_confidence
            return DocumentType.INNY, score
        mapping = {
            "odwolanie": DocumentType.ODWOLANIE,
            "pelnomocnictwo": DocumentType.PELNOMOCNICTWO,
            "przelew": DocumentType.PRZELEW,
            "dowod_wplaty": DocumentType.PRZELEW,
            "inny": DocumentType.INNY,
        }
        doc_type = mapping.get(label, DocumentType.INNY)
        if doc_type == DocumentType.PELNOMOCNICTWO and _looks_like_appeal_over_proxy(text):
            return DocumentType.ODWOLANIE, score
        return doc_type, score
    except FileNotFoundError:
        return fallback_classify(text)


def softmax(scores: list[float]) -> list[float]:
    largest = max(scores)
    exponents = [math.exp(score - largest) for score in scores]
    total = sum(exponents)
    return [value / total for value in exponents]


_DIACRITIC_MAP = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _strip_accents(text: str) -> str:
    return text.translate(_DIACRITIC_MAP)


def fallback_classify(text: str) -> tuple[DocumentType, float]:
    normalized = normalize_text(text)
    if not normalized or _is_out_of_scope_header(normalized):
        return DocumentType.INNY, 0.95

    folded = _strip_accents(normalized)
    phrase_map = _phrases_by_doc_type()
    matches: dict[DocumentType, int] = {
        doc_type: sum(1 for phrase in phrases if _strip_accents(phrase) in folded)
        for doc_type, phrases in phrase_map.items()
    }
    best_type = max(matches, key=matches.__getitem__)
    count = matches[best_type]
    min_hits = 2 if len(normalized) > 400 else 1
    if count < min_hits:
        return DocumentType.INNY, 1.0
    confidence = min(0.95, 0.50 + 0.10 * count)
    if best_type == DocumentType.PELNOMOCNICTWO and _looks_like_appeal_over_proxy(normalized):
        return DocumentType.ODWOLANIE, confidence
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
