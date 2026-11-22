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
    "potwierdzenie wykonania operacji kartowej",
    "zakup przy uzyciu karty",
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
    head = _strip_accents(normalized_text[:800])
    if any(_strip_accents(marker) in head for marker in _NEGATIVE_HEADER_MARKERS):
        return True
    is_kio_ruling = (
        ("sygn. akt" in head or "krajowa izba odwolawcza" in head)
        and ("w skladzie:" in head or "przewodniczacy:" in head)
        and ("wyrok" in head or "postanowienie" in head)
    )
    return is_kio_ruling


def _looks_like_appeal_over_proxy(normalized_text: str) -> bool:
    """Detect when a KIO appeal mentions a representative/proxy in its header without being a standalone proxy."""
    folded = _strip_accents(normalized_text)
    if "pelnomocnictwo" in folded[:350] and any(
        grant in folded[:900] for grant in ("udziela", "upowazni", "niniejszym ustanawia")
    ):
        return False
    if folded[:120].startswith("pelnomocnictwo"):
        return False
    has_parties = "odwolujac" in folded and "zamawiajac" in folded
    has_appeal_body = any(
        marker in folded
        for marker in ("zarzut", "zarzuca", "wnosze o")
    )
    return has_parties and has_appeal_body


def _looks_like_proxy_over_appeal(normalized_text: str) -> bool:
    """Detect standalone proxies that mention KIO/appeal scope and lack appeal grounds/demands."""
    folded = _strip_accents(normalized_text)
    has_proxy_header = "pelnomocnictwo" in folded[:500] or any(
        grant in folded[:900] for grant in ("udziela pelnomocnictwa", "udzielam pelnomocnictwa", "upowaznia", "niniejszym ustanawia")
    )
    has_appeal_body = any(marker in folded for marker in ("zarzut", "zarzuca", "wnosze o"))
    return has_proxy_header and not has_appeal_body


def _resolve_doc_type(label: str) -> DocumentType | str:
    """Resolve a predicted label to a built-in DocumentType or a dynamic YAML class id."""
    if label == "dowod_wplaty":
        return DocumentType.PRZELEW
    try:
        return DocumentType(label)
    except ValueError:
        try:
            if label in get_all_document_configs():
                return label
        except Exception:
            pass
        return DocumentType.INNY


def _phrases_by_doc_type() -> dict[DocumentType | str, tuple[str, ...]]:
    """Load classification phrases from YAML configs, including dynamically added classes."""
    phrases: dict[DocumentType | str, tuple[str, ...]] = {}
    try:
        configs = get_all_document_configs()
    except Exception:
        configs = {}
    for doc_id, cfg in configs.items():
        if cfg.klasyfikacja.frazy_kluczowe:
            try:
                key: DocumentType | str = DocumentType(doc_id)
            except ValueError:
                key = doc_id
            phrases[key] = tuple(p.lower() for p in cfg.klasyfikacja.frazy_kluczowe)
    for doc_type, default_phrases in FALLBACK_PHRASES.items():
        phrases.setdefault(doc_type, default_phrases)
    return phrases


def _match_untrained_yaml_classes(
    normalized_text: str,
    trained_labels: set[str],
) -> tuple[str, float] | None:
    """Match newly added YAML document classes even before retraining LinearSVC."""
    try:
        configs = get_all_document_configs()
    except Exception:
        return None

    folded = _strip_accents(normalized_text)
    best_id: str | None = None
    best_conf = 0.0
    for doc_id, cfg in configs.items():
        if doc_id in trained_labels or not cfg.klasyfikacja.frazy_kluczowe:
            continue
        phrases = tuple(_strip_accents(p.lower()) for p in cfg.klasyfikacja.frazy_kluczowe)
        hits = sum(1 for phrase in phrases if phrase in folded)
        min_hits = min(len(phrases), 2 if len(normalized_text) > 400 else 1)
        if hits >= min_hits and hits > 0:
            conf = min(0.95, 0.50 + 0.15 * hits)
            if conf >= cfg.klasyfikacja.prog_pewnosci and conf > best_conf:
                best_id = doc_id
                best_conf = conf
    return (best_id, best_conf) if best_id is not None else None


def classify_document(
    text: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
) -> tuple[DocumentType | str, float]:
    text = normalize_text(text)
    if not text:
        return DocumentType.INNY, 1.0
    if _is_out_of_scope_header(text):
        return DocumentType.INNY, 0.95

    try:
        model = load_classifier(model_path)
        labels = model["classifier"].classes_
        untrained_match = _match_untrained_yaml_classes(text, {str(lbl) for lbl in labels})
        if untrained_match is not None:
            return untrained_match

        matrix = model["vectorizer"].transform([text])
        scores = model["classifier"].decision_function(matrix)[0]

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
        doc_type = _resolve_doc_type(label)
        if doc_type == DocumentType.PELNOMOCNICTWO and _looks_like_appeal_over_proxy(text):
            return DocumentType.ODWOLANIE, score
        if doc_type == DocumentType.ODWOLANIE and _looks_like_proxy_over_appeal(text):
            return DocumentType.PELNOMOCNICTWO, score
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


def fallback_classify(text: str) -> tuple[DocumentType | str, float]:
    normalized = normalize_text(text)
    if not normalized or _is_out_of_scope_header(normalized):
        return DocumentType.INNY, 0.95

    folded = _strip_accents(normalized)
    phrase_map = _phrases_by_doc_type()
    matches: dict[DocumentType | str, int] = {
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
    if best_type == DocumentType.ODWOLANIE and _looks_like_proxy_over_appeal(normalized):
        return DocumentType.PELNOMOCNICTWO, confidence
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
