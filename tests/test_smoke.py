"""Basic smoke tests for the backend API."""

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import DocumentType
from backend.services.classifier import classify_document, classify_and_segment

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classifier_fallback_examples() -> None:
    appeal, _ = classify_document("Odwołanie do Krajowej Izby Odwoławczej")
    proxy, _ = classify_document("Pełnomocnictwo dla pełnomocnika i mocodawcy")
    transfer, _ = classify_document("Przelew opłaty na rachunek bankowy")

    assert appeal == DocumentType.ODWOLANIE
    assert proxy == DocumentType.PELNOMOCNICTWO
    assert transfer == DocumentType.DOWOD_WPLATY


def test_segmenter_returns_empty_for_empty_text() -> None:
    assert classify_and_segment("") == []
