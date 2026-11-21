"""Validate canonical Pydantic schemas against all UZP training JSON files."""
import json
from pathlib import Path
import pytest
from backend.schemas import OdwolanieSchema, PelnomocnictwoSchema, PrzelewSchema
DATA_DIR = Path("data/uzp-hacknation-2026/dane")
@pytest.mark.skipif(not DATA_DIR.exists(), reason="UZP training dataset not cloned locally")
@pytest.mark.parametrize(
    ("filename", "model_cls"),
    [
        ("odwolanie.json", OdwolanieSchema),
        ("pelnomocnictwo.json", PelnomocnictwoSchema),
        ("przelew.json", PrzelewSchema),
    ],
)
def test_training_jsons_match_schema(filename: str, model_cls: type) -> None:
    paths = sorted(DATA_DIR.glob(f"przyklad_*/{filename}"))
    assert paths, f"No files found for {filename}"
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parsed = model_cls.model_validate(payload)
        assert parsed.model_dump(by_alias=True) == payload