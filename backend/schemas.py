"""Pydantic schemas representing the canonical UZP JSON output."""

from enum import StrEnum
from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """Supported document classes in an appeal case."""

    ODWOLANIE = "odwolanie"
    PELNOMOCNICTWO = "pelnomocnictwo"
    DOWOD_WPLATY = "dowod_wplaty"
    INNY = "inny"


class DocumentSegment(BaseModel):
    """Classified text segment extracted from an input PDF."""

    doc_type: DocumentType
    text: str


class ExtractionResult(BaseModel):
    """Canonical structured output returned by the extraction pipeline."""

    filename: str
    documents: list[DocumentSegment] = Field(default_factory=list)
    data: dict = Field(default_factory=dict, description="Docelowa struktura JSON zgodna ze schematem UZP")
