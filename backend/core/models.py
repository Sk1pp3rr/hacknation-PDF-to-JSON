"""Internal pipeline domain models and DTOs."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """Supported document classes in an appeal case."""

    ODWOLANIE = "odwolanie"
    PELNOMOCNICTWO = "pelnomocnictwo"
    PRZELEW = "przelew"
    DOWOD_WPLATY = "przelew"
    INNY = "inny"


class DocumentSegment(BaseModel):
    """Classified text segment extracted from an input PDF."""

    doc_type: DocumentType
    text: str


class ExtractionResult(BaseModel):
    """Structured output returned by the API."""

    filename: str
    documents: list[DocumentSegment] = Field(default_factory=list)
    data: dict = Field(default_factory=dict, description="Docelowa struktura JSON zgodna ze schematem UZP")
