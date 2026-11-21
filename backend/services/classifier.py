"""Document classification and segmentation module."""

from backend.schemas import DocumentSegment, DocumentType


def classify_and_segment(raw_text: str) -> list[DocumentSegment]:
    """Split raw case text into classified document segments."""
    # TODO: Implement feature-matrix / classifier segmentation
    return [DocumentSegment(doc_type=DocumentType.INNY, text=raw_text)]
