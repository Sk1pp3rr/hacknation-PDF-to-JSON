"""Normalization, checksum validation, and zero-token augmentation services."""

from backend.services.normalizer.augmenter import (
    FieldEvidence,
    annotate_odwolanie_header,
    augment_post_extraction,
    extract_header_city_and_date,
    locate_field_evidence,
    strip_kio_addressee,
    strip_person_titles,
)
from backend.services.normalizer.validator import (
    IdentifierHints,
    ValidationOutcome,
    extract_identifier_hints,
    is_valid_krs,
    is_valid_nip,
    is_valid_nrb,
    is_valid_pesel,
    is_valid_regon,
    normalize_date_str,
    normalize_payload,
    validate_extraction,
)

__all__ = [
    "FieldEvidence",
    "IdentifierHints",
    "ValidationOutcome",
    "annotate_odwolanie_header",
    "augment_post_extraction",
    "extract_header_city_and_date",
    "extract_identifier_hints",
    "is_valid_krs",
    "is_valid_nip",
    "is_valid_nrb",
    "is_valid_pesel",
    "is_valid_regon",
    "locate_field_evidence",
    "normalize_date_str",
    "normalize_payload",
    "strip_kio_addressee",
    "strip_person_titles",
    "validate_extraction",
]
