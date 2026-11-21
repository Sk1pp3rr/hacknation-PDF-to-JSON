"""FastAPI entrypoint for Odwołania2json."""

from fastapi import FastAPI, File, UploadFile

from backend.config import settings
from backend.schemas import ExtractionResult
from backend.services.classifier import classify_and_segment
from backend.services.llm_extractor import extract_structured_data

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple liveness endpoint."""
    return {"status": "ok"}


@app.post("/api/extract", response_model=ExtractionResult)
async def extract_case(file: UploadFile = File(...)) -> ExtractionResult:
    """Accept a PDF file and return extracted canonical JSON (stub pipeline)."""
    _ = await file.read()
    # TODO: Replace placeholder text with extract_text_from_pdf(_)
    segments = classify_and_segment("")
    data = await extract_structured_data(segments)
    return ExtractionResult(
        filename=file.filename or "unknown.pdf",
        documents=segments,
        data=data,
    )
