"""EU LLM API client for structured JSON extraction."""

from backend.core.models import DocumentSegment


async def extract_structured_data(segments: list[DocumentSegment]) -> dict:
    """Send classified segments to EU-hosted LLM API with enforced JSON schema."""
    # TODO: Call Mistral / Scaleway UE endpoint with structured output
    return {"segments_count": len(segments), "status": "stub"}
