import uuid
from datetime import datetime, timezone

from fastapi import FastAPI

from app.schemas import SubmitRequest, SubmitResponse

MIN_TEXT_CHARS = 100

app = FastAPI(title="Provenance Guard")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/submit", response_model=SubmitResponse)
async def submit(req: SubmitRequest) -> SubmitResponse:
    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if len(req.text.strip()) < MIN_TEXT_CHARS:
        return SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution="insufficient_text",
            status="classified",
        )

    # TODO: fan out to stylometry + LLM, calibrate, fuse, derive label, audit-log.
    return SubmitResponse(
        content_id=content_id,
        creator_id=req.creator_id,
        timestamp=timestamp,
        attribution="uncertain",
        confidence=0.5,
        stylometry_score=0.5,
        llm_score=0.5,
        status="classified",
    )
