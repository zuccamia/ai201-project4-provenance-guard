import uuid
from datetime import datetime, timezone

from fastapi import FastAPI

from app.audit import append as audit_append, read as audit_read
from app.schemas import SubmitRequest, SubmitResponse

MIN_TEXT_CHARS = 100

app = FastAPI(title="Provenance Guard")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/log")
async def log(limit: int = 100) -> dict[str, list[dict]]:
    return {"entries": audit_read(limit=limit)}


@app.post("/submit", response_model=SubmitResponse)
async def submit(req: SubmitRequest) -> SubmitResponse:
    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if len(req.text.strip()) < MIN_TEXT_CHARS:
        response = SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution="insufficient_text",
            status="classified",
        )
    else:
        # TODO: fan out to stylometry + LLM, calibrate, fuse, derive label.
        response = SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution="uncertain",
            confidence=0.5,
            stylometry_score=0.5,
            llm_score=0.5,
            status="classified",
        )

    audit_append(response.model_dump())
    return response
