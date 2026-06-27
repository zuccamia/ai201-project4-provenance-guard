import asyncio
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI

from app import confidence, llm_signal, stylometry
from app.audit import append as audit_append, read as audit_read
from app.schemas import SubmitRequest, SubmitResponse

load_dotenv()

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
        # Fan out: stylometry on CPU, LLM over HTTP. asyncio.to_thread lets the
        # sync stylometry call yield the loop while the Groq round-trip runs.
        sty_features, llm_result = await asyncio.gather(
            asyncio.to_thread(stylometry.score, req.text),
            llm_signal.score(req.text),
        )

        p_sty = confidence.calibrate_stylometry(sty_features.raw_score)
        p_llm = confidence.calibrate_llm(llm_result.vote)
        fused = confidence.fuse(p_sty, p_llm)

        response = SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution=fused.attribution,
            confidence=round(fused.confidence, 4),
            stylometry_score=round(fused.stylometry_score, 4) if fused.stylometry_score is not None else None,
            llm_score=round(fused.llm_score, 4) if fused.llm_score is not None else None,
            status="classified",
        )

    audit_append(response.model_dump())
    return response
