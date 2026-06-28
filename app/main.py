import asyncio
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import confidence, label, llm_signal, stylometry
from app.audit import (
    append as audit_append,
    find_by_content_id as audit_find,
    read as audit_read,
)
from app.schemas import AppealRequest, AppealResponse, SubmitRequest, SubmitResponse

load_dotenv()

# Word-based gate for the current two-signal runtime. Very short text is noisy
# for both stylometry and the LLM signal, so we short-circuit early.
MIN_TEXT_WORDS = 50

# Keyed by client IP. In-memory store is fine for a single-process deployment;
# if we ever run multi-worker, swap this to a Redis-backed limiter.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Provenance Guard")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/log")
async def log(limit: int = 100) -> dict[str, list[dict]]:
    return {"entries": audit_read(limit=limit)}


@app.post("/submit", response_model=SubmitResponse)
@limiter.limit("10/minute")
@limiter.limit("100/hour")
async def submit(request: Request, req: SubmitRequest) -> SubmitResponse:
    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if len(req.text.split()) < MIN_TEXT_WORDS:
        attribution = "insufficient_text"
        response = SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution=attribution,
            label_text=label.derive(attribution),
            status="classified",
        )
    else:
        # Fan out the two runtime signals concurrently: stylometry (CPU) and
        # the LLM vote (Groq HTTP). Slowest of the two sets the latency.
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
            label_text=label.derive(fused.attribution),
            status="classified",
        )

    audit_append(response.model_dump())
    return response


@app.post("/appeal", response_model=AppealResponse)
async def appeal(req: AppealRequest) -> AppealResponse:
    if not audit_find(req.content_id):
        raise HTTPException(status_code=404, detail=f"content_id not found: {req.content_id}")

    appeal_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    status = "under_review"

    # Append a new entry rather than mutating the original — the audit log is
    # append-only and the original classification must stay intact.
    audit_append({
        "event": "appeal",
        "content_id": req.content_id,
        "appeal_id": appeal_id,
        "timestamp": timestamp,
        "appeal_reasoning": req.appeal_reasoning,
        "status": status,
    })

    return AppealResponse(
        content_id=req.content_id,
        appeal_id=appeal_id,
        timestamp=timestamp,
        status=status,
    )
