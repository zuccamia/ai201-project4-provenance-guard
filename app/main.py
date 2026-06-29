import asyncio
import logging
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import analytics, binoculars_signal, confidence, label, llm_signal, provenance, stylometry
from app.audit import (
    append as audit_append,
    find_by_content_id as audit_find,
    read as audit_read,
)
from app.schemas import (
    AppealRequest,
    AppealResponse,
    SubmitRequest,
    SubmitResponse,
    VerifyRequest,
    VerifyResponse,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Global minimum for any classification attempt.
MIN_TEXT_CHARS = 100

# HF Binoculars endpoint needs longer text than the local two-signal runtime.
MIN_BINOCULARS_WORDS = 65

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


@app.get("/analytics")
async def analytics_summary() -> dict:
    return analytics.summarize(audit_read(limit=None))


@app.get("/analytics/view", response_class=HTMLResponse)
async def analytics_view() -> HTMLResponse:
    summary = analytics.summarize(audit_read(limit=None))
    return HTMLResponse(content=analytics.render_html(summary))


@app.get("/badge/{creator_id}", response_class=HTMLResponse)
async def badge_view(creator_id: str) -> HTMLResponse:
    creator_status = provenance.get_creator_status(creator_id)
    status = creator_status.get("status")
    if status == "verified_human":
        text = "Verified human creator"
        bg = "#e8f7ec"
        fg = "#166534"
        border = "#86efac"
    elif status == "pending":
        text = "Verification pending"
        bg = "#fff7ed"
        fg = "#9a3412"
        border = "#fdba74"
    else:
        text = "Not verified"
        bg = "#f3f4f6"
        fg = "#374151"
        border = "#d1d5db"
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <style>
    html, body {{ margin: 0; padding: 0; background: transparent; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .badge {{
      display: inline-flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 0.25rem;
      padding: 0.55rem 0.75rem;
      border-radius: 14px;
      border: 1px solid {border};
      background: {bg};
      color: {fg};
      min-width: 220px;
      box-sizing: border-box;
    }}
    .topline {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font: 600 14px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
      white-space: nowrap;
    }}
    .dot {{
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 999px;
      background: {fg};
      display: inline-block;
    }}
    .creator {{
      font: 500 12px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
      color: #111827;
    }}
    .issuer {{
      font: 400 11px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <span class=\"badge\">
    <span class=\"topline\"><span class=\"dot\"></span>{text}</span>
    <span class=\"creator\">Creator: {creator_id}</span>
    <span class=\"issuer\">Issued by Provenance Guard</span>
  </span>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/verify/request", response_model=VerifyResponse)
async def verify_request(req: VerifyRequest) -> VerifyResponse:
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "creator_id": req.creator_id,
        "status": "pending",
        "reason": req.reason,
        "requested_at": timestamp,
        "method": "manual_review",
    }
    provenance.append_status(entry)
    audit_append({"event": "verify_request", **entry})
    return VerifyResponse(creator_id=req.creator_id, status="pending")


@app.post("/verify/approve", response_model=VerifyResponse)
async def verify_approve(req: VerifyRequest) -> VerifyResponse:
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "creator_id": req.creator_id,
        "status": "verified_human",
        "reason": req.reason,
        "issued_at": timestamp,
        "method": "manual_review",
    }
    provenance.append_status(entry)
    audit_append({"event": "verify_approve", **entry})
    return VerifyResponse(creator_id=req.creator_id, status="verified_human", issued_at=timestamp)


@app.post("/submit", response_model=SubmitResponse)
@limiter.limit("10/minute")
@limiter.limit("100/hour")
async def submit(request: Request, req: SubmitRequest) -> SubmitResponse:
    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    creator_status = provenance.get_creator_status(req.creator_id)
    verification_status = creator_status.get("status")
    provenance_badge = "Verified human creator" if verification_status == "verified_human" else None

    if len(req.text.strip()) < MIN_TEXT_CHARS:
        attribution = "insufficient_text"
        response = SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution=attribution,
            verification_status=verification_status,
            provenance_badge=provenance_badge,
            label_text=label.derive(attribution),
            status="classified",
        )
    else:
        binoculars_eligible = len(req.text.split()) >= MIN_BINOCULARS_WORDS
        if binoculars_eligible:
            sty_features, llm_result, bino_result = await asyncio.gather(
                asyncio.to_thread(stylometry.score, req.text),
                llm_signal.score(req.text),
                binoculars_signal.score(req.text),
            )
        else:
            logger.info(
                "binoculars skipped: text below %d-word minimum for HF endpoint; falling back to 2-signal fusion",
                MIN_BINOCULARS_WORDS,
            )
            sty_features, llm_result = await asyncio.gather(
                asyncio.to_thread(stylometry.score, req.text),
                llm_signal.score(req.text),
            )
            bino_result = binoculars_signal.BinocularsResult(tier=None)

        p_sty = confidence.calibrate_stylometry(sty_features.raw_score)
        p_llm = confidence.calibrate_llm(llm_result.vote)
        p_bino = confidence.calibrate_binoculars(bino_result.tier)
        if p_bino is None:
            if binoculars_eligible:
                logger.warning("binoculars unavailable; falling back to 2-signal fusion")
            fused = confidence.fuse(p_sty, p_llm, weight_profile="two_signal")
        else:
            fused = confidence.fuse(p_sty, p_llm, p_bino, weight_profile="three_signal")

        response = SubmitResponse(
            content_id=content_id,
            creator_id=req.creator_id,
            timestamp=timestamp,
            attribution=fused.attribution,
            confidence=round(fused.confidence, 4),
            stylometry_score=round(fused.stylometry_score, 4) if fused.stylometry_score is not None else None,
            llm_score=round(fused.llm_score, 4) if fused.llm_score is not None else None,
            binoculars_score=round(fused.binoculars_score, 4) if fused.binoculars_score is not None else None,
            binoculars_tier=bino_result.tier,
            verification_status=verification_status,
            provenance_badge=provenance_badge,
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
