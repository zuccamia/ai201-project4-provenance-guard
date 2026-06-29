from typing import Literal, Optional

from pydantic import BaseModel, Field


Attribution = Literal["likely_ai", "likely_human", "uncertain", "insufficient_text"]


class SubmitRequest(BaseModel):
    text: str = Field(..., description="Content to analyze.")
    creator_id: str = Field(..., description="Submitter identifier.")


class SubmitResponse(BaseModel):
    content_id: str
    creator_id: str
    timestamp: str
    attribution: Attribution
    confidence: Optional[float] = None
    stylometry_score: Optional[float] = None
    llm_score: Optional[float] = None
    binoculars_score: Optional[float] = None
    binoculars_tier: Optional[str] = None
    verification_status: Optional[str] = None
    provenance_badge: Optional[str] = None
    label_text: str
    status: str = "classified"


class AppealRequest(BaseModel):
    content_id: str = Field(..., description="content_id from a prior /submit response.")
    appeal_reasoning: str = Field(..., min_length=1, max_length=2000,
                                   description="Why the creator believes the result is wrong.")


class AppealResponse(BaseModel):
    content_id: str
    appeal_id: str
    timestamp: str
    status: str = "under_review"


class VerifyRequest(BaseModel):
    creator_id: str = Field(..., description="Creator identifier requesting verification.")
    reason: str = Field(..., min_length=1, max_length=1000,
                        description="Why the creator is requesting verification.")


class VerifyResponse(BaseModel):
    creator_id: str
    status: str
    issued_at: Optional[str] = None
