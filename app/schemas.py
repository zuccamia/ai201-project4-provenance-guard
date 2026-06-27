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
    status: str = "classified"
