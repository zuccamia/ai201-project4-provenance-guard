"""Transparency-label text derived from attribution. Exact wording per
planning.md §"Transparency label design"."""
from app.schemas import Attribution

_LABELS: dict[str, str] = {
    "likely_ai": (
        "Likely AI-generated. Our automated checks agree this text shows "
        "machine-generation patterns. This is an automated estimate, not "
        "proof. You can appeal this result."
    ),
    "likely_human": (
        "Likely human-written. Our automated checks agree this text shows "
        "human-writing patterns. This is an automated estimate, not proof."
    ),
    "uncertain": (
        "Inconclusive. Our checks did not agree, so we are not confident "
        "either way. Treat this as no result, not as a verdict. This case "
        "is a good candidate for human review."
    ),
    "insufficient_text": (
        "Not enough text to analyze reliably. Please submit a longer passage."
    ),
}


def derive(attribution: Attribution) -> str:
    return _LABELS[attribution]
