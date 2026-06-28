import math
from dataclasses import dataclass
from typing import Literal, Optional

FusedAttribution = Literal["likely_ai", "likely_human", "uncertain"]

# --- Calibration placeholders. Refit on the labeled validation set via
# scripts/calibrate.py whenever the prompts or feature set change. ---

# Platt logistic for stylometry: P(AI) = sigmoid(A * raw + B).
# Current values fit from data/ai_batch_*.jsonl + data/human_batch.jsonl on
# 2026-06-27 with the six-feature stylometry heuristic.
STYLOMETRY_PLATT_A = 2.1585
STYLOMETRY_PLATT_B = -0.6547

# LLM vote -> P(AI). Empirical bucket rates from the same calibration run.
# Will be re-fit after any prompt change in app/llm_signal.py.
LLM_BUCKET_P_AI: dict[str, float] = {"low": 0.15, "medium": 0.50, "high": 0.85}

# Current runtime weights for the two active signals.
WEIGHTS = {"stylometry": 0.60, "llm": 0.40}

# Attribution thresholds on the fused P(AI).
THRESHOLD_LOW = 0.35
THRESHOLD_HIGH = 0.65

# If the most disagreeing pair of available signals exceeds this gap, the
# attribution is forced to "uncertain" even when the weighted average sits
# in a confident band.
DISAGREE_THRESHOLD = 0.40


@dataclass
class FusedResult:
    attribution: FusedAttribution
    confidence: float
    stylometry_score: Optional[float]
    llm_score: Optional[float]


def calibrate_stylometry(raw: float) -> float:
    return _sigmoid(STYLOMETRY_PLATT_A * raw + STYLOMETRY_PLATT_B)


def calibrate_llm(vote: Optional[str]) -> Optional[float]:
    if vote is None:
        return None
    return LLM_BUCKET_P_AI.get(vote)


def fuse(
    p_sty: Optional[float],
    p_llm: Optional[float],
) -> FusedResult:
    contributions: list[tuple[float, float]] = []
    if p_sty is not None:
        contributions.append((WEIGHTS["stylometry"], p_sty))
    if p_llm is not None:
        contributions.append((WEIGHTS["llm"], p_llm))

    if not contributions:
        # No signal survived. Caller should not normally reach fuse() in this
        # state, but report neutral uncertainty rather than crash.
        return FusedResult("uncertain", 0.5, p_sty, p_llm)

    total_w = sum(w for w, _ in contributions)
    fused = sum(w * p for w, p in contributions) / total_w

    # Agreement override across all available signals: max-pairwise spread.
    # With one signal the spread is 0 and the override is a no-op.
    available_ps = [p for _, p in contributions]
    spread = max(available_ps) - min(available_ps)
    if len(available_ps) >= 2 and spread > DISAGREE_THRESHOLD:
        return FusedResult("uncertain", fused, p_sty, p_llm)

    if fused < THRESHOLD_LOW:
        attribution: FusedAttribution = "likely_human"
    elif fused > THRESHOLD_HIGH:
        attribution = "likely_ai"
    else:
        attribution = "uncertain"

    return FusedResult(attribution, fused, p_sty, p_llm)


def _sigmoid(z: float) -> float:
    # Numerically stable form so large |z| doesn't overflow exp().
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
