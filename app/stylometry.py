import re
import statistics
from dataclasses import dataclass


FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "to", "in",
    "on", "at", "by", "for", "with", "from", "as", "into", "about", "over",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "there", "here",
    "not", "no", "so", "than", "then", "though", "because", "when", "where",
    "what", "which", "who", "whom", "how", "why",
})

# Phrases empirically overused by modern instruction-tuned LLMs across genres.
# This is a high-precision starter set, not exhaustive — add markers you spot
# in your own outputs. Multi-word patterns are safer than bare words like
# "ember" or "golden" which over-flag human poetry.
AI_CLICHE_PATTERNS = [
    # Essay / blog register
    r"\bdelve(?:s|d|ing)?\s+into\b",
    r"\bin the realm of\b",
    r"\bnavigate(?:s|d|ing)?\s+the\s+(?:complex|nuanc|intric)",
    r"\brich tapestry\b",
    r"\bintricate(?:ly)?\b",
    r"\bmultifaceted\b",
    r"\bunderscore(?:s|d|ing)?\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\bnonetheless\b",
    r"\bdelicate (?:balance|dance|interplay)\b",
    r"\bstands? as a\b",
    r"\bplays? a (?:crucial|pivotal|significant|vital) role\b",
    r"\bit(?:'?s| is) (?:important|worth) (?:to note|noting)\b",
    r"\bin conclusion\b",
    r"\bcornerstone of\b",
    r"\btestament to\b",
    r"\bmyriad\b",
    r"\bplethora\b",
    # Poetry register
    r"\bwhisper(?:s|ed|ing)?\s+of\b",
    r"\bdance of (?:light|shadow|color)\b",
    r"\bgolden (?:hue|light|glow|warmth)\b",
    r"\bgentle (?:embrace|reminder|whisper)\b",
    r"\beternal (?:dance|embrace|whisper)\b",
]
AI_CLICHE_RE = re.compile("|".join(f"(?:{p})" for p in AI_CLICHE_PATTERNS), re.IGNORECASE)

# Apostrophe-contractions (won't, you're, didn't, weather'd). Common in human
# casual prose and in older verse; rare in instruction-tuned LLM output.
CONTRACTION_RE = re.compile(r"\b[A-Za-z]+'[a-z]+\b")

# Punctuation marks beyond the basic period/comma. Diverse punctuation use is
# a steady human marker across formal and informal writing.
DIVERSE_PUNCT_RE = re.compile(r"—|–|;|:|\.{3}|\([^)]+\)")

# ALL-CAPS words (length >= 2) used mid-prose for emphasis — almost only
# happens in informal human writing.
ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[\w']+\b")
_PUNCT_RE = re.compile(r"[.,;:!?\-—\"'()]")


@dataclass
class StylometryFeatures:
    burstiness: float
    sentence_length_mean: float
    sentence_length_std: float
    function_word_ratio: float
    punctuation_density: float
    cliche_density: float
    type_token_ratio: float
    contraction_rate: float
    caps_irregularity: float
    punctuation_diversity: float
    raw_score: float


def score(text: str) -> StylometryFeatures:
    """Return stylometric features plus a raw 0-1 AI-likeness score (uncalibrated)."""
    stripped = text.strip()
    sentences = [s for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]
    if not sentences:
        return _empty_features()

    sentence_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    sentence_lengths = [n for n in sentence_lengths if n > 0]
    if not sentence_lengths:
        return _empty_features()

    words_orig = _WORD_RE.findall(stripped)
    words_lower = [w.lower() for w in words_orig]
    n_words = max(len(words_lower), 1)
    per_100 = n_words / 100.0  # denominator for per-100-words rates

    mean_len = statistics.fmean(sentence_lengths)
    std_len = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    denom = std_len + mean_len
    burstiness = (std_len - mean_len) / denom if denom else 0.0

    fn_ratio = sum(1 for w in words_lower if w in FUNCTION_WORDS) / n_words
    punct_density = len(_PUNCT_RE.findall(stripped)) / n_words

    cliche_density = len(AI_CLICHE_RE.findall(stripped)) / per_100
    type_token_ratio = len(set(words_lower)) / n_words
    contraction_rate = len(CONTRACTION_RE.findall(stripped)) / per_100

    all_caps = len(ALL_CAPS_RE.findall(stripped))
    lowercase_sentence_starts = sum(
        1 for s in sentences if s.strip() and s.strip()[0].islower()
    )
    caps_irregularity = (all_caps + lowercase_sentence_starts) / per_100

    punctuation_diversity = float(len(set(DIVERSE_PUNCT_RE.findall(stripped))))

    # All six contributions are normalized to [0, 1] in the AI-leaning
    # direction, so the weighted sum stays in [0, 1]. Platt calibration in
    # confidence.py refits the final probability, so these weights only need
    # to be *roughly* right.
    low_burstiness = 1.0 - _clip01((burstiness + 1.0) / 2.0)
    cliche_norm = min(cliche_density / 2.0, 1.0)
    contraction_norm = min(contraction_rate / 3.0, 1.0)
    caps_norm = min(caps_irregularity / 2.0, 1.0)
    punct_div_norm = min(punctuation_diversity / 3.0, 1.0)

    raw_score = _clip01(
        0.35 * cliche_norm
        + 0.15 * type_token_ratio
        + 0.10 * low_burstiness
        + 0.15 * (1.0 - contraction_norm)
        + 0.15 * (1.0 - caps_norm)
        + 0.10 * (1.0 - punct_div_norm)
    )

    return StylometryFeatures(
        burstiness=burstiness,
        sentence_length_mean=mean_len,
        sentence_length_std=std_len,
        function_word_ratio=fn_ratio,
        punctuation_density=punct_density,
        cliche_density=cliche_density,
        type_token_ratio=type_token_ratio,
        contraction_rate=contraction_rate,
        caps_irregularity=caps_irregularity,
        punctuation_diversity=punctuation_diversity,
        raw_score=raw_score,
    )


def _empty_features() -> StylometryFeatures:
    return StylometryFeatures(
        burstiness=0.0,
        sentence_length_mean=0.0,
        sentence_length_std=0.0,
        function_word_ratio=0.0,
        punctuation_density=0.0,
        cliche_density=0.0,
        type_token_ratio=0.0,
        contraction_rate=0.0,
        caps_irregularity=0.0,
        punctuation_diversity=0.0,
        raw_score=0.5,
    )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
