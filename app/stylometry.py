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
    raw_score: float


def score(text: str) -> StylometryFeatures:
    """Return stylometric features plus a raw 0-1 AI-likeness score (uncalibrated)."""
    stripped = text.strip()
    sentences = [s for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]
    if not sentences:
        return StylometryFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.5)

    sentence_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    sentence_lengths = [n for n in sentence_lengths if n > 0]
    if not sentence_lengths:
        return StylometryFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.5)

    words = [w.lower() for w in _WORD_RE.findall(stripped)]
    n_words = len(words)

    mean_len = statistics.fmean(sentence_lengths)
    std_len = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    # Burstiness in [-1, 1]: higher means more sentence-length variation (human-like).
    denom = std_len + mean_len
    burstiness = (std_len - mean_len) / denom if denom else 0.0

    fn_ratio = sum(1 for w in words if w in FUNCTION_WORDS) / n_words
    punct_density = len(_PUNCT_RE.findall(stripped)) / n_words

    # Heuristic raw score. Low burstiness, function-word ratio near a "polished" band,
    # and low punctuation diversity each push toward AI. Replaced by Platt calibration
    # once a labeled validation set is in place.
    low_burstiness = 1.0 - _clip01((burstiness + 1.0) / 2.0)
    polished_fn = 1.0 - min(1.0, abs(fn_ratio - 0.45) * 2.5)
    sparse_punct = 1.0 - _clip01(punct_density * 8.0)
    raw_score = _clip01(0.5 * low_burstiness + 0.3 * polished_fn + 0.2 * sparse_punct)

    return StylometryFeatures(
        burstiness=burstiness,
        sentence_length_mean=mean_len,
        sentence_length_std=std_len,
        function_word_ratio=fn_ratio,
        punctuation_density=punct_density,
        raw_score=raw_score,
    )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
