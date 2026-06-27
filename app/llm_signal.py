import json
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from groq import AsyncGroq

log = logging.getLogger(__name__)

Vote = Literal["low", "medium", "high"]

MODEL = "llama-3.3-70b-versatile"

# Call A: describe surface style. Must never use language that frames the text
# as authored by a human or an AI — that taint would leak into Call B's vote.
# Verbatim quoting matters: Call B can only recognize stock phrasings if Call A
# preserves them, not if it paraphrases them as "specific imagery".
PROMPT_A_SYSTEM = (
    "You are a style analyst. Read the provided text and describe only its "
    "observable surface features:\n"
    "- Sentence rhythm and length variation\n"
    "- Distinctive vocabulary — quote the 3-5 most notable individual words "
    "or short phrases VERBATIM (in single quotes)\n"
    "- Imagery — quote the imagery phrases VERBATIM, do not paraphrase\n"
    "- Punctuation habits, noting any em-dashes, semicolons, ellipses, "
    "ALL-CAPS words, or lowercase sentence starts\n"
    "- Any proper nouns (specific people, places, brands, dates)\n"
    "- Register mixing, fragments, contractions, or apparent imperfections\n"
    "- Transitions and connectives\n\n"
    "Be concrete and brief. Do not speculate about the author or audience. "
    "Do not classify the text or use the words 'human', 'AI', 'generated', "
    "'machine', 'model', or 'detection'. "
    "Return strict JSON: {\"observations\": [\"...\", \"...\"]}."
)

# Call B: vote from observations alone. It MUST NOT see the source text —
# that decoupling is the whole point. Self-reported confidence is discarded
# upstream, so we don't ask for it. Anchored rubric below: Call B knows
# which patterns push high vs low, Call A does not.
PROMPT_B_SYSTEM = (
    "You will receive a JSON list of style observations about a piece of "
    "writing. You did NOT see the writing itself. Based only on these "
    "observations, output a single coarse vote for how AI-generated the "
    "writing appears.\n\n"
    "Vote HIGH if observations mention any of:\n"
    "- Stock imagery phrases: 'scattered stars', 'whispers of', 'dance of "
    "light' or 'dance of shadow', 'ancient', 'eternal', 'golden hues', "
    "'embers', 'tapestry', 'gentle embrace'\n"
    "- Regular meter, end-stopped lines, or polished symmetric structure\n"
    "- Abstract universals or generic categories rather than concrete "
    "specifics\n"
    "- Polished prose with no awkward phrasings, fragments, or "
    "irregularities\n"
    "- Connectives like 'moreover', 'furthermore', 'it is important to "
    "note', 'in conclusion', 'delve into', 'in the realm of'\n\n"
    "Vote LOW if observations mention any of:\n"
    "- Proper nouns of specific real-sounding places, people, brands, "
    "or dates\n"
    "- Register mixing (formal alongside slang, contractions, fragments)\n"
    "- Voice quirks (lowercase 'i', ALL-CAPS for emphasis, em-dashes "
    "for asides, ellipses for trailing thoughts)\n"
    "- Surprising or idiosyncratic word choices outside common poetic or "
    "essay vocabulary\n"
    "- Awkwardness, hedging in unexpected places, or apparent "
    "imperfections\n\n"
    "Vote MEDIUM when observations describe imagery without revealing "
    "whether the phrasings are stock or fresh, OR when high and low "
    "signals are mixed.\n\n"
    "Return strict JSON: {\"vote\": \"low\" | \"medium\" | \"high\"}."
)


@dataclass
class LLMResult:
    vote: Optional[Vote]
    observations: list[str] = field(default_factory=list)


_client: Optional[AsyncGroq] = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def score(text: str) -> LLMResult:
    """Two-call LLM signal. Returns LLMResult(vote=None) if anything fails — the
    fusion layer treats that as a missing signal."""
    try:
        observations = await _call_a(text)
        vote = await _call_b(observations)
    except Exception as exc:
        # Silent in production, but log type+message so we can tell a missing
        # API key from a network blip from a malformed-JSON response.
        log.warning("llm_signal failed: %s: %s", type(exc).__name__, exc)
        return LLMResult(vote=None)
    if vote is None:
        log.warning("llm_signal: Call B returned no valid vote")
    return LLMResult(vote=vote, observations=observations)


async def _call_a(text: str) -> list[str]:
    client = _get_client()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_A_SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    obs = data.get("observations", [])
    return [str(x) for x in obs if isinstance(x, (str, int, float))]


async def _call_b(observations: list[str]) -> Optional[Vote]:
    client = _get_client()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_B_SYSTEM},
            {"role": "user", "content": json.dumps({"observations": observations})},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    vote = data.get("vote")
    if vote in ("low", "medium", "high"):
        return vote
    return None
