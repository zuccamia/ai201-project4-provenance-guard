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
    "- Distinctive vocabulary — quote 2-4 exact words or short phrases "
    "VERBATIM in single quotes\n"
    "- Imagery — quote 1-3 imagery phrases VERBATIM in single quotes; do not "
    "paraphrase them as merely vivid or specific\n"
    "- Punctuation habits, noting any em-dashes, semicolons, ellipses, "
    "ALL-CAPS words, or lowercase sentence starts\n"
    "- Concrete specifics: note any proper nouns, named places, brands, dates, "
    "or other unmistakably specific real-world references\n"
    "- Irregularities: note fragments, contractions, slang, register mixing, "
    "awkward phrasing, or other imperfections\n"
    "- Transitions/connectives only if they are explicitly present — quote them "
    "VERBATIM\n\n"
    "Return 4-7 short bullet-like observations total. Prefer quoted evidence "
    "over summary adjectives. Do not speculate about the author or audience. "
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
    "Decision rule:\n"
    "- Vote HIGH only when the observations contain positive machine-like "
    "evidence, especially quoted stock phrases, generic abstractions, cliché "
    "imagery, or formulaic transitions.\n"
    "- Vote LOW only when the observations contain positive human-leaning "
    "evidence, especially concrete named specifics or genuinely idiosyncratic "
    "details.\n"
    "- Vote MEDIUM when evidence is mixed, weak, too generic, or based mainly "
    "on overall polish/formality. When in doubt, choose MEDIUM.\n\n"
    "HIGH evidence includes:\n"
    "- Quoted stock imagery or overused LLM phrases such as 'scattered stars', "
    "'whispers of', 'dance of light', 'dance of shadow', 'ancient', 'eternal', "
    "'golden hues', 'embers', 'tapestry', 'gentle embrace', 'delve into', 'in "
    "the realm of', 'it is important to note', or 'in conclusion'\n"
    "- Generic abstractions or universal categories with few concrete specifics\n"
    "- Formulaic connective-heavy exposition\n"
    "- Polished regularity PLUS cliché phrasing\n\n"
    "LOW evidence includes:\n"
    "- Specific proper nouns, dates, brands, or real-world references\n"
    "- Idiosyncratic or surprising wording that does not sound stock\n"
    "- Concrete scene details that feel particular rather than generic\n"
    "- Irregularities or register mixing may support LOW, but are not enough by "
    "themselves\n\n"
    "Fairness rule: careful, grammatical, or non-native English is NOT evidence "
    "of AI by itself. Lack of slang, contractions, or obvious voice quirks is "
    "NOT enough for HIGH.\n\n"
    "Important: regular meter, polished structure, poetic language, or clean "
    "prose ALONE are not enough for HIGH, because human poetry and careful "
    "second-language writing can have those too. Do not infer HIGH from polish "
    "alone.\n\n"
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
