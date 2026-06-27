# Provenance Guard

A service that classifies a piece of text as human-written or AI-generated and returns a calibrated confidence score, per-signal detail, and a transparency label. The output is a triage signal, not a final ruling about a person — false positives fall hardest on non-native and stylized writing, so the label and appeal path are core features.

## Architecture overview

A `POST /submit` flows through five stages:

1. **Length gate** (`app/main.py:33`) — text under 100 characters short-circuits to `attribution = insufficient_text`. Below that threshold no signal has enough material to be meaningful.
2. **Fan-out** — stylometry runs on CPU via `asyncio.to_thread`; the LLM signal makes two Groq calls. They run concurrently with `asyncio.gather`.
3. **Calibration + fusion** (`app/confidence.py`) — each raw signal is mapped to P(AI), then combined as a weighted average (0.60 stylometry, 0.40 LLM). If a signal is missing, its weight is dropped and the remainder is renormalized.
4. **Agreement override + thresholds** — if the two calibrated probabilities disagree by more than 0.40, attribution is forced to `uncertain`. Otherwise the fused score is thresholded: `< 0.35 → likely_human`, `> 0.65 → likely_ai`, else `uncertain`.
5. **Audit log + response** — one structured JSON line appended to `logs/audit.jsonl`, response returned with content_id, attribution, calibrated confidence, per-signal scores, and status.

`GET /log` exposes recent audit entries for grading visibility. `POST /appeal` (planned) records creator reasoning and updates the audit log.

## Detection signals

Two signals, picked to fail differently.

**Signal A — Stylometry** (`app/stylometry.py`). Mechanical CPU measurement. Six features: `cliche_density` (LLM-overused phrases per 100 words — "delve into", "in the realm of", "rich tapestry", "whispers of"…), `type_token_ratio`, `low_burstiness` (inverse of sentence-length variance), `contraction_rate`, `caps_irregularity` (ALL-CAPS + lowercase sentence starts), and `punctuation_diversity` (em-dash, semicolon, colon, ellipsis). The first three push toward AI; the last three push toward human.

*Why chosen:* deterministic, no API cost, fails in a different direction than the LLM signal. *What it misses:* voice-driven AI prose that avoids stock phrasings; very short text where per-100-word rates are noisy; classical human poetry whose formal features overlap with AI poetry.

**Signal B — LLM**, Groq `llama-3.3-70b-versatile` (`app/llm_signal.py`). Two decoupled calls. Call A reads the text and returns surface-style observations with distinctive vocabulary and imagery quoted verbatim; it is banned from using the words *human*, *AI*, *generated*, *machine*, *model*, *detection*. Call B sees only Call A's observations (never the source text) and votes `low`/`medium`/`high` against an anchored rubric that names specific cliché patterns and voice quirks. Self-reported confidence is discarded.

*Why chosen:* a coarse second opinion that catches things stylometry can't (semantic clichés, register). *What it misses:* modern AI prose with specific imagery — when the model can write voice-driven content, the same model can't detect it.

## Confidence scoring

Each raw signal is calibrated to P(AI):
- Stylometry: Platt logistic `sigmoid(A · raw + B)`, with `A`, `B` fit on the labeled set by max-likelihood gradient ascent.
- LLM: empirical bucket rates `P(AI | vote)` counted directly from the labeled set.

Both fit by `scripts/calibrate.py` against `data/ai_batch_*.jsonl` (Claude/GPT/Gemini outputs, ~90 rows) + `data/human_batch.jsonl` (51 mixed-source human rows). The script also picks attribution thresholds from the fused-score distributions: `THRESHOLD_LOW` = 75th percentile of human fused, `THRESHOLD_HIGH` = 25th percentile of AI fused — so the uncertain band tracks where the distributions actually overlap.

Validation is honest: the script prints the confusion matrix under the fitted constants. If `THRESHOLD_LOW > THRESHOLD_HIGH`, it warns that the signals don't cleanly separate the dataset.

**Fusion:**

```
P_fused = (w_sty · P_sty + w_llm · P_llm) / (w_sty + w_llm)        # weights = {0.60, 0.40}
if |P_sty − P_llm| > 0.40:  attribution = "uncertain"               # disagreement override
```

### Example submissions

(Scores to be filled in after the next `scripts/calibrate.py` run and real `/submit` calls against the deployed app.)

**High-confidence AI** — a stock-imagery poem:
```json
{
  "text": "City lights like scattered stars below, a thousand windows glowing in the dark...",
  "attribution": "likely_ai",
  "confidence": <TBD>,
  "stylometry_score": <TBD>,
  "llm_score": <TBD>
}
```
Both signals expected to agree. Spread = <TBD>, expected well under the 0.40 override threshold.

**Lower-confidence (uncertain)** — short casual prose:
```json
{
  "text": "ok so i finally tried that new ramen place downtown and honestly? underwhelming...",
  "attribution": "uncertain",
  "confidence": <TBD>,
  "stylometry_score": <TBD>,
  "llm_score": <TBD>
}
```
55-word casual review. Stylometry's cliche feature is 0 (no clichés in casual prose); LLM correctly votes `low`. Fused score expected to land in the uncertain band or trigger the spread override. System refuses to commit, which is the honest call.

## Transparency label

Derived from `attribution` and shown verbatim to the user:

| attribution | label text |
|---|---|
| `likely_ai` | "Likely AI-generated (confidence: <TBD>%). This is a triage signal, not a final ruling. If this assessment is wrong, you can appeal." |
| `likely_human` | "Likely human-written (confidence: <TBD>%)." |
| `uncertain` | "Could not determine with confidence (P(AI) = <TBD>%). The two detection signals disagreed or the score is borderline. No attribution is being made." |
| `insufficient_text` | "Text too short to evaluate — need at least 100 characters." |

Each label embeds the calibrated probability so a reader can sanity-check the call rather than trust the verdict on faith.

## Rate limiting

Planned (to be wired via `slowapi`), with the following limits and reasoning:

| endpoint | limit | reasoning |
|---|---|---|
| `POST /submit` | 10 / min per creator_id | Each call costs 2 Groq requests. A human-driven workflow rarely exceeds this; the cap contains accidental loops and bulk abuse without blocking legitimate testing. |
| `POST /appeal` | 5 / min per creator_id | Appeals require human-written reasoning; rate-limit lower to discourage automation. |
| `GET /log`, `GET /health` | 60 / min per IP | Read-only and cheap; loose limit just to deter scraping. |

## Known limitations

**Short casual prose** is the worst case. A 50–80 word review with lowercase `i`, occasional contractions, and no clichés sits between distributions: stylometry's strongest feature (`cliche_density`) returns 0, per-100-word rates have high variance, and the LLM struggles because it has too few sentences to characterize. These submissions tend to land in `uncertain` — honest but not useful.

**Non-native English writing** carries the highest false-positive risk for AI labels. Many features that flag AI (consistent register, fewer contractions, careful transitions) also describe a thoughtful second-language writer. The appeal path exists specifically because this failure mode falls on real users.

## Spec reflection

**Where the spec helped:** the two-decoupled-LLM-call architecture made debug-driven prompt iteration possible. When `scripts/inspect_llm.py` revealed Call A was treating AI clichés ("scattered stars", "glowing windows") as "specific imagery", we could fix Call B's rubric in isolation without retraining anything. The decoupling kept the rationale honest enough to inspect.

**Where the implementation diverged:** the spec's stylometry features (burstiness, function-word ratio, punctuation density) turned out to be *anti-correlated* with AI on our labeled set — the first Platt fit came back with `A = −1.22`. The data was saying the heuristic was reading the wrong direction, because the human batch is heavy on classical poetry (formal meter, dense punctuation) and the AI batch is mostly modern free verse. We diverged by adding `cliche_density`, `caps_irregularity`, `contraction_rate`, and `punctuation_diversity` — features that discriminate across genres rather than within poetic register. Re-fitting yielded `A = +2.16`, the intended direction. A separate bug fix: `max(n_words / 100.0, 1.0)` was floored too aggressively, so per-100-word rates underreported for any text under 100 words; removing the floor moved short-text scores closer to neutral.

## AI usage

I used Claude (Opus 4.7) as a pair programmer throughout. `planning.md` was the source of truth; Claude generated implementation, and I revised whenever empirical results contradicted what Claude initially produced.

**Instance 1 — Stylometry heuristic.** I directed Claude to implement Signal A per `planning.md` (burstiness, function-word ratio, punctuation density). Claude wrote a hand-tuned combination with magic constants (weights `0.5/0.3/0.2`, multipliers `*2.5` and `*8`). After the first calibration came back with a negative Platt slope, I overrode the feature set: directed Claude to add `cliche_density`, `type_token_ratio`, `contraction_rate`, `caps_irregularity`, and `punctuation_diversity`, and rewrote the raw-score combination. I also caught a `max(per_100, 1.0)` denominator floor Claude introduced and had it removed.

**Instance 2 — LLM prompt non-discrimination.** I directed Claude to write the two-call LLM signal per the spec. Initial calibration showed flat bucket rates (`P(AI | low/medium/high) ≈ 0.63 / 0.60 / 0.67`) — the LLM was predicting the dataset prior, not the label. I directed Claude to build `scripts/inspect_llm.py` to surface what Call A was actually producing. The inspector revealed Call A framed AI clichés ("scattered stars", "glowing windows") as "specific imagery" — *positive* descriptors that Call B was reading as human-leaning. I overrode the original prompts: required Call A to quote distinctive vocabulary and imagery *verbatim* so Call B could pattern-match cliché strings, and added an explicit anchored rubric to Call B naming specific stock-imagery patterns and voice quirks. Re-running `scripts/calibrate.py --rebuild` after that change is the immediate next step to validate the fix.
