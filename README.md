# Provenance Guard

A service that classifies a piece of text as human-written or AI-generated and returns a calibrated confidence score, per-signal detail, and a transparency label. The output is a triage signal, not a final ruling about a person — false positives fall hardest on non-native and stylized writing, so the label and appeal path are core features.

## Architecture overview

```text
                              POST /submit  (raw text)
                                     |
                                     v
                          +---------------------+
                          |     Length Gate     |
                          |  <100 chars =>      |
                          |  insufficient_text  |
                          +----+-----------+----+
              too short        |           |  raw text (length OK)
              short-circuit <---+           v
                              fans out to 2 or 3 signals (async)
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
  +----------------+         +----------------------+   +----------------------+
  | Stylometry     |         | LLM (Groq)           |   | Binoculars HF Space |
  | CPU, local     |         | Call A: observations |   | default-on for      |
  | raw_score      |         | Call B: low/med/high |   | 65+ words; cached   |
  +-------+--------+         +----------+-----------+   | tier-only response   |
          |                             |               | fallback to 2-signal |
          | p_sty                       | p_llm         +----------+-----------+
          |                             |                          | p_bino
          +-----------------------------+--------------------------+
                                        v
                           +----------------------------+
                           | Confidence Scoring         |
                           | calibrate -> P(AI)         |
                           | fuse by weight profile     |
                           | disagreement override      |
                           | thresholds: 0.35 / 0.65    |
                           +-------------+--------------+
                                         |
                                         v
                           +----------------------------+
                           | Transparency Label         |
                           | derive from attribution    |
                           +-------------+--------------+
                                         |
                                         v
                           +----------------------------+
                           | Audit Log                  |
                           | append-only JSONL          |
                           +-------------+--------------+
                                         |
                                         v
                           +----------------------------+
                           | Response                   |
                           | content_id, attribution,   |
                           | confidence, per-signal     |
                           | scores, label_text, status |
                           +----------------------------+
```

`GET /log` exposes recent audit entries for grading visibility. `POST /appeal` (`app/main.py:87`) accepts a `content_id` and `appeal_reasoning`, appends an appeal record to the audit log, and flips the content's status to `under_review` — original attribution and scores are never overwritten.

## APIs

OpenAPI-style summary of the exposed endpoints:

### `POST /submit`
- **Summary:** Classify a text submission and return fused attribution plus per-signal detail.
- **Rate limits:** `10/minute` and `100/hour` per IP.
- **Request body:**

```json
{
  "text": "Content to analyze.",
  "creator_id": "user-123"
}
```

- **Response `200`:**

```json
{
  "content_id": "uuid",
  "creator_id": "user-123",
  "timestamp": "2026-06-29T00:54:59.988117+00:00",
  "attribution": "uncertain",
  "confidence": 0.2836,
  "stylometry_score": 0.6553,
  "llm_score": 0.1500,
  "binoculars_score": 0.2000,
  "binoculars_tier": "likely_human",
  "verification_status": "verified_human",
  "provenance_badge": "Verified human creator",
  "label_text": "Inconclusive. Our checks did not agree, so we are not confident either way. Treat this as no result, not as a verdict. This case is a good candidate for human review.",
  "status": "classified"
}
```

- **Response semantics:**
  - `attribution ∈ {likely_ai, likely_human, uncertain, insufficient_text}`
  - `confidence` is the fused calibrated `P(AI)` when classification runs; `null` for `insufficient_text`
  - `binoculars_score` / `binoculars_tier` are `null` when Signal C is skipped or unavailable
  - `verification_status` / `provenance_badge` reflect creator-level provenance credential state, not the detector verdict

### `POST /appeal`
- **Summary:** Append an appeal event for an earlier classification.
- **Request body:**

```json
{
  "content_id": "uuid-from-submit",
  "appeal_reasoning": "I wrote this myself; English is my second language."
}
```

- **Response `200`:**

```json
{
  "content_id": "uuid-from-submit",
  "appeal_id": "appeal-uuid",
  "timestamp": "2026-06-29T01:00:00+00:00",
  "status": "under_review"
}
```

- **Response `404`:** when `content_id` is not found in the audit log.

### `POST /verify/request`
- **Summary:** Request a manual-review provenance certificate for a creator.
- **Request body:**

```json
{
  "creator_id": "user-123",
  "reason": "I want my original writing to display a verified-human badge."
}
```

- **Response `200`:**

```json
{
  "creator_id": "user-123",
  "status": "pending",
  "issued_at": null
}
```

### `POST /verify/approve`
- **Summary:** Demo/admin endpoint to approve a creator as `verified_human`.
- **Request body:**

```json
{
  "creator_id": "user-123",
  "reason": "Manual review completed."
}
```

- **Response `200`:**

```json
{
  "creator_id": "user-123",
  "status": "verified_human",
  "issued_at": "2026-06-29T02:00:00+00:00"
}
```

The resulting certificate is displayed on later `/submit` responses as `verification_status` and `provenance_badge`.

### `GET /analytics`
- **Summary:** Return aggregate analytics derived from the append-only audit log.
- **Response `200`:**

```json
{
  "detections": {
    "likely_ai": 0,
    "likely_human": 0,
    "uncertain": 12,
    "insufficient_text": 2,
    "total_classified": 14
  },
  "appeals": {
    "total_appeals": 0,
    "overall_appeal_rate": 0.0,
    "by_original_attribution": {
      "likely_ai": 0.0,
      "likely_human": 0.0,
      "uncertain": 0.0
    }
  },
  "signal_c_usage": {
    "eligible_submissions": 12,
    "used_binoculars": 5,
    "fallback_to_2_signal": 7,
    "binoculars_usage_rate": 0.4167
  }
}
```

### `GET /analytics/view`
- **Summary:** Render a simple HTML dashboard for the same analytics summary.
- **Response `200`:** server-rendered HTML page with cards and tables for detection patterns, appeals, and Signal C usage.

### `GET /log`
- **Summary:** Return recent audit-log entries for inspection.
- **Query parameters:**
  - `limit` (`integer`, optional, default `100`) — maximum number of entries to return.
- **Response `200`:**

```json
{
  "entries": [
    {
      "content_id": "...",
      "creator_id": "...",
      "timestamp": "...",
      "attribution": "uncertain",
      "confidence": 0.51,
      "stylometry_score": 0.67,
      "llm_score": 0.15,
      "binoculars_score": 0.20,
      "binoculars_tier": "likely_human",
      "label_text": "...",
      "status": "classified"
    }
  ]
}
```

### `GET /health`
- **Summary:** Liveness check.
- **Response `200`:**

```json
{ "status": "ok" }
```

## Detection signals

Three runtime signals are wired into the orchestrator for the current demo, picked to fail differently. Signal C is best-effort: when the Hugging Face Space is unavailable or the text is too short for that endpoint, the service falls back to the 2-signal system and logs the fallback.

**Signal A — Stylometry** (`app/stylometry.py`). Mechanical CPU measurement. Six features: `cliche_density` (LLM-overused phrases per 100 words — "delve into", "in the realm of", "rich tapestry", "whispers of"…), `type_token_ratio`, `low_burstiness` (inverse of sentence-length variance), `contraction_rate`, `caps_irregularity` (ALL-CAPS + lowercase sentence starts), and `punctuation_diversity` (em-dash, semicolon, colon, ellipsis). The first three push toward AI; the last three push toward human.

*Why chosen:* deterministic, no API cost, fails in a different direction than the LLM signal. *What it misses:* voice-driven AI prose that avoids stock phrasings; very short text where per-100-word rates are noisy; classical human poetry whose formal features overlap with AI poetry.

**Signal B — LLM**, Groq `llama-3.3-70b-versatile` (`app/llm_signal.py`). Two decoupled calls. Call A reads the text and returns surface-style observations with distinctive vocabulary and imagery quoted verbatim; it is banned from using the words *human*, *AI*, *generated*, *machine*, *model*, *detection*. Call B sees only Call A's observations (never the source text) and votes `low`/`medium`/`high` against an anchored rubric that names specific cliché patterns and voice quirks. Self-reported confidence is discarded.

*Why chosen:* a coarse second opinion that catches things stylometry can't (semantic clichés, register). *What it misses:* modern AI prose with specific imagery — when the model can write voice-driven content, the same model can't detect it.

**Signal C — Binoculars**, via the public Hugging Face Space (`app/binoculars_signal.py`). For the demo, the service calls the Space over HTTP, caches successful responses locally, and maps its coarse output tiers to fixed probabilities: `likely_human -> 0.20`, `uncertain -> 0.50`, `likely_ai -> 0.80`. The endpoint is only attempted for texts with at least 65 whitespace-token words; shorter eligible texts stay on the 2-signal path.

*Why chosen:* in a self-hosted form Binoculars would likely be the strongest signal, so the demo treats it as the dominant vote when available. *What it misses:* the public Space is rate-limited and returns only tiers rather than a continuous score, so this integration is best-effort and intentionally conservative.

## Confidence scoring

Each raw signal is calibrated to P(AI):
- Stylometry: Platt logistic `sigmoid(A · raw + B)`, with `A`, `B` fit on the labeled set by max-likelihood gradient ascent.
- LLM: empirical bucket rates `P(AI | vote)` counted directly from the labeled set.
- Binoculars: fixed manual tier mapping because the public HF endpoint exposes only coarse labels.

Both fit by `scripts/calibrate.py` against `data/ai_batch_*.jsonl` (Claude/GPT/Gemini outputs, ~90 rows) + `data/human_batch.jsonl` (51 mixed-source human rows). The script also picks attribution thresholds from the fused-score distributions: `THRESHOLD_LOW` = 75th percentile of human fused, `THRESHOLD_HIGH` = 25th percentile of AI fused — so the uncertain band tracks where the distributions actually overlap.

Validation is honest: the script prints the confusion matrix under the fitted constants. If `THRESHOLD_LOW > THRESHOLD_HIGH`, it warns that the signals don't cleanly separate the dataset.

**Fusion:**

```
weights = {stylometry: 0.60, llm: 0.40}                            # app/confidence.py:WEIGHTS

P_fused = weighted average of available calibrated P(AI) values    # missing signals are dropped + remaining renormalized
if |P_i − P_j| > 0.40:  attribution = "uncertain"                  # disagreement override (pairwise)
```

The current runtime uses stylometry + LLM only, with stylometry slightly heavier because it is deterministic and available on every request. If one signal is missing at request time, its weight is dropped and the remainder is renormalized.

### Reading the score fields

All three numeric fields in the response (`confidence`, `stylometry_score`, `llm_score`) are calibrated probabilities on the same scale: `0.0` means "highly confident human," `1.0` means "highly confident AI," `0.5` is a coin flip. `stylometry_score` is continuous (a Platt logistic over six features). `llm_score` is a categorical lookup, so it can only take as many distinct values as there are vote buckets. The exact numbers in `LLM_BUCKET_P_AI` in `app/confidence.py` are refit by `scripts/calibrate.py`.

### Example submissions

**AI-labeled dataset example with live Binoculars participation** — `demo-ai-strong`:
> Rain hammered the window of my office on the forty-third floor. The dame walked in like trouble wearing a red dress, and trouble was exactly what she was. \"I need you to find someone,\" she said, her voice smoky and low. I lit a cigarette and leaned back. In this city, everybody was looking for someone. The question was always the same: who was looking for them?
```json
{
  "content_id": "1039a9de-6745-455f-9a42-0df47c38279c",
  "creator_id": "demo-ai-strong",
  "timestamp": "2026-06-29T06:07:26.896633+00:00",
  "attribution": "uncertain",
  "confidence": 0.669,
  "stylometry_score": 0.6323,
  "llm_score": 0.15,
  "binoculars_score": 0.8,
  "binoculars_tier": "likely_ai",
  "label_text": "Inconclusive. Our checks did not agree, so we are not confident either way. Treat this as no result, not as a verdict. This case is a good candidate for human review.",
  "status": "classified"
}
```
This is a useful demo case because Binoculars votes strongly `likely_ai` (`0.8`), stylometry also leans AI (`0.6323`), but the LLM vote stays low (`0.15`). The fused score lands above the AI threshold, yet the pairwise spread exceeds the 0.40 disagreement override, so the final attribution is still `uncertain`.

**Human-labeled dataset example with live Binoculars participation** — `demo-human-strong`:
> Edgy technologies and smart designs, cleanliness and convenience, pop culture and the famous cuisine: there are many reasons why Japan is an attractive place to visit and to live. I have worked with candidates from overseas, developing and developed countries, seeking for job opportunities to move to Japan. Many have fallen in love with Japan since childhood through the unique pop culture of games, anime, manga. Others (me included) were impressed by the perfectionism in hospitality, the exciting metropolitan scenes and technology advancement. Living here in long-term, however, would be a different story.
```json
{
  "content_id": "9748f621-5f97-4d36-83f4-f9e14937c779",
  "creator_id": "demo-human-strong",
  "timestamp": "2026-06-29T06:08:51.814309+00:00",
  "attribution": "uncertain",
  "confidence": 0.2763,
  "stylometry_score": 0.6188,
  "llm_score": 0.15,
  "binoculars_score": 0.2,
  "binoculars_tier": "likely_human",
  "label_text": "Inconclusive. Our checks did not agree, so we are not confident either way. Treat this as no result, not as a verdict. This case is a good candidate for human review.",
  "status": "classified"
}
```
This is the mirror-image demo case: Binoculars votes `likely_human` (`0.2`), the LLM vote is also low (`0.15`), but stylometry stays relatively AI-leaning (`0.6188`). The fused score lands below the human threshold, yet the pairwise spread again exceeds the 0.40 disagreement override, so the system refuses to issue a confident `likely_human` verdict.

## Transparency label

Derived from `attribution` and returned to the client as `label_text`. The exact wording per variant comes from `planning.md` §"Transparency label design"; the per-attribution mapping is implemented in `app/label.py`:

| attribution | label text |
|---|---|
| `likely_ai` | "Likely AI-generated. Our automated checks agree this text shows machine-generation patterns. This is an automated estimate, not proof. You can appeal this result." |
| `likely_human` | "Likely human-written. Our automated checks agree this text shows human-writing patterns. This is an automated estimate, not proof." |
| `uncertain` | "Inconclusive. Our checks did not agree, so we are not confident either way. Treat this as no result, not as a verdict. This case is a good candidate for human review." |
| `insufficient_text` | "Not enough text to analyze reliably. Please submit a longer passage." |

The label varies with `attribution`, and `attribution` comes from the fused confidence thresholds (`< 0.35` → likely_human, `> 0.65` → likely_ai, between → uncertain), the spread-based disagreement override into `uncertain`, and the short-text gate into `insufficient_text`. The numeric scores (`confidence`, `stylometry_score`, `llm_score`) ship alongside `label_text` in the response so a reader can sanity-check the verdict rather than trust the label on faith.

## Rate limiting

`POST /submit` is rate-limited via `slowapi` (in-process, IP-keyed) with two stacked limits:

| endpoint | limit | status |
|---|---|---|
| `POST /submit` | **10 requests / minute** and **100 requests / hour** per IP | Implemented (`app/main.py:43-44`) |

**Why two stacked limits on `/submit`.** A short window catches bursts; a long window caps sustained use. Either trip returns `429 Too Many Requests` with a `Retry-After` header.

**Why these specific numbers.**

*Realistic upper bound for a human writer iterating on their own work.* Each submission needs ~10–30 seconds of "read result, decide, edit" before the next one — so a real ceiling is **~6 submissions per minute** during active drafting, dropping to far less when the writer pauses to think. Over an hour of intensive work, **30–60 submissions** is a heavy session; sustained 6/min for a full hour is uncommon and probably reflects iteration on a very rough draft.

*The chosen numbers sit just above that.* `10/min` gives a writer headroom over their realistic peak (60% slack) without leaving room for a script firing once per second. `100/hour` accommodates intense sessions (up to ~10 minutes of peak-rate iteration, plus pauses) but stops a script from sustaining 6/min for hours.

*Why the limits matter.* Each `/submit` triggers two Groq calls against `llama-3.3-70b-versatile`, so a runaway script could exhaust the API quota in minutes. The audit log would also fill with noise that drowns out real submissions. Rate-limiting per IP is the standard defense — `creator_id` lives in the request body and is trivially spoofable, so it isn't a useful key.

*What we accept by choosing per-IP.* Users sharing an IP (school WiFi, corporate NAT) share the limit. For a class demo this is fine. In production this would either need per-account keys behind authentication or X-Forwarded-For handling with a trusted proxy.

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
