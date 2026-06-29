# Provenance Guard — Planning

A service that judges whether a piece of text is human-written or AI-generated.
It accepts a poem, story excerpt, or blog post. It returns an attribution, a calibrated
confidence score, per-signal scores, and a status. A transparency label is shown to the
user, derived from the attribution.

The output is a suspicion score, not a final ruling about a person. It is meant for triage, not
for final trust decisions about a person. False positives fall hardest on non-native and
stylized writing. So the label and the appeal path are core features, not extras.

---

## Architecture

A CPU orchestrator handles each request. It also calls Groq over HTTP.

This document mixes current implementation notes with planned extensions. In
the active build, the orchestrator ships with Signals A and B only; Signal C
below remains design-only.

```
                              POST /submit  (raw text)
                                     |
                                     v
                          +---------------------+
                          |   Length Gate       |
                          +----+-----------+----+
              too short        |           |  raw text (length OK)
              "insufficient"   |           |
                  label  <-----+           v
                              fans out to 2 signals (async)
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
        +----------------+                     +----------------------+
        | Stylometry     |                     | LLM (Groq)           |
        | CPU, local     |                     | Call A: features     |
        |                |                     | Call B: vote         |
        +-------+--------+                     +----------+-----------+
                | s_sty (raw)                             | vote + observations
                +--------------------+--------------------+
                                     v
                          +---------------------+
                          | Confidence Scoring  |
                          | calibrate -> P(AI); |
                          | fuse; check         |
                          | agreement; threshold|
                          +----------+----------+
                          confidence, attribution,
                          stylometry_score, llm_score,
                          observations
                                     v
                          +---------------------+
                          | Transparency Label  |
                          | (derive from        |
                          |  attribution)       |
                          +----------+----------+
                          label_text (for display + log)
                                     v
                          +---------------------+
                          | Audit Log (external |
                          | DB, append-only)    |
                          +----------+----------+
                          content_id |
                                     v
                          +---------------------+
                          | Response            |
                          | {content_id,        |
                          |  creator_id,        |
                          |  timestamp,         |
                          |  attribution,       |
                          |  confidence,        |
                          |  stylometry_score,  |
                          |  llm_score,         |
                          |  status}            |
                          +---------------------+


                              POST /appeal  ({content_id, appeal_reasoning})
                                     |
                                     v
                          +---------------------+
                          | Status Update       |
                          | fetch record;       |
                          | mark "under_review";|
                          | attach reasoning    |
                          +----------+----------+
                          updated_status, reasoning
                                     v
                          +---------------------+
                          | Audit Log           |
                          | append appeal event |
                          +----------+----------+
                          appeal_id  |
                                     v
                          +---------------------+
                          | Response            |
                          | {content_id,        |
                          |  status,            |
                          |  appeal_id,         |
                          |  timestamp}         |
                          +---------------------+
```

**Submission flow.** A user posts text. The orchestrator gates on length, scores it with
two signals, fuses them, derives the label, logs the decision, and returns the result.

**Appeal flow.** A user posts a `content_id` and their `appeal_reasoning`. The system marks that
record as `under_review`, appends an appeal event to the log, and returns a confirmation. The
original record is never overwritten.

---

## API structures

Two endpoints. Both take and return JSON.

### POST /submit

Classifies one piece of text.

Request fields:
- `text` (string): the content to analyze.
- `creator_id` (string): who submitted it.

Response fields:
- `content_id` (string): unique id for this submission; used to appeal.
- `creator_id` (string): echoed back.
- `timestamp` (string): ISO-8601 UTC time of classification.
- `attribution` (string): one of `likely_ai`, `likely_human`, `uncertain`, `insufficient_text`.
- `confidence` (float): the calibrated 0–1 score behind the attribution.
- `stylometry_score` (float): the stylometry signal's calibrated score (per-signal detail).
- `llm_score` (float): the LLM signal's calibrated score (per-signal detail).
- `label_text` (string): the transparency label text derived from `attribution`, shown to the user.
- `status` (string): `classified` on success.

```bash
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The sun dipped below the horizon, painting the sky in hues of amber and rose. I sat on the porch, coffee in hand, watching the neighborhood slowly go quiet.", "creator_id": "test-user-1"}'
```

```json
{
  "content_id": "3f7a2b1e-...",
  "creator_id": "test-user-1",
  "timestamp": "2025-04-01T14:32:10.123Z",
  "attribution": "likely_ai",
  "confidence": 0.78,
  "stylometry_score": 0.72,
  "llm_score": 0.81,
  "label_text": "Likely AI-generated. Our automated checks agree this text shows machine-generation patterns. This is an automated estimate, not proof. You can appeal this result.",
  "status": "classified"
}
```

### POST /appeal

Files an appeal against a past classification.

Request fields:
- `content_id` (string): the id from the `/submit` response.
- `appeal_reasoning` (string): why the creator believes the result is wrong.

Response fields:
- `content_id` (string): echoed back.
- `status` (string): now `under_review`.
- `appeal_id` (string): id for this appeal event.
- `timestamp` (string): ISO-8601 UTC time of the appeal.

```bash
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE", "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."}'
```

```json
{
  "content_id": "3f7a2b1e-...",
  "appeal_id": "a91c4d7f-...",
  "status": "under_review",
  "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
  "original_attribution": "likely_ai",
  "timestamp": "2025-04-01T15:10:44.456Z"
}
```

The appeal never changes the original `attribution` or scores. It only adds an appeal
event and flips the status.

---

## Detection signals

Three signals. They are chosen to fail in different ways. Signals A and B are wired into
the current orchestrator; Signal C is part of the design but not part of the active
build. When present, the fusion weights rebalance as described below.

**Signal A — Stylometry.** Measures the shape of the writing. The current implementation
extracts six features: AI-cliche density (overused LLM phrasings like "delve into",
"whispers of"), type-token ratio, low-burstiness (inverse of sentence-length variance),
contraction rate, capitalization irregularity (ALL-CAPS plus lowercase sentence starts),
and punctuation diversity (em-dashes, semicolons, ellipses). Runs on CPU. Output is a
feature vector summarized into a raw 0–1 score. It is a mechanical measurement, not a
judgment. The original spec listed only burstiness, function-word use, and punctuation;
calibration against the labeled set showed those alone were anti-correlated with AI on
poetry-heavy data, which is what drove the feature-set expansion.

**Signal B — LLM (Groq `llama-3.3-70b-versatile`).** Gives a coarse vote plus
human-readable observations. Output is a low/medium/high vote. Its self-reported
confidence number is discarded.

The LLM uses two decoupled calls. Call A reports neutral style facts and never mentions
human, AI, or detection. Call B maps those facts to a vote. This keeps the rationale from
being bent to fit a conclusion.

**Signal C — Binoculars.** A likelihood signal. Measures how predictable the text is to
a reference LLM — lower means more machine-like, higher means more human. It is the
strongest single AI-detection signal in the literature.

It is not in the current build because it needs two scoring models running together with
full access to their token probabilities. That requires a GPU and cannot run on a chat
API like Groq. So it ships as its own service, called over HTTP exactly like Groq.

The reference implementation we plan to call is the
[Binoculars HF Space by tomg-group-umd](https://huggingface.co/spaces/tomg-group-umd/Binoculars),
likely via `gradio_client`. The GPU service returns a raw ratio; the orchestrator
calibrates it to `P(AI)` with a fitted logistic, the same way it handles the other
signals. The raw ratio has the opposite direction (higher = more human) and is not a
probability, so the calibration step is required before fusion.

A known failure mode of Signal C: it scores well-memorized text as machine-like. A famous
public-domain poem or a widely-quoted passage can be flagged as AI. This is a structural
failure of the likelihood approach, not a tuning bug, and it makes the appeal path matter
more.

**Combining them.** The orchestrator calibrates each raw output to `P(AI)`, a value
between 0 and 1, then takes a weighted average. The design weights:

```
weights = {binoculars: 0.50, stylometry: 0.35, llm: 0.15}
fused   = weighted average of the calibrated P(AI) values
```

Binoculars carries the most weight because it's the strongest single signal. The LLM
weight is kept low because it shares a likelihood prior with Binoculars (both reason
from token probabilities, even if one is implicit), so without the down-weight the
ensemble would double-count that kind of evidence.

If a signal is missing at request time, it is dropped and the remaining weights are
renormalized over their sum. A missing signal is never treated as a vote.

Until Signal C is wired in, the orchestrator runs A and B only, with hardcoded weights
`{stylometry: 0.60, llm: 0.40}` (see `app/confidence.py:WEIGHTS`). Adding Signal C is a
two-step change: bring up the Binoculars HTTP client and switch the weight table to the
three-signal design above.

---

## Uncertainty representation

A confidence of 0.6 means the system estimates a 60% chance the text is AI-generated. It
is a calibrated probability, not a raw signal value. Calibration makes the number honest.

**Mapping raw outputs to a score.** Each raw signal goes through a fitted logistic
(Platt) function. This puts both signals on the same 0–1 scale. We fit these functions on
a small labeled set.

**Thresholds.** Two cutoffs split the range. They are set from validation accuracy, not
guessed.

```
fused < 0.35            -> likely human
0.35 <= fused <= 0.65   -> uncertain
fused > 0.65            -> likely AI
```

**Agreement overrides the average.** Agreement is the gap between the two signals. If
they disagree a lot, the attribution is `uncertain`. This holds even if the average lands in
a confident band. Disagreement is itself a signal of low confidence.

```
spread = abs(p_stylometry - p_llm)
if spread > DISAGREE_THRESHOLD: attribution = "uncertain"
```

**Prep step: build a validation set first.** Confidence numbers only mean something when
measured against known answers. So before calibrating, gather a labeled set of texts we
know are human and texts we know are AI (a few dozen per side to start). Run it through
once to fit all three things at once: the stylometry curve, the LLM bucket rates, and the
thresholds. Until then the code uses placeholder numbers. One rule: freeze the LLM prompt
before measuring, since the bucket rates only describe that exact prompt.

---

## Transparency label design

Three variants. Each one should change how the reader treats the result. The label is
derived from `attribution`: `likely_ai`, `likely_human`, `uncertain`, and the short-input
case map one-to-one to the four variants below. It is returned as the `label_text` field
on the `/submit` response and is also written into the audit log entry.

**High-confidence AI:**
> Likely AI-generated. Our automated checks agree this text shows machine-generation
> patterns. This is an automated estimate, not proof. You can appeal this result.

**High-confidence human:**
> Likely human-written. Our automated checks agree this text shows human-writing
> patterns. This is an automated estimate, not proof.

**Uncertain:**
> Inconclusive. Our checks did not agree, so we are not confident either way. Treat this
> as no result, not as a verdict. This case is a good candidate for human review.

A fourth case exists for very short input:
> Not enough text to analyze reliably. Please submit a longer passage.

---

## Appeals workflow

**Who can appeal.** Anyone who received a result. They use the `content_id` from the
response.

**What they provide.** The `content_id` and a short reason (`appeal_reasoning`). For example: "this is a
public-domain poem," or "I wrote this; English is my second language."

**What the system does.** It looks up the record. It marks the status as `under_review`. It
attaches the user's reason. It appends an appeal event to the audit log. It returns an
`appeal_id`. It never edits the original attribution or scores.

**What gets logged.** The `content_id`, a timestamp, the user's reason, and the new
status. The original decision stays intact next to it.

**What a reviewer sees.** A queue of appealed records. Each entry shows the text, the
attribution, the confidence, and the per-signal scores. It shows which signals were used. It
shows the user's stated reason. The per-signal breakdown tells the reviewer which signal
likely drove the attribution. That points them to the failure mode to check first.

---

## Anticipated edge cases

These are specific cases the system will handle poorly.

**A short, stylized poem.** Poems use deliberate repetition and odd rhythm. Stylometry
expects human writing to be irregular. A stylized human poem can look like an outlier and
get misread. Short length also makes the signal noisy.

**Non-native English writing.** The LLM judge tends to read simpler, more regular prose
as AI. A fluent but non-native human writer can be flagged as AI. This is the fairness
hazard that lands on real people.

**Heavily edited AI text.** A user can paraphrase AI output to smooth its style.
Stylometry may still catch some uniformity. But a careful edit can defeat both signals.
The honest output here is often `uncertain`.

---

## AI Tool Plan

How I will use an AI coding tool across the three milestones. The pattern is the same
each time: give it the relevant spec sections, ask for a small piece, verify it in
isolation, then wire it in.

### M3 — Submission endpoint + first signal
- **Provide:** the Detection signals section (stylometry) and the Architecture diagram.
- **Ask for:** a FastAPI app skeleton with the `/submit` endpoint, plus the stylometry
  signal function.
- **Verify:** run the stylometry function on a few texts on its own first. Check the
  scores look sane. Only then wire it into the endpoint.

### M4 — Second signal + confidence scoring
- **Provide:** Detection signals, Uncertainty representation, and the diagram.
- **Ask for:** the LLM signal function (Call A then Call B) and the scoring logic
  (calibrate each signal, fuse, agreement check, threshold).
- **Check:** feed it clearly-AI text and clearly-human text. Confirm the scores differ
  meaningfully. Confirm that signal disagreement produces `uncertain`.

### M5 — Production layer
- **Provide:** Transparency label design, Appeals workflow, and the diagram.
- **Ask for:** the label generation logic and the `/appeal` endpoint.
- **Verify:** craft inputs that land in each band so all three labels appear. Submit an
  appeal and confirm the record status changes to `under_review` and a log entry is written.

**Prompting note.** Give one milestone at a time. Paste only the named sections. Ask for
small functions, not whole files. Small asks are easier to verify and less likely to
drift from the spec.
