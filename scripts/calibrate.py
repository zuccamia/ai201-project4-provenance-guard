"""Fit calibration constants for app/confidence.py against labeled data.

Reads data/ai_batch_*.jsonl + data/human_batch.jsonl, runs both signals over
every row (cached), then fits:
  - Stylometry Platt logistic (a, b) by max-likelihood gradient ascent.
  - LLM bucket rates P(AI | vote) by empirical counts.
  - Attribution thresholds from fused-score percentiles.

Prints the resulting constants ready to paste into app/confidence.py.

Usage:
    python scripts/calibrate.py
    python scripts/calibrate.py --rebuild         # re-run signals, ignore cache
    python scripts/calibrate.py --re-stylometry-only   # refresh raw_sty only

The cache (data/calibration_cache.jsonl) stores partial results, so a re-run
fills in only the signals that are missing per row — useful when Groq
rate-limits us partway through.
"""
import argparse
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app import llm_signal, stylometry  # noqa: E402

DATA_DIR = Path("data")
AI_FILES = ["ai_batch_claude.jsonl", "ai_batch_gemini.jsonl", "ai_batch_gpt.jsonl"]
HUMAN_FILES = ["human_batch.jsonl"]
CACHE_PATH = DATA_DIR / "calibration_cache.jsonl"
MIN_TEXT_CHARS = 100

DEFAULT_W_STY = 0.60
DEFAULT_W_LLM = 0.40
WEIGHT_CANDIDATES: list[tuple[float, float]] = [
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
]
MANUAL_THRESHOLD_CANDIDATES: list[tuple[float, float]] = [
    (0.35, 0.65),
    (0.35, 0.70),
    (0.30, 0.70),
    (0.30, 0.75),
]


# ---------- data loading -----------------------------------------------------

def iter_records(path: Path) -> Iterator[dict]:
    """Handles both real JSONL (one obj per line) and concatenated pretty-
    printed JSON (the AI batch format)."""
    raw = path.read_text(encoding="utf-8")
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    if first.strip().startswith("{") and first.strip().endswith("}"):
        for line in raw.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    decoder = json.JSONDecoder()
    idx, n = 0, len(raw)
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(raw, idx)
        yield obj
        idx = end


def load_dataset() -> list[dict]:
    rows: list[dict] = []
    for fname in AI_FILES:
        for rec in iter_records(DATA_DIR / fname):
            rows.append({"text": rec["text"], "label": rec.get("label", "ai")})
    for fname in HUMAN_FILES:
        for rec in iter_records(DATA_DIR / fname):
            rows.append({"text": rec["text"], "label": rec.get("label", "human")})
    return rows


# ---------- signal cache -----------------------------------------------------

def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    cache: dict[str, dict] = {}
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["hash"]] = entry
            except json.JSONDecodeError:
                continue
    return cache


def rewrite_cache(cache: dict[str, dict]) -> None:
    """Atomically replace the cache file. Used to compact the file after a
    run (or to update raw_sty in place during --re-stylometry-only)."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for entry in cache.values():
            f.write(json.dumps(entry) + "\n")
    tmp.replace(CACHE_PATH)


def append_cache(entry: dict) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _word_count(text: str) -> int:
    return len(text.split())


async def collect_signals(rows: list[dict], rebuild: bool, concurrency: int,
                          re_stylometry_only: bool = False) -> None:
    """Populate row["raw_sty"] and row["vote"] for each row. Successful LLM
    results and below-gate rows are cached; LLM *failures* are deliberately
    NOT cached so the next run retries them. This means re-running picks up
    exactly where Groq rate-limited us last time.

    re_stylometry_only=True recomputes raw_sty for every row using the current
    stylometry heuristic but reuses cached LLM votes — no Groq calls. Use this
    after editing app/stylometry.py to refresh stale raw_sty values without
    burning API quota."""
    cache = {} if rebuild else load_cache()
    if rebuild and CACHE_PATH.exists():
        CACHE_PATH.unlink()

    if re_stylometry_only:
        if not cache:
            print("  WARNING: cache is empty; no LLM votes to reuse. "
                  "All rows will end up vote=None. Run a normal pass first.")
        have_votes = 0
        for i, row in enumerate(rows):
            h = text_hash(row["text"])
            row["raw_sty"] = stylometry.score(row["text"]).raw_score
            cached = cache.get(h)
            if cached is not None:
                row["vote"] = cached.get("vote")
                cache[h] = {**cached, "raw_sty": row["raw_sty"]}
                if row["vote"] is not None:
                    have_votes += 1
            else:
                row["vote"] = None
            print(f"  [{i+1:>3}/{len(rows)}]  {row['label']:<5}  "
                  f"sty={row['raw_sty']:.3f}  vote={row['vote']}  (no Groq calls)")
        rewrite_cache(cache)
        print(f"\n  recomputed stylometry for {len(rows)} rows; "
              f"kept {have_votes} LLM votes")
        return

    sem = asyncio.Semaphore(concurrency)

    cached_hits = 0
    partial_cache_hits = 0
    new_success = 0
    new_short = 0
    new_failure = 0

    async def process(row: dict, i: int) -> None:
        nonlocal cached_hits, partial_cache_hits, new_success, new_short, new_failure
        h = text_hash(row["text"])
        cached = cache.get(h, {})

        # Known-short rows: no signals to run, just propagate cached marker.
        if cached.get("short"):
            row["raw_sty"] = cached.get("raw_sty", 0.5)
            row["vote"] = None
            cached_hits += 1
            return

        # Stylometry: reuse if cached, recompute otherwise (cheap either way).
        row["raw_sty"] = cached.get("raw_sty") or stylometry.score(row["text"]).raw_score

        short = len(row["text"].strip()) < MIN_TEXT_CHARS
        had_llm = cached.get("vote") is not None

        if short:
            row["vote"] = None
            if not cached.get("short"):
                new_short += 1
        else:
            # LLM: reuse cache if present, else fetch.
            if had_llm:
                row["vote"] = cached["vote"]
            else:
                async with sem:
                    result = await llm_signal.score(row["text"])
                row["vote"] = result.vote
                if row["vote"] is None:
                    new_failure += 1
                else:
                    new_success += 1
        # Bookkeeping for the summary line.
        if cached and had_llm:
            cached_hits += 1
        elif cached:
            partial_cache_hits += 1

        # Cache the entry if something is worth saving.
        keep = short or row["vote"] is not None
        if keep:
            entry: dict = {
                "hash": h, "label": row["label"],
                "raw_sty": row["raw_sty"],
                "vote": row["vote"],
            }
            if short:
                entry["short"] = True
            cache[h] = entry
            append_cache(entry)

        marker = "short" if short else (
            "OK" if row["vote"] is not None
            else "partial"
        )
        print(f"  [{i+1:>3}/{len(rows)}]  {row['label']:<5}  "
              f"sty={row['raw_sty']:.3f}  vote={row['vote']}  {marker}")

    await asyncio.gather(*[process(r, i) for i, r in enumerate(rows)])

    # Compact the cache file after the run.
    rewrite_cache(cache)

    print(
        f"\n  cache hits: {cached_hits}   partial cache hits: {partial_cache_hits}   "
        f"new ok: {new_success}   "
        f"new short: {new_short}   "
        f"FAILED (will retry next run): {new_failure}"
    )
    if new_failure:
        print("  (failures aren't cached; rerun the same command to retry them.)")


# ---------- fitting ----------------------------------------------------------

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(xs: list[float], ys: list[int], iters: int = 3000, lr: float = 0.1) -> tuple[float, float]:
    """Maximum-likelihood fit of P(y=1|x) = sigmoid(a*x + b) by batch gradient
    ascent. Small dataset, low dimension — no need for scipy/sklearn."""
    a, b = 1.0, 0.0
    n = max(len(xs), 1)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            p = sigmoid(a * x + b)
            err = y - p
            ga += err * x
            gb += err
        a += lr * ga / n
        b += lr * gb / n
    return a, b


def fit_bucket_rates(rows: list[dict], field: str, buckets: tuple[str, ...]
                     ) -> tuple[dict[str, Optional[float]], dict[str, int]]:
    """Empirical P(AI | bucket) for any categorical signal field.
    `field` is the row dict key (e.g. 'vote', 'bino_verdict'); `buckets` is
    the list of valid bucket labels."""
    grouped: dict[str, list[int]] = {b: [] for b in buckets}
    for row in rows:
        v = row.get(field)
        if v in grouped:
            grouped[v].append(1 if row["label"] == "ai" else 0)
    rates: dict[str, Optional[float]] = {}
    counts: dict[str, int] = {}
    for b in buckets:
        ys = grouped[b]
        counts[b] = len(ys)
        rates[b] = (sum(ys) / len(ys)) if ys else None
    return rates, counts


def fuse(p_sty: Optional[float], p_llm: Optional[float],
         w_sty: float, w_llm: float) -> float:
    contribs = []
    if p_sty is not None:
        contribs.append((w_sty, p_sty))
    if p_llm is not None:
        contribs.append((w_llm, p_llm))
    if not contribs:
        return 0.5
    total = sum(w for w, _ in contribs)
    return sum(w * p for w, p in contribs) / total


def calibrated_fused(row: dict, a: float, b: float,
                     llm_rates: dict[str, Optional[float]],
                     w_sty: float, w_llm: float) -> float:
    vote = row.get("vote")
    p_sty = sigmoid(a * row["raw_sty"] + b)
    p_llm = llm_rates.get(vote) if vote else None
    return fuse(p_sty, p_llm, w_sty, w_llm)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.5
    k = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def pick_thresholds(rows: list[dict], a: float, b: float,
                    llm_rates: dict[str, Optional[float]],
                    w_sty: float, w_llm: float
                    ) -> tuple[float, float, list[float], list[float]]:
    """THRESHOLD_LOW = 75th-percentile of human fused scores; THRESHOLD_HIGH =
    25th-percentile of AI fused scores. Where the two distributions overlap,
    the gap is the 'uncertain' band — honest about ambiguity."""
    ai_scores: list[float] = []
    hu_scores: list[float] = []
    for row in rows:
        s = calibrated_fused(row, a, b, llm_rates, w_sty, w_llm)
        (ai_scores if row["label"] == "ai" else hu_scores).append(s)
    ai_scores.sort()
    hu_scores.sort()
    return percentile(hu_scores, 75), percentile(ai_scores, 25), ai_scores, hu_scores


def confusion(rows: list[dict], a: float, b: float,
              llm_rates: dict[str, Optional[float]],
              t_lo: float, t_hi: float,
              w_sty: float, w_llm: float) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        "ai": {"likely_ai": 0, "likely_human": 0, "uncertain": 0},
        "human": {"likely_ai": 0, "likely_human": 0, "uncertain": 0},
    }
    for row in rows:
        s = calibrated_fused(row, a, b, llm_rates, w_sty, w_llm)
        if s < t_lo:
            attr = "likely_human"
        elif s > t_hi:
            attr = "likely_ai"
        else:
            attr = "uncertain"
        out[row["label"]][attr] += 1
    return out


def separation_score(ai_scores: list[float], hu_scores: list[float]) -> float:
    if not ai_scores or not hu_scores:
        return float("-inf")
    return percentile(ai_scores, 50) - percentile(hu_scores, 50)


# ---------- entrypoint -------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore cache, re-run all signals")
    parser.add_argument("--re-stylometry-only", action="store_true",
                        help="recompute stylometry only; keep cached LLM votes (no Groq calls)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="concurrent Groq calls (default 2; lower if rate-limited)")
    args = parser.parse_args()
    if args.rebuild and args.re_stylometry_only:
        parser.error("--rebuild and --re-stylometry-only are mutually exclusive")

    print("Loading dataset...")
    rows = load_dataset()
    print(f"  total: {len(rows)}  (ai={sum(1 for r in rows if r['label']=='ai')}, "
          f"human={sum(1 for r in rows if r['label']=='human')})")

    print(f"\nCollecting signals (cache: {CACHE_PATH})...")
    await collect_signals(rows, rebuild=args.rebuild, concurrency=args.concurrency,
                          re_stylometry_only=args.re_stylometry_only)

    long_rows = [r for r in rows if len(r["text"].strip()) >= MIN_TEXT_CHARS]
    short = len(rows) - len(long_rows)
    if short:
        print(f"  skipping {short} short rows (< {MIN_TEXT_CHARS} chars)")

    print("\nFitting stylometry Platt logistic...")
    xs = [r["raw_sty"] for r in long_rows]
    ys = [1 if r["label"] == "ai" else 0 for r in long_rows]
    platt_a, platt_b = fit_platt(xs, ys)
    print(f"  A = {platt_a:.4f}   B = {platt_b:.4f}")

    print("\nFitting LLM bucket rates...")
    rates, counts = fit_bucket_rates(long_rows, "vote", ("low", "medium", "high"))
    for v in ("low", "medium", "high"):
        rate = rates[v]
        rate_s = f"{rate:.4f}" if rate is not None else "n/a"
        print(f"  P(AI | {v:<6}) = {rate_s}   (n = {counts[v]})")
    n_none = sum(1 for r in long_rows if r.get("vote") is None)
    if n_none:
        print(f"  {n_none} rows had no LLM vote — dropped from bucket fit, "
              f"still scored stylometry-only for thresholds")

    print("\nEvaluating fusion-weight candidates...")
    best: Optional[dict] = None
    for w_sty, w_llm in WEIGHT_CANDIDATES:
        t_lo, t_hi, fused_ai, fused_hu = pick_thresholds(
            long_rows, platt_a, platt_b, rates, w_sty, w_llm
        )
        sep = separation_score(fused_ai, fused_hu)
        crossed = t_lo > t_hi
        gap = t_hi - t_lo
        print(f"\n  weights stylometry={w_sty:.2f}  llm={w_llm:.2f}")
        print(f"    THRESHOLD_LOW  = {t_lo:.4f}   (75th-pctile of human fused)")
        print(f"    THRESHOLD_HIGH = {t_hi:.4f}   (25th-pctile of AI fused)")
        print(f"    gap = {gap:.4f}   median separation = {sep:.4f}")
        if crossed:
            print("    WARNING: thresholds cross — the fused distributions don't cleanly separate.")
        if fused_hu:
            print(f"    human fused: min={min(fused_hu):.3f} median={percentile(fused_hu, 50):.3f} max={max(fused_hu):.3f}")
        if fused_ai:
            print(f"    ai    fused: min={min(fused_ai):.3f} median={percentile(fused_ai, 50):.3f} max={max(fused_ai):.3f}")

        print("    confusion:")
        mat = confusion(long_rows, platt_a, platt_b, rates, t_lo, t_hi, w_sty, w_llm)
        header = f"{'':12}{'likely_ai':>12}{'likely_human':>14}{'uncertain':>12}"
        print("    " + header)
        for label in ("ai", "human"):
            r = mat[label]
            print(f"      {label:<8}{r['likely_ai']:>12}{r['likely_human']:>14}{r['uncertain']:>12}")

        candidate = {
            "w_sty": w_sty,
            "w_llm": w_llm,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "sep": sep,
            "crossed": crossed,
            "gap": gap,
        }
        if best is None:
            best = candidate
        else:
            best_key = (best["crossed"], -best["gap"], -best["sep"])
            cand_key = (candidate["crossed"], -candidate["gap"], -candidate["sep"])
            if cand_key < best_key:
                best = candidate

    assert best is not None
    t_lo = best["t_lo"]
    t_hi = best["t_hi"]
    best_w_sty = best["w_sty"]
    best_w_llm = best["w_llm"]

    print("\nSelected weight setting for pasted constants:")
    print(f"  stylometry={best_w_sty:.2f}  llm={best_w_llm:.2f}")
    if best["crossed"]:
        print("  WARNING: even the best candidate still has crossed thresholds.")

    print("\nEvaluating manual threshold candidates under selected weights...")
    header = f"{'':12}{'likely_ai':>12}{'likely_human':>14}{'uncertain':>12}"
    for manual_t_lo, manual_t_hi in MANUAL_THRESHOLD_CANDIDATES:
        print(f"\n  THRESHOLD_LOW={manual_t_lo:.2f}  THRESHOLD_HIGH={manual_t_hi:.2f}")
        mat = confusion(
            long_rows, platt_a, platt_b, rates,
            manual_t_lo, manual_t_hi,
            best_w_sty, best_w_llm,
        )
        print("    " + header)
        for label in ("ai", "human"):
            r = mat[label]
            print(f"      {label:<8}{r['likely_ai']:>12}{r['likely_human']:>14}{r['uncertain']:>12}")

    print("\n" + "=" * 60)
    print("Paste into app/confidence.py (replace the matching constants):")
    print("=" * 60)
    print(f"WEIGHTS = {{'stylometry': {best_w_sty:.2f}, 'llm': {best_w_llm:.2f}}}")
    print(f"STYLOMETRY_PLATT_A = {platt_a:.4f}")
    print(f"STYLOMETRY_PLATT_B = {platt_b:.4f}")
    print("LLM_BUCKET_P_AI: dict[str, float] = {")
    for v in ("low", "medium", "high"):
        rate = rates[v]
        if rate is None:
            print(f'    "{v}": 0.50,  # no samples in this bucket; kept neutral')
        else:
            print(f'    "{v}": {rate:.4f},')
    print("}")
    print(f"THRESHOLD_LOW = {t_lo:.4f}")
    print(f"THRESHOLD_HIGH = {t_hi:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
