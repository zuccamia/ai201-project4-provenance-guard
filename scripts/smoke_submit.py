"""Smoke-test /submit against labeled rows from data/ai_batch_*.jsonl.

Prints expected label vs the system's attribution and per-signal scores so
you can eyeball calibration drift before fitting Platt + bucket rates.

Usage:
    python scripts/smoke_submit.py
    python scripts/smoke_submit.py --file data/ai_batch_gpt.jsonl --limit 5
    python scripts/smoke_submit.py --file data/ai_batch_gemini.jsonl --limit 3
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

# Allow running as `python scripts/smoke_submit.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app


def iter_records(path: Path) -> Iterator[dict]:
    """The data files are concatenated pretty-printed JSON objects, not real
    JSONL — peel them off with raw_decode."""
    raw = path.read_text(encoding="utf-8")
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


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  -  "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/ai_batch_claude.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--creator", default="smoke-test")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    client = TestClient(app)
    print(f"file: {path}  limit: {args.limit}")
    print(f"{'expected':<9} {'attribution':<18} {'conf':>7} {'sty':>7} {'llm':>7}  prompt_id")
    print("-" * 72)

    agree = disagree = 0
    for i, rec in enumerate(iter_records(path)):
        if i >= args.limit:
            break
        resp = client.post(
            "/submit",
            json={"text": rec["text"], "creator_id": args.creator},
        )
        resp.raise_for_status()
        out = resp.json()
        expected = rec.get("label", "?")
        attribution = out["attribution"]
        # "agreement" here is just expected-vs-attribution direction, ignoring
        # the uncertain band — a rough eyeball, not a real accuracy metric.
        if expected == "ai" and attribution == "likely_ai":
            agree += 1
        elif expected == "human" and attribution == "likely_human":
            agree += 1
        elif attribution in ("likely_ai", "likely_human"):
            disagree += 1
        print(
            f"{expected:<9} {attribution:<18} "
            f"{fmt(out.get('confidence')):>7} "
            f"{fmt(out.get('stylometry_score')):>7} "
            f"{fmt(out.get('llm_score')):>7}  "
            f"{rec.get('prompt_id', '')}"
        )

    print("-" * 72)
    print(f"directional match: {agree}   directional miss: {disagree}   "
          f"(rest were uncertain or insufficient)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
