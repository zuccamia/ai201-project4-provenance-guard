"""Inspect what the LLM signal produces on labeled rows.

Samples N AI and N human rows, re-runs Call A (and Call B unless --reuse-vote)
on each, and prints text + observations + vote alongside any cached vote.
Use this when bucket rates come out flat to see whether Call A surfaces
useful features and whether Call B acts on them.

Usage:
    python scripts/inspect_llm.py                       # 5 AI + 5 human, fresh A+B
    python scripts/inspect_llm.py --ai 3 --human 3
    python scripts/inspect_llm.py --reuse-vote          # re-run only Call A; show cached vote
    python scripts/inspect_llm.py --filter-vote low     # only sample rows whose cached vote == low
"""
import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # for sibling import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for `app`

from dotenv import load_dotenv

load_dotenv()

from app import llm_signal  # noqa: E402
from calibrate import (  # noqa: E402
    AI_FILES, DATA_DIR, HUMAN_FILES, MIN_TEXT_CHARS,
    iter_records, load_cache, text_hash,
)


def load_all_rows() -> list[dict]:
    rows: list[dict] = []
    for fname in AI_FILES:
        for rec in iter_records(DATA_DIR / fname):
            rows.append({"text": rec["text"], "label": "ai"})
    for fname in HUMAN_FILES:
        for rec in iter_records(DATA_DIR / fname):
            rows.append({"text": rec["text"], "label": "human"})
    return rows


def truncate(s: str, n: int) -> str:
    flat = " ".join(s.split())
    return flat if len(flat) <= n else flat[:n].rstrip() + "..."


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", type=int, default=5)
    parser.add_argument("--human", type=int, default=5)
    parser.add_argument("--reuse-vote", action="store_true",
                        help="re-run only Call A; show cached vote instead of a fresh Call B")
    parser.add_argument("--filter-vote", choices=["low", "medium", "high"], default=None,
                        help="only sample rows whose cached vote matches this")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    rows = load_all_rows()
    cache = load_cache()

    def sample(label: str, k: int) -> list[dict]:
        pool = [r for r in rows
                if r["label"] == label
                and len(r["text"].strip()) >= MIN_TEXT_CHARS]
        if args.filter_vote:
            pool = [r for r in pool
                    if cache.get(text_hash(r["text"]), {}).get("vote") == args.filter_vote]
        return random.sample(pool, min(k, len(pool)))

    samples = sample("ai", args.ai) + sample("human", args.human)
    if not samples:
        print("no rows matched the sampling criteria", file=sys.stderr)
        return 1

    mode = "reuse cached vote" if args.reuse_vote else "fresh Call A + Call B"
    print(f"Inspecting {len(samples)} rows  (mode: {mode})\n")

    for i, row in enumerate(samples):
        h = text_hash(row["text"])
        cached_vote = cache.get(h, {}).get("vote")
        try:
            if args.reuse_vote:
                obs = await llm_signal._call_a(row["text"])
                fresh_vote = cached_vote
            else:
                result = await llm_signal.score(row["text"])
                obs, fresh_vote = result.observations, result.vote
        except Exception as e:
            print(f"--- {row['label'].upper()} #{i+1}  ERROR: {type(e).__name__}: {e}\n")
            continue

        marker = "==" if cached_vote == fresh_vote else "!="
        print(f"--- {row['label'].upper()} #{i+1}   cached={cached_vote}  fresh={fresh_vote}  [{marker}]")
        print(f"    text: {truncate(row['text'], 120)}")
        print(f"    Call A observations:")
        if obs:
            for o in obs:
                print(f"      - {o}")
        else:
            print("      (none — Call A returned empty)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
