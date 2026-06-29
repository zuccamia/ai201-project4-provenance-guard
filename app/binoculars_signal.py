import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import httpx

log = logging.getLogger(__name__)

BinocularsTier = Literal["likely_human", "uncertain", "likely_ai"]

SPACE_API_URL = "https://tomg-group-umd-binoculars.hf.space/call/run_detector"
CACHE_PATH = Path("data") / "binoculars_cache.jsonl"
TIMEOUT_SECONDS = 20.0

_sem = asyncio.Semaphore(1)
_cache_lock = asyncio.Lock()


@dataclass
class BinocularsResult:
    tier: Optional[BinocularsTier]
    raw_label: Optional[str] = None


def _text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


async def score(text: str) -> BinocularsResult:
    h = _text_hash(text)
    cached = await _load_cached(h)
    if cached is not None:
        return cached

    try:
        async with _sem:
            result = await _call_space(text)
    except Exception as exc:
        log.warning("binoculars_signal failed: %s: %s", type(exc).__name__, exc)
        return BinocularsResult(tier=None)

    if result.tier is not None:
        await _append_cache(h, result)
    else:
        log.warning("binoculars_signal: remote returned no recognizable tier")
    return result


async def _call_space(text: str) -> BinocularsResult:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        resp = await client.post(SPACE_API_URL, json={"data": [text]})
        resp.raise_for_status()
        payload = resp.json()

        event_id = payload.get("event_id")
        if not event_id:
            return _parse_payload(payload)

        stream_url = f"{SPACE_API_URL}/{event_id}"
        async with client.stream("GET", stream_url) as stream:
            stream.raise_for_status()
            async for line in stream.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                parsed = _parse_payload(payload)
                if parsed.tier is not None:
                    return parsed

    return BinocularsResult(tier=None)


def _parse_payload(payload: object) -> BinocularsResult:
    candidates: list[str] = []

    def walk(obj: object) -> None:
        if isinstance(obj, str):
            candidates.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)

    walk(payload)
    for text in candidates:
        norm = _normalize_tier(text)
        if norm is not None:
            return BinocularsResult(tier=norm, raw_label=text)
    return BinocularsResult(tier=None)


def _normalize_tier(text: str) -> Optional[BinocularsTier]:
    t = " ".join(text.strip().lower().replace("_", " ").split())
    if "likely ai" in t or "likely machine" in t:
        return "likely_ai"
    if "likely human" in t:
        return "likely_human"
    if "uncertain" in t or "inconclusive" in t or "not sure" in t:
        return "uncertain"
    return None


async def _load_cached(text_hash: str) -> Optional[BinocularsResult]:
    if not CACHE_PATH.exists():
        return None
    async with _cache_lock:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("hash") == text_hash:
                    tier = entry.get("tier")
                    if tier in ("likely_human", "uncertain", "likely_ai"):
                        return BinocularsResult(tier=tier, raw_label=entry.get("raw_label"))
    return None


async def _append_cache(text_hash: str, result: BinocularsResult) -> None:
    async with _cache_lock:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "hash": text_hash,
                "tier": result.tier,
                "raw_label": result.raw_label,
            }) + "\n")