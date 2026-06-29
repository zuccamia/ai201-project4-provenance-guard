import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Optional

PROVENANCE_PATH = Path(os.getenv("PROVENANCE_CREATOR_PATH", "data/creator_certificates.jsonl"))
_lock = Lock()


def get_creator_status(creator_id: str) -> dict[str, Any]:
    latest: Optional[dict[str, Any]] = None
    if not PROVENANCE_PATH.exists():
        return {"creator_id": creator_id, "status": "unverified"}
    with PROVENANCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("creator_id") == creator_id:
                latest = entry
    if latest is None:
        return {"creator_id": creator_id, "status": "unverified"}
    return latest


def append_status(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
    with _lock:
        PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROVENANCE_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")