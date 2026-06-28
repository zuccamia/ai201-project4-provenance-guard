import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

# Default to repo-local logs/. Overridable so tests can redirect to a tmpdir.
AUDIT_PATH = Path(os.getenv("PROVENANCE_AUDIT_PATH", "logs/audit.jsonl"))

# Serialize writes within the process. POSIX append is atomic for small payloads,
# but uvicorn may run the handler in a threadpool — the lock keeps lines intact.
_lock = Lock()


def append(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
    with _lock:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_by_content_id(content_id: str) -> list[dict[str, Any]]:
    """Return every audit entry referencing this content_id, oldest first.
    Used by /appeal to confirm the content exists before recording the appeal."""
    if not AUDIT_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("content_id") == content_id:
                entries.append(entry)
    return entries


def read(limit: int | None = None) -> list[dict[str, Any]]:
    """Return audit entries newest-first, optionally capped at `limit`."""
    if not AUDIT_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupt lines rather than poison the whole read.
                continue
    entries.reverse()
    if limit is not None:
        entries = entries[:limit]
    return entries
