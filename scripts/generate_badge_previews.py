from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.badge import render_svg


OUT_DIR = Path("docs")
CREATOR_ID = "test-user-1"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    originals = Path("data/creator_certificates.jsonl")
    existing = originals.read_text(encoding="utf-8") if originals.exists() else ""

    cases = {
        "badge-verified.svg": '{"creator_id":"test-user-1","status":"verified_human","issued_at":"2026-06-29T02:00:00+00:00","method":"manual_review"}\n',
        "badge-pending.svg": '{"creator_id":"test-user-1","status":"pending","requested_at":"2026-06-29T02:00:00+00:00","method":"manual_review"}\n',
        "badge-unverified.svg": "",
    }

    for filename, fixture in cases.items():
        if fixture:
            originals.write_text(fixture, encoding="utf-8")
        elif originals.exists():
            originals.unlink()
        (OUT_DIR / filename).write_text(render_svg(CREATOR_ID), encoding="utf-8")

    if existing:
        originals.write_text(existing, encoding="utf-8")
    elif originals.exists():
        originals.unlink()


if __name__ == "__main__":
    main()