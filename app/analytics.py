from collections import Counter
from typing import Any


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [e for e in entries if e.get("status") == "classified" and e.get("content_id")]
    appeals = [e for e in entries if e.get("event") == "appeal"]

    detection_counts = Counter(e.get("attribution", "unknown") for e in classified)
    total_classified = len(classified)
    total_appeals = len(appeals)

    appealed_content_ids = {e.get("content_id") for e in appeals if e.get("content_id")}
    appealed_originals = [e for e in classified if e.get("content_id") in appealed_content_ids]
    appealed_by_attr = Counter(e.get("attribution", "unknown") for e in appealed_originals)

    bino_eligible = 0
    bino_used = 0
    for e in classified:
        if e.get("attribution") == "insufficient_text":
            continue
        # Best-effort proxy: any classified non-short text is eligible for local runtime,
        # and binoculars participation is visible via non-null score.
        bino_eligible += 1
        if e.get("binoculars_score") is not None:
            bino_used += 1

    fallback_count = max(bino_eligible - bino_used, 0)

    return {
        "detections": {
            "likely_ai": detection_counts.get("likely_ai", 0),
            "likely_human": detection_counts.get("likely_human", 0),
            "uncertain": detection_counts.get("uncertain", 0),
            "insufficient_text": detection_counts.get("insufficient_text", 0),
            "total_classified": total_classified,
        },
        "appeals": {
            "total_appeals": total_appeals,
            "overall_appeal_rate": _safe_rate(total_appeals, total_classified),
            "by_original_attribution": {
                "likely_ai": _safe_rate(appealed_by_attr.get("likely_ai", 0), detection_counts.get("likely_ai", 0)),
                "likely_human": _safe_rate(appealed_by_attr.get("likely_human", 0), detection_counts.get("likely_human", 0)),
                "uncertain": _safe_rate(appealed_by_attr.get("uncertain", 0), detection_counts.get("uncertain", 0)),
            },
        },
        "signal_c_usage": {
            "eligible_submissions": bino_eligible,
            "used_binoculars": bino_used,
            "fallback_to_2_signal": fallback_count,
            "binoculars_usage_rate": _safe_rate(bino_used, bino_eligible),
        },
    }


def render_html(summary: dict[str, Any]) -> str:
    det = summary["detections"]
    appeals = summary["appeals"]
    signal_c = summary["signal_c_usage"]
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Provenance Guard Analytics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; color: #1f2937; }}
    h1 {{ margin-bottom: 1rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 12px; padding: 1rem; background: #f9fafb; }}
    .metric {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Provenance Guard Analytics</h1>
  <p class=\"muted\">Simple dashboard derived from the append-only audit log.</p>

  <div class=\"cards\">
    <div class=\"card\"><div>Total classified</div><div class=\"metric\">{det['total_classified']}</div></div>
    <div class=\"card\"><div>Total appeals</div><div class=\"metric\">{appeals['total_appeals']}</div></div>
    <div class=\"card\"><div>Appeal rate</div><div class=\"metric\">{_pct(appeals['overall_appeal_rate'])}</div></div>
    <div class=\"card\"><div>Binoculars usage</div><div class=\"metric\">{_pct(signal_c['binoculars_usage_rate'])}</div></div>
  </div>

  <h2>Detection patterns</h2>
  <table>
    <tr><th>Attribution</th><th>Count</th></tr>
    <tr><td>likely_ai</td><td>{det['likely_ai']}</td></tr>
    <tr><td>likely_human</td><td>{det['likely_human']}</td></tr>
    <tr><td>uncertain</td><td>{det['uncertain']}</td></tr>
    <tr><td>insufficient_text</td><td>{det['insufficient_text']}</td></tr>
  </table>

  <h2>Appeals</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total appeals</td><td>{appeals['total_appeals']}</td></tr>
    <tr><td>Overall appeal rate</td><td>{_pct(appeals['overall_appeal_rate'])}</td></tr>
    <tr><td>Appeal rate for likely_ai</td><td>{_pct(appeals['by_original_attribution']['likely_ai'])}</td></tr>
    <tr><td>Appeal rate for likely_human</td><td>{_pct(appeals['by_original_attribution']['likely_human'])}</td></tr>
    <tr><td>Appeal rate for uncertain</td><td>{_pct(appeals['by_original_attribution']['uncertain'])}</td></tr>
  </table>

  <h2>Signal C usage</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Eligible submissions</td><td>{signal_c['eligible_submissions']}</td></tr>
    <tr><td>Used Binoculars</td><td>{signal_c['used_binoculars']}</td></tr>
    <tr><td>Fallback to 2-signal</td><td>{signal_c['fallback_to_2_signal']}</td></tr>
    <tr><td>Binoculars usage rate</td><td>{_pct(signal_c['binoculars_usage_rate'])}</td></tr>
  </table>
</body>
</html>
"""


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"