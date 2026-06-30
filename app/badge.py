from app import provenance


def style(creator_id: str) -> dict[str, str]:
    creator_status = provenance.get_creator_status(creator_id)
    status = creator_status.get("status")
    if status == "verified_human":
        return {
            "text": "Verified human creator",
            "bg": "#e8f7ec",
            "fg": "#166534",
            "border": "#86efac",
        }
    if status == "pending":
        return {
            "text": "Verification pending",
            "bg": "#fff7ed",
            "fg": "#9a3412",
            "border": "#fdba74",
        }
    return {
        "text": "Not verified",
        "bg": "#f3f4f6",
        "fg": "#374151",
        "border": "#d1d5db",
    }


def render_svg(creator_id: str) -> str:
    palette = style(creator_id)
    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"240\" height=\"82\" viewBox=\"0 0 240 82\">\n  <rect x=\"0.5\" y=\"0.5\" width=\"239\" height=\"81\" rx=\"14\" fill=\"{palette['bg']}\" stroke=\"{palette['border']}\"/>\n  <circle cx=\"18\" cy=\"20\" r=\"5\" fill=\"{palette['fg']}\"/>\n  <text x=\"30\" y=\"24\" font-family=\"-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif\" font-size=\"14\" font-weight=\"600\" fill=\"{palette['fg']}\">{palette['text']}</text>\n  <text x=\"12\" y=\"48\" font-family=\"-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif\" font-size=\"12\" font-weight=\"500\" fill=\"#111827\">Creator: {creator_id}</text>\n  <text x=\"12\" y=\"67\" font-family=\"-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif\" font-size=\"11\" font-weight=\"400\" fill=\"#6b7280\">Issued by Provenance Guard</text>\n</svg>"""


def render_html(creator_id: str) -> str:
    palette = style(creator_id)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <style>
    html, body {{ margin: 0; padding: 0; background: transparent; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .badge {{
      display: inline-flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 0.25rem;
      padding: 0.55rem 0.75rem;
      border-radius: 14px;
      border: 1px solid {palette['border']};
      background: {palette['bg']};
      color: {palette['fg']};
      min-width: 220px;
      box-sizing: border-box;
    }}
    .topline {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font: 600 14px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
      white-space: nowrap;
    }}
    .dot {{
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 999px;
      background: {palette['fg']};
      display: inline-block;
    }}
    .creator {{
      font: 500 12px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
      color: #111827;
    }}
    .issuer {{
      font: 400 11px/1.2 -apple-system, BlinkMacSystemFont, sans-serif;
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <span class=\"badge\">
    <span class=\"topline\"><span class=\"dot\"></span>{palette['text']}</span>
    <span class=\"creator\">Creator: {creator_id}</span>
    <span class=\"issuer\">Issued by Provenance Guard</span>
  </span>
</body>
</html>"""