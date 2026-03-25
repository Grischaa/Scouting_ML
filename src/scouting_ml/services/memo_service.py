"""Professional PDF memo export for player scouting reports."""

from __future__ import annotations

import html
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _safe_float(value: Any) -> float | None:
    """Return a float when possible, otherwise ``None``."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_currency(value: Any) -> str:
    """Render a euro amount for memo display."""
    amount = _safe_float(value)
    if amount is None:
        return "-"
    return f"EUR {amount:,.0f}"


def _fmt_pct(value: Any) -> str:
    """Render a percentage-like value for memo display."""
    pct = _safe_float(value)
    if pct is None:
        return "-"
    if abs(pct) <= 1.0:
        pct *= 100.0
    return f"{pct:.1f}%"


def _svg_path_points(values: list[float], width: int, height: int, padding: int) -> str:
    """Convert chart values into an SVG polyline point string."""
    if not values:
        return ""
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 1.0)
    usable_width = max(width - (padding * 2), 1)
    usable_height = max(height - (padding * 2), 1)
    points: list[str] = []
    for idx, value in enumerate(values):
        x = padding + (usable_width * idx / max(len(values) - 1, 1))
        y = padding + usable_height - (((value - minimum) / span) * usable_height)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _build_trajectory_svg(trajectory: dict[str, Any] | None) -> str:
    """Build an inline SVG trajectory chart from season payloads."""
    seasons = trajectory.get("seasons") if isinstance(trajectory, dict) else []
    if not isinstance(seasons, list) or not seasons:
        return ""
    values = [_safe_float(item.get("predicted_value")) for item in seasons]
    plotted = [value for value in values if value is not None]
    if not plotted:
        return ""
    width = 720
    height = 190
    padding = 26
    points = _svg_path_points([float(value) for value in plotted], width, height, padding)
    next_value = _safe_float(trajectory.get("projected_next_value"))
    extension = ""
    if next_value is not None and len(plotted) >= 1:
        ext_points = _svg_path_points([float(plotted[-1]), float(next_value)], 160, height, padding)
        if ext_points:
            start, end = ext_points.split(" ")
            start_x, start_y = start.split(",")
            end_x, end_y = end.split(",")
            extension = (
                f'<line x1="{float(start_x) + (width - 160):.1f}" y1="{start_y}" '
                f'x2="{float(end_x) + (width - 160):.1f}" y2="{end_y}" class="memo-chart__projection" />'
            )
    labels = "".join(
        f'<text x="{padding + ((width - (padding * 2)) * idx / max(len(seasons) - 1, 1)):.1f}" y="{height - 6}" '
        f'class="memo-chart__label">{html.escape(str(item.get("season") or "-"))}</text>'
        for idx, item in enumerate(seasons)
    )
    return f"""
    <svg class="memo-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Player trajectory">
      <rect x="0" y="0" width="{width}" height="{height}" rx="18" class="memo-chart__bg"></rect>
      <polyline points="{points}" class="memo-chart__line"></polyline>
      {extension}
      {labels}
    </svg>
    """


@dataclass(frozen=True)
class PlayerMemoService:
    """Generate single-page PDF memos from scouting report payloads."""

    accent_color: str = "#1a1a2e"

    def render_pdf(
        self,
        *,
        report: dict[str, Any],
        trajectory: dict[str, Any] | None,
        similar_players: list[dict[str, Any]] | None,
        model_manifest: dict[str, Any] | None,
    ) -> bytes:
        """Render the provided player payloads into a PDF byte stream."""
        html_text = self._render_html(
            report=report,
            trajectory=trajectory,
            similar_players=similar_players,
            model_manifest=model_manifest,
        )
        try:
            from weasyprint import HTML

            return HTML(string=html_text).write_pdf()
        except Exception:
            return self._render_reportlab_fallback(
                report=report,
                trajectory=trajectory,
                similar_players=similar_players,
                model_manifest=model_manifest,
            )

    def _render_html(
        self,
        *,
        report: dict[str, Any],
        trajectory: dict[str, Any] | None,
        similar_players: list[dict[str, Any]] | None,
        model_manifest: dict[str, Any] | None,
        ) -> str:
        """Render the memo as HTML suitable for WeasyPrint conversion."""
        player = report.get("player") if isinstance(report.get("player"), dict) else {}
        guardrails = report.get("valuation_guardrails") if isinstance(report.get("valuation_guardrails"), dict) else {}
        confidence = report.get("confidence") if isinstance(report.get("confidence"), dict) else {}
        strengths = list(report.get("strengths") or [])[:4]
        weaknesses = list(report.get("weaknesses") or [])[:4]
        levers = list(report.get("development_levers") or [])[:4]
        risk_flags = list(report.get("risk_flags") or [])[:4]
        similar = list(similar_players or [])[:3]
        trajectory_svg = _build_trajectory_svg(trajectory)
        model_label = (
            str(model_manifest.get("label") or model_manifest.get("model_version") or "current_bundle")
            if isinstance(model_manifest, dict)
            else "current_bundle"
        )
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        def bullet_list(items: list[dict[str, Any]], empty_message: str) -> str:
            """Render a compact HTML bullet list from memo items."""
            if not items:
                return f"<li>{html.escape(empty_message)}</li>"
            return "".join(f"<li>{html.escape(str(item.get('label') or item.get('message') or '-'))}</li>" for item in items)

        similar_rows = "".join(
            """
            <tr>
              <td>{name}</td>
              <td>{club}</td>
              <td>{league}</td>
              <td>{similarity}</td>
              <td>{market}</td>
              <td>{predicted}</td>
            </tr>
            """.format(
                name=html.escape(str(item.get("name") or item.get("player_id") or "-")),
                club=html.escape(str(item.get("club") or "-")),
                league=html.escape(str(item.get("league") or "-")),
                similarity=_fmt_pct(item.get("similarity_score")),
                market=html.escape(_fmt_currency(item.get("market_value_eur"))),
                predicted=html.escape(_fmt_currency(item.get("predicted_value"))),
            )
            for item in similar
        ) or "<tr><td colspan='6'>No similar-player matches available.</td></tr>"

        return f"""
        <html>
          <head>
            <meta charset="utf-8" />
            <style>
              @page {{
                size: A4;
                margin: 14mm;
              }}
              body {{
                font-family: Arial, sans-serif;
                color: #1f2933;
                font-size: 11px;
                line-height: 1.35;
              }}
              .memo {{
                border-top: 10px solid {self.accent_color};
                padding-top: 10px;
              }}
              .memo__header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 20px;
                margin-bottom: 12px;
              }}
              .memo__title h1 {{
                margin: 0 0 4px;
                font-size: 24px;
                color: {self.accent_color};
              }}
              .memo__meta {{
                color: #52606d;
              }}
              .memo__badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 999px;
                background: #ecfdf3;
                color: #065f46;
                font-weight: bold;
              }}
              .memo__grid {{
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 14px;
              }}
              .memo__card {{
                border: 1px solid #d9e2ec;
                border-radius: 12px;
                padding: 10px 12px;
                margin-bottom: 10px;
              }}
              .memo__card h2 {{
                margin: 0 0 8px;
                color: {self.accent_color};
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
              }}
              .memo__metrics {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 8px 10px;
              }}
              .memo__metric strong {{
                display: block;
                font-size: 14px;
              }}
              .memo__risk-list li {{
                margin-bottom: 4px;
              }}
              table {{
                width: 100%;
                border-collapse: collapse;
              }}
              th, td {{
                border-bottom: 1px solid #e5e7eb;
                padding: 4px 0;
                text-align: left;
                vertical-align: top;
                font-size: 10px;
              }}
              .memo-chart__bg {{
                fill: #f8fafc;
                stroke: #d9e2ec;
              }}
              .memo-chart__line {{
                fill: none;
                stroke: {self.accent_color};
                stroke-width: 3;
              }}
              .memo-chart__projection {{
                stroke: #f59e0b;
                stroke-width: 2;
                stroke-dasharray: 6 4;
              }}
              .memo-chart__label {{
                fill: #52606d;
                font-size: 9px;
                text-anchor: middle;
              }}
              .memo__footer {{
                margin-top: 10px;
                color: #7b8794;
                font-size: 10px;
                display: flex;
                justify-content: space-between;
              }}
            </style>
          </head>
          <body>
            <main class="memo">
              <section class="memo__header">
                <div class="memo__title">
                  <h1>{html.escape(str(player.get("name") or player.get("player_id") or "Player"))}</h1>
                  <div class="memo__meta">
                    {html.escape(str(player.get("club") or "-"))} | {html.escape(str(player.get("league") or "-"))} |
                    {html.escape(str(player.get("position") or player.get("model_position") or "-"))} |
                    Age {html.escape(str(player.get("age") or "-"))} |
                    {html.escape(str(player.get("nationality") or "-"))}
                  </div>
                </div>
                <span class="memo__badge">{html.escape(str(confidence.get("label") or "medium").title())} confidence</span>
              </section>

              <section class="memo__grid">
                <div>
                  <article class="memo__card">
                    <h2>Value View</h2>
                    <div class="memo__metrics">
                      <div class="memo__metric"><span>Market value</span><strong>{html.escape(_fmt_currency(guardrails.get("market_value_eur")))}</strong></div>
                      <div class="memo__metric"><span>Predicted value</span><strong>{html.escape(_fmt_currency(guardrails.get("fair_value_eur")))}</strong></div>
                      <div class="memo__metric"><span>Gap</span><strong>{html.escape(_fmt_currency(guardrails.get("value_gap_conservative_eur")))}</strong></div>
                      <div class="memo__metric"><span>Interval</span><strong>{html.escape(_fmt_currency(confidence.get("interval_low_eur")))} to {html.escape(_fmt_currency(confidence.get("interval_high_eur")))}</strong></div>
                    </div>
                  </article>

                  <article class="memo__card">
                    <h2>Trajectory</h2>
                    {trajectory_svg or '<p>No trajectory chart available.</p>'}
                    <p>
                      {html.escape(str(trajectory.get("trajectory_label") if isinstance(trajectory, dict) else "stable").title())}
                      | slope {_fmt_pct(trajectory.get("slope_pct") if isinstance(trajectory, dict) else None)}
                      | projected next {_fmt_currency(trajectory.get("projected_next_value") if isinstance(trajectory, dict) else None)}
                    </p>
                  </article>

                  <article class="memo__card">
                    <h2>Similar Players</h2>
                    <table>
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Club</th>
                          <th>League</th>
                          <th>Similarity</th>
                          <th>Market</th>
                          <th>Predicted</th>
                        </tr>
                      </thead>
                      <tbody>{similar_rows}</tbody>
                    </table>
                  </article>
                </div>

                <div>
                  <article class="memo__card">
                    <h2>Strengths</h2>
                    <ul class="memo__risk-list">{bullet_list(strengths, "No standout strengths recorded.")}</ul>
                  </article>
                  <article class="memo__card">
                    <h2>Weaknesses</h2>
                    <ul class="memo__risk-list">{bullet_list(weaknesses, "No major weaknesses recorded.")}</ul>
                  </article>
                  <article class="memo__card">
                    <h2>Development Levers</h2>
                    <ul class="memo__risk-list">{bullet_list(levers, "No major development levers recorded.")}</ul>
                  </article>
                  <article class="memo__card">
                    <h2>Risk Flags</h2>
                    <ul class="memo__risk-list">{bullet_list(risk_flags, "No immediate risk flags triggered.")}</ul>
                  </article>
                </div>
              </section>

              <footer class="memo__footer">
                <span>Generated by ScoutML</span>
                <span>{generated_at} | {html.escape(model_label)}</span>
              </footer>
            </main>
          </body>
        </html>
        """

    def _render_reportlab_fallback(
        self,
        *,
        report: dict[str, Any],
        trajectory: dict[str, Any] | None,
        similar_players: list[dict[str, Any]] | None,
        model_manifest: dict[str, Any] | None,
    ) -> bytes:
        """Fallback PDF renderer when WeasyPrint is unavailable."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.pdfgen import canvas
        except Exception:
            return self._render_minimal_pdf(
                report=report,
                trajectory=trajectory,
                similar_players=similar_players,
                model_manifest=model_manifest,
            )

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 18 * mm

        player = report.get("player") if isinstance(report.get("player"), dict) else {}
        guardrails = report.get("valuation_guardrails") if isinstance(report.get("valuation_guardrails"), dict) else {}
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(16 * mm, y, str(player.get("name") or player.get("player_id") or "Player"))
        y -= 7 * mm
        pdf.setFont("Helvetica", 10)
        meta = " | ".join(
            str(part)
            for part in (
                player.get("club") or "-",
                player.get("league") or "-",
                player.get("position") or player.get("model_position") or "-",
                f"Age {player.get('age') or '-'}",
            )
        )
        pdf.drawString(16 * mm, y, meta)
        y -= 10 * mm

        lines = [
            f"Market value: {_fmt_currency(guardrails.get('market_value_eur'))}",
            f"Predicted value: {_fmt_currency(guardrails.get('fair_value_eur'))}",
            f"Conservative gap: {_fmt_currency(guardrails.get('value_gap_conservative_eur'))}",
            f"Trajectory: {str(trajectory.get('trajectory_label') if isinstance(trajectory, dict) else 'stable').title()}",
            f"Projected next value: {_fmt_currency(trajectory.get('projected_next_value') if isinstance(trajectory, dict) else None)}",
            f"Model bundle: {str(model_manifest.get('label') or model_manifest.get('model_version') or 'current_bundle') if isinstance(model_manifest, dict) else 'current_bundle'}",
        ]
        for line in lines:
            pdf.drawString(16 * mm, y, line)
            y -= 6 * mm

        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(16 * mm, y, "Similar Players")
        y -= 6 * mm
        pdf.setFont("Helvetica", 10)
        for item in list(similar_players or [])[:3]:
            pdf.drawString(
                18 * mm,
                y,
                f"{item.get('name') or item.get('player_id')} | {item.get('club') or '-'} | {item.get('league') or '-'} | {_fmt_pct(item.get('similarity_score'))}",
            )
            y -= 5 * mm
        pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    def _render_minimal_pdf(
        self,
        *,
        report: dict[str, Any],
        trajectory: dict[str, Any] | None,
        similar_players: list[dict[str, Any]] | None,
        model_manifest: dict[str, Any] | None,
    ) -> bytes:
        """Render a minimal valid PDF without optional third-party PDF libraries."""
        player = report.get("player") if isinstance(report.get("player"), dict) else {}
        guardrails = report.get("valuation_guardrails") if isinstance(report.get("valuation_guardrails"), dict) else {}
        model_label = (
            str(model_manifest.get("label") or model_manifest.get("model_version") or "current_bundle")
            if isinstance(model_manifest, dict)
            else "current_bundle"
        )
        similar = list(similar_players or [])[:3]
        lines = [
            str(player.get("name") or player.get("player_id") or "Player Memo"),
            "ScoutML Player Memo",
            f"Club: {player.get('club') or '-'}",
            f"League: {player.get('league') or '-'}",
            f"Position: {player.get('position') or player.get('model_position') or '-'}",
            f"Market value: {_fmt_currency(guardrails.get('market_value_eur'))}",
            f"Predicted value: {_fmt_currency(guardrails.get('fair_value_eur'))}",
            f"Conservative gap: {_fmt_currency(guardrails.get('value_gap_conservative_eur'))}",
            f"Trajectory: {str(trajectory.get('trajectory_label') if isinstance(trajectory, dict) else 'stable').title()}",
            f"Projected next value: {_fmt_currency(trajectory.get('projected_next_value') if isinstance(trajectory, dict) else None)}",
            f"Model bundle: {model_label}",
        ]
        if similar:
            lines.append("Similar players:")
            for item in similar:
                lines.append(
                    f"- {item.get('name') or item.get('player_id')} | "
                    f"{item.get('club') or '-'} | {item.get('league') or '-'} | "
                    f"{_fmt_pct(item.get('similarity_score'))}"
                )
        text = "\n".join(line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines)
        stream = f"BT /F1 12 Tf 50 780 Td 14 TL ({text.replace(chr(10), ') Tj T* (')}) Tj ET"
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            + f"5 0 obj << /Length {len(stream.encode('latin-1', errors='replace'))} >> stream\n{stream}\nendstream endobj\n".encode(
                "latin-1",
                errors="replace",
            )
            + b"xref\n0 6\n0000000000 65535 f \n"
            + b"trailer << /Root 1 0 R /Size 6 >>\nstartxref\n0\n%%EOF"
        )
        return pdf


_MEMO_SERVICE = PlayerMemoService()


def get_memo_service() -> PlayerMemoService:
    """Return the singleton memo service."""
    return _MEMO_SERVICE


__all__ = ["PlayerMemoService", "get_memo_service"]
