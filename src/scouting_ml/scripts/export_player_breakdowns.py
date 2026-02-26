from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from scouting_ml.services.market_value_service import query_player_reports


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _build_flat_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        report = item.get("report", {}) if isinstance(item.get("report"), dict) else {}
        player = report.get("player", {}) if isinstance(report.get("player"), dict) else {}
        guardrails = (
            report.get("valuation_guardrails", {})
            if isinstance(report.get("valuation_guardrails"), dict)
            else {}
        )
        confidence = report.get("confidence", {}) if isinstance(report.get("confidence"), dict) else {}
        history = item.get("history_strength", {}) if isinstance(item.get("history_strength"), dict) else {}
        player_type = report.get("player_type", {}) if isinstance(report.get("player_type"), dict) else {}
        formation_fit = report.get("formation_fit", {}) if isinstance(report.get("formation_fit"), dict) else {}
        formation_recommended = (
            formation_fit.get("recommended", [])
            if isinstance(formation_fit.get("recommended"), list)
            else []
        )
        best_formation = formation_recommended[0] if formation_recommended else {}

        strengths = report.get("strengths", []) if isinstance(report.get("strengths"), list) else []
        weaknesses = report.get("weaknesses", []) if isinstance(report.get("weaknesses"), list) else []
        levers = (
            report.get("development_levers", [])
            if isinstance(report.get("development_levers"), list)
            else []
        )
        risks = report.get("risk_flags", []) if isinstance(report.get("risk_flags"), list) else []

        rows.append(
            {
                "player_id": player.get("player_id") or item.get("player_id"),
                "name": player.get("name"),
                "season": player.get("season") or item.get("season"),
                "league": player.get("league"),
                "club": player.get("club"),
                "position": player.get("model_position") or player.get("position_group"),
                "age": player.get("age"),
                "minutes": player.get("minutes") or player.get("sofa_minutesPlayed"),
                "market_value_eur": guardrails.get("market_value_eur"),
                "fair_value_eur": guardrails.get("fair_value_eur"),
                "value_gap_conservative_eur": guardrails.get("value_gap_conservative_eur"),
                "value_gap_capped_eur": guardrails.get("value_gap_capped_eur"),
                "cap_applied": guardrails.get("cap_applied"),
                "undervaluation_confidence": confidence.get("undervaluation_confidence"),
                "confidence_label": confidence.get("label"),
                "confidence_score": confidence.get("score"),
                "history_strength_score": history.get("score_0_to_100"),
                "history_strength_tier": history.get("tier"),
                "player_archetype": player_type.get("archetype"),
                "player_archetype_confidence": player_type.get("confidence_0_to_1"),
                "best_formation": best_formation.get("formation"),
                "best_role": best_formation.get("role"),
                "best_formation_fit_score": best_formation.get("fit_score_0_to_1"),
                "top_strengths": "|".join(str(m.get("label")) for m in strengths[:3] if m.get("label")),
                "top_weaknesses": "|".join(str(m.get("label")) for m in weaknesses[:3] if m.get("label")),
                "top_development_levers": "|".join(str(m.get("label")) for m in levers[:3] if m.get("label")),
                "risk_codes": "|".join(str(r.get("code")) for r in risks if isinstance(r, dict) and r.get("code")),
                "summary_text": report.get("summary_text"),
            }
        )
    return rows


def export_player_breakdowns(
    *,
    predictions_path: str,
    split: str = "test",
    out_json: str = "data/model/reports/player_breakdowns.json",
    out_jsonl: str | None = "data/model/reports/player_breakdowns.jsonl",
    out_csv: str | None = "data/model/reports/player_breakdowns.csv",
    season: str | None = None,
    league: str | None = None,
    club: str | None = None,
    position: str | None = None,
    min_minutes: float | None = None,
    max_age: float | None = None,
    player_ids: list[str] | None = None,
    top_metrics: int = 5,
    include_history: bool = True,
    sort_by: str = "undervaluation_score",
    sort_order: str = "desc",
    limit: int = 0,
    offset: int = 0,
) -> dict[str, Any]:
    split_key = str(split).strip().lower()
    if split_key not in {"test", "val"}:
        raise ValueError("split must be 'test' or 'val'.")

    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    if split_key == "test":
        os.environ["SCOUTING_TEST_PREDICTIONS_PATH"] = str(pred_path)
    else:
        os.environ["SCOUTING_VAL_PREDICTIONS_PATH"] = str(pred_path)

    effective_limit = int(limit) if int(limit) > 0 else 1_000_000
    payload = query_player_reports(
        split=split_key,  # type: ignore[arg-type]
        season=season,
        league=league,
        club=club,
        position=position,
        min_minutes=min_minutes,
        max_age=max_age,
        player_ids=player_ids,
        top_metrics=int(top_metrics),
        include_history=bool(include_history),
        sort_by=sort_by,
        sort_order="asc" if str(sort_order).lower() == "asc" else "desc",  # type: ignore[arg-type]
        limit=effective_limit,
        offset=max(int(offset), 0),
    )

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[player-breakdowns] wrote json -> {out_json_path}")

    if out_jsonl:
        out_jsonl_path = Path(out_jsonl)
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl_path.open("w", encoding="utf-8") as handle:
            for item in payload.get("items", []):
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        print(f"[player-breakdowns] wrote jsonl -> {out_jsonl_path}")

    if out_csv:
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        flat_rows = _build_flat_rows(payload.get("items", []))
        pd.DataFrame(flat_rows).to_csv(out_csv_path, index=False)
        print(f"[player-breakdowns] wrote csv -> {out_csv_path}")

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export precise scouting breakdowns for all/filtered players into JSON/JSONL/CSV."
    )
    parser.add_argument("--predictions", required=True, help="Predictions CSV for the selected split.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--out-json", default="data/model/reports/player_breakdowns.json")
    parser.add_argument("--out-jsonl", default="data/model/reports/player_breakdowns.jsonl")
    parser.add_argument("--out-csv", default="data/model/reports/player_breakdowns.csv")
    parser.add_argument("--season", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--club", default=None)
    parser.add_argument("--position", default=None, help="GK/DF/MF/FW")
    parser.add_argument("--min-minutes", type=float, default=None)
    parser.add_argument("--max-age", type=float, default=None)
    parser.add_argument("--player-ids", default="", help="Comma-separated player_ids filter.")
    parser.add_argument("--top-metrics", type=int, default=5)
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-by", default="undervaluation_score")
    parser.add_argument("--sort-order", default="desc", choices=["asc", "desc"])
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Rows to export after filters. Use <=0 to export all matching rows.",
    )
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_player_breakdowns(
        predictions_path=args.predictions,
        split=args.split,
        out_json=args.out_json,
        out_jsonl=args.out_jsonl,
        out_csv=args.out_csv,
        season=args.season,
        league=args.league,
        club=args.club,
        position=args.position,
        min_minutes=args.min_minutes,
        max_age=args.max_age,
        player_ids=_parse_csv_tokens(args.player_ids),
        top_metrics=args.top_metrics,
        include_history=args.include_history,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        limit=args.limit,
        offset=args.offset,
    )


if __name__ == "__main__":
    main()
