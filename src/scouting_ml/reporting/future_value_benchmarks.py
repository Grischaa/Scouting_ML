from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scouting_ml.scripts.build_future_value_targets import build_future_value_targets_frame


BIG5_LEAGUES = {
    "english premier league",
    "premier league",
    "spanish la liga",
    "la liga",
    "laliga",
    "italian serie a",
    "serie a",
    "german bundesliga",
    "bundesliga",
    "french ligue 1",
    "ligue 1",
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _parse_int_k_values(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            out.append(parsed)
    return sorted({*out}) or [10, 25, 50]


def _player_key(frame: pd.DataFrame) -> pd.Series:
    if "player_id" in frame.columns:
        key = frame["player_id"].astype(str).str.strip()
        if key.notna().any() and (key != "").any():
            return key
    if all(col in frame.columns for col in ("name", "dob")):
        return frame["name"].astype(str).str.strip() + "|" + frame["dob"].astype(str).str.strip()
    if "name" in frame.columns:
        return frame["name"].astype(str).str.strip()
    raise ValueError("Frame needs player_id or name/dob columns to build a future-benchmark join key.")


def _season_key(frame: pd.DataFrame) -> pd.Series:
    if "season" not in frame.columns:
        raise ValueError("Frame needs a 'season' column to build a future-benchmark join key.")
    return frame["season"].astype(str).str.strip()


def _position_series(frame: pd.DataFrame) -> pd.Series:
    for col in ("model_position", "position_group", "position_main"):
        if col in frame.columns:
            return frame[col].astype(str).str.upper().str.strip()
    return pd.Series("UNKNOWN", index=frame.index, dtype=object)


def _value_segment_series(frame: pd.DataFrame) -> pd.Series:
    if "value_segment" in frame.columns:
        return frame["value_segment"].astype(str).str.strip()

    market = pd.to_numeric(frame.get("market_value_eur"), errors="coerce")
    out = pd.Series("unknown", index=frame.index, dtype=object)
    out.loc[(market >= 0.0) & (market < 5_000_000.0)] = "under_5m"
    out.loc[(market >= 5_000_000.0) & (market < 20_000_000.0)] = "5m_to_20m"
    out.loc[market >= 20_000_000.0] = "over_20m"
    return out.astype(str)


def _minutes_series(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "sofa_minutesPlayed" in frame.columns:
        return pd.to_numeric(frame["sofa_minutesPlayed"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _resolve_score_col(frame: pd.DataFrame, explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            "scout_target_score",
            "scout_score",
            "undervaluation_score",
            "value_gap_capped_eur",
            "value_gap_conservative_eur",
            "value_gap_eur",
        ]
    )
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError("No suitable score column found for future benchmark.")


def _load_future_targets(
    *,
    future_targets_path: str | None,
    dataset_path: str | None,
    min_next_minutes: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if future_targets_path:
        path = Path(future_targets_path)
        if path.exists():
            return (
                pd.read_parquet(path),
                {
                    "source": "future_targets_parquet",
                    "future_targets_path": str(path),
                    "dataset_path": dataset_path,
                    "min_next_minutes": float(min_next_minutes),
                },
            )
    if not dataset_path:
        raise FileNotFoundError("Need either an existing future-target parquet or a source dataset path.")

    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    frame = pd.read_parquet(dataset)
    return (
        build_future_value_targets_frame(frame, min_next_minutes=min_next_minutes, drop_na_target=False),
        {
            "source": "dataset_built_in_memory",
            "future_targets_path": future_targets_path,
            "dataset_path": str(dataset),
            "min_next_minutes": float(min_next_minutes),
        },
    )


def load_future_targets_frame(
    *,
    future_targets_path: str | None = None,
    dataset_path: str | None = None,
    min_next_minutes: float = 450.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets_raw, meta = _load_future_targets(
        future_targets_path=future_targets_path,
        dataset_path=dataset_path,
        min_next_minutes=min_next_minutes,
    )
    return _prepare_target_frame(targets_raw), meta


def _prepare_target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "season",
        "next_season",
        "next_market_value_eur",
        "next_minutes",
        "has_next_season_target",
        "value_growth_next_season_eur",
        "value_growth_next_season_pct",
        "value_growth_next_season_log_delta",
        "value_growth_positive_flag",
        "value_growth_gt25pct_flag",
    ]
    available = [col for col in keep_cols if col in frame.columns]
    if "season" not in available:
        raise ValueError("Future-target frame is missing 'season'.")

    work = frame[available].copy()
    work["_player_key"] = _player_key(frame)
    work["_season_key"] = _season_key(frame)
    work["_has_target_sort"] = pd.to_numeric(work.get("has_next_season_target"), errors="coerce").fillna(0.0)
    if "minutes" in frame.columns:
        work["_minutes_sort"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(-1.0)
    else:
        work["_minutes_sort"] = -1.0
    if "market_value_eur" in frame.columns:
        work["_market_value_sort"] = pd.to_numeric(frame["market_value_eur"], errors="coerce").fillna(-1.0)
    else:
        work["_market_value_sort"] = -1.0

    work = work.sort_values(
        ["_player_key", "_season_key", "_minutes_sort", "_market_value_sort"],
        ascending=[True, True, False, False],
    )
    work = work.sort_values(
        ["_player_key", "_season_key", "_has_target_sort", "_minutes_sort", "_market_value_sort"],
        ascending=[True, True, False, False, False],
    )
    work = work.drop_duplicates(subset=["_player_key", "_season_key"], keep="first").reset_index(drop=True)
    return work.drop(columns=["_has_target_sort", "_minutes_sort", "_market_value_sort"], errors="ignore")


def _prepare_prediction_frame(
    frame: pd.DataFrame,
    *,
    min_minutes: float,
    max_age: float | None,
    non_big5_only: bool,
) -> pd.DataFrame:
    work = frame.copy()
    minutes = _minutes_series(work)
    work = work[minutes.fillna(0.0) >= float(min_minutes)].copy()
    if max_age is not None and "age" in work.columns:
        age = pd.to_numeric(work["age"], errors="coerce")
        work = work[age.fillna(999.0) <= float(max_age)].copy()
    if non_big5_only and "league" in work.columns:
        league_norm = work["league"].astype(str).str.strip().str.casefold()
        work = work[~league_norm.isin(BIG5_LEAGUES)].copy()
    if work.empty:
        return work

    work["_player_key"] = _player_key(work)
    work["_season_key"] = _season_key(work)
    work["_position"] = _position_series(work)
    work["_value_segment"] = _value_segment_series(work)
    return work


def _join_predictions_with_targets(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_cols = [
        "_player_key",
        "_season_key",
        "next_season",
        "next_market_value_eur",
        "next_minutes",
        "has_next_season_target",
        "value_growth_next_season_eur",
        "value_growth_next_season_pct",
        "value_growth_next_season_log_delta",
        "value_growth_positive_flag",
        "value_growth_gt25pct_flag",
    ]
    work = predictions.drop(
        columns=[col for col in target_cols if col in predictions.columns and col not in {"_player_key", "_season_key"}],
        errors="ignore",
    ).copy()
    available = [col for col in target_cols if col in targets.columns]
    merged = work.merge(targets[available], on=["_player_key", "_season_key"], how="left")
    matched = int(merged["has_next_season_target"].notna().sum()) if "has_next_season_target" in merged.columns else 0
    labeled = (
        int((pd.to_numeric(merged["has_next_season_target"], errors="coerce") == 1).sum())
        if "has_next_season_target" in merged.columns
        else 0
    )
    return merged, {
        "prediction_rows": int(len(predictions)),
        "target_rows": int(len(targets)),
        "matched_rows": matched,
        "matched_share": float(matched / max(len(predictions), 1)),
        "labeled_rows": labeled,
        "labeled_share": float(labeled / max(len(predictions), 1)),
    }


def attach_future_targets(frame: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    existing_target_cols = [
        "next_season",
        "next_market_value_eur",
        "next_minutes",
        "has_next_season_target",
        "value_growth_next_season_eur",
        "value_growth_next_season_pct",
        "value_growth_next_season_log_delta",
        "value_growth_positive_flag",
        "value_growth_gt25pct_flag",
    ]
    work = work.drop(columns=[col for col in existing_target_cols if col in work.columns], errors="ignore")
    work["_player_key"] = _player_key(work)
    work["_season_key"] = _season_key(work)
    return _join_predictions_with_targets(work, targets)


def _pearson(series_a: pd.Series, series_b: pd.Series) -> float | None:
    mask = series_a.notna() & series_b.notna()
    if int(mask.sum()) < 3:
        return None
    return _safe_float(series_a.loc[mask].astype(float).corr(series_b.loc[mask].astype(float), method="pearson"))


def _spearman(series_a: pd.Series, series_b: pd.Series) -> float | None:
    mask = series_a.notna() & series_b.notna()
    if int(mask.sum()) < 3:
        return None
    return _safe_float(series_a.loc[mask].astype(float).corr(series_b.loc[mask].astype(float), method="spearman"))


def _precision_rows_for_group(
    frame: pd.DataFrame,
    *,
    group_type: str,
    group_name: str,
    score_col: str,
    label_col: str,
    k_values: Sequence[int],
) -> list[dict[str, Any]]:
    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(work[label_col], errors="coerce")
    work["_growth_eur"] = pd.to_numeric(work.get("value_growth_next_season_eur"), errors="coerce")
    work["_growth_pct"] = pd.to_numeric(work.get("value_growth_next_season_pct"), errors="coerce")
    work = work[np.isfinite(work["_score"]) & work["_label"].notna()].copy()
    if work.empty:
        return []

    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    label_rate = float((work["_label"] > 0).mean())
    rows: list[dict[str, Any]] = []
    for k in _parse_int_k_values(k_values):
        top = work.head(k)
        if top.empty:
            continue
        rows.append(
            {
                "cohort_type": group_type,
                "cohort": group_name,
                "k": int(k),
                "n_pool": int(len(frame)),
                "n_labeled": int(len(work)),
                "n_eval": int(len(top)),
                "positive_rate": label_rate,
                "precision_at_k": float((top["_label"] > 0).mean()),
                "lift_vs_base": float((top["_label"] > 0).mean() - label_rate),
                "avg_growth_eur_top_k": _safe_float(top["_growth_eur"].mean()),
                "median_growth_eur_top_k": _safe_float(top["_growth_eur"].median()),
                "avg_growth_pct_top_k": _safe_float(top["_growth_pct"].mean()),
            }
        )
    return rows


def _cohort_precision_rows(
    frame: pd.DataFrame,
    *,
    score_col: str,
    label_col: str,
    k_values: Sequence[int],
    cohort_min_labeled: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _precision_rows_for_group(
            frame,
            group_type="overall",
            group_name="ALL",
            score_col=score_col,
            label_col=label_col,
            k_values=k_values,
        )
    )

    specs = [
        ("league", "league" if "league" in frame.columns else None),
        ("position", "_position"),
        ("value_segment", "_value_segment"),
    ]
    for group_type, col in specs:
        if not col or col not in frame.columns:
            continue
        grouped = frame.groupby(col, dropna=False)
        for key, group in grouped:
            n_labeled = int(pd.to_numeric(group[label_col], errors="coerce").notna().sum())
            if n_labeled < int(cohort_min_labeled):
                continue
            rows.extend(
                _precision_rows_for_group(
                    group,
                    group_type=group_type,
                    group_name=str(key),
                    score_col=score_col,
                    label_col=label_col,
                    k_values=k_values,
                )
            )
    return rows


def _top_realized_rows(frame: pd.DataFrame, *, score_col: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_growth_eur"] = pd.to_numeric(work.get("value_growth_next_season_eur"), errors="coerce")
    work["_growth_pct"] = pd.to_numeric(work.get("value_growth_next_season_pct"), errors="coerce")
    work = work[np.isfinite(work["_score"]) & work["_growth_eur"].notna()].copy()
    if work.empty:
        return []

    keep_cols = [
        "player_id",
        "name",
        "club",
        "league",
        "season",
        "next_season",
        "market_value_eur",
        "next_market_value_eur",
        "value_growth_next_season_eur",
        "value_growth_next_season_pct",
        score_col,
        "_position",
        "_value_segment",
    ]
    out_cols = [col for col in keep_cols if col in work.columns]
    work = work.sort_values("_score", ascending=False).head(limit)
    rows = work[out_cols].to_dict(orient="records")
    for row in rows:
        if "_position" in row:
            row["position"] = row.pop("_position")
        if "_value_segment" in row:
            row["value_segment"] = row.pop("_value_segment")
    return rows


def build_future_value_split_payload(
    *,
    predictions_path: str,
    targets: pd.DataFrame,
    split_label: str,
    score_col: str | None = None,
    k_values: Sequence[int] = (10, 25, 50),
    cohort_min_labeled: int = 25,
    min_minutes: float = 900.0,
    max_age: float | None = None,
    non_big5_only: bool = False,
    top_realized_limit: int = 25,
) -> dict[str, Any]:
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    predictions = pd.read_csv(pred_path, low_memory=False)
    predictions = _prepare_prediction_frame(
        predictions,
        min_minutes=min_minutes,
        max_age=max_age,
        non_big5_only=non_big5_only,
    )
    if predictions.empty:
        raise ValueError(f"No rows remain after filters for split '{split_label}'.")

    resolved_score_col = _resolve_score_col(predictions, explicit=score_col)
    merged, join_meta = _join_predictions_with_targets(predictions, targets)

    merged["_score"] = pd.to_numeric(merged[resolved_score_col], errors="coerce")
    labeled_mask = pd.to_numeric(
        merged["has_next_season_target"] if "has_next_season_target" in merged.columns else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    ) == 1
    labeled = merged[labeled_mask].copy()

    positive_col = "value_growth_positive_flag"
    gt25_col = "value_growth_gt25pct_flag"
    positive_rows = _cohort_precision_rows(
        labeled,
        score_col=resolved_score_col,
        label_col=positive_col,
        k_values=k_values,
        cohort_min_labeled=cohort_min_labeled,
    )
    gt25_rows = _cohort_precision_rows(
        labeled,
        score_col=resolved_score_col,
        label_col=gt25_col,
        k_values=k_values,
        cohort_min_labeled=cohort_min_labeled,
    )

    warnings: list[str] = []
    if join_meta["labeled_rows"] < 20:
        warnings.append("low_labeled_rows")
    if join_meta["labeled_share"] < 0.10:
        warnings.append("low_label_coverage")

    return {
        "predictions_path": str(pred_path),
        "split": split_label,
        "score_column": resolved_score_col,
        "k_values": _parse_int_k_values(k_values),
        "filters": {
            "min_minutes": float(min_minutes),
            "max_age": None if max_age is None else float(max_age),
            "non_big5_only": bool(non_big5_only),
            "cohort_min_labeled": int(cohort_min_labeled),
        },
        "join": join_meta,
        "warnings": warnings,
        "growth_summary": {
            "mean_growth_eur": _safe_float(pd.to_numeric(labeled.get("value_growth_next_season_eur"), errors="coerce").mean()),
            "median_growth_eur": _safe_float(pd.to_numeric(labeled.get("value_growth_next_season_eur"), errors="coerce").median()),
            "mean_growth_pct": _safe_float(pd.to_numeric(labeled.get("value_growth_next_season_pct"), errors="coerce").mean()),
            "median_growth_pct": _safe_float(pd.to_numeric(labeled.get("value_growth_next_season_pct"), errors="coerce").median()),
            "positive_growth_rate": _safe_float(pd.to_numeric(labeled.get(positive_col), errors="coerce").mean()),
            "growth_gt25pct_rate": _safe_float(pd.to_numeric(labeled.get(gt25_col), errors="coerce").mean()),
        },
        "score_correlation": {
            "pearson_growth_eur": _pearson(merged["_score"], pd.to_numeric(merged.get("value_growth_next_season_eur"), errors="coerce")),
            "spearman_growth_eur": _spearman(merged["_score"], pd.to_numeric(merged.get("value_growth_next_season_eur"), errors="coerce")),
            "pearson_growth_pct": _pearson(merged["_score"], pd.to_numeric(merged.get("value_growth_next_season_pct"), errors="coerce")),
            "spearman_growth_pct": _spearman(merged["_score"], pd.to_numeric(merged.get("value_growth_next_season_pct"), errors="coerce")),
        },
        "precision_at_k": {
            "positive_growth": positive_rows,
            "growth_gt25pct": gt25_rows,
        },
        "top_realized_rows": _top_realized_rows(labeled, score_col=resolved_score_col, limit=top_realized_limit),
    }


def build_future_value_benchmark_payload(
    *,
    test_predictions_path: str | None = None,
    val_predictions_path: str | None = None,
    future_targets_path: str | None = None,
    dataset_path: str | None = None,
    score_col: str | None = None,
    k_values: Sequence[int] = (10, 25, 50),
    cohort_min_labeled: int = 25,
    min_next_minutes: float = 450.0,
    min_minutes: float = 900.0,
    max_age: float | None = None,
    non_big5_only: bool = False,
    top_realized_limit: int = 25,
) -> dict[str, Any]:
    if not test_predictions_path and not val_predictions_path:
        raise ValueError("Need at least one predictions path for the future benchmark.")

    targets_raw, target_meta = _load_future_targets(
        future_targets_path=future_targets_path,
        dataset_path=dataset_path,
        min_next_minutes=min_next_minutes,
    )
    targets = _prepare_target_frame(targets_raw)

    splits: dict[str, Any] = {}
    if val_predictions_path:
        splits["val"] = build_future_value_split_payload(
            predictions_path=val_predictions_path,
            targets=targets,
            split_label="val",
            score_col=score_col,
            k_values=k_values,
            cohort_min_labeled=cohort_min_labeled,
            min_minutes=min_minutes,
            max_age=max_age,
            non_big5_only=non_big5_only,
            top_realized_limit=top_realized_limit,
        )
    if test_predictions_path:
        splits["test"] = build_future_value_split_payload(
            predictions_path=test_predictions_path,
            targets=targets,
            split_label="test",
            score_col=score_col,
            k_values=k_values,
            cohort_min_labeled=cohort_min_labeled,
            min_minutes=min_minutes,
            max_age=max_age,
            non_big5_only=non_big5_only,
            top_realized_limit=top_realized_limit,
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_source": target_meta,
        "splits": splits,
    }


def _precision_rows_for_markdown(rows: list[dict[str, Any]], *, k: int, cohort_type: str) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if str(row.get("cohort_type")) == str(cohort_type) and int(row.get("k") or 0) == int(k)
    ]
    return sorted(
        filtered,
        key=lambda row: (
            -(row.get("precision_at_k") or -1.0),
            -(row.get("lift_vs_base") or -999.0),
            str(row.get("cohort") or ""),
        ),
    )


def write_future_value_benchmark_report(
    payload: dict[str, Any],
    *,
    out_json: str,
    out_md: str,
    cohort_top_n: int = 5,
) -> dict[str, str]:
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Future Value Benchmark")
    lines.append("")
    source = payload.get("target_source", {})
    lines.append(f"- Generated: `{payload.get('generated_at_utc')}`")
    lines.append(f"- Target source: `{source.get('source', 'unknown')}`")
    if source.get("dataset_path"):
        lines.append(f"- Dataset: `{source.get('dataset_path')}`")
    if source.get("future_targets_path"):
        lines.append(f"- Future targets parquet: `{source.get('future_targets_path')}`")
    lines.append(f"- Min next-season minutes: `{source.get('min_next_minutes')}`")
    lines.append("")

    for split_name, split in (payload.get("splits") or {}).items():
        join = split.get("join", {})
        growth = split.get("growth_summary", {})
        corr = split.get("score_correlation", {})
        lines.append(f"## {str(split_name).upper()} Split")
        lines.append("")
        lines.append(f"- Predictions: `{split.get('predictions_path')}`")
        lines.append(f"- Score column: `{split.get('score_column')}`")
        lines.append(
            f"- Labeled coverage: `{join.get('labeled_rows', 0)}/{join.get('prediction_rows', 0)}` "
            f"(`{(join.get('labeled_share') or 0.0) * 100:.2f}%`)"
        )
        lines.append(
            f"- Positive growth rate: `{((growth.get('positive_growth_rate') or 0.0) * 100):.2f}%` | "
            f"`>=25%` growth rate: `{((growth.get('growth_gt25pct_rate') or 0.0) * 100):.2f}%`"
        )
        lines.append(
            f"- Score correlation vs growth: `spearman_eur={corr.get('spearman_growth_eur')}` | "
            f"`spearman_pct={corr.get('spearman_growth_pct')}`"
        )
        warnings = split.get("warnings") or []
        if warnings:
            lines.append(f"- Warnings: `{', '.join(str(item) for item in warnings)}`")
        lines.append("")

        for label_key, title in [
            ("positive_growth", "Precision@K: Positive Growth"),
            ("growth_gt25pct", "Precision@K: Growth >= 25%"),
        ]:
            rows = split.get("precision_at_k", {}).get(label_key, [])
            lines.append(f"### {title}")
            overall_rows = _precision_rows_for_markdown(rows, k=25, cohort_type="overall")
            if overall_rows:
                row = overall_rows[0]
                lines.append(
                    f"- Overall `precision@25={((row.get('precision_at_k') or 0.0) * 100):.2f}%` "
                    f"vs base `{((row.get('positive_rate') or 0.0) * 100):.2f}%`"
                )
            else:
                lines.append("- No overall precision rows available.")
            league_rows = _precision_rows_for_markdown(rows, k=25, cohort_type="league")
            if league_rows:
                lines.append("")
                lines.append("| league | precision@25 | base rate | lift | labeled n |")
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for row in league_rows[: max(int(cohort_top_n), 1)]:
                    lines.append(
                        f"| {row.get('cohort')} | "
                        f"{((row.get('precision_at_k') or 0.0) * 100):.2f}% | "
                        f"{((row.get('positive_rate') or 0.0) * 100):.2f}% | "
                        f"{((row.get('lift_vs_base') or 0.0) * 100):.2f}% | "
                        f"{int(row.get('n_labeled') or 0)} |"
                    )
            else:
                lines.append("")
                lines.append("No league cohort rows available at `k=25`.")
            lines.append("")

    md_path = Path(out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
