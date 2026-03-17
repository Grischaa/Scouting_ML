from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scouting_ml.utils.metrics import regression_metrics


VALUE_SEGMENTS = [
    ("under_5m", 0.0, 5_000_000.0),
    ("5m_to_20m", 5_000_000.0, 20_000_000.0),
    ("over_20m", 20_000_000.0, float("inf")),
]


def season_slug(season: str) -> str:
    return season.replace("/", "-")


def slugify(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def resolve_position_series(frame: pd.DataFrame) -> pd.Series:
    for col in ("model_position", "position_group", "position_main"):
        if col in frame.columns:
            return frame[col].astype(str).str.upper().str.strip()
    return pd.Series("UNKNOWN", index=frame.index, dtype=object)


def resolve_league_series(frame: pd.DataFrame) -> pd.Series:
    if "league" in frame.columns:
        return frame["league"].astype(str).str.strip()
    return pd.Series("UNKNOWN", index=frame.index, dtype=object)


def value_segment_series(frame: pd.DataFrame) -> pd.Series:
    if "value_segment" in frame.columns:
        raw = frame["value_segment"].astype(str).str.strip()
        valid = {label for label, _, _ in VALUE_SEGMENTS}
        return raw.where(raw.isin(valid), "unknown").astype(str)

    market = pd.to_numeric(frame.get("market_value_eur"), errors="coerce")
    out = pd.Series("unknown", index=frame.index, dtype=object)
    for label, lo, hi in VALUE_SEGMENTS:
        out.loc[(market >= lo) & (market < hi)] = label
    return out.astype(str)


def load_prediction_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return frame
    out = frame.copy()
    out["market_value_eur"] = pd.to_numeric(out["market_value_eur"], errors="coerce")
    out["expected_value_eur"] = pd.to_numeric(out["expected_value_eur"], errors="coerce")
    out = out[np.isfinite(out["market_value_eur"]) & np.isfinite(out["expected_value_eur"])].copy()
    out["_position"] = resolve_position_series(out)
    out["_league"] = resolve_league_series(out)
    out["_value_segment"] = value_segment_series(out)
    return out


def metric_row(
    frame: pd.DataFrame,
    *,
    config: str,
    split: str,
    slice_type: str,
    slice_key: str,
    slice_label: str,
    note: str,
    position: str = "",
    league: str = "",
    value_segment: str = "",
    mape_min_denom_eur: float = 1_000_000.0,
) -> dict[str, Any]:
    metrics = regression_metrics(
        frame["market_value_eur"].to_numpy(),
        frame["expected_value_eur"].to_numpy(),
        mape_min_denom=mape_min_denom_eur,
    )
    return {
        "config": config,
        "note": note,
        "split": split,
        "slice_type": slice_type,
        "slice_key": slice_key,
        "slice_label": slice_label,
        "position": position,
        "league": league,
        "value_segment": value_segment,
        "n_samples": int(len(frame)),
        **metrics,
    }


def group_metric_rows(
    frame: pd.DataFrame,
    *,
    config: str,
    split: str,
    slice_type: str,
    group_cols: Sequence[str],
    note: str,
    min_samples: int,
    mape_min_denom_eur: float = 1_000_000.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows

    work = frame.copy()
    for col in group_cols:
        work[col] = work[col].fillna("unknown").astype(str)

    for key, group in work.groupby(list(group_cols), dropna=False):
        if len(group) < int(min_samples):
            continue
        if not isinstance(key, tuple):
            key = (key,)
        parts = dict(zip(group_cols, key))
        position = str(parts.get("_position", ""))
        league = str(parts.get("_league", ""))
        value_segment = str(parts.get("_value_segment", ""))
        label_parts: list[str] = []
        if position:
            label_parts.append(position)
        if value_segment:
            label_parts.append(value_segment)
        if league:
            label_parts.append(league)
        slice_label = " | ".join(label_parts) or "ALL"
        slice_key = "||".join(label_parts) or "ALL"
        rows.append(
            metric_row(
                group,
                config=config,
                split=split,
                slice_type=slice_type,
                slice_key=slice_key,
                slice_label=slice_label,
                note=note,
                position=position,
                league=league,
                value_segment=value_segment,
                mape_min_denom_eur=mape_min_denom_eur,
            )
        )
    return rows


def build_slice_matrix(
    frame: pd.DataFrame,
    *,
    config: str,
    split: str,
    note: str,
    slice_min_samples: int,
    league_min_samples: int,
    mape_min_denom_eur: float = 1_000_000.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows

    rows.append(
        metric_row(
            frame,
            config=config,
            split=split,
            slice_type="overall",
            slice_key="ALL",
            slice_label="ALL",
            note=note,
            mape_min_denom_eur=mape_min_denom_eur,
        )
    )
    rows.extend(
        group_metric_rows(
            frame,
            config=config,
            split=split,
            slice_type="position",
            group_cols=("_position",),
            note=note,
            min_samples=slice_min_samples,
            mape_min_denom_eur=mape_min_denom_eur,
        )
    )
    rows.extend(
        group_metric_rows(
            frame,
            config=config,
            split=split,
            slice_type="value_segment",
            group_cols=("_value_segment",),
            note=note,
            min_samples=slice_min_samples,
            mape_min_denom_eur=mape_min_denom_eur,
        )
    )
    rows.extend(
        group_metric_rows(
            frame,
            config=config,
            split=split,
            slice_type="league",
            group_cols=("_league",),
            note=note,
            min_samples=league_min_samples,
            mape_min_denom_eur=mape_min_denom_eur,
        )
    )
    rows.extend(
        group_metric_rows(
            frame,
            config=config,
            split=split,
            slice_type="position_value_segment",
            group_cols=("_position", "_value_segment"),
            note=note,
            min_samples=slice_min_samples,
            mape_min_denom_eur=mape_min_denom_eur,
        )
    )

    lowmid = frame[frame["_value_segment"].isin(["under_5m", "5m_to_20m"])].copy()
    if len(lowmid) >= int(slice_min_samples):
        rows.append(
            metric_row(
                lowmid,
                config=config,
                split=split,
                slice_type="value_focus",
                slice_key="under_20m",
                slice_label="under_20m",
                note=note,
                value_segment="under_20m",
                mape_min_denom_eur=mape_min_denom_eur,
            )
        )
    return rows


def lookup_slice_metric(
    slice_df: pd.DataFrame,
    *,
    config: str,
    split: str,
    slice_type: str,
    slice_key: str,
    metric: str,
) -> float:
    if slice_df.empty:
        return float("nan")
    mask = (
        (slice_df["config"].astype(str) == str(config))
        & (slice_df["split"].astype(str) == str(split))
        & (slice_df["slice_type"].astype(str) == str(slice_type))
        & (slice_df["slice_key"].astype(str) == str(slice_key))
    )
    if not mask.any():
        return float("nan")
    return safe_float(slice_df.loc[mask, metric].iloc[0])


def decorate_slice_matrix(slice_df: pd.DataFrame) -> pd.DataFrame:
    if slice_df.empty:
        return slice_df

    out = slice_df.copy()
    ref = out.loc[out["config"].astype(str) == "full"].copy()
    metric_cols = ["r2", "mae_eur", "mape", "wmape"]
    if not ref.empty:
        ref = ref[
            ["split", "slice_type", "slice_key", *metric_cols]
        ].rename(columns={col: f"{col}_full" for col in metric_cols})
        out = out.merge(ref, on=["split", "slice_type", "slice_key"], how="left")
        for col in metric_cols:
            out[f"delta_{col}_vs_full"] = pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(
                out.get(f"{col}_full"),
                errors="coerce",
            )
    else:
        for col in metric_cols:
            out[f"delta_{col}_vs_full"] = float("nan")

    out["wmape_rank_within_slice"] = (
        out.groupby(["split", "slice_type", "slice_key"], dropna=False)["wmape"]
        .rank(method="dense", ascending=True, na_option="bottom")
    )
    out["r2_rank_within_slice"] = (
        out.groupby(["split", "slice_type", "slice_key"], dropna=False)["r2"]
        .rank(method="dense", ascending=False, na_option="bottom")
    )
    return out


def build_overall_summary(run_rows: list[dict[str, Any]], slice_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for base in run_rows:
        config = str(base["config"])
        row = dict(base)
        for split in ("val", "test"):
            overall_mask = (
                (slice_df["config"].astype(str) == config)
                & (slice_df["split"].astype(str) == split)
                & (slice_df["slice_type"].astype(str) == "overall")
                & (slice_df["slice_key"].astype(str) == "ALL")
            )
            overall = slice_df.loc[overall_mask]
            if overall.empty:
                row.update(
                    {
                        f"{split}_n_samples": 0,
                        f"{split}_r2": float("nan"),
                        f"{split}_mae_eur": float("nan"),
                        f"{split}_mape": float("nan"),
                        f"{split}_wmape": float("nan"),
                    }
                )
            else:
                item = overall.iloc[0]
                row.update(
                    {
                        f"{split}_n_samples": int(item["n_samples"]),
                        f"{split}_r2": safe_float(item["r2"]),
                        f"{split}_mae_eur": safe_float(item["mae_eur"]),
                        f"{split}_mape": safe_float(item["mape"]),
                        f"{split}_wmape": safe_float(item["wmape"]),
                    }
                )

            for slice_type, slice_key, suffix in [
                ("value_segment", "under_5m", "under_5m"),
                ("value_segment", "5m_to_20m", "5m_to_20m"),
                ("value_segment", "over_20m", "over_20m"),
                ("value_focus", "under_20m", "under_20m"),
            ]:
                row[f"{split}_{suffix}_wmape"] = lookup_slice_metric(
                    slice_df,
                    config=config,
                    split=split,
                    slice_type=slice_type,
                    slice_key=slice_key,
                    metric="wmape",
                )
                row[f"{split}_{suffix}_r2"] = lookup_slice_metric(
                    slice_df,
                    config=config,
                    split=split,
                    slice_type=slice_type,
                    slice_key=slice_key,
                    metric="r2",
                )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    metric_cols = [
        col
        for col in summary.columns
        if col.endswith("_r2") or col.endswith("_mae_eur") or col.endswith("_mape") or col.endswith("_wmape")
    ]
    if "full" in set(summary["config"].astype(str)):
        ref = summary.loc[summary["config"].astype(str) == "full"].iloc[0]
        for col in metric_cols:
            summary[f"delta_{col}_vs_full"] = pd.to_numeric(summary[col], errors="coerce") - safe_float(ref[col])

    summary["test_wmape_rank"] = summary["test_wmape"].rank(method="dense", ascending=True, na_option="bottom")
    summary["test_under_20m_wmape_rank"] = summary["test_under_20m_wmape"].rank(
        method="dense",
        ascending=True,
        na_option="bottom",
    )
    summary = summary.sort_values(
        ["test_wmape", "test_under_20m_wmape", "test_r2"],
        ascending=[True, True, False],
        na_position="last",
    )
    return summary


def pick_best_overall(summary: pd.DataFrame, metric_col: str, *, exclude_configs: Sequence[str] | None = None) -> dict[str, Any] | None:
    if summary.empty or metric_col not in summary.columns:
        return None
    work = summary[np.isfinite(pd.to_numeric(summary[metric_col], errors="coerce"))].copy()
    if exclude_configs:
        excluded = {str(item) for item in exclude_configs}
        work = work[~work["config"].astype(str).isin(excluded)].copy()
    if work.empty:
        return None
    work = work.sort_values([metric_col, "test_r2"], ascending=[True, False], na_position="last")
    row = work.iloc[0]
    payload = {
        "config": str(row["config"]),
        "metric": metric_col,
        "value": safe_float(row[metric_col]),
        "test_r2": safe_float(row.get("test_r2")),
        "note": str(row.get("note", "")),
    }
    for extra in (
        "column",
        "under_5m_weight",
        "mid_5m_to_20m_weight",
        "over_20m_weight",
        "optimize_metric",
    ):
        if extra in row and pd.notna(row[extra]):
            payload[extra] = row[extra]
    return payload
