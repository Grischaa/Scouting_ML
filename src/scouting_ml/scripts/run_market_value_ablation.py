from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from scouting_ml.utils.metrics import regression_metrics


VALUE_SEGMENTS = [
    ("under_5m", 0.0, 5_000_000.0),
    ("5m_to_20m", 5_000_000.0, 20_000_000.0),
    ("over_20m", 20_000_000.0, float("inf")),
]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    exclude_prefixes: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()
    note: str = ""


DEFAULT_CONFIGS: Dict[str, AblationConfig] = {
    "full": AblationConfig("full", note="All available features."),
    "no_provider": AblationConfig(
        "no_provider",
        exclude_prefixes=("sb_", "avail_", "fixture_", "odds_"),
        note="Remove provider-derived StatsBomb / availability / fixture / odds features.",
    ),
    "no_contract": AblationConfig(
        "no_contract",
        exclude_prefixes=("contract_",),
        note="Remove contract-related features.",
    ),
    "no_injury": AblationConfig(
        "no_injury",
        exclude_prefixes=("injury_",),
        note="Remove injury-history features.",
    ),
    "no_transfer": AblationConfig(
        "no_transfer",
        exclude_prefixes=("transfer_",),
        note="Remove transfer-history features.",
    ),
    "no_national": AblationConfig(
        "no_national",
        exclude_prefixes=("nt_",),
        note="Remove national-team features.",
    ),
    "no_context": AblationConfig(
        "no_context",
        exclude_prefixes=("clubctx_", "leaguectx_", "uefa_coeff_"),
        note="Remove club/league environment features.",
    ),
    "no_profile_context": AblationConfig(
        "no_profile_context",
        exclude_prefixes=("contract_", "injury_", "transfer_", "nt_"),
        note="Remove external player-profile context features.",
    ),
    "baseline_stats_only": AblationConfig(
        "baseline_stats_only",
        exclude_prefixes=(
            "contract_",
            "injury_",
            "transfer_",
            "nt_",
            "clubctx_",
            "leaguectx_",
            "uefa_coeff_",
            "sb_",
            "avail_",
            "fixture_",
            "odds_",
        ),
        note="Only baseline player/season stat features.",
    ),
}


def _train_market_value_main(**kwargs) -> None:
    from scouting_ml.models.train_market_value_full import main as train_market_value_main

    train_market_value_main(**kwargs)


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _season_slug(season: str) -> str:
    return season.replace("/", "-")


def _slugify(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _resolve_position_series(frame: pd.DataFrame) -> pd.Series:
    for col in ("model_position", "position_group", "position_main"):
        if col in frame.columns:
            return frame[col].astype(str).str.upper().str.strip()
    return pd.Series("UNKNOWN", index=frame.index, dtype=object)


def _resolve_league_series(frame: pd.DataFrame) -> pd.Series:
    if "league" in frame.columns:
        return frame["league"].astype(str).str.strip()
    return pd.Series("UNKNOWN", index=frame.index, dtype=object)


def _value_segment_series(frame: pd.DataFrame) -> pd.Series:
    if "value_segment" in frame.columns:
        raw = frame["value_segment"].astype(str).str.strip()
        valid = {label for label, _, _ in VALUE_SEGMENTS}
        out = raw.where(raw.isin(valid), "unknown")
        return out.astype(str)

    market = pd.to_numeric(frame.get("market_value_eur"), errors="coerce")
    out = pd.Series("unknown", index=frame.index, dtype=object)
    for label, lo, hi in VALUE_SEGMENTS:
        out.loc[(market >= lo) & (market < hi)] = label
    return out.astype(str)


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return frame
    out = frame.copy()
    out["market_value_eur"] = pd.to_numeric(out["market_value_eur"], errors="coerce")
    out["expected_value_eur"] = pd.to_numeric(out["expected_value_eur"], errors="coerce")
    out = out[np.isfinite(out["market_value_eur"]) & np.isfinite(out["expected_value_eur"])].copy()
    out["_position"] = _resolve_position_series(out)
    out["_league"] = _resolve_league_series(out)
    out["_value_segment"] = _value_segment_series(out)
    return out


def _metric_row(
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


def _group_metric_rows(
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
            _metric_row(
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


def _build_slice_matrix(
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
        _metric_row(
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
        _group_metric_rows(
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
        _group_metric_rows(
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
        _group_metric_rows(
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
        _group_metric_rows(
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
            _metric_row(
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


def _lookup_slice_metric(
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
    return _safe_float(slice_df.loc[mask, metric].iloc[0])


def _decorate_slice_matrix(slice_df: pd.DataFrame) -> pd.DataFrame:
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


def _build_overall_summary(
    run_rows: list[dict[str, Any]],
    slice_df: pd.DataFrame,
) -> pd.DataFrame:
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
                        f"{split}_r2": _safe_float(item["r2"]),
                        f"{split}_mae_eur": _safe_float(item["mae_eur"]),
                        f"{split}_mape": _safe_float(item["mape"]),
                        f"{split}_wmape": _safe_float(item["wmape"]),
                    }
                )

            for slice_type, slice_key, suffix in [
                ("value_segment", "under_5m", "under_5m"),
                ("value_segment", "5m_to_20m", "5m_to_20m"),
                ("value_segment", "over_20m", "over_20m"),
                ("value_focus", "under_20m", "under_20m"),
            ]:
                row[f"{split}_{suffix}_wmape"] = _lookup_slice_metric(
                    slice_df,
                    config=config,
                    split=split,
                    slice_type=slice_type,
                    slice_key=slice_key,
                    metric="wmape",
                )
                row[f"{split}_{suffix}_r2"] = _lookup_slice_metric(
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
            summary[f"delta_{col}_vs_full"] = pd.to_numeric(summary[col], errors="coerce") - _safe_float(ref[col])

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


def _best_config_rows(
    frame: pd.DataFrame,
    *,
    split: str,
    slice_type: str,
    min_samples: int,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame[
        (frame["split"].astype(str) == str(split))
        & (frame["slice_type"].astype(str) == str(slice_type))
        & (pd.to_numeric(frame["n_samples"], errors="coerce") >= int(min_samples))
    ].copy()
    if work.empty:
        return []
    work = work.sort_values(
        ["slice_key", "wmape", "r2", "n_samples"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    winners = work.groupby("slice_key", dropna=False).head(1).reset_index(drop=True)
    return winners.to_dict(orient="records")


def _top_full_weak_slices(
    frame: pd.DataFrame,
    *,
    split: str,
    min_samples: int,
    limit: int,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame[
        (frame["config"].astype(str) == "full")
        & (frame["split"].astype(str) == str(split))
        & (frame["slice_type"].astype(str) != "overall")
        & (pd.to_numeric(frame["n_samples"], errors="coerce") >= int(min_samples))
    ].copy()
    if work.empty:
        return []
    work = work.sort_values(
        ["wmape", "mae_eur", "n_samples"],
        ascending=[False, False, False],
        na_position="last",
    )
    cols = [
        "slice_type",
        "slice_key",
        "slice_label",
        "position",
        "league",
        "value_segment",
        "n_samples",
        "r2",
        "mae_eur",
        "mape",
        "wmape",
    ]
    return work.loc[:, cols].head(int(limit)).to_dict(orient="records")


def _pick_best_overall(summary: pd.DataFrame, metric_col: str) -> dict[str, Any] | None:
    if summary.empty or metric_col not in summary.columns:
        return None
    work = summary[np.isfinite(pd.to_numeric(summary[metric_col], errors="coerce"))].copy()
    if work.empty:
        return None
    work = work.sort_values([metric_col, "test_r2"], ascending=[True, False], na_position="last")
    row = work.iloc[0]
    return {
        "config": str(row["config"]),
        "metric": metric_col,
        "value": _safe_float(row[metric_col]),
        "test_r2": _safe_float(row.get("test_r2")),
        "note": str(row.get("note", "")),
    }


def _build_report_bundle(
    *,
    dataset_path: str,
    val_season: str,
    test_season: str,
    out_dir: Path,
    summary: pd.DataFrame,
    slice_df: pd.DataFrame,
    report_top_n: int,
) -> dict[str, Any]:
    overall_best = _pick_best_overall(summary, "test_wmape")
    cheap_best = _pick_best_overall(summary, "test_under_20m_wmape")
    under5_best = _pick_best_overall(summary, "test_under_5m_wmape")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "val_season": val_season,
        "test_season": test_season,
        "artifact_dir": str(out_dir),
        "best_overall_test": overall_best,
        "best_under_20m_test": cheap_best,
        "best_under_5m_test": under5_best,
        "position_winners_test": _best_config_rows(
            slice_df,
            split="test",
            slice_type="position",
            min_samples=1,
        ),
        "value_segment_winners_test": _best_config_rows(
            slice_df,
            split="test",
            slice_type="value_segment",
            min_samples=1,
        ),
        "weakest_full_slices_test": _top_full_weak_slices(
            slice_df,
            split="test",
            min_samples=1,
            limit=report_top_n,
        ),
        "artifacts": {
            "overall_summary_csv": str(out_dir / f"ablation_summary_{_season_slug(test_season)}.csv"),
            "slice_matrix_csv": str(out_dir / f"ablation_slices_{_season_slug(test_season)}.csv"),
            "bundle_json": str(out_dir / f"ablation_bundle_{_season_slug(test_season)}.json"),
            "report_md": str(out_dir / f"ablation_report_{_season_slug(test_season)}.md"),
        },
    }
    return payload


def _write_markdown_report(
    *,
    path: Path,
    dataset_path: str,
    val_season: str,
    test_season: str,
    summary: pd.DataFrame,
    bundle: dict[str, Any],
    report_top_n: int,
) -> None:
    lines: list[str] = []
    lines.append("# Market Value Ablation Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Validation season: `{val_season}`")
    lines.append(f"- Test season: `{test_season}`")
    lines.append(f"- Generated: `{bundle.get('generated_at_utc')}`")
    lines.append("")

    def _best_line(title: str, item: dict[str, Any] | None) -> None:
        if not item:
            lines.append(f"- {title}: unavailable")
            return
        lines.append(
            f"- {title}: `{item['config']}` | {item['metric']}={item['value']:.4f} | "
            f"test_r2={item['test_r2']:.4f}"
        )

    lines.append("## Best Configs")
    _best_line("Best overall test config", bundle.get("best_overall_test"))
    _best_line("Best under-20m test config", bundle.get("best_under_20m_test"))
    _best_line("Best under-5m test config", bundle.get("best_under_5m_test"))
    lines.append("")

    lines.append("## Overall Ranking")
    if summary.empty:
        lines.append("No ablation rows produced.")
    else:
        cols = [
            "config",
            "test_wmape",
            "test_r2",
            "test_under_20m_wmape",
            "test_under_5m_wmape",
            "note",
        ]
        work = summary.loc[:, [col for col in cols if col in summary.columns]].copy()
        work = work.head(max(int(report_top_n), 1))
        lines.append("| config | test_wmape | test_r2 | test_under_20m_wmape | test_under_5m_wmape | note |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for _, row in work.iterrows():
            lines.append(
                f"| {row['config']} | {row['test_wmape']:.4f} | {row['test_r2']:.4f} | "
                f"{_safe_float(row.get('test_under_20m_wmape')):.4f} | "
                f"{_safe_float(row.get('test_under_5m_wmape')):.4f} | "
                f"{row.get('note', '')} |"
            )
    lines.append("")

    position_winners = bundle.get("position_winners_test", [])
    lines.append("## Best By Position")
    if position_winners:
        lines.append("| position | best config | n | wmape | r2 | delta_wmape_vs_full |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in position_winners:
            lines.append(
                f"| {row.get('position') or row.get('slice_key')} | {row.get('config')} | "
                f"{int(row.get('n_samples', 0))} | { _safe_float(row.get('wmape')):.4f} | "
                f"{ _safe_float(row.get('r2')):.4f} | { _safe_float(row.get('delta_wmape_vs_full')):.4f} |"
            )
    else:
        lines.append("No position winners available.")
    lines.append("")

    value_winners = bundle.get("value_segment_winners_test", [])
    lines.append("## Best By Value Segment")
    if value_winners:
        lines.append("| segment | best config | n | wmape | r2 | delta_wmape_vs_full |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in value_winners:
            lines.append(
                f"| {row.get('value_segment') or row.get('slice_key')} | {row.get('config')} | "
                f"{int(row.get('n_samples', 0))} | { _safe_float(row.get('wmape')):.4f} | "
                f"{ _safe_float(row.get('r2')):.4f} | { _safe_float(row.get('delta_wmape_vs_full')):.4f} |"
            )
    else:
        lines.append("No value-segment winners available.")
    lines.append("")

    weak_rows = bundle.get("weakest_full_slices_test", [])
    lines.append("## Weakest Full-Model Test Slices")
    if weak_rows:
        lines.append("| slice_type | slice | n | wmape | r2 |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in weak_rows[: max(int(report_top_n), 1)]:
            lines.append(
                f"| {row.get('slice_type')} | {row.get('slice_label')} | "
                f"{int(row.get('n_samples', 0))} | { _safe_float(row.get('wmape')):.4f} | "
                f"{ _safe_float(row.get('r2')):.4f} |"
            )
    else:
        lines.append("No weak-slice diagnostics available.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(
    dataset_path: str,
    val_season: str,
    test_season: str,
    out_dir: str,
    config_names: Sequence[str],
    trials: int = 60,
    recency_half_life: float = 2.0,
    under_5m_weight: float = 1.0,
    mid_5m_to_20m_weight: float = 1.0,
    over_20m_weight: float = 1.0,
    optimize_metric: str = "hybrid_wmape",
    min_feature_coverage: float = 0.01,
    min_provider_feature_coverage: float = 0.05,
    slice_min_samples: int = 25,
    league_min_samples: int = 40,
    report_top_n: int = 8,
    mape_min_denom_eur: float = 1_000_000.0,
) -> dict[str, Any]:
    selected: List[AblationConfig] = []
    for name in config_names:
        key = name.strip()
        if key not in DEFAULT_CONFIGS:
            raise ValueError(
                f"Unknown ablation config '{key}'. Available: {sorted(DEFAULT_CONFIGS.keys())}"
            )
        selected.append(DEFAULT_CONFIGS[key])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for cfg in selected:
        print("\n==============================")
        print(f"[ablation] running: {cfg.name}")
        print(f"[ablation] note: {cfg.note}")
        print("==============================")

        stem = f"{cfg.name}_{_season_slug(test_season)}"
        pred_path = out_path / f"{stem}.csv"
        val_pred_path = out_path / f"{stem}_val.csv"
        metrics_path = out_path / f"{stem}.metrics.json"

        _train_market_value_main(
            dataset_path=dataset_path,
            val_season=val_season,
            test_season=test_season,
            output_path=str(pred_path),
            val_output_path=str(val_pred_path),
            metrics_output_path=str(metrics_path),
            n_optuna_trials=trials,
            recency_half_life=recency_half_life,
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
            exclude_prefixes=list(cfg.exclude_prefixes),
            exclude_columns=list(cfg.exclude_columns),
            optimize_metric=optimize_metric,
            min_feature_coverage=min_feature_coverage,
            min_provider_feature_coverage=min_provider_feature_coverage,
        )

        if not metrics_path.exists():
            raise RuntimeError(f"Ablation run '{cfg.name}' did not produce metrics file: {metrics_path}")
        if not pred_path.exists() or not val_pred_path.exists():
            raise RuntimeError(
                f"Ablation run '{cfg.name}' did not produce prediction outputs: "
                f"test={pred_path.exists()} val={val_pred_path.exists()}"
            )

        run_rows.append(
            {
                "config": cfg.name,
                "exclude_prefixes": ",".join(cfg.exclude_prefixes),
                "exclude_columns": ",".join(cfg.exclude_columns),
                "note": cfg.note,
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_path),
                "val_predictions_path": str(val_pred_path),
            }
        )

        for split, path in [("val", val_pred_path), ("test", pred_path)]:
            frame = _load_predictions(path)
            split_rows = _build_slice_matrix(
                frame,
                config=cfg.name,
                split=split,
                note=cfg.note,
                slice_min_samples=slice_min_samples,
                league_min_samples=league_min_samples,
                mape_min_denom_eur=mape_min_denom_eur,
            )
            slice_rows.extend(split_rows)

    slice_df = pd.DataFrame(slice_rows)
    if slice_df.empty:
        raise RuntimeError("Ablation study produced no slice diagnostics.")
    slice_df = _decorate_slice_matrix(slice_df)

    summary = _build_overall_summary(run_rows, slice_df)

    summary_path = out_path / f"ablation_summary_{_season_slug(test_season)}.csv"
    slices_path = out_path / f"ablation_slices_{_season_slug(test_season)}.csv"
    bundle_path = out_path / f"ablation_bundle_{_season_slug(test_season)}.json"
    report_path = out_path / f"ablation_report_{_season_slug(test_season)}.md"

    summary.to_csv(summary_path, index=False)
    slice_df.to_csv(slices_path, index=False)

    bundle = _build_report_bundle(
        dataset_path=dataset_path,
        val_season=val_season,
        test_season=test_season,
        out_dir=out_path,
        summary=summary,
        slice_df=slice_df,
        report_top_n=report_top_n,
    )
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    _write_markdown_report(
        path=report_path,
        dataset_path=dataset_path,
        val_season=val_season,
        test_season=test_season,
        summary=summary,
        bundle=bundle,
        report_top_n=report_top_n,
    )

    print("\n========== ABLATION SUMMARY ==========")
    for _, row in summary.iterrows():
        print(
            f"{row['config']:>20} | "
            f"test R2 {row['test_r2']*100:6.2f}% | "
            f"WMAPE {row['test_wmape']*100:6.2f}% | "
            f"under_20m WMAPE {_safe_float(row.get('test_under_20m_wmape'))*100:6.2f}% | "
            f"under_5m WMAPE {_safe_float(row.get('test_under_5m_wmape'))*100:6.2f}%"
        )
    print(f"[ablation] wrote overall summary -> {summary_path}")
    print(f"[ablation] wrote slice matrix -> {slices_path}")
    print(f"[ablation] wrote bundle -> {bundle_path}")
    print(f"[ablation] wrote report -> {report_path}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature-group ablations for the market-value pipeline."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument(
        "--configs",
        default="full,no_provider,no_contract,no_injury,no_transfer,no_national,no_context,no_profile_context,baseline_stats_only",
        help=f"Comma-separated config names. Available: {','.join(sorted(DEFAULT_CONFIGS.keys()))}",
    )
    parser.add_argument("--out-dir", default="data/model/ablation")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
    parser.add_argument(
        "--optimize-metric",
        default="hybrid_wmape",
        choices=["mae", "rmse", "overall_wmape", "band_wmape", "lowmid_wmape", "hybrid_wmape"],
    )
    parser.add_argument("--min-feature-coverage", type=float, default=0.01)
    parser.add_argument("--min-provider-feature-coverage", type=float, default=0.05)
    parser.add_argument("--slice-min-samples", type=int, default=25)
    parser.add_argument("--league-min-samples", type=int, default=40)
    parser.add_argument("--report-top-n", type=int, default=8)
    parser.add_argument("--mape-min-denom-eur", type=float, default=1_000_000.0)
    args = parser.parse_args()

    run_ablation(
        dataset_path=args.dataset,
        val_season=args.val_season,
        test_season=args.test_season,
        out_dir=args.out_dir,
        config_names=_parse_csv_tokens(args.configs),
        trials=args.trials,
        recency_half_life=args.recency_half_life,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_20m_weight,
        over_20m_weight=args.over_20m_weight,
        optimize_metric=args.optimize_metric,
        min_feature_coverage=args.min_feature_coverage,
        min_provider_feature_coverage=args.min_provider_feature_coverage,
        slice_min_samples=args.slice_min_samples,
        league_min_samples=args.league_min_samples,
        report_top_n=args.report_top_n,
        mape_min_denom_eur=args.mape_min_denom_eur,
    )


if __name__ == "__main__":
    main()
