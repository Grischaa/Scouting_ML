from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from scouting_ml.models.train_market_value_full import main as train_market_value_main


def normalize_season_label(value: str | float | int | None) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    season = str(value).strip().replace("\\", "/")
    if not season:
        return None
    if "-" in season and "/" not in season:
        season = season.replace("-", "/")

    m_full = re.match(r"^(\d{4})/(\d{2}|\d{4})$", season)
    if m_full:
        start = int(m_full.group(1))
        end = m_full.group(2)
        end2 = end[-2:] if len(end) == 4 else end
        return f"{start}/{end2}"

    m_year = re.match(r"^(\d{4})$", season)
    if m_year:
        year = int(m_year.group(1))
        return f"{year-1}/{str(year)[-2:]}"
    return season


def season_start_year(season: str | None) -> int | None:
    if season is None:
        return None
    m = re.match(r"^(\d{4})/\d{2}$", season)
    return int(m.group(1)) if m else None


def _season_slug(season: str) -> str:
    return season.replace("/", "-")


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _load_dataset_seasons(dataset_path: str) -> list[str]:
    df = pd.read_parquet(dataset_path, columns=["season"])
    seasons = (
        df["season"]
        .map(normalize_season_label)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    seasons = [s for s in seasons if season_start_year(s) is not None]
    seasons.sort(key=lambda s: season_start_year(s) or -1)
    return seasons


def _load_test_split_snapshots(dataset_path: str) -> dict[str, dict[str, int]]:
    cols = ["season", "market_value_eur"]
    df = pd.read_parquet(dataset_path, columns=cols)
    if df.empty:
        return {}

    season = df["season"].map(normalize_season_label)
    value = pd.to_numeric(df["market_value_eur"], errors="coerce")
    snap = pd.DataFrame({"season": season, "value": value}).dropna(subset=["season"])
    if snap.empty:
        return {}

    snap["under_5m"] = ((snap["value"] >= 0.0) & (snap["value"] < 5_000_000.0)).astype(int)
    snap["over_20m"] = (snap["value"] >= 20_000_000.0).astype(int)

    grouped = (
        snap.groupby("season", dropna=False)
        .agg(
            test_n_samples=("season", "size"),
            test_under_5m_n_samples=("under_5m", "sum"),
            test_over_20m_n_samples=("over_20m", "sum"),
        )
        .reset_index()
    )
    out: dict[str, dict[str, int]] = {}
    for _, row in grouped.iterrows():
        key = str(row["season"])
        out[key] = {
            "test_n_samples": int(row["test_n_samples"]),
            "test_under_5m_n_samples": int(row["test_under_5m_n_samples"]),
            "test_over_20m_n_samples": int(row["test_over_20m_n_samples"]),
        }
    return out


def _weighted_by_n(rows: list[tuple[int, float]]) -> float:
    total_n = 0
    weighted_sum = 0.0
    for n, value in rows:
        if int(n) <= 0 or not np.isfinite(float(value)):
            continue
        total_n += int(n)
        weighted_sum += int(n) * float(value)
    if total_n <= 0:
        return float("nan")
    return float(weighted_sum / total_n)


def _read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    def _as_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    out = {}
    for split in ("val", "test"):
        overall = payload.get("overall", {}).get(split, {})
        out[f"{split}_n_samples"] = int(overall.get("n_samples", 0))
        out[f"{split}_r2"] = _as_float(overall.get("r2", float("nan")))
        out[f"{split}_mae_eur"] = _as_float(overall.get("mae_eur", float("nan")))
        out[f"{split}_mape"] = _as_float(overall.get("mape", float("nan")))
        out[f"{split}_mape_raw"] = _as_float(overall.get("mape_raw", float("nan")))
        out[f"{split}_mape_min_denom_eur"] = _as_float(
            overall.get("mape_min_denom_eur", payload.get("mape_min_denom_eur", float("nan")))
        )
        out[f"{split}_wmape"] = _as_float(overall.get("wmape", float("nan")))

    test_segments = {
        str(row.get("segment")): row
        for row in payload.get("segments", {}).get("test", [])
        if isinstance(row, dict)
    }
    for seg in ("under_5m", "5m_to_20m", "over_20m"):
        row = test_segments.get(seg, {})
        out[f"test_{seg}_n_samples"] = int(row.get("n_samples", 0) or 0)
        out[f"test_{seg}_r2"] = _as_float(row.get("r2", float("nan")))
        out[f"test_{seg}_mape"] = _as_float(row.get("mape", float("nan")))
        out[f"test_{seg}_mape_raw"] = _as_float(row.get("mape_raw", float("nan")))
        out[f"test_{seg}_wmape"] = _as_float(row.get("wmape", float("nan")))
    out["test_lowmid_weighted_wmape"] = _weighted_by_n(
        [
            (out.get("test_under_5m_n_samples", 0), out.get("test_under_5m_wmape", float("nan"))),
            (out.get("test_5m_to_20m_n_samples", 0), out.get("test_5m_to_20m_wmape", float("nan"))),
        ]
    )
    out["test_segment_weighted_wmape"] = _weighted_by_n(
        [
            (out.get("test_under_5m_n_samples", 0), out.get("test_under_5m_wmape", float("nan"))),
            (out.get("test_5m_to_20m_n_samples", 0), out.get("test_5m_to_20m_wmape", float("nan"))),
            (out.get("test_over_20m_n_samples", 0), out.get("test_over_20m_wmape", float("nan"))),
        ]
    )
    return out


def run_rolling_backtest(
    dataset_path: str,
    out_dir: str,
    trials: int = 60,
    recency_half_life: float = 2.0,
    under_5m_weight: float = 1.0,
    mid_5m_to_20m_weight: float = 1.0,
    over_20m_weight: float = 1.0,
    optimize_metric: str = "lowmid_wmape",
    interval_q: float = 0.8,
    strict_leakage_guard: bool = True,
    strict_quality_gate: bool = False,
    two_stage_band_model: bool = True,
    band_min_samples: int = 160,
    band_blend_alpha: float = 0.35,
    enforce_quality_gate: bool = False,
    min_test_r2: float = 0.60,
    max_test_mape: float | None = None,
    max_test_wmape: float = 0.42,
    max_under5m_wmape: float = 0.50,
    max_lowmid_weighted_wmape: float = 0.48,
    max_segment_weighted_wmape: float = 0.45,
    min_test_samples: int = 300,
    min_test_under5m_samples: int = 50,
    min_test_over20m_samples: int = 25,
    skip_incomplete_test_seasons: bool = True,
    drop_incomplete_league_seasons: bool = True,
    min_league_season_rows: int = 40,
    min_league_season_completeness: float = 0.55,
    residual_calibration_min_samples: int = 30,
    mape_min_denom_eur: float = 1_000_000.0,
    min_train_seasons: int = 2,
    test_seasons: Sequence[str] | None = None,
    exclude_prefixes: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
) -> None:
    seasons = _load_dataset_seasons(dataset_path)
    split_snapshots = _load_test_split_snapshots(dataset_path)
    if len(seasons) < 3:
        raise ValueError("Need at least 3 seasons in dataset for rolling train/val/test backtests.")

    requested_tests = set(normalize_season_label(s) for s in (test_seasons or []))
    requested_tests.discard(None)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped_runs: list[dict] = []

    for idx in range(1, len(seasons)):
        val_season = seasons[idx - 1]
        test_season = seasons[idx]
        train_seasons = seasons[: idx - 1]

        if len(train_seasons) < min_train_seasons:
            continue
        if requested_tests and test_season not in requested_tests:
            continue

        snapshot = split_snapshots.get(test_season, {})
        precheck_reasons: list[str] = []
        test_n_snapshot = int(snapshot.get("test_n_samples", 0))
        under5_snapshot = int(snapshot.get("test_under_5m_n_samples", 0))
        over20_snapshot = int(snapshot.get("test_over_20m_n_samples", 0))
        if int(min_test_samples) > 0 and test_n_snapshot < int(min_test_samples):
            precheck_reasons.append(
                f"test_n_samples={test_n_snapshot} < min_test_samples={int(min_test_samples)}"
            )
        if int(min_test_under5m_samples) > 0 and under5_snapshot < int(min_test_under5m_samples):
            precheck_reasons.append(
                "test_under_5m_n_samples="
                f"{under5_snapshot} < min_test_under5m_samples={int(min_test_under5m_samples)}"
            )
        if int(min_test_over20m_samples) > 0 and over20_snapshot < int(min_test_over20m_samples):
            precheck_reasons.append(
                "test_over_20m_n_samples="
                f"{over20_snapshot} < min_test_over20m_samples={int(min_test_over20m_samples)}"
            )
        if precheck_reasons and skip_incomplete_test_seasons:
            print(
                f"[backtest] skipped {test_season} due to incomplete test split (precheck): "
                + "; ".join(precheck_reasons)
            )
            skipped_runs.append(
                {
                    "test_season": test_season,
                    "val_season": val_season,
                    "reasons": precheck_reasons,
                    "test_n_samples": test_n_snapshot,
                    "test_under_5m_n_samples": under5_snapshot,
                    "test_over_20m_n_samples": over20_snapshot,
                    "source": "precheck",
                }
            )
            continue

        stem = f"rolling_{_season_slug(test_season)}"
        test_pred_path = out_path / f"{stem}.csv"
        val_pred_path = out_path / f"{stem}_val.csv"
        metrics_path = out_path / f"{stem}.metrics.json"

        print("\n====================================")
        print(f"[backtest] train={train_seasons} | val={val_season} | test={test_season}")
        print("====================================")

        train_market_value_main(
            dataset_path=dataset_path,
            val_season=val_season,
            test_season=test_season,
            output_path=str(test_pred_path),
            val_output_path=str(val_pred_path),
            metrics_output_path=str(metrics_path),
            n_optuna_trials=trials,
            recency_half_life=recency_half_life,
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
            optimize_metric=optimize_metric,
            interval_q=interval_q,
            strict_leakage_guard=strict_leakage_guard,
            strict_quality_gate=strict_quality_gate,
            two_stage_band_model=two_stage_band_model,
            band_min_samples=band_min_samples,
            band_blend_alpha=band_blend_alpha,
            exclude_prefixes=list(exclude_prefixes or []),
            exclude_columns=list(exclude_columns or []),
            drop_incomplete_league_seasons=drop_incomplete_league_seasons,
            min_league_season_rows=min_league_season_rows,
            min_league_season_completeness=min_league_season_completeness,
            residual_calibration_min_samples=residual_calibration_min_samples,
            mape_min_denom_eur=mape_min_denom_eur,
        )

        if not metrics_path.exists():
            raise RuntimeError(f"Backtest run for {test_season} did not produce metrics file: {metrics_path}")

        metrics = _read_metrics(metrics_path)
        test_n = int(metrics.get("test_n_samples", 0) or 0)
        test_under5_n = int(metrics.get("test_under_5m_n_samples", 0) or 0)
        test_over20_n = int(metrics.get("test_over_20m_n_samples", 0) or 0)
        incomplete_reasons: list[str] = []
        if int(min_test_samples) > 0 and test_n < int(min_test_samples):
            incomplete_reasons.append(
                f"test_n_samples={test_n} < min_test_samples={int(min_test_samples)}"
            )
        if int(min_test_under5m_samples) > 0 and test_under5_n < int(min_test_under5m_samples):
            incomplete_reasons.append(
                f"test_under_5m_n_samples={test_under5_n} < min_test_under5m_samples={int(min_test_under5m_samples)}"
            )
        if int(min_test_over20m_samples) > 0 and test_over20_n < int(min_test_over20m_samples):
            incomplete_reasons.append(
                f"test_over_20m_n_samples={test_over20_n} < min_test_over20m_samples={int(min_test_over20m_samples)}"
            )

        if incomplete_reasons and skip_incomplete_test_seasons:
            print(
                f"[backtest] skipped {test_season} due to incomplete test split: "
                + "; ".join(incomplete_reasons)
            )
            skipped_runs.append(
                {
                    "test_season": test_season,
                    "val_season": val_season,
                    "reasons": incomplete_reasons,
                    "test_n_samples": test_n,
                    "test_under_5m_n_samples": test_under5_n,
                    "test_over_20m_n_samples": test_over20_n,
                    "metrics_path": str(metrics_path),
                    "source": "metrics",
                }
            )
            continue

        row = {
            "test_season": test_season,
            "val_season": val_season,
            "n_train_seasons": len(train_seasons),
            "train_seasons": ",".join(train_seasons),
            "metrics_path": str(metrics_path),
            "test_predictions_path": str(test_pred_path),
            "val_predictions_path": str(val_pred_path),
        }
        row.update(metrics)
        if incomplete_reasons:
            row["incomplete_split_reasons"] = "; ".join(incomplete_reasons)
        rows.append(row)

    if not rows:
        if skipped_runs:
            raise ValueError(
                "No rolling backtest runs executed after skipping incomplete test seasons. "
                "Relax min-test sample thresholds or disable --skip-incomplete-test-seasons."
            )
        raise ValueError("No rolling backtest runs executed. Check season filters/min-train-seasons.")

    summary = pd.DataFrame(rows).sort_values(
        by="test_season",
        key=lambda s: s.map(lambda v: season_start_year(str(v)) or -1),
    )
    summary_path = out_path / "rolling_backtest_summary.csv"
    summary.to_csv(summary_path, index=False)

    agg = {
        "runs": int(len(summary)),
        "mean_test_r2": float(summary["test_r2"].mean()),
        "std_test_r2": float(summary["test_r2"].std()),
        "mean_test_mae_eur": float(summary["test_mae_eur"].mean()),
        "std_test_mae_eur": float(summary["test_mae_eur"].std()),
        "mean_test_mape": float(summary["test_mape"].mean()),
        "std_test_mape": float(summary["test_mape"].std()),
        "mean_test_mape_raw": float(summary["test_mape_raw"].mean()),
        "std_test_mape_raw": float(summary["test_mape_raw"].std()),
        "mean_test_wmape": float(summary["test_wmape"].mean()),
        "std_test_wmape": float(summary["test_wmape"].std()),
        "mean_test_lowmid_weighted_wmape": float(summary["test_lowmid_weighted_wmape"].mean()),
        "std_test_lowmid_weighted_wmape": float(summary["test_lowmid_weighted_wmape"].std()),
        "mean_test_segment_weighted_wmape": float(summary["test_segment_weighted_wmape"].mean()),
        "std_test_segment_weighted_wmape": float(summary["test_segment_weighted_wmape"].std()),
    }

    gate_failures: list[str] = []
    if enforce_quality_gate:
        for _, row in summary.iterrows():
            season = str(row.get("test_season", "unknown"))
            r2 = float(row.get("test_r2", float("nan")))
            mape = float(row.get("test_mape", float("nan")))
            wmape = float(row.get("test_wmape", float("nan")))
            under5_wmape = float(row.get("test_under_5m_wmape", float("nan")))
            lowmid_weighted = float(row.get("test_lowmid_weighted_wmape", float("nan")))
            seg_weighted = float(row.get("test_segment_weighted_wmape", float("nan")))
            if np.isfinite(r2) and r2 < float(min_test_r2):
                gate_failures.append(
                    f"{season}: test_r2 {r2:.4f} < min_test_r2 {float(min_test_r2):.4f}"
                )
            if max_test_mape is not None and np.isfinite(mape) and mape > float(max_test_mape):
                gate_failures.append(
                    f"{season}: test_mape {mape:.4f} > max_test_mape {float(max_test_mape):.4f}"
                )
            if np.isfinite(wmape) and wmape > float(max_test_wmape):
                gate_failures.append(
                    f"{season}: test_wmape {wmape:.4f} > max_test_wmape {float(max_test_wmape):.4f}"
                )
            if np.isfinite(under5_wmape) and under5_wmape > float(max_under5m_wmape):
                gate_failures.append(
                    f"{season}: test_under_5m_wmape {under5_wmape:.4f} > "
                    f"max_under5m_wmape {float(max_under5m_wmape):.4f}"
                )
            if np.isfinite(lowmid_weighted) and lowmid_weighted > float(max_lowmid_weighted_wmape):
                gate_failures.append(
                    f"{season}: test_lowmid_weighted_wmape {lowmid_weighted:.4f} > "
                    f"max_lowmid_weighted_wmape {float(max_lowmid_weighted_wmape):.4f}"
                )
            if np.isfinite(seg_weighted) and seg_weighted > float(max_segment_weighted_wmape):
                gate_failures.append(
                    f"{season}: test_segment_weighted_wmape {seg_weighted:.4f} > "
                    f"max_segment_weighted_wmape {float(max_segment_weighted_wmape):.4f}"
                )

    agg["quality_gate"] = {
        "enabled": bool(enforce_quality_gate),
        "passed": len(gate_failures) == 0,
        "min_test_r2": float(min_test_r2),
        "max_test_mape": None if max_test_mape is None else float(max_test_mape),
        "max_test_wmape": float(max_test_wmape),
        "max_under5m_wmape": float(max_under5m_wmape),
        "max_lowmid_weighted_wmape": float(max_lowmid_weighted_wmape),
        "max_segment_weighted_wmape": float(max_segment_weighted_wmape),
        "failures": gate_failures,
    }
    agg["minimum_test_split_requirements"] = {
        "min_test_samples": int(min_test_samples),
        "min_test_under5m_samples": int(min_test_under5m_samples),
        "min_test_over20m_samples": int(min_test_over20m_samples),
        "skip_incomplete_test_seasons": bool(skip_incomplete_test_seasons),
    }
    agg["training_completeness_gate"] = {
        "drop_incomplete_league_seasons": bool(drop_incomplete_league_seasons),
        "min_league_season_rows": int(min_league_season_rows),
        "min_league_season_completeness": float(min_league_season_completeness),
        "residual_calibration_min_samples": int(residual_calibration_min_samples),
        "mape_min_denom_eur": float(mape_min_denom_eur),
    }
    agg["skipped_runs"] = skipped_runs
    agg_path = out_path / "rolling_backtest_summary.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print("\n========== ROLLING BACKTEST SUMMARY ==========")
    for _, row in summary.iterrows():
        print(
            f"test={row['test_season']} | val={row['val_season']} | "
            f"R² {row['test_r2']*100:6.2f}% | "
            f"MAE €{row['test_mae_eur']:,.0f} | "
            f"MAPE {row['test_mape']*100:6.2f}% | "
            f"WMAPE {row['test_wmape']*100:6.2f}%"
        )
    print("----------------------------------------------")
    print(
        f"mean test R² {agg['mean_test_r2']*100:,.2f}% | "
        f"mean test MAE €{agg['mean_test_mae_eur']:,.0f} | "
        f"mean test MAPE {agg['mean_test_mape']*100:,.2f}%"
    )
    if skipped_runs:
        print("[backtest] skipped incomplete seasons:")
        for item in skipped_runs:
            reasons = "; ".join(item.get("reasons", []))
            print(f"  - {item['test_season']}: {reasons}")
    if enforce_quality_gate:
        if gate_failures:
            print("[backtest] quality gate: FAILED")
            for msg in gate_failures:
                print(f"  - {msg}")
        else:
            print("[backtest] quality gate: PASSED")
    print(f"[backtest] wrote summary csv -> {summary_path}")
    print(f"[backtest] wrote summary json -> {agg_path}")
    if enforce_quality_gate and gate_failures:
        raise RuntimeError("Rolling backtest quality gate failed. See summary json for details.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rolling season backtests for the market-value pipeline."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--out-dir", default="data/model/backtests")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
    parser.add_argument(
        "--optimize-metric",
        default="lowmid_wmape",
        choices=["mae", "rmse", "band_wmape", "lowmid_wmape"],
        help="Optuna objective metric passed to train_market_value_full.",
    )
    parser.add_argument(
        "--interval-q",
        type=float,
        default=0.80,
        help="Conformal interval quantile passed to train_market_value_full.",
    )
    parser.add_argument(
        "--strict-leakage-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict leakage guard in train_market_value_full.",
    )
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help="Fail backtest run if quality flags are present.",
    )
    parser.add_argument(
        "--two-stage-band-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable value-band routing model during backtests.",
    )
    parser.add_argument(
        "--band-min-samples",
        type=int,
        default=160,
        help="Minimum samples required to train each value-band expert.",
    )
    parser.add_argument(
        "--band-blend-alpha",
        type=float,
        default=0.35,
        help="Blend factor between global and band-expert predictions.",
    )
    parser.add_argument(
        "--enforce-quality-gate",
        action="store_true",
        help="Fail backtest if global/segment thresholds are breached.",
    )
    parser.add_argument(
        "--min-test-r2",
        type=float,
        default=0.60,
        help="Quality gate threshold: minimum acceptable test R².",
    )
    parser.add_argument(
        "--max-test-mape",
        type=float,
        default=None,
        help="Optional quality-gate threshold for test MAPE (set to disable MAPE gate).",
    )
    parser.add_argument(
        "--max-test-wmape",
        type=float,
        default=0.42,
        help="Quality gate threshold: maximum acceptable test WMAPE.",
    )
    parser.add_argument(
        "--max-under5m-wmape",
        type=float,
        default=0.50,
        help="Quality gate threshold: maximum acceptable under_5m test WMAPE.",
    )
    parser.add_argument(
        "--max-lowmid-weighted-wmape",
        type=float,
        default=0.48,
        help="Quality gate threshold: max weighted WMAPE over under_5m + 5m_to_20m.",
    )
    parser.add_argument(
        "--max-segment-weighted-wmape",
        type=float,
        default=0.45,
        help="Quality gate threshold: max weighted WMAPE across all value segments.",
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=300,
        help="Minimum test rows required for a season to be included in summary/gate checks.",
    )
    parser.add_argument(
        "--min-test-under5m-samples",
        type=int,
        default=50,
        help="Minimum under_5m test rows required for a season to be included.",
    )
    parser.add_argument(
        "--min-test-over20m-samples",
        type=int,
        default=25,
        help="Minimum over_20m test rows required for a season to be included.",
    )
    parser.add_argument(
        "--skip-incomplete-test-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seasons with test splits below minimum sample requirements.",
    )
    parser.add_argument(
        "--drop-incomplete-league-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop incomplete league-season groups before each training run.",
    )
    parser.add_argument(
        "--min-league-season-rows",
        type=int,
        default=40,
        help="Minimum rows required per league-season when dropping incomplete groups.",
    )
    parser.add_argument(
        "--min-league-season-completeness",
        type=float,
        default=0.55,
        help="Minimum ratio vs league max-season row count to keep a league-season.",
    )
    parser.add_argument(
        "--residual-calibration-min-samples",
        type=int,
        default=30,
        help="Minimum validation samples for residual-calibration buckets.",
    )
    parser.add_argument(
        "--mape-min-denom-eur",
        type=float,
        default=1_000_000.0,
        help="MAPE denominator floor in EUR passed to training runs.",
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=2,
        help="Minimum number of seasons required in train split before running a backtest.",
    )
    parser.add_argument(
        "--test-seasons",
        default="",
        help="Optional comma-separated test seasons (e.g. 2022/23,2023/24,2024/25).",
    )
    parser.add_argument(
        "--exclude-prefixes",
        default="",
        help="Optional comma-separated feature prefixes passed through to train_market_value_full.",
    )
    parser.add_argument(
        "--exclude-columns",
        default="",
        help="Optional comma-separated exact feature columns passed through to train_market_value_full.",
    )
    args = parser.parse_args()

    run_rolling_backtest(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        trials=args.trials,
        recency_half_life=args.recency_half_life,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_20m_weight,
        over_20m_weight=args.over_20m_weight,
        optimize_metric=args.optimize_metric,
        interval_q=args.interval_q,
        strict_leakage_guard=args.strict_leakage_guard,
        strict_quality_gate=args.strict_quality_gate,
        two_stage_band_model=args.two_stage_band_model,
        band_min_samples=args.band_min_samples,
        band_blend_alpha=args.band_blend_alpha,
        enforce_quality_gate=args.enforce_quality_gate,
        min_test_r2=args.min_test_r2,
        max_test_mape=args.max_test_mape,
        max_test_wmape=args.max_test_wmape,
        max_under5m_wmape=args.max_under5m_wmape,
        max_lowmid_weighted_wmape=args.max_lowmid_weighted_wmape,
        max_segment_weighted_wmape=args.max_segment_weighted_wmape,
        min_test_samples=args.min_test_samples,
        min_test_under5m_samples=args.min_test_under5m_samples,
        min_test_over20m_samples=args.min_test_over20m_samples,
        skip_incomplete_test_seasons=args.skip_incomplete_test_seasons,
        drop_incomplete_league_seasons=args.drop_incomplete_league_seasons,
        min_league_season_rows=args.min_league_season_rows,
        min_league_season_completeness=args.min_league_season_completeness,
        residual_calibration_min_samples=args.residual_calibration_min_samples,
        mape_min_denom_eur=args.mape_min_denom_eur,
        min_train_seasons=args.min_train_seasons,
        test_seasons=_parse_csv_tokens(args.test_seasons),
        exclude_prefixes=_parse_csv_tokens(args.exclude_prefixes),
        exclude_columns=_parse_csv_tokens(args.exclude_columns),
    )


if __name__ == "__main__":
    main()
