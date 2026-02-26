from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scouting_ml.scripts.run_rolling_backtest import run_rolling_backtest


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_int_grid(raw: str) -> list[int]:
    out = []
    for tok in _parse_csv_tokens(raw):
        out.append(int(tok))
    if not out:
        raise ValueError("band_min_samples grid is empty.")
    return sorted(set(out))


def _parse_float_grid(raw: str) -> list[float]:
    out = []
    for tok in _parse_csv_tokens(raw):
        out.append(float(tok))
    if not out:
        raise ValueError("band_blend_alpha grid is empty.")
    return sorted(set(out))


def _season_start_year(value: str | float | int | None) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    season = str(value).strip().replace("\\", "/")
    if not season:
        return None
    if "-" in season and "/" not in season:
        season = season.replace("-", "/")
    m = re.match(r"^(\d{4})/\d{2}$", season)
    if m:
        return int(m.group(1))
    return None


def _alpha_slug(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def _coerce_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def run_band_hyper_sweep(
    dataset_path: str,
    out_dir: str,
    trials: int,
    recency_half_life: float,
    under_5m_weight: float,
    mid_5m_to_20m_weight: float,
    over_20m_weight: float,
    optimize_metric: str,
    interval_q: float,
    strict_leakage_guard: bool,
    strict_quality_gate: bool,
    test_seasons: Sequence[str],
    min_train_seasons: int,
    exclude_prefixes: Sequence[str],
    exclude_columns: Sequence[str],
    band_min_samples_grid: Sequence[int],
    band_blend_alpha_grid: Sequence[float],
    two_stage_band_model: bool,
    enforce_quality_gate: bool,
    min_test_r2: float,
    max_test_mape: float,
    max_test_wmape: float,
    max_under5m_wmape: float,
    continue_on_error: bool,
    sort_by: str,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    combos = list(itertools.product(band_min_samples_grid, band_blend_alpha_grid))
    print(
        f"[sweep] {len(combos)} combos | min_samples={list(band_min_samples_grid)} "
        f"| alpha={list(band_blend_alpha_grid)}"
    )

    for idx, (band_min_samples, band_blend_alpha) in enumerate(combos, start=1):
        combo_slug = f"min{int(band_min_samples)}_alpha{_alpha_slug(float(band_blend_alpha))}"
        combo_out_dir = out_path / combo_slug
        combo_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n====================================")
        print(
            f"[sweep] combo {idx}/{len(combos)} | "
            f"band_min_samples={int(band_min_samples)} | "
            f"band_blend_alpha={float(band_blend_alpha):.2f}"
        )
        print("====================================")

        row = {
            "status": "ok",
            "combo": combo_slug,
            "band_min_samples": int(band_min_samples),
            "band_blend_alpha": float(band_blend_alpha),
            "out_dir": str(combo_out_dir),
            "optimize_metric": optimize_metric,
            "trials": int(trials),
            "two_stage_band_model": bool(two_stage_band_model),
        }

        try:
            run_rolling_backtest(
                dataset_path=dataset_path,
                out_dir=str(combo_out_dir),
                trials=trials,
                recency_half_life=recency_half_life,
                under_5m_weight=under_5m_weight,
                mid_5m_to_20m_weight=mid_5m_to_20m_weight,
                over_20m_weight=over_20m_weight,
                optimize_metric=optimize_metric,
                interval_q=interval_q,
                strict_leakage_guard=strict_leakage_guard,
                strict_quality_gate=strict_quality_gate,
                two_stage_band_model=two_stage_band_model,
                band_min_samples=int(band_min_samples),
                band_blend_alpha=float(band_blend_alpha),
                enforce_quality_gate=enforce_quality_gate,
                min_test_r2=min_test_r2,
                max_test_mape=max_test_mape,
                max_test_wmape=max_test_wmape,
                max_under5m_wmape=max_under5m_wmape,
                min_train_seasons=min_train_seasons,
                test_seasons=list(test_seasons),
                exclude_prefixes=list(exclude_prefixes),
                exclude_columns=list(exclude_columns),
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            rows.append(row)
            print(f"[sweep] failed combo {combo_slug}: {exc}")
            if not continue_on_error:
                raise
            continue

        summary_json = combo_out_dir / "rolling_backtest_summary.json"
        summary_csv = combo_out_dir / "rolling_backtest_summary.csv"
        if not summary_json.exists() or not summary_csv.exists():
            row["status"] = "error"
            row["error"] = "missing summary outputs"
            rows.append(row)
            print(f"[sweep] missing outputs for {combo_slug}")
            continue

        agg = json.loads(summary_json.read_text(encoding="utf-8"))
        df = pd.read_csv(summary_csv)
        if df.empty:
            row["status"] = "error"
            row["error"] = "empty summary csv"
            rows.append(row)
            print(f"[sweep] empty summary for {combo_slug}")
            continue

        season_years = df["test_season"].map(_season_start_year)
        latest_idx = season_years.fillna(-1).astype(int).idxmax()
        latest = df.loc[latest_idx]

        row.update(
            {
                "runs": int(agg.get("runs", len(df))),
                "mean_test_r2": _coerce_float(agg.get("mean_test_r2")),
                "mean_test_mae_eur": _coerce_float(agg.get("mean_test_mae_eur")),
                "mean_test_mape": _coerce_float(agg.get("mean_test_mape")),
                "mean_test_wmape": _coerce_float(agg.get("mean_test_wmape")),
                "latest_test_season": str(latest.get("test_season")),
                "latest_test_r2": _coerce_float(latest.get("test_r2")),
                "latest_test_mae_eur": _coerce_float(latest.get("test_mae_eur")),
                "latest_test_mape": _coerce_float(latest.get("test_mape")),
                "latest_test_wmape": _coerce_float(latest.get("test_wmape")),
                "latest_test_under_5m_wmape": _coerce_float(latest.get("test_under_5m_wmape")),
                "latest_test_5m_to_20m_wmape": _coerce_float(latest.get("test_5m_to_20m_wmape")),
                "latest_test_over_20m_wmape": _coerce_float(latest.get("test_over_20m_wmape")),
            }
        )
        rows.append(row)

    if not rows:
        raise RuntimeError("No sweep rows were produced.")

    summary = pd.DataFrame(rows)
    sort_col = str(sort_by)
    if sort_col not in summary.columns:
        raise ValueError(f"Unknown sort column '{sort_col}'. Available columns: {list(summary.columns)}")

    asc = sort_col not in {"mean_test_r2", "latest_test_r2"}
    summary = summary.sort_values(
        by=[sort_col, "mean_test_wmape", "mean_test_mae_eur"],
        ascending=[asc, True, True],
        na_position="last",
    )

    summary_csv_path = out_path / "band_sweep_summary.csv"
    summary_json_path = out_path / "band_sweep_summary.json"
    summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n========== BAND SWEEP TOP ==========")
    cols = [
        "status",
        "combo",
        "mean_test_r2",
        "mean_test_mae_eur",
        "mean_test_mape",
        "mean_test_wmape",
        "latest_test_season",
        "latest_test_r2",
        "latest_test_wmape",
    ]
    preview = summary[cols].head(10)
    for _, r in preview.iterrows():
        print(
            f"{r['status']:>5} | {r['combo']:>20} | "
            f"mean R² {(_coerce_float(r['mean_test_r2'])*100):6.2f}% | "
            f"mean MAE €{_coerce_float(r['mean_test_mae_eur']):,.0f} | "
            f"mean WMAPE {(_coerce_float(r['mean_test_wmape'])*100):6.2f}% | "
            f"latest {r['latest_test_season']} R² {(_coerce_float(r['latest_test_r2'])*100):6.2f}%"
        )
    print(f"[sweep] wrote summary csv -> {summary_csv_path}")
    print(f"[sweep] wrote summary json -> {summary_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid sweep band-routing hyperparameters for rolling market-value backtests."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--out-dir", default="data/model/band_sweep")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
    parser.add_argument(
        "--optimize-metric",
        default="lowmid_wmape",
        choices=["mae", "rmse", "band_wmape", "lowmid_wmape"],
    )
    parser.add_argument("--interval-q", type=float, default=0.80)
    parser.add_argument(
        "--strict-leakage-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strict-quality-gate", action="store_true")
    parser.add_argument(
        "--two-stage-band-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--band-min-samples-grid", default="160,220,300")
    parser.add_argument("--band-blend-alpha-grid", default="0.20,0.35,0.50")
    parser.add_argument("--min-train-seasons", type=int, default=2)
    parser.add_argument("--test-seasons", default="2022/23,2023/24,2024/25")
    parser.add_argument("--exclude-prefixes", default="")
    parser.add_argument("--exclude-columns", default="")
    parser.add_argument(
        "--enforce-quality-gate",
        action="store_true",
        help="Pass-through to each backtest run. Off by default for broad sweeps.",
    )
    parser.add_argument("--min-test-r2", type=float, default=0.60)
    parser.add_argument("--max-test-mape", type=float, default=0.58)
    parser.add_argument("--max-test-wmape", type=float, default=0.42)
    parser.add_argument("--max-under5m-wmape", type=float, default=0.52)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue sweep even if one combo fails.",
    )
    parser.add_argument(
        "--sort-by",
        default="mean_test_wmape",
        choices=[
            "mean_test_wmape",
            "mean_test_mae_eur",
            "mean_test_mape",
            "mean_test_r2",
            "latest_test_wmape",
            "latest_test_mae_eur",
            "latest_test_mape",
            "latest_test_r2",
        ],
    )
    args = parser.parse_args()

    run_band_hyper_sweep(
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
        test_seasons=_parse_csv_tokens(args.test_seasons),
        min_train_seasons=args.min_train_seasons,
        exclude_prefixes=_parse_csv_tokens(args.exclude_prefixes),
        exclude_columns=_parse_csv_tokens(args.exclude_columns),
        band_min_samples_grid=_parse_int_grid(args.band_min_samples_grid),
        band_blend_alpha_grid=_parse_float_grid(args.band_blend_alpha_grid),
        two_stage_band_model=args.two_stage_band_model,
        enforce_quality_gate=args.enforce_quality_gate,
        min_test_r2=args.min_test_r2,
        max_test_mape=args.max_test_mape,
        max_test_wmape=args.max_test_wmape,
        max_under5m_wmape=args.max_under5m_wmape,
        continue_on_error=args.continue_on_error,
        sort_by=args.sort_by,
    )


if __name__ == "__main__":
    main()

