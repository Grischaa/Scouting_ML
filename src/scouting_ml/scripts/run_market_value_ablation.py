from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from scouting_ml.models.train_market_value_full import main as train_market_value_main


@dataclass(frozen=True)
class AblationConfig:
    name: str
    exclude_prefixes: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()
    note: str = ""


DEFAULT_CONFIGS: Dict[str, AblationConfig] = {
    "full": AblationConfig("full", note="All available features."),
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
    "baseline_stats_only": AblationConfig(
        "baseline_stats_only",
        exclude_prefixes=("contract_", "injury_", "transfer_", "nt_", "clubctx_", "leaguectx_", "uefa_coeff_"),
        note="Only baseline player/season stat features.",
    ),
}


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _season_slug(season: str) -> str:
    return season.replace("/", "-")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _overall(metrics_payload: dict, split: str) -> dict:
    overall = metrics_payload.get("overall", {}).get(split, {})
    return {
        f"{split}_n_samples": int(overall.get("n_samples", 0)),
        f"{split}_r2": float(overall.get("r2", float("nan"))),
        f"{split}_mae_eur": float(overall.get("mae_eur", float("nan"))),
        f"{split}_mape": float(overall.get("mape", float("nan"))),
        f"{split}_wmape": float(overall.get("wmape", float("nan"))),
    }


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
) -> None:
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
    rows: list[dict] = []

    for cfg in selected:
        print("\n==============================")
        print(f"[ablation] running: {cfg.name}")
        print(f"[ablation] note: {cfg.note}")
        print("==============================")

        stem = f"{cfg.name}_{_season_slug(test_season)}"
        pred_path = out_path / f"{stem}.csv"
        val_pred_path = out_path / f"{stem}_val.csv"
        metrics_path = out_path / f"{stem}.metrics.json"

        train_market_value_main(
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
        )

        if not metrics_path.exists():
            raise RuntimeError(f"Ablation run '{cfg.name}' did not produce metrics file: {metrics_path}")

        payload = _load_json(metrics_path)
        row = {
            "config": cfg.name,
            "exclude_prefixes": ",".join(cfg.exclude_prefixes),
            "exclude_columns": ",".join(cfg.exclude_columns),
            "note": cfg.note,
            "metrics_path": str(metrics_path),
            "predictions_path": str(pred_path),
            "val_predictions_path": str(val_pred_path),
        }
        row.update(_overall(payload, "val"))
        row.update(_overall(payload, "test"))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["test_mae_eur", "test_mape"], ascending=[True, True], na_position="last")
    summary_path = out_path / f"ablation_summary_{_season_slug(test_season)}.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== ABLATION SUMMARY ==========")
    for _, row in summary.iterrows():
        print(
            f"{row['config']:>20} | "
            f"test R² {row['test_r2']*100:6.2f}% | "
            f"MAE €{row['test_mae_eur']:,.0f} | "
            f"MAPE {row['test_mape']*100:6.2f}% | "
            f"WMAPE {row['test_wmape']*100:6.2f}%"
        )
    print(f"[ablation] wrote summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature-group ablations for the market-value pipeline."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument(
        "--configs",
        default="full,no_contract,no_injury,no_transfer,no_national,no_context,baseline_stats_only",
        help=f"Comma-separated config names. Available: {','.join(sorted(DEFAULT_CONFIGS.keys()))}",
    )
    parser.add_argument("--out-dir", default="data/model/ablation")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
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
    )


if __name__ == "__main__":
    main()
