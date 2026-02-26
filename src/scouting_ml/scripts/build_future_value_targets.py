from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _season_end_year_from_season(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    start = pd.to_numeric(raw.str.extract(r"^(\d{4})")[0], errors="coerce")
    end_part = raw.str.extract(r"/(\d{2,4})")[0]
    end_num = pd.to_numeric(end_part, errors="coerce")

    end = end_num.copy()
    two_digit_mask = end_num.notna() & (end_num < 100) & start.notna()
    end.loc[two_digit_mask] = (
        (start.loc[two_digit_mask] // 100) * 100 + end_num.loc[two_digit_mask]
    )

    fallback = start + 1
    return end.fillna(fallback)


def _player_key(df: pd.DataFrame) -> pd.Series:
    if "player_id" in df.columns:
        key = df["player_id"].astype(str).str.strip()
        if key.notna().any() and (key != "").any():
            return key
    if all(col in df.columns for col in ("name", "dob")):
        return df["name"].astype(str).str.strip() + "|" + df["dob"].astype(str).str.strip()
    if "name" in df.columns:
        return df["name"].astype(str).str.strip()
    raise ValueError("Dataset needs player_id or name/dob columns to build next-season target.")


def build_future_value_targets(
    *,
    input_path: str,
    output_path: str,
    min_next_minutes: float = 450.0,
    drop_na_target: bool = False,
) -> None:
    df = pd.read_parquet(input_path)
    if "market_value_eur" not in df.columns:
        raise ValueError("Input dataset missing 'market_value_eur'.")

    out = df.copy()
    out["market_value_eur"] = _to_numeric(out["market_value_eur"])
    out["minutes"] = _to_numeric(out["minutes"]) if "minutes" in out.columns else np.nan

    if "season_end_year" in out.columns:
        out["season_end_year"] = _to_numeric(out["season_end_year"])
    elif "season" in out.columns:
        out["season_end_year"] = _season_end_year_from_season(out["season"])
    else:
        raise ValueError("Input dataset needs season_end_year or season columns.")

    out["_player_key"] = _player_key(out)
    sort_cols = ["_player_key", "season_end_year"]
    if "minutes" in out.columns:
        sort_cols.append("minutes")
    out = out.sort_values(sort_cols, ascending=[True, True, True], na_position="last").reset_index(drop=True)

    grouped = out.groupby("_player_key", dropna=False)
    out["next_market_value_eur"] = grouped["market_value_eur"].shift(-1)
    out["next_season_end_year"] = grouped["season_end_year"].shift(-1)
    if "season" in out.columns:
        out["next_season"] = grouped["season"].shift(-1)
    if "minutes" in out.columns:
        out["next_minutes"] = grouped["minutes"].shift(-1)

    year_delta = out["next_season_end_year"] - out["season_end_year"]
    has_consecutive_next = year_delta == 1
    has_next_value = out["next_market_value_eur"].notna()
    has_min_next_minutes = pd.Series(True, index=out.index)
    if "next_minutes" in out.columns:
        has_min_next_minutes = _to_numeric(out["next_minutes"]).fillna(0.0) >= float(min_next_minutes)

    out["has_next_season_target"] = (has_consecutive_next & has_next_value & has_min_next_minutes).astype(int)

    current_value = out["market_value_eur"].clip(lower=1.0)
    growth_eur = out["next_market_value_eur"] - out["market_value_eur"]
    growth_pct = growth_eur / current_value
    growth_log = np.log1p(out["next_market_value_eur"].clip(lower=0.0)) - np.log1p(
        out["market_value_eur"].clip(lower=0.0)
    )

    valid_mask = out["has_next_season_target"] == 1
    out["value_growth_next_season_eur"] = growth_eur.where(valid_mask)
    out["value_growth_next_season_pct"] = growth_pct.where(valid_mask)
    out["value_growth_next_season_log_delta"] = growth_log.where(valid_mask)
    out["value_growth_positive_flag"] = (out["value_growth_next_season_eur"] > 0).astype(int)
    out["value_growth_gt25pct_flag"] = (out["value_growth_next_season_pct"] >= 0.25).astype(int)

    if drop_na_target:
        out = out[out["has_next_season_target"] == 1].copy()

    out = out.drop(columns=["_player_key"], errors="ignore")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    total_rows = int(len(df))
    target_rows = int(valid_mask.sum())
    coverage = target_rows / max(total_rows, 1)
    print(f"[future-target] wrote {len(out):,} rows -> {output}")
    print(
        "[future-target] target coverage: "
        f"{target_rows:,}/{total_rows:,} ({coverage*100:.2f}%) | "
        f"min_next_minutes={min_next_minutes:g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build next-season market-value growth targets for scouting outcome modeling."
    )
    parser.add_argument("--input", default="data/model/big5_players_clean.parquet")
    parser.add_argument(
        "--output",
        default="data/model/big5_players_future_targets.parquet",
    )
    parser.add_argument("--min-next-minutes", type=float, default=450.0)
    parser.add_argument(
        "--drop-na-target",
        action="store_true",
        help="Keep only rows that have a valid next-season target.",
    )
    args = parser.parse_args()

    build_future_value_targets(
        input_path=args.input,
        output_path=args.output,
        min_next_minutes=args.min_next_minutes,
        drop_na_target=args.drop_na_target,
    )


if __name__ == "__main__":
    main()
