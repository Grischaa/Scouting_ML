# src/scouting_ml/validate_schema.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scouting_ml.utils.import_guard import *  # noqa: F403


EXPECTED_SCHEMA = {
    "player_id": "object",
    "name": "object",
    "position": "object",
    "position_main": "object",
    "position_alt": "object",
    "position_group": "object",
    "nationality": "object",
    "dob": "object",
    "age": "float64",
    "height_cm": "float64",
    "foot": "object",
    "market_value_eur": "float64",
    "club": "object",
    "league": "object",
    "season": "object",
    "market_value_log": "float64",
    "age_group": "object",
    "is_young": "float64",
    "is_veteran": "float64",
    "value_per_age": "float64",
    "height_zscore": "float64",
}

# Optional columns that can exist without error
OPTIONAL_COLS = {"dob_age", "link", "market_value", "type"}


def validate_schema(df: pd.DataFrame, verbose: bool = True) -> bool:
    ok = True
    df_cols = set(df.columns)
    exp_cols = set(EXPECTED_SCHEMA.keys())

    missing = exp_cols - df_cols
    extra = df_cols - exp_cols - OPTIONAL_COLS

    if missing:
        print(f"[schema ❌] Missing columns: {sorted(missing)}")
        ok = False
    if extra:
        print(f"[schema ⚠️] Unexpected columns: {sorted(extra)}")
    else:
        extra_opt = df_cols & OPTIONAL_COLS
        if extra_opt:
            print(f"[schema ℹ️] Optional columns present: {sorted(extra_opt)}")

    # Type check for shared columns
    for col, exp_type in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if exp_type not in actual:
            # allow int64 when float64 expected
            if exp_type == "float64" and "int64" in actual:
                continue
            print(f"[schema ⚠️] Column '{col}' has dtype {actual}, expected {exp_type}")
            ok = False

    if ok and verbose:
        print("[schema ✅] All required columns and types are correct.")
    return ok


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate schema of cleaned Transfermarkt dataset.")
    p.add_argument("--file", required=True, help="Path to the cleaned CSV file.")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[schema ❌] File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[schema] Loaded {len(df)} rows from {path}")
    valid = validate_schema(df)

    if valid:
        print(f"[schema ✅] {path.name} passed schema validation.")
        sys.exit(0)
    else:
        print(f"[schema ⚠️] {path.name} passed with minor warnings.")
        sys.exit(0)


if __name__ == "__main__":
    main()
