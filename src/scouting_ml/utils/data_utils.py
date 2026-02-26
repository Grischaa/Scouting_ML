from __future__ import annotations
import pandas as pd
from typing import Iterable, List, Set


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def split_by_season(df: pd.DataFrame, test_season: str):
    train = df[df["season"] != test_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, test


def split_train_val_test_by_season(
    df: pd.DataFrame,
    val_season: str,
    test_season: str,
):
    train = df[(df["season"] != val_season) & (df["season"] != test_season)].copy()
    val = df[df["season"] == val_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, val, test


def infer_numeric_columns(df: pd.DataFrame, blocked: Iterable[str] | None = None) -> List[str]:
    blocked_set: Set[str] = set(blocked or [])
    cols: List[str] = []
    for c in df.columns:
        if c in blocked_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 1:
            cols.append(c)
    return cols


def infer_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
