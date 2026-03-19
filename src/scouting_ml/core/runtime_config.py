"""Shared runtime configuration helpers for API and pipeline entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def read_str_env(name: str, default: str) -> str:
    """Return a non-empty string from the environment or the provided default."""
    value = os.getenv(name, default).strip()
    return value or default


def read_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable with a validated fallback."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}.")


def read_csv_env(name: str, default: str) -> tuple[str, ...]:
    """Parse a comma-separated env var into a normalized tuple of values."""
    raw = read_str_env(name, default)
    if raw == "*":
        return ("*",)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class ApiRuntimeConfig:
    """Environment-backed API settings used across app initialization."""

    cors_origins: tuple[str, ...] = (
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    )
    strict_artifacts: bool = False
    experimental_nlp_routes: bool = False


def load_api_runtime_config() -> ApiRuntimeConfig:
    """Resolve API runtime settings from environment variables."""
    return ApiRuntimeConfig(
        cors_origins=read_csv_env(
            "SCOUTING_API_CORS_ORIGINS",
            "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500",
        ),
        strict_artifacts=read_bool_env("SCOUTING_STRICT_ARTIFACTS", default=False),
        experimental_nlp_routes=read_bool_env(
            "SCOUTING_ENABLE_EXPERIMENTAL_NLP_ROUTES",
            default=False,
        ),
    )


@dataclass(frozen=True)
class ProductionPipelineDefaults:
    """Single source of truth for production pipeline CLI defaults."""

    players_source: str = "data/processed/Clubs combined"
    data_dir: str = "data/processed/Clubs combined"
    external_dir: str = "data/external"
    dataset_output: str = "data/model/champion_players.parquet"
    clean_output: str = "data/model/champion_players_clean.parquet"
    predictions_output: str = "data/model/champion_predictions_2024-25.csv"
    val_season: str = "2023/24"
    test_season: str = "2024/25"
    start_season: str = "2019/20"
    end_season: str = "2024/25"
    min_minutes: float = 450.0
    trials: int = 40
    optimize_metric: str = "lowmid_wmape"
    band_min_samples: int = 160
    band_blend_alpha: float = 0.35
    mape_min_denom_eur: float = 1_000_000.0
    with_backtest: bool = True
    backtest_test_seasons: str = "2022/23,2023/24,2024/25,2025/26"
    backtest_enforce_quality_gate: bool = False
    backtest_min_test_r2: float = 0.60
    backtest_max_test_wmape: float = 0.42
    backtest_max_under5m_wmape: float = 0.50
    backtest_max_lowmid_weighted_wmape: float = 0.48
    backtest_max_segment_weighted_wmape: float = 0.45
    backtest_min_test_samples: int = 300
    backtest_min_test_under5m_samples: int = 50
    backtest_min_test_over20m_samples: int = 25
    backtest_skip_incomplete_test_seasons: bool = True
    drop_incomplete_league_seasons: bool = True
    min_league_season_rows: int = 40
    min_league_season_completeness: float = 0.55
    residual_calibration_min_samples: int = 30
    provider_config_json: str = ""
    provider_audit_json: str = "data/external/provider_link_audit.json"
    provider_audit_csv: str = "data/external/provider_link_audit.csv"
    lock_manifest_out: str = "data/model/model_manifest.json"
    lock_env_out: str = "data/model/model_artifacts.env"
    lock_label: str = "production_market_value_bundle"
    run_weekly_ops: bool = True
    weekly_split: str = "test"
    weekly_reports_out_dir: str = "data/model/reports"
    weekly_non_big5_only: bool = True
    weekly_min_minutes: float = 900.0
    weekly_max_age: float = 23.0
    weekly_watchlist_tag: str = "u23_non_big5"
    production_summary_out: str = "data/model/production/production_pipeline_summary.json"


PRODUCTION_PIPELINE_DEFAULTS = ProductionPipelineDefaults()


__all__ = [
    "ApiRuntimeConfig",
    "PRODUCTION_PIPELINE_DEFAULTS",
    "ProductionPipelineDefaults",
    "load_api_runtime_config",
    "read_bool_env",
    "read_csv_env",
    "read_str_env",
]
