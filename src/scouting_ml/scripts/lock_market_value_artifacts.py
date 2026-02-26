from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEST_ENV = "SCOUTING_TEST_PREDICTIONS_PATH"
VAL_ENV = "SCOUTING_VAL_PREDICTIONS_PATH"
METRICS_ENV = "SCOUTING_METRICS_PATH"
MANIFEST_ENV = "SCOUTING_MODEL_MANIFEST_PATH"
STRICT_ENV = "SCOUTING_STRICT_ARTIFACTS"

DEFAULT_TEST = Path("data/model/big5_predictions_full_v2.csv")
DEFAULT_VAL = Path("data/model/big5_predictions_full_v2_val.csv")
DEFAULT_METRICS = Path("data/model/big5_predictions_full_v2.metrics.json")
DEFAULT_MANIFEST = Path("data/model/model_manifest.json")
DEFAULT_ENV_OUT = Path("data/model/model_artifacts.env")


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256_file(path),
    }


def _resolve_path(cli_value: str | None, env_name: str, default: Path) -> Path:
    if cli_value and cli_value.strip():
        return Path(cli_value.strip())
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return Path(env_value)
    return default


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_lock_bundle(
    *,
    test_predictions: Path,
    val_predictions: Path,
    metrics_path: Path,
    manifest_out: Path,
    env_out: Path,
    strict_artifacts: bool,
    label: str,
) -> None:
    for required in (test_predictions, val_predictions, metrics_path):
        if not required.exists():
            raise FileNotFoundError(f"Required artifact not found: {required}")

    metrics = _load_metrics(metrics_path)
    payload = {
        "registry_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "artifacts": {
            "test_predictions": _meta(test_predictions),
            "val_predictions": _meta(val_predictions),
            "metrics": _meta(metrics_path),
        },
        "config": {
            "dataset": metrics.get("dataset"),
            "val_season": metrics.get("val_season"),
            "test_season": metrics.get("test_season"),
            "trials_per_position": metrics.get("trials_per_position"),
            "optimize_metric": metrics.get("optimize_metric"),
            "interval_q": metrics.get("interval_q"),
            "two_stage_band_model": metrics.get("two_stage_band_model"),
            "band_min_samples": metrics.get("band_min_samples"),
            "band_blend_alpha": metrics.get("band_blend_alpha"),
            "strict_leakage_guard": metrics.get("strict_leakage_guard"),
        },
        "summary": {
            "overall": metrics.get("overall"),
            "segments": metrics.get("segments"),
            "league_holdout": metrics.get("league_holdout"),
        },
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env_lines = [
        f"{TEST_ENV}={test_predictions}",
        f"{VAL_ENV}={val_predictions}",
        f"{METRICS_ENV}={metrics_path}",
        f"{MANIFEST_ENV}={manifest_out}",
        f"{STRICT_ENV}={'1' if strict_artifacts else '0'}",
    ]

    env_out.parent.mkdir(parents=True, exist_ok=True)
    env_out.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print(f"[lock] wrote manifest -> {manifest_out}")
    print(f"[lock] wrote env file -> {env_out}")
    print("[lock] export this file before starting the API:")
    print(f"       source {env_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze model artifacts into a manifest + env lock file for reproducible API serving."
    )
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument("--val-predictions", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--env-out", default=str(DEFAULT_ENV_OUT))
    parser.add_argument("--label", default="market_value_main")
    parser.add_argument(
        "--strict-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write SCOUTING_STRICT_ARTIFACTS=1 in lock env file.",
    )
    args = parser.parse_args()

    test_predictions = _resolve_path(args.test_predictions, TEST_ENV, DEFAULT_TEST)
    val_predictions = _resolve_path(args.val_predictions, VAL_ENV, DEFAULT_VAL)
    metrics_path = _resolve_path(args.metrics, METRICS_ENV, DEFAULT_METRICS)

    build_lock_bundle(
        test_predictions=test_predictions,
        val_predictions=val_predictions,
        metrics_path=metrics_path,
        manifest_out=Path(args.manifest_out),
        env_out=Path(args.env_out),
        strict_artifacts=bool(args.strict_artifacts),
        label=str(args.label),
    )


if __name__ == "__main__":
    main()
