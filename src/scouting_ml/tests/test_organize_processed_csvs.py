from __future__ import annotations

from pathlib import Path

import pandas as pd

from scouting_ml.scripts.organize_processed_csvs import organize_processed_files


def test_organize_processed_files_ignores_existing_combined_copy_when_source_has_newer_content(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    combined_dir = processed_root / "Clubs combined"
    country_root = processed_root / "by_country"
    season_root = processed_root / "by_season"
    manifest = processed_root / "organization_manifest.csv"

    processed_root.mkdir(parents=True)
    combined_dir.mkdir(parents=True)

    basename = "scottish_premiership_2025-26_with_sofa.csv"
    fresh = processed_root / basename
    stale = combined_dir / basename

    pd.DataFrame(
        [
            {"player_id": "p1", "season": "2025/26", "league": "Scottish Premiership", "market_value_eur": 1_000_000},
            {"player_id": "p2", "season": "2025/26", "league": "Scottish Premiership", "market_value_eur": 2_000_000},
        ]
    ).to_csv(fresh, index=False)
    pd.DataFrame(
        [
            {"player_id": "old", "season": "2025/26", "league": "Scottish Premiership", "market_value_eur": 999_999},
        ]
    ).to_csv(stale, index=False)

    organize_processed_files(
        source_dir=str(processed_root),
        combined_dir=str(combined_dir),
        country_root=str(country_root),
        season_root=str(season_root),
        manifest_path=str(manifest),
        clean_targets=True,
    )

    combined = pd.read_csv(combined_dir / basename)
    assert len(combined) == 2
    assert set(combined["player_id"]) == {"p1", "p2"}

    manifest_text = manifest.read_text(encoding="utf-8")
    assert "duplicate_collision" in manifest_text
    assert "False" in manifest_text
