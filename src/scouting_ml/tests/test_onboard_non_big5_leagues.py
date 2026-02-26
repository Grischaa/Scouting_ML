from __future__ import annotations

import json

import pandas as pd

from scouting_ml.scripts.onboard_non_big5_leagues import build_onboarding_report


def test_onboarding_report_uses_manifest_and_holdout_metrics(tmp_path) -> None:
    manifest_path = tmp_path / "organization_manifest.csv"
    holdout_path = tmp_path / "rolling_2024-25.holdout_eredivisie.metrics.json"
    out_json = tmp_path / "non_big5_onboarding_report.json"
    out_csv = tmp_path / "non_big5_onboarding_report.csv"

    pd.DataFrame(
        [
            {
                "basename": "dutch_eredivisie_2024-25_with_sofa.csv",
                "league_slug": "dutch_eredivisie",
                "season": "2024-25",
                "country": "netherlands",
                "source_path": "x",
                "combined_path": "x",
                "country_path": "x",
                "season_path": "x",
                "duplicate_count": 1,
                "duplicate_collision": False,
            },
            {
                "basename": "dutch_eredivisie_2023-24_with_sofa.csv",
                "league_slug": "dutch_eredivisie",
                "season": "2023-24",
                "country": "netherlands",
                "source_path": "x",
                "combined_path": "x",
                "country_path": "x",
                "season_path": "x",
                "duplicate_count": 1,
                "duplicate_collision": False,
            },
            {
                "basename": "english_premier_league_2024-25_with_sofa.csv",
                "league_slug": "english_premier_league",
                "season": "2024-25",
                "country": "england",
                "source_path": "x",
                "combined_path": "x",
                "country_path": "x",
                "season_path": "x",
                "duplicate_count": 1,
                "duplicate_collision": False,
            },
        ]
    ).to_csv(manifest_path, index=False)

    holdout_payload = {
        "league": "Eredivisie",
        "overall": {"r2": 0.46, "wmape": 0.41, "n_samples": 380},
        "domain_shift": {"mean_abs_shift_z": 0.72},
    }
    holdout_path.write_text(json.dumps(holdout_payload), encoding="utf-8")

    payload = build_onboarding_report(
        manifest_path=str(manifest_path),
        holdout_metrics_glob=str(tmp_path / "*.metrics.json"),
        out_json=str(out_json),
        out_csv=str(out_csv),
        min_seasons=2,
        min_files=2,
        max_domain_shift_z=1.25,
        min_holdout_r2=0.35,
    )

    assert out_json.exists()
    assert out_csv.exists()
    assert payload["status_counts"]["ready"] == 1
    item = payload["items"][0]
    assert item["league_slug"] == "dutch_eredivisie"
    assert item["holdout_available"] is True
    assert item["status"] == "ready"
