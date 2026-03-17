from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from scouting_ml.league_registry import get_league
from scouting_ml.scripts import backfill_future_league_files as bff
from scouting_ml.tests.summary_contract import assert_common_summary_contract


def test_extend_league_for_season_infers_2025_26_defaults() -> None:
    config = get_league("dutch_eredivisie")
    extended = bff._extend_league_for_season(config, "2025/26")

    assert extended.tm_season_ids["2025/26"] == 2025
    assert extended.sofa_season_map["2025/26"] == "25/26"
    assert extended.seasons[0] == "2025/26"


def test_run_future_league_backfill_stages_outputs(tmp_path: Path, monkeypatch) -> None:
    base_config = get_league("dutch_eredivisie")
    config = replace(
        bff._extend_league_for_season(base_config, "2025/26"),
        processed_dataset_pattern=str(tmp_path / "processed" / "dutch_eredivisie_{season_slug}_with_sofa.csv"),
    )

    monkeypatch.setattr(bff, "get_league", lambda slug: config)

    processed_root = tmp_path / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, payload: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path

    def _tm_stub(config_arg, season, *, force, python_executable):
        season_slug = season.replace("/", "-")
        combined_path = _write(processed_root / f"{config_arg.slug}_{season_slug}_players.csv", "player_id,name\np1,Player One\n")
        clean_path = _write(processed_root / f"{config_arg.slug}_{season_slug}_clean.csv", "player_id,name,club\np1,Player One,Ajax\n")
        return type(
            "TransfermarktResultStub",
            (),
            {
                "combined_path": combined_path,
                "clean_path": clean_path,
            },
        )()

    def _sofa_stub(config_arg, season, *, force, python_executable):
        season_slug = season.replace("/", "-")
        return _write(processed_root / f"sofa_{config_arg.slug}_{season_slug}.csv", "player_name,team_name,sofa_player_id\nPlayer One,Ajax,10\n")

    def _merge_stub(config_arg, season, *, tm_clean_path, sofa_path, force):
        out_path = Path(config_arg.guess_processed_dataset(season))
        _write(
            out_path,
            "player_id,name,sofa_player_id,sofa_matched\np1,Player One,10,1\np2,Player Two,,0\n",
        )
        return out_path

    monkeypatch.setattr(bff, "_run_transfermarkt_pipeline", _tm_stub)
    monkeypatch.setattr(bff, "_run_sofascore_pipeline", _sofa_stub)
    monkeypatch.setattr(bff, "_merge_tm_sofa_pipeline", _merge_stub)

    summary_path = tmp_path / "future_backfill_summary.json"
    summary = bff.run_future_league_backfill(
        leagues=["dutch_eredivisie"],
        season="2025/26",
        import_dir=str(tmp_path / "incoming_future"),
        summary_json=str(summary_path),
    )

    assert_common_summary_contract(summary)
    staged = tmp_path / "incoming_future" / "dutch_eredivisie_2025-26_with_sofa.csv"
    assert summary["status"] == "ok"
    assert summary["counts"]["ok"] == 1
    assert staged.exists()
    assert summary_path.exists()
    assert summary["results"][0]["merge_summary"]["rows"] == 2
    assert summary["results"][0]["merge_summary"]["sofa_matched_rows"] == 1

    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert_common_summary_contract(disk_summary)
    assert disk_summary["counts"]["ok"] == 1
    assert disk_summary["artifacts"]["summary_json"]["exists"] is True
