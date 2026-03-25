from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scouting_ml.league_registry import LEAGUES
from scouting_ml.sofa import league_pull


def test_ensure_league_registered_writes_scraperfc_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(comps={})
    monkeypatch.setattr(
        league_pull,
        "_load_sofascore_runtime",
        lambda: (fake_module, object),
    )

    league_pull.ensure_league_registered("Scottish Premiership", 36)

    assert fake_module.comps["Scottish Premiership"] == {"SOFASCORE": 36}


def test_ensure_league_registered_normalizes_legacy_int_entry(monkeypatch) -> None:
    fake_module = SimpleNamespace(comps={"Scottish Premiership": 36})
    monkeypatch.setattr(
        league_pull,
        "_load_sofascore_runtime",
        lambda: (fake_module, object),
    )

    league_pull.ensure_league_registered("Scottish Premiership", 36)

    assert fake_module.comps["Scottish Premiership"] == {"SOFASCORE": 36}


def test_pull_writes_header_only_csv_for_empty_frame(monkeypatch, tmp_path) -> None:
    fake_module = SimpleNamespace(comps={})

    class FakeSofascore:
        def get_valid_seasons(self, league_key: str) -> dict[str, object]:
            assert league_key == "Ekstraklasa"
            return {"24/25": object()}

        def scrape_player_league_stats(self, **kwargs) -> pd.DataFrame:
            assert kwargs["year"] == "24/25"
            return pd.DataFrame()

    monkeypatch.setattr(
        league_pull,
        "_load_sofascore_runtime",
        lambda: (fake_module, FakeSofascore),
    )

    out_path = tmp_path / "sofa_polish.csv"
    league_pull.pull(
        league="Ekstraklasa",
        season="2024/25",
        sofa_season="24/25",
        sofa_league_key="Ekstraklasa",
        tournament_id=59,
        outfile=out_path,
    )

    frame = pd.read_csv(out_path)
    assert frame.empty
    assert list(frame.columns) == ["player", "team", "player id", "team id"]


def test_pull_resolves_calendar_year_alias_for_split_season(monkeypatch, tmp_path) -> None:
    fake_module = SimpleNamespace(comps={})

    class FakeSofascore:
        def get_valid_seasons(self, league_key: str) -> dict[str, object]:
            assert league_key == "Super League"
            return {"2024": object()}

        def scrape_player_league_stats(self, **kwargs) -> pd.DataFrame:
            assert kwargs["year"] == "2024"
            return pd.DataFrame([{"player": "A", "team": "B", "player id": 1, "team id": 2}])

    monkeypatch.setattr(
        league_pull,
        "_load_sofascore_runtime",
        lambda: (fake_module, FakeSofascore),
    )

    out_path = tmp_path / "sofa_swiss.csv"
    league_pull.pull(
        league="Super League",
        season="2024/25",
        sofa_season="24/25",
        sofa_league_key="Super League",
        tournament_id=215,
        outfile=out_path,
    )

    frame = pd.read_csv(out_path)
    assert len(frame) == 1
    assert frame.loc[0, "player"] == "A"


def test_registry_pins_current_swiss_danish_and_czech_sofa_mappings() -> None:
    assert LEAGUES["swiss_super_league"].sofa_tournament_id == 215
    assert LEAGUES["swiss_super_league"].sofa_season_map["2024/25"] == "2024"
    assert LEAGUES["swiss_super_league"].sofa_season_map["2023/24"] == "2023"

    assert LEAGUES["danish_superliga"].sofa_tournament_id == 39
    assert LEAGUES["danish_superliga"].sofa_season_map["2024/25"] == "24/25"
    assert LEAGUES["danish_superliga"].sofa_season_map["2023/24"] == "23/24"

    assert LEAGUES["czech_first_league"].sofa_tournament_id == 172
