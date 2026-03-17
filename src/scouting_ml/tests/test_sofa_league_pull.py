from __future__ import annotations

from types import SimpleNamespace

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
