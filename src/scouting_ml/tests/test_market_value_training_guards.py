from __future__ import annotations

import pandas as pd
import pytest

from scouting_ml.models.train_market_value_full import (
    _maybe_save_tree_shap_bar,
    _drop_low_coverage_features,
    _holdout_optuna_namespace,
    _resolve_holdout_leagues,
    _validate_no_leakage_features,
    apply_confidence_scoring,
)


def test_leakage_guard_blocks_value_proxy_features() -> None:
    features = ["sofa_goals_per90", "clubctx_club_strength_proxy", "market_value_proxy"]
    with pytest.raises(ValueError, match="Leakage guard failed"):
        _validate_no_leakage_features(features, strict=True)


def test_confidence_scoring_adds_fair_value_and_score_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_position": "FW",
                "value_segment": "5m_to_20m",
                "market_value_eur": 10_000_000,
                "expected_value_eur": 12_000_000,
                "minutes": 1500,
                "age": 22,
            }
        ]
    )
    priors = pd.DataFrame(
        [
            {
                "model_position": "FW",
                "value_segment": "5m_to_20m",
                "prior_mae_eur": 1_000_000,
                "prior_medae_eur": 800_000,
                "prior_p75ae_eur": 1_200_000,
                "prior_qae_eur": 1_100_000,
                "prior_interval_q": 0.8,
                "n_samples": 100,
            }
        ]
    )

    out = apply_confidence_scoring(frame, priors, interval_q=0.8)
    assert "fair_value_eur" in out.columns
    assert "undervaluation_score" in out.columns
    assert "interval_contains_truth" in out.columns
    assert out.loc[0, "fair_value_eur"] == pytest.approx(out.loc[0, "expected_value_eur"])
    assert out.loc[0, "undervaluation_score"] >= 0.0


def test_drop_low_coverage_features_uses_stricter_provider_threshold() -> None:
    train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "contract_years_left": [2.0, None, 3.0, None],
            "sb_completed_passes_per90": [1.0, None, None, None],
            "league": ["A", "A", "A", "A"],
        }
    )

    keep_num, keep_cat = _drop_low_coverage_features(
        train,
        numeric_cols=["age", "contract_years_left", "sb_completed_passes_per90"],
        categorical_cols=["league"],
        pos="FW",
        min_feature_coverage=0.10,
        min_provider_feature_coverage=0.50,
    )

    assert "age" in keep_num
    assert "contract_years_left" in keep_num
    assert "sb_completed_passes_per90" not in keep_num
    assert "league" in keep_cat


def test_maybe_save_tree_shap_bar_respects_flag(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_save_tree_shap_bar(model, preprocessor, X, out_path) -> None:
        calls.append(str(out_path))

    monkeypatch.setattr(
        "scouting_ml.models.train_market_value_full.save_tree_shap_bar",
        fake_save_tree_shap_bar,
    )

    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    out_path = tmp_path / "shap.png"

    _maybe_save_tree_shap_bar(
        enabled=False,
        model=object(),
        preprocessor=object(),
        X=frame,
        out_path=out_path,
        pos="DF",
    )
    assert calls == []

    _maybe_save_tree_shap_bar(
        enabled=True,
        model=object(),
        preprocessor=object(),
        X=frame,
        out_path=out_path,
        pos="DF",
    )
    assert calls == [str(out_path)]


def test_holdout_optuna_namespace_is_unique_per_league() -> None:
    assert (
        _holdout_optuna_namespace("cheap_aggressive_prod60", "Primeira Liga")
        == "cheap_aggressive_prod60_holdout_primeira_liga"
    )
    assert (
        _holdout_optuna_namespace(None, "Turkish Super Lig")
        == "holdout_turkish_super_lig"
    )


def test_resolve_holdout_leagues_accepts_slug_and_display_name() -> None:
    frame = pd.DataFrame(
        [
            {"league": "Austrian Bundesliga", "season": "2024/25"},
            {"league": "Austrian Bundesliga", "season": "2023/24"},
            {"league": "Eredivisie", "season": "2024/25"},
        ]
    )

    resolved = _resolve_holdout_leagues(
        frame,
        ["austrian_bundesliga", "Austrian Bundesliga", "eredivisie"],
        test_season="2024/25",
    )

    assert [(item.league, item.league_slug) for item in resolved] == [
        ("Austrian Bundesliga", "austrian_bundesliga"),
        ("Eredivisie", "dutch_eredivisie"),
    ]
    assert resolved[0].resolved_from == "slug"
    assert resolved[1].resolved_from == "name"


def test_resolve_holdout_leagues_raises_on_unknown_token() -> None:
    frame = pd.DataFrame([{"league": "Austrian Bundesliga", "season": "2024/25"}])

    with pytest.raises(ValueError, match="Unknown league holdout token"):
        _resolve_holdout_leagues(frame, ["made_up_league"], test_season="2024/25")


def test_resolve_holdout_leagues_raises_when_test_season_missing() -> None:
    frame = pd.DataFrame([{"league": "Austrian Bundesliga", "season": "2023/24"}])

    with pytest.raises(ValueError, match="zero rows for test season 2024/25"):
        _resolve_holdout_leagues(frame, ["austrian_bundesliga"], test_season="2024/25")
