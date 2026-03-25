from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd

from scouting_ml.features import optuna_tuner as tuner


class _FakeTrial:
    def __init__(self) -> None:
        self.params: dict[str, float | int | str] = {}
        self._reports: list[tuple[float, int]] = []

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, log=False):
        value = (float(low) + float(high)) / 2.0
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high):
        value = int(low) if name == "max_depth" else int(high)
        self.params[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        self._reports.append((float(value), int(step)))

    def should_prune(self) -> bool:
        return False


class _FakeStudy:
    def __init__(self) -> None:
        self.study_name = "fake-study"
        self.direction = SimpleNamespace(name="MINIMIZE")
        self.user_attrs: dict[str, object] = {}
        self.trials: list[SimpleNamespace] = []
        self.best_trial: SimpleNamespace | None = None
        self.best_value: float | None = None

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

    def optimize(self, objective, n_trials: int) -> None:
        trial = _FakeTrial()
        try:
            value = objective(trial)
        except optuna.TrialPruned:
            self.trials.append(
                SimpleNamespace(
                    state=optuna.trial.TrialState.PRUNED,
                    value=None,
                    number=0,
                    params=dict(trial.params),
                )
            )
            return
        self.best_value = float(value)
        self.best_trial = SimpleNamespace(number=0, params=dict(trial.params))
        self.trials.append(
            SimpleNamespace(
                state=optuna.trial.TrialState.COMPLETE,
                value=float(value),
                number=0,
                params=dict(trial.params),
            )
        )


class _MemoryFailPipe:
    def fit(self, X, y, **kwargs):
        raise MemoryError("simulated LightGBM memory failure")

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class _SuccessPipe:
    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class _FakeLightGBMError(Exception):
    pass


class _LightGBMFailPipe:
    def fit(self, X, y, **kwargs):
        raise _FakeLightGBMError("Model file doesn't specify the number of classes")

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def _sample_frame() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "num": np.linspace(0.0, 1.0, 12),
            "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    y = np.linspace(0.0, 0.5, 12)
    return X, y


def test_tune_lgbm_uses_fallback_when_all_trials_prune_on_memory(monkeypatch, tmp_path: Path) -> None:
    X, y = _sample_frame()
    study = _FakeStudy()

    monkeypatch.setattr(tuner.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(tuner, "_make_lgbm_regressor", lambda **params: object())
    monkeypatch.setattr(tuner, "clone", lambda pipe: _MemoryFailPipe())

    metadata_path = tmp_path / "optuna_fallback.json"
    params = tuner.tune_lgbm(
        X,
        y,
        numeric_cols=["num"],
        categorical_cols=["cat"],
        n_trials=1,
        study_metadata_path=str(metadata_path),
    )

    assert params["n_jobs"] == 1
    assert params["verbosity"] == -1
    assert params["n_estimators"] == 900
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["fallback_used"] is True


def test_tune_lgbm_returns_constrained_successful_params(monkeypatch) -> None:
    X, y = _sample_frame()
    study = _FakeStudy()

    monkeypatch.setattr(tuner.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(tuner, "_make_lgbm_regressor", lambda **params: object())
    monkeypatch.setattr(tuner, "clone", lambda pipe: _SuccessPipe())

    params = tuner.tune_lgbm(
        X,
        y,
        numeric_cols=["num"],
        categorical_cols=["cat"],
        n_trials=1,
    )

    assert params["n_jobs"] == 1
    assert params["verbosity"] == -1
    assert params["max_depth"] <= 8
    assert params["num_leaves"] <= 64
    assert params["max_depth"] == 3
    assert params["num_leaves"] == 8


def test_tune_lgbm_prunes_retryable_lightgbm_internal_error(monkeypatch, tmp_path: Path) -> None:
    X, y = _sample_frame()
    study = _FakeStudy()

    monkeypatch.setattr(tuner.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(tuner, "_make_lgbm_regressor", lambda **params: object())
    monkeypatch.setattr(tuner, "clone", lambda pipe: _LightGBMFailPipe())

    metadata_path = tmp_path / "optuna_lgbm_retryable.json"
    params = tuner.tune_lgbm(
        X,
        y,
        numeric_cols=["num"],
        categorical_cols=["cat"],
        n_trials=1,
        study_metadata_path=str(metadata_path),
    )

    assert params["n_jobs"] == 1
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["fallback_used"] is True


def test_tune_lgbm_prunes_retryable_row_selection_memory_error(monkeypatch, tmp_path: Path) -> None:
    X, y = _sample_frame()
    study = _FakeStudy()

    monkeypatch.setattr(tuner.optuna, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(tuner, "_make_lgbm_regressor", lambda **params: object())
    monkeypatch.setattr(tuner, "clone", lambda pipe: _SuccessPipe())
    monkeypatch.setattr(
        tuner,
        "_select_rows",
        lambda X, idx: (_ for _ in ()).throw(MemoryError("simulated row selection memory failure")),
    )

    metadata_path = tmp_path / "optuna_row_select_retryable.json"
    params = tuner.tune_lgbm(
        X,
        y,
        numeric_cols=["num"],
        categorical_cols=["cat"],
        n_trials=1,
        study_metadata_path=str(metadata_path),
    )

    assert params["n_jobs"] == 1
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["fallback_used"] is True
