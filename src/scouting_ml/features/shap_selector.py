from __future__ import annotations
from typing import List, Tuple, Sequence, Set
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
import matplotlib
matplotlib.use("Agg")


def _base_feature_names(preprocessor: ColumnTransformer, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> List[str]:
    """
    Build a list of base feature names aligned to preprocessor.get_feature_names_out(),
    without relying on fragile string splitting of one-hot outputs.
    """
    bases: List[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            bases.extend(list(cols))
        elif name == "cat":
            ohe = transformer
            if hasattr(transformer, "named_steps"):
                ohe = transformer.named_steps.get("onehot") or transformer.named_steps.get("oh") or transformer.named_steps.get("encoder") or ohe
            if hasattr(ohe, "categories_"):
                for col, cats in zip(categorical_cols, ohe.categories_):
                    bases.extend([col] * len(cats))
            else:
                bases.extend(list(categorical_cols))

    return bases


def select_top_features(
    model,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    top_n: int = 25,
) -> Tuple[List[str], List[str]]:
    """
    Use SHAP to select top_n *base* features (shared across numeric & categorical),
    then split into numeric and categorical lists.
    """
    sample = X.sample(min(500, len(X)), random_state=42)
    transformed = preprocessor.transform(sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(transformed)

    shap_vals_arr = np.array(shap_vals)
    if shap_vals_arr.ndim == 3:
        shap_vals_arr = shap_vals_arr[..., 0]

    importances = np.mean(np.abs(shap_vals_arr), axis=0)

    feature_names = preprocessor.get_feature_names_out()
    base_names = _base_feature_names(preprocessor, numeric_cols, categorical_cols)
    if len(base_names) != len(feature_names):
        # Fallback to original heuristic if lengths mismatch
        base_names = []
        for name in feature_names:
            if name.startswith("num__"):
                base_names.append(name.split("__", 1)[1])
            elif name.startswith("cat__"):
                base_names.append(name.split("__", 1)[1].split("_", 1)[0])
            else:
                base_names.append(name)

    order = np.argsort(importances)[::-1]

    selected_bases: List[str] = []
    seen: Set[str] = set()

    for idx in order:
        base = base_names[idx]
        if base in seen:
            continue
        if base not in numeric_cols and base not in categorical_cols:
            continue
        selected_bases.append(base)
        seen.add(base)
        if len(selected_bases) >= top_n:
            break

    selected_num = [c for c in numeric_cols if c in seen]
    selected_cat = [c for c in categorical_cols if c in seen]

    print(f"[shap-selector] selected {len(selected_num)} numeric and {len(selected_cat)} categorical features")
    return selected_num, selected_cat
