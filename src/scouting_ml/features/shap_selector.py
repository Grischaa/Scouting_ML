from __future__ import annotations
from typing import List, Tuple, Sequence, Set

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
import matplotlib
matplotlib.use("Agg")


def _base_from_feature_name(name: str) -> str:
    """
    Map transformed feature name back to base column name.
    Examples:
      'num__age' -> 'age'
      'cat__league_English Premier League' -> 'league'
      'cat__age_bucket_u21' -> 'age_bucket'
    """
    if name.startswith("num__"):
        return name.split("__", 1)[1]
    if name.startswith("cat__"):
        rest = name.split("__", 1)[1]
        # remove last _category
        if "_" in rest:
            return rest.rsplit("_", 1)[0]
        return rest
    return name


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

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(transformed)

    shap_vals_arr = np.array(shap_vals)
    if shap_vals_arr.ndim == 3:
        shap_vals_arr = shap_vals_arr[..., 0]

    importances = np.mean(np.abs(shap_vals_arr), axis=0)

    feature_names = preprocessor.get_feature_names_out()
    order = np.argsort(importances)[::-1]

    selected_bases: List[str] = []
    seen: Set[str] = set()

    for idx in order:
        base = _base_from_feature_name(feature_names[idx])
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
