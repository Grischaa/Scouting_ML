from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer


def save_tree_shap_bar(
    model,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    out_path: Path,
    max_display: int = 25,
) -> None:
    """
    Compute SHAP values for a tree-based model and save a global bar plot.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = X.sample(min(400, len(X)), random_state=42)
    transformed = preprocessor.transform(sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(transformed)

    feature_names = preprocessor.get_feature_names_out()
    shap_expl = shap.Explanation(values=shap_vals, data=transformed, feature_names=feature_names)

    shap.plots.bar(shap_expl, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[shap] saved → {out_path}")
