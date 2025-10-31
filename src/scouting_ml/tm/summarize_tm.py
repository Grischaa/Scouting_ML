# scouting_ml/summarize_tm.py
from __future__ import annotations
import pandas as pd
import numpy as np
import typer
from scouting_ml.utils.import_guard import *  # noqa: F403

app = typer.Typer(add_completion=False)


@app.command()
def team(
    infile: str = typer.Option(..., "--in"),
    outfile: str = typer.Option(..., "--out")
):
    """
    Produce lightweight team summary KPIs from normalized CSV.
    """
    df = pd.read_csv(infile)

    def _num_mean(series):
        if series is None:
            return None
        s = pd.to_numeric(series, errors="coerce")
        m = s.dropna().mean()
        return None if np.isnan(m) else float(m)

    age_mean = _num_mean(df.get("age"))
    height_mean = _num_mean(df.get("height_cm"))
    mv_sum = (
        pd.to_numeric(df.get("market_value_eur"), errors="coerce")
        .dropna()
        .sum()
        if "market_value_eur" in df
        else None
    )
    mv_sum = (
        None
        if mv_sum is None or (isinstance(mv_sum, float) and np.isnan(mv_sum))
        else int(mv_sum)
    )

    kpis = {
        "n_players": int(len(df)),
        "avg_age": round(age_mean, 2) if age_mean is not None else None,
        "avg_height_cm": round(height_mean, 1) if height_mean is not None else None,
        "total_market_value_eur": mv_sum,
        "position_distribution": df["position_std"].value_counts().to_dict()
        if "position_std" in df
        else {},
        "foot_distribution": df["foot_std"].value_counts().to_dict()
        if "foot_std" in df
        else {},
        "nationality_distribution": df["nationality_std"].value_counts().to_dict()
        if "nationality_std" in df
        else {},
    }

    pd.Series(kpis, name="value").to_json(outfile, indent=2, force_ascii=False)
    typer.echo(f"[summarize_tm] Wrote: {outfile}")


if __name__ == "__main__":
    app()
