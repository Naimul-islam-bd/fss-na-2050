"""Phase 5 scenario projection: climate-change response under SSP2-4.5 and SSP5-8.5.

The observed-trend projection continues each station's measured trajectory. This
adds the climate-change contribution: the CMIP6 2050 deltas shift each station's
climate covariates to their mid-century values under each SSP, the trained level
model re-predicts the equilibrium salinity, and the ratio to the baseline
prediction gives a climate response factor that is applied to the trend
projection. Management is held constant, so this isolates the climate signal and
gives the forecast a scenario range.

Run from the project root after run_phase5.py and the CMIP6 deltas exist:
    python run_scenarios.py
Inputs (06_outputs/phase5/): projection_2050.parquet, station_covariates.parquet,
cmip6_deltas.parquet. Output: scenario_projection.parquet, scenario_results.json,
fig_chloride_scenarios.png.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PH5 = ROOT / "06_outputs" / "phase5"
NAVY, BLACK, GREY = "#12355B", "#111111", "#7A8699"
PET_SENS = 0.02   # potential ET rises about 2 percent per degree C of warming


def load(proj_p=None, cov_p=None, del_p=None):
    proj = pd.read_parquet(proj_p or PH5 / "projection_2050.parquet")
    cov = pd.read_parquet(cov_p or PH5 / "station_covariates.parquet")
    dlt = pd.read_parquet(del_p or PH5 / "cmip6_deltas.parquet")
    for d in (proj, cov, dlt):
        d["station_id"] = d["station_id"].astype(str)
    df = proj.merge(cov, on="station_id", how="inner").merge(dlt, on="station_id", how="inner")
    return df


def run(df: pd.DataFrame, outdir: Path, make_fig=True) -> dict:
    import lightgbm as lgb
    df = df.copy()
    df["ion"] = df["ion"].astype("category")
    df["log_level"] = np.log1p(df["base_level"])

    delta_cols = ["dtemp_ssp245", "dprecip_ssp245", "dtemp_ssp585", "dprecip_ssp585"]
    drop = {"station_id", "base_level", "base_year", "slope_log_yr", "mk_p",
            "value_2050", "pct_change_2050", "log_level", "source", "state_prov",
            "country", "median_mgL", "n_year_obs"} | set(delta_cols)
    features = [c for c in df.columns if c not in drop]  # covariates + latitude/longitude + ion

    params = dict(objective="regression", learning_rate=0.04, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, n_estimators=500, random_state=0, n_jobs=-1, verbose=-1)
    model = lgb.LGBMRegressor(**params)
    model.fit(df[features], df["log_level"], categorical_feature=["ion"])
    base_pred = model.predict(df[features])

    def shifted(ssp):
        x = df.copy()
        dt, dp = x[f"dtemp_{ssp}"], x[f"dprecip_{ssp}"]
        for col in ("temp_c",):
            if col in x:
                x[col] = x[col] + dt
        for col in ("precip_mm", "pr_tc_mm"):
            if col in x:
                x[col] = x[col] + dp
        if "pet_mm" in x:
            x["pet_mm"] = x["pet_mm"] * (1 + PET_SENS * dt)
        if "aridity" in x and "pet_mm" in x and "pr_tc_mm" in x:
            x["aridity"] = x["pr_tc_mm"] / x["pet_mm"].clip(lower=1)
        return model.predict(x[features])

    res = {"scenarios": {}}
    df["value_2050_trend"] = df["value_2050"]
    for ssp in ("ssp245", "ssp585"):
        factor = np.exp(shifted(ssp) - base_pred)
        df[f"climate_factor_{ssp}"] = factor
        df[f"value_2050_{ssp}"] = df["value_2050"] * factor

    cl = df[df["ion"] == "cl"].dropna(subset=["value_2050"])
    thr = [120, 230, 860]
    def exc(col):
        return {str(t): round((cl[col] > t).mean() * 100, 1) for t in thr}
    res["chloride_exceedance"] = {
        "recent": exc("base_level"),
        "2050_trend": exc("value_2050_trend"),
        "2050_ssp245": exc("value_2050_ssp245"),
        "2050_ssp585": exc("value_2050_ssp585"),
    }
    res["climate_factor_median"] = {
        "ssp245": round(float(df["climate_factor_ssp245"].median()), 4),
        "ssp585": round(float(df["climate_factor_ssp585"].median()), 4),
    }
    res["chloride_median_pct_change_2050"] = {
        "trend": round(float((cl["value_2050_trend"] / cl["base_level"] - 1).median() * 100), 1),
        "ssp245": round(float((cl["value_2050_ssp245"] / cl["base_level"] - 1).median() * 100), 1),
        "ssp585": round(float((cl["value_2050_ssp585"] / cl["base_level"] - 1).median() * 100), 1),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    keep = ["station_id", "ion", "latitude", "longitude", "base_level",
            "value_2050_trend", "value_2050_ssp245", "value_2050_ssp585",
            "climate_factor_ssp245", "climate_factor_ssp585"]
    df[keep].to_parquet(outdir / "scenario_projection.parquet", index=False)
    (outdir / "scenario_results.json").write_text(json.dumps(res, indent=2))

    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = ["Aquatic-life\nchronic (120)", "EPA chronic\n(230)", "EPA acute\n(860)"]
        series = [("Recent", "base_level", GREY), ("2050 trend", "value_2050_trend", BLACK),
                  ("2050 SSP2-4.5", "value_2050_ssp245", NAVY), ("2050 SSP5-8.5", "value_2050_ssp585", "#3E6DA3")]
        x = np.arange(len(thr)); w = 0.2
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for j, (lab, col, color) in enumerate(series):
            vals = [(cl[col] > t).mean() * 100 for t in thr]
            ax.bar(x + (j - 1.5) * w, vals, w, color=color, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Share of chloride stations exceeding (%)")
        ax.set_title(f"Chloride exceedance to 2050, trend and climate scenarios\n{len(cl)} stations", fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        plt.tight_layout(); plt.savefig(outdir / "fig_chloride_scenarios.png", dpi=150); plt.close()
    return res


def main():
    df = load()
    res = run(df, PH5)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
