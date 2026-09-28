"""Phase 5: driver attribution and 2050 salinization projection.

From the harmonized master table and the station covariates extracted through
Earth Engine, this builds the per-station trend targets, models salinity level
and trend from catchment and climate drivers with honest spatial cross
validation, attributes drivers with SHAP, and projects each station to 2050 by
continuing its observed trend. Figures are navy and black.

Run from the project root after the covariate extraction has finished:
    pip install lightgbm shap scikit-learn scipy matplotlib
    python run_phase5.py

Inputs:
    01_data/processed/master_harmonized.parquet
    06_outputs/phase5/station_covariates.parquet
Outputs (06_outputs/phase5/):
    station_trends.parquet, projection_2050.parquet/.csv,
    driver_importance_level.csv, phase5_results.json,
    fig_drivers_level.png, fig_chloride_2050.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "01_data" / "processed" / "master_harmonized.parquet"
PH5 = ROOT / "06_outputs" / "phase5"
COVARS = PH5 / "station_covariates.parquet"
NAVY, BLACK = "#12355B", "#111111"
ION_KEYS = ("cl", "na", "ca", "mg", "so4", "alk")
MIN_YEARS = 10


def station_trends(master: pd.DataFrame) -> pd.DataFrame:
    """Per station-ion Theil-Sen slope, Mann-Kendall p, and recent level."""
    from scipy.stats import theilslopes, kendalltau
    df = master.copy()
    df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["dt"].notna() & df["value_mg_L"].notna() & (df["value_mg_L"] > 0)]
    df["year"] = df["dt"].dt.year
    ann = df.groupby(["station_id", "ion", "year"]).agg(
        value=("value_mg_L", "mean"), latitude=("latitude", "first"),
        longitude=("longitude", "first"), source=("source", "first")).reset_index()

    rows = []
    for (sid, ion), g in ann.groupby(["station_id", "ion"], sort=False):
        yrs, val = g["year"].to_numpy(), g["value"].to_numpy()
        if len(np.unique(yrs)) < MIN_YEARS:
            continue
        sl, *_ = theilslopes(val, yrs)
        lsl, *_ = theilslopes(np.log1p(val), yrs)
        _, p = kendalltau(yrs, val)
        recent = g[g["year"] >= yrs.max() - 4]["value"].mean()
        st = g.iloc[0]
        rows.append((str(sid), ion, len(g), int(yrs.max()), float(np.median(val)),
                     float(sl), float(lsl), float(p), float(recent),
                     float(st["latitude"]), float(st["longitude"]), st["source"]))
    return pd.DataFrame(rows, columns=[
        "station_id", "ion", "n_year_obs", "base_year", "median_mgL",
        "slope_mgL_yr", "slope_log_yr", "mk_p", "base_level",
        "latitude", "longitude", "source"])


def _spatial_cv(d, features, target, params, n_splits=5):
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb
    m = d[target].notna().values
    dd = d[m].reset_index(drop=True)
    oof = np.full(len(dd), np.nan)
    for tri, tei in GroupKFold(n_splits).split(dd, groups=dd["block"]):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(dd.iloc[tri][features], dd.iloc[tri][target], categorical_feature=["ion"])
        oof[tei] = mdl.predict(dd.iloc[tei][features])
    return dd, oof


def _r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss = np.nansum((y - p) ** 2); st = np.nansum((y - np.nanmean(y)) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")


def main() -> int:
    import lightgbm as lgb
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not MASTER.exists() or not COVARS.exists():
        print("need master_harmonized.parquet and station_covariates.parquet")
        return 1
    PH5.mkdir(parents=True, exist_ok=True)

    print("computing station trends")
    tr = station_trends(pd.read_parquet(MASTER))
    tr.to_parquet(PH5 / "station_trends.parquet", index=False)
    print(f"  {len(tr)} station-ion series with {MIN_YEARS}+ years")

    cov = pd.read_parquet(COVARS); cov["station_id"] = cov["station_id"].astype(str)
    d = tr.merge(cov, on="station_id", how="inner")
    d["ion"] = d["ion"].astype("category")
    d["log_level"] = np.log1p(d["base_level"])
    d["block"] = (np.floor(d["latitude"] / 2).astype(int).astype(str) + "_"
                  + np.floor(d["longitude"] / 2).astype(int).astype(str))

    meta = {"station_id", "ion", "n_year_obs", "base_year", "median_mgL", "slope_mgL_yr",
            "slope_log_yr", "mk_p", "base_level", "log_level", "source", "state_prov",
            "country", "block"}
    covs = [c for c in d.columns if c not in meta]
    features = covs + ["ion"]  # covs already include latitude, longitude
    params = dict(objective="regression", learning_rate=0.04, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, n_estimators=500, random_state=0, n_jobs=-1, verbose=-1)

    print("level model, spatial-block CV")
    dd, oofL = _spatial_cv(d, features, "log_level", params)
    level_by_ion = {ion: round(_r2(dd.loc[dd["ion"] == ion, "log_level"],
                                   oofL[(dd["ion"] == ion).values]), 3) for ion in ION_KEYS}
    level_overall = round(_r2(dd["log_level"], oofL), 3)
    print(f"  level R2 {level_overall} | by ion {level_by_ion}")

    print("trend model, spatial-block CV")
    ddT, oofT = _spatial_cv(d, features, "slope_log_yr", params)
    trend_overall = round(_r2(ddT["slope_log_yr"], oofT), 3)
    print(f"  trend R2 {trend_overall} (expected weak; trends are heterogeneous)")

    print("driver attribution, SHAP on the level model")
    mL = lgb.LGBMRegressor(**params); mL.fit(dd[features], dd["log_level"], categorical_feature=["ion"])
    samp = dd.sample(min(2000, len(dd)), random_state=0)
    sv = shap.TreeExplainer(mL).shap_values(samp[features])
    impL = (pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(sv).mean(0)})
            .sort_values("mean_abs_shap", ascending=False))
    impL.to_csv(PH5 / "driver_importance_level.csv", index=False)

    print("2050 projection, observed-trend continuation")
    d["yrs_ahead"] = 2050 - d["base_year"]
    d["value_2050"] = d["base_level"] * np.exp(d["slope_log_yr"] * d["yrs_ahead"])
    d["pct_change_2050"] = 100 * (d["value_2050"] / d["base_level"] - 1)
    keep = ["station_id", "ion", "latitude", "longitude", "base_level", "base_year",
            "slope_log_yr", "mk_p", "value_2050", "pct_change_2050"]
    d[keep].to_parquet(PH5 / "projection_2050.parquet", index=False)
    d[keep].to_csv(PH5 / "projection_2050.csv", index=False)

    cl = d[d["ion"] == "cl"].dropna(subset=["base_level", "value_2050"])
    thr = [(120, "Aquatic-life\nchronic (120)"), (230, "EPA chronic\n(230)"), (860, "EPA acute\n(860)")]
    now = [(cl["base_level"] > t).mean() * 100 for t, _ in thr]
    fut = [(cl["value_2050"] > t).mean() * 100 for t, _ in thr]

    # figure 1: drivers
    top = impL.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), top["mean_abs_shap"], color=NAVY)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"Drivers of freshwater salinity level\nspatial-block CV R2 = {level_overall}", fontsize=11)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(PH5 / "fig_drivers_level.png", dpi=150); plt.close()

    # figure 2: chloride exceedance
    x = np.arange(len(thr)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - w / 2, now, w, color=BLACK, label="Recent")
    ax.bar(x + w / 2, fut, w, color=NAVY, label="Projected 2050")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in thr])
    ax.set_ylabel("Share of chloride stations exceeding (%)")
    ax.set_title(f"Chloride threshold exceedance, recent vs 2050\ntrend continuation, {len(cl)} stations", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(PH5 / "fig_chloride_2050.png", dpi=150); plt.close()

    results = {
        "level_model_R2_spatialCV": level_overall, "level_R2_by_ion": level_by_ion,
        "trend_model_R2_spatialCV": trend_overall,
        "chloride_median_pct_change_2050": round(float(cl["pct_change_2050"].median()), 1),
        "chloride_exceedance_now_pct": {str(t): round(n, 1) for (t, _), n in zip(thr, now)},
        "chloride_exceedance_2050_pct": {str(t): round(f, 1) for (t, _), f in zip(thr, fut)},
        "top_level_drivers": impL.head(10)["feature"].tolist(),
    }
    (PH5 / "phase5_results.json").write_text(json.dumps(results, indent=2))
    print("\nRESULTS:", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
