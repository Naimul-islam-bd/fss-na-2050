"""Seasonal road-salt fingerprint from the observation-level record.

The trend model shows that the rate of chloride change is not set by static catchment
properties, which points to local de-icing management. This test turns that inference
into direct evidence. Road salt is applied in winter, so a road-salt-driven stream
carries more chloride in winter than in summer. For every station with enough
sub-annual sampling we compute the mean winter (December to February) and mean summer
(June to August) chloride and their ratio, then show that the winter enrichment rises
with latitude and with urban intensity, the two conditions under which road salt is
applied. We also compare the winter and summer chloride trends: if winter chloride is
rising faster than summer chloride, de-icing intensity is increasing.

Run from the project root (or a folder with master_harmonized.parquet):
    python analysis_seasonality.py
Inputs: master_harmonized.parquet, station_covariates.parquet, na_boundaries.geojson.
Outputs (06_outputs/phase5/): seasonality_chloride.parquet, seasonality_summary.json;
figures Figure_7 (winter:summer ratio map) and Figure_8 (ratio vs latitude).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import joh_style as S

warnings.filterwarnings("ignore")

MIN_SEASON_OBS = 8          # need enough winter and summer samples for a stable ratio
MIN_YEARS_TREND = 10        # years with both seasons, for a winter/summer trend


def theilsen(years, vals):
    y = np.asarray(years, float); v = np.asarray(vals, float)
    if len(y) < 5:
        return np.nan
    from scipy.stats import theilslopes
    return float(theilslopes(v, y)[0])


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    m = pd.read_parquet(S.find_master(), columns=["station_id", "ion", "datetime",
                                                  "value_mg_L", "latitude", "longitude"])
    cl = m[m.ion == "cl"].dropna(subset=["value_mg_L", "datetime"]).copy()
    cl["station_id"] = cl["station_id"].astype(str)
    cl["month"] = cl["datetime"].dt.month
    cl["year"] = cl["datetime"].dt.year
    cl["season"] = np.where(cl["month"].isin([12, 1, 2]), "winter",
                            np.where(cl["month"].isin([6, 7, 8]), "summer", "other"))

    # station-level winter and summer means, over the whole record
    def season_mean(s):
        g = cl[cl.season == s].groupby("station_id")
        return g["value_mg_L"].agg(["mean", "size"]).rename(columns={"mean": f"{s}_mean", "size": f"{s}_n"})
    w = season_mean("winter"); su = season_mean("summer")
    coords = cl.groupby("station_id")[["latitude", "longitude"]].first()
    d = coords.join(w).join(su).dropna(subset=["winter_mean", "summer_mean"])
    d = d[(d["winter_n"] >= MIN_SEASON_OBS) & (d["summer_n"] >= MIN_SEASON_OBS)].copy()
    d["ws_ratio"] = d["winter_mean"] / d["summer_mean"].clip(lower=0.1)

    # winter and summer trends, per station-year seasonal means
    sy = (cl[cl.season.isin(["winter", "summer"])]
          .groupby(["station_id", "season", "year"])["value_mg_L"].mean().reset_index())
    rows = []
    for (sid, seas), g in sy.groupby(["station_id", "season"]):
        if g["year"].nunique() >= MIN_YEARS_TREND:
            rows.append((sid, seas, theilsen(g["year"], np.log1p(g["value_mg_L"]))))
    tr = pd.DataFrame(rows, columns=["station_id", "season", "slope_log"]).pivot(
        index="station_id", columns="season", values="slope_log")
    d = d.join(tr.rename(columns={"winter": "winter_slope", "summer": "summer_slope"}))

    cov = pd.read_parquet(S.find("station_covariates.parquet"))
    cov["station_id"] = cov["station_id"].astype(str)
    d = d.join(cov.set_index("station_id")[["night_lights_5km"]])
    d.reset_index().to_parquet(S.find("station_covariates.parquet").parent / "seasonality_chloride.parquet")

    cold = d[d["latitude"] >= 40]
    warm = d[d["latitude"] < 40]
    urban = d[d["night_lights_5km"] > d["night_lights_5km"].median()]
    both = d.dropna(subset=["winter_slope", "summer_slope"])
    cold_both = both[both["latitude"] >= 40]
    summary = {
        "n_stations": int(len(d)),
        "ws_ratio_median_all": round(float(d["ws_ratio"].median()), 2),
        "ws_ratio_median_cold_lat_ge40": round(float(cold["ws_ratio"].median()), 2),
        "ws_ratio_median_warm_lat_lt40": round(float(warm["ws_ratio"].median()), 2),
        "ws_ratio_median_urban": round(float(urban["ws_ratio"].median()), 2),
        "corr_ratio_latitude": round(float(d[["ws_ratio", "latitude"]].corr().iloc[0, 1]), 3),
        "corr_ratio_nightlights": round(float(d[["ws_ratio", "night_lights_5km"]].corr().iloc[0, 1]), 3),
        "share_winter_enriched_pct": round(float((d["ws_ratio"] > 1).mean() * 100), 1),
        "share_winter_enriched_cold_pct": round(float((cold["ws_ratio"] > 1).mean() * 100), 1),
        "winter_slope_median_cold": round(float(cold_both["winter_slope"].median()), 4),
        "summer_slope_median_cold": round(float(cold_both["summer_slope"].median()), 4),
        "n_trend_stations": int(len(both)),
    }
    (S.find("station_covariates.parquet").parent / "seasonality_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    bounds = S.load_boundaries()

    # Figure 7: winter-to-summer chloride ratio map (navy = winter enriched)
    dd = d.copy(); dd["ws_ratio"] = dd["ws_ratio"].clip(0.3, 3.0)
    pts = S.to_points(dd.reset_index()).sort_values("ws_ratio")
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.5)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    S.map_axes(ax, bounds, extent_pts=pts)
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts["ws_ratio"], cmap=S.DIVERGE,
                    norm=norm, s=8, linewidths=0, alpha=0.85, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, extend="both")
    cb.set_label("Winter-to-summer chloride ratio", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.text(0.01, 0.02, f"n = {len(pts):,} stations", transform=ax.transAxes,
            fontsize=8, color=S.BLACK)
    S.save(fig, "Figure_S3")

    # Figure 8: ratio vs latitude, binned medians, showing the cold-region rise
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    dl = d[(d["ws_ratio"] > 0.2) & (d["ws_ratio"] < 6)]
    ax.scatter(dl["latitude"], dl["ws_ratio"], s=6, color=S.GREY, alpha=0.35, linewidths=0)
    bins = np.arange(25, 62, 3)
    dl = dl.assign(band=pd.cut(dl["latitude"], bins))
    med = dl.groupby("band")["ws_ratio"].median()
    centers = [b.mid for b in med.index.categories]
    ax.plot(centers, med.values, color=S.NAVY, linewidth=1.8, marker="o", markersize=4,
            label="Median by 3 deg latitude band")
    ax.axhline(1.0, color=S.BLACK, linewidth=0.7, linestyle="--")
    ax.set_xlabel("Latitude (degrees north)", fontsize=9)
    ax.set_ylabel("Winter-to-summer chloride ratio", fontsize=9)
    ax.set_ylim(0, 4)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    S.save(fig, "Figure_S4")
    print("done seasonality")


if __name__ == "__main__":
    main()
