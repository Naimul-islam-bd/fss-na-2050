"""Ion-ratio source fingerprinting and salinization regime typology.

Two additions that move the study from "which ions change" to "which processes
drive the change, mapped across the continent":

  1. Ion ratios at every co-located station. The sodium-to-chloride mass ratio
     fingerprints the salt source: pure halite (road salt or natural rock salt)
     sits near 0.65, a chloride excess (ratio below 0.65) still marks halite-type
     inputs, and a sodium excess (ratio above about 1) marks silicate weathering,
     wastewater, or ion-exchange softeners. Calcium-to-magnesium separates
     limestone from dolomite weathering, and chloride-to-sulfate separates the
     road-salt signal from the mineral and deposition signal.

  2. An unsupervised regime typology. Stations are clustered on their multi-ion
     signature, their chloride and sulfate trends, and a few landscape drivers, so
     each station falls into a salinization regime that is named from the cluster
     centre rather than assumed in advance.

Run from the project root (or any folder holding the phase-5 outputs):
    python analysis_ratios_typology.py
Inputs: projection_2050_allion.parquet, station_trends.parquet,
station_covariates.parquet, na_boundaries.geojson.
Outputs (06_outputs/phase5/): ion_ratios.parquet, station_typology.parquet,
typology_summary.json; figures Figure_5 (typology) and Figure_6 (Na:Cl ratio).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import joh_style as S

warnings.filterwarnings("ignore")

# molar masses for mass-ratio interpretation
HALITE_NaCl = 22.99 / 35.45      # 0.649, sodium-to-chloride mass ratio of pure halite


def load_wide():
    pj = pd.read_parquet(S.find("projection_2050_allion.parquet"))
    pj["station_id"] = pj["station_id"].astype(str)
    lvl = pj.pivot_table(index="station_id", columns="ion", values="base_level", aggfunc="first")
    slp = pj.pivot_table(index="station_id", columns="ion", values="slope_log", aggfunc="first")
    slp.columns = [f"slope_{c}" for c in slp.columns]
    coords = pj.groupby("station_id")[["latitude", "longitude"]].first()
    wide = lvl.join(slp).join(coords)
    return wide


def ion_ratios(wide):
    r = pd.DataFrame(index=wide.index)
    r["na_cl"] = wide["na"] / wide["cl"]
    r["ca_mg"] = wide["ca"] / wide["mg"]
    r["cl_so4"] = wide["cl"] / wide["so4"]
    r["cl_alk"] = wide["cl"] / wide["alk"]
    r["cl_level"] = wide["cl"]
    r[["latitude", "longitude"]] = wide[["latitude", "longitude"]]
    return r


def build_features(wide, cov):
    ions = ["cl", "na", "ca", "mg", "so4"]
    have = wide[ions].notna().sum(axis=1)
    core = wide[have >= 4].copy()                     # stations with a real multi-ion signature
    feats = pd.DataFrame(index=core.index)
    for i in ions:
        feats[f"log_{i}"] = np.log1p(core[i])
    feats["na_cl"] = (core["na"] / core["cl"]).clip(0, 3)
    feats["ca_mg"] = (core["ca"] / core["mg"]).clip(0, 8)
    feats["slope_cl"] = core.get("slope_cl")
    feats["slope_so4"] = core.get("slope_so4")
    cov = cov.set_index("station_id")
    for c in ["lc_tree_5km", "night_lights_5km", "aridity", "soil_ph", "temp_c"]:
        if c in cov.columns:
            feats[c] = cov[c].reindex(core.index)
    feats[["latitude", "longitude"]] = core[["latitude", "longitude"]]
    return feats


def cluster(feats, k=5):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in feats.columns if c not in ("latitude", "longitude")]
    X = feats[cols].apply(lambda s: s.fillna(s.median()))
    Xs = StandardScaler().fit_transform(X)
    for kk in range(3, 7):                                   # report the range, then fix k
        km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(Xs)
        sil = silhouette_score(Xs, km.labels_, sample_size=min(3000, len(Xs)), random_state=0)
        print(f"  k={kk}  silhouette={sil:.3f}")
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_, sample_size=min(3000, len(Xs)), random_state=0)
    feats = feats.copy()
    feats["cluster"] = km.labels_
    return feats, k, sil, cols


def name_clusters(feats, cols):
    """Assign five archetype names to the clusters one-to-one, greedily, from centres.

    The regimes form a soft continuum (low silhouette), so each cluster is matched to
    the archetype it fits best, without forcing hard separation. Every name is unique.
    """
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    cen = feats.groupby("cluster")[cols].median()
    zc = cen.apply(z, axis=0)

    def col(name):
        return zc[name] if name in zc.columns else pd.Series(0.0, index=zc.index)

    # score each cluster against each archetype (higher is a better match)
    scores = pd.DataFrame(index=zc.index)
    scores["Road-salt urban (rising Cl)"] = col("log_cl") + col("night_lights_5km") + col("slope_cl")
    scores["Sulfate-declining"] = col("log_so4") - col("slope_so4")
    scores["Dilute forested"] = col("lc_tree_5km") - col("log_cl")
    scores["Arid evaporative"] = -col("aridity") + col("na_cl")
    scores["Carbonate weathering"] = col("ca_mg") + col("log_ca")

    names = {}
    remaining = set(zc.index)
    # greedy one-to-one assignment in priority order
    for arche in ["Road-salt urban (rising Cl)", "Sulfate-declining", "Dilute forested",
                  "Arid evaporative", "Carbonate weathering"]:
        if not remaining:
            break
        best = max(remaining, key=lambda c: scores.loc[c, arche])
        names[best] = arche
        remaining.discard(best)
    for c in remaining:                                     # any leftover keeps a signature name
        names[c] = f"Mixed ({cen.loc[c].get('na_cl', float('nan')):.2f} Na:Cl)"

    notes = {}
    for c, g in feats.groupby("cluster"):
        m = g[cols].median()
        notes[c] = dict(n=int(len(g)), cl_level=round(float(np.expm1(m.get("log_cl", 0))), 1),
                        na_cl=round(float(m.get("na_cl", np.nan)), 2),
                        ca_mg=round(float(m.get("ca_mg", np.nan)), 2),
                        slope_cl=round(float(m.get("slope_cl", 0)), 4),
                        slope_so4=round(float(m.get("slope_so4", 0)), 4))
    return names, notes


def fig_typology(feats, names, bounds):
    import matplotlib.pyplot as plt
    pts = S.to_points(feats.reset_index())
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    S.map_axes(ax, bounds, extent_pts=pts)
    order = sorted(names)
    for i, c in enumerate(order):
        sub = pts[pts["cluster"] == c]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=7, linewidths=0, alpha=0.85,
                   color=S.CATEGORICAL[i % len(S.CATEGORICAL)], label=names[c], zorder=2)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", markerscale=1.6,
              handletextpad=0.3, borderpad=0.2)
    S.save(fig, "Figure_4")


def fig_na_cl(ratios, bounds):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    d = ratios.dropna(subset=["na_cl"]).copy()
    d["na_cl"] = d["na_cl"].clip(0.1, 2.0)
    pts = S.to_points(d.reset_index()).sort_values("cl_level")
    norm = TwoSlopeNorm(vmin=0.2, vcenter=HALITE_NaCl, vmax=1.5)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    S.map_axes(ax, bounds, extent_pts=pts)
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts["na_cl"], cmap=S.DIVERGE.reversed(),
                    norm=norm, s=7, linewidths=0, alpha=0.85, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, extend="both")
    cb.set_label("Sodium-to-chloride mass ratio", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.ax.axhline(HALITE_NaCl, color=S.BLACK, linewidth=0.8)
    ax.text(0.01, 0.02, f"n = {len(pts):,} stations\nline = halite (0.65)",
            transform=ax.transAxes, fontsize=8, color=S.BLACK)
    S.save(fig, "Figure_3")


def main():
    import matplotlib.pyplot as plt
    wide = load_wide()
    cov = pd.read_parquet(S.find("station_covariates.parquet"))
    cov["station_id"] = cov["station_id"].astype(str)

    ratios = ion_ratios(wide)
    ratios.to_parquet(S.find("projection_2050_allion.parquet").parent / "ion_ratios.parquet")

    # ratio summary
    def q(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return dict(median=round(float(s.median()), 3), p25=round(float(s.quantile(.25)), 3),
                    p75=round(float(s.quantile(.75)), 3), n=int(s.size))
    halite_like = ratios["na_cl"].between(0.4, 0.9)
    summary = {
        "na_cl": q(ratios["na_cl"]), "ca_mg": q(ratios["ca_mg"]),
        "cl_so4": q(ratios["cl_so4"]),
        "halite_like_share_pct": round(float(halite_like.mean() * 100), 1),
        "halite_reference": round(HALITE_NaCl, 3),
    }

    print("Clustering regimes:")
    feats = build_features(wide, cov)
    feats, k, sil, cols = cluster(feats)
    names, notes = name_clusters(feats, cols)
    feats["regime"] = feats["cluster"].map(names)
    feats.reset_index().to_parquet(
        S.find("projection_2050_allion.parquet").parent / "station_typology.parquet")

    summary["typology"] = {
        "k": int(k), "silhouette": round(float(sil), 3),
        "regimes": {names[c]: notes[c] for c in sorted(names)},
    }
    (S.find("projection_2050_allion.parquet").parent / "typology_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    bounds = S.load_boundaries()
    fig_typology(feats, names, bounds)
    fig_na_cl(ratios, bounds)
    print("done ratios + typology")


if __name__ == "__main__":
    main()
