"""Figure generation for the continental salinization manuscript, Journal of Hydrology format.

Every figure is written as a separate file (Figure_1, Figure_2, ...) in both TIFF
and PNG, following the Elsevier artwork rules: colour raster at 300 dpi or more,
single-column width at least 1063 px and full-page width at least 2244 px, and no
title drawn inside the figure, because the caption (a brief title plus description)
belongs in the manuscript text, not on the artwork. Axis labels, legends, and
colour-bar labels are kept, since those are part of the plot rather than a caption.

Palette: navy (#12355B) is the house colour and marks increases; a muted orange
(#B8621B) marks decreases. Navy and orange form a colour-blind-safe diverging pair.
Bar charts use a navy ramp with black text only.

Run from the folder that holds the phase-5 outputs:
    python make_figures_joh.py
Inputs: station_trends.parquet, driver_importance_level.csv, scenario_results.json,
scenario_projection.parquet, na_boundaries.geojson.
Outputs (./figures_joh/): Figure_1..Figure_4 (.tif and .png) and a graphical abstract.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_joh"
OUT.mkdir(exist_ok=True)

NAVY = "#12355B"
ORANGE = "#B8621B"
BLACK = "#111111"
GREY = "#9AA6B2"
PAPER = "#F7F7F5"
LAND = "#EDEDE7"
LINE = "#BEBEB6"

# navy = increase, orange = decrease, near-white at no change
DIVERGE = LinearSegmentedColormap.from_list("NavyOrange", [ORANGE, "#F3EFE7", NAVY])
# navy monochrome ramp for the four exceedance series
RAMP = [GREY, "#3E6DA3", NAVY, "#0A1E38"]
PROJ = "ESRI:102008"          # North America Albers Equal Area
BBOX = dict(lon=(-141, -52), lat=(24, 62))   # continental clip, removes stray points

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

PRETTY = {
    "ion": "Ion identity", "lc_tree_5km": "Forest cover (5 km)",
    "precip_mm": "Precipitation", "longitude": "Longitude",
    "lc_bare_5km": "Bare ground (5 km)", "pet_mm": "Potential evapotranspiration",
    "night_lights_5km": "Night-time lights (5 km)", "elevation": "Elevation",
    "temp_trend_c_yr": "Warming rate", "soil_ph": "Soil pH", "latitude": "Latitude",
    "lc_crop_5km": "Cropland (5 km)", "lc_water_500m": "Open water (500 m)",
    "pr_tc_mm": "Precipitation (TerraClimate)", "soil_clay": "Soil clay",
    "pop_density_5km": "Population density (5 km)", "lc_grass_5km": "Grassland (5 km)",
    "temp_c": "Air temperature", "soil_soc": "Soil organic carbon", "aridity": "Aridity index",
}


def save(fig, stem, png_dpi=600, tif_dpi=500):
    """Write a figure as TIFF and PNG, no title, at print resolution."""
    png = OUT / f"{stem}.png"
    tif = OUT / f"{stem}.tif"
    fig.savefig(png, dpi=png_dpi, facecolor="white")
    fig.savefig(tif, dpi=tif_dpi, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  wrote {png.name} and {tif.name}")


def load_boundaries():
    import geopandas as gpd
    g = gpd.read_file(ROOT / "na_boundaries.geojson")
    g = g[g["admin"].isin(["United States of America", "Canada"])].copy()
    g = g[~g["name"].isin(["Hawaii", "Alaska"])]        # keep the studied continent
    return g.to_crs(PROJ)


def to_points(df):
    import geopandas as gpd
    d = df[(df.longitude.between(*BBOX["lon"])) & (df.latitude.between(*BBOX["lat"]))].copy()
    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude), crs="EPSG:4326")
    return gdf.to_crs(PROJ)


def map_axes(ax, bounds, extent_pts=None, pad=0.06):
    bounds.plot(ax=ax, facecolor=LAND, edgecolor=LINE, linewidth=0.35, zorder=1)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if extent_pts is not None:
        minx, miny, maxx, maxy = extent_pts.total_bounds
        dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)


def figure_1(bounds):
    """Observed chloride trend, per cent per year."""
    tr = pd.read_parquet(ROOT / "station_trends.parquet")
    cl = tr[tr.ion == "cl"].copy()
    cl["pct_yr"] = (np.exp(cl["slope_log_yr"]) - 1) * 100
    pts = to_points(cl).sort_values("pct_yr", key=lambda s: s.abs())
    vlim = float(np.nanpercentile(np.abs(pts["pct_yr"]), 97))
    vlim = max(1.0, round(vlim, 1))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    map_axes(ax, bounds, extent_pts=pts)
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts["pct_yr"], cmap=DIVERGE, norm=norm,
                    s=7, linewidths=0, alpha=0.85, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, extend="both")
    cb.set_label("Chloride trend (% yr$^{-1}$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.text(0.01, 0.02, f"n = {len(pts):,} stations", transform=ax.transAxes,
            fontsize=8, color=BLACK)
    save(fig, "Figure_1", png_dpi=800, tif_dpi=650)


def figure_2():
    """Drivers of salinity level, mean absolute SHAP value."""
    imp = pd.read_csv(ROOT / "driver_importance_level.csv").head(15).iloc[::-1]
    labels = [PRETTY.get(f, f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(3.54, 4.6))
    ax.barh(range(len(imp)), imp["mean_abs_shap"], color=NAVY, height=0.72)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP| (log mg L$^{-1}$)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(y=0.01)
    save(fig, "Figure_2")


def figure_3():
    """Chloride threshold exceedance, recent versus 2050 under trend and climate scenarios."""
    res = json.loads((ROOT / "scenario_results.json").read_text())["chloride_exceedance"]
    thr = ["120", "230", "860"]
    labels = ["Aquatic-life\nchronic (120)", "EPA chronic\n(230)", "EPA acute\n(860)"]
    series = [("Recent", "recent"), ("2050 trend", "2050_trend"),
              ("2050 SSP2-4.5", "2050_ssp245"), ("2050 SSP5-8.5", "2050_ssp585")]
    x = np.arange(len(thr)); w = 0.2
    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    for j, (lab, key) in enumerate(series):
        vals = [res[key][t] for t in thr]
        ax.bar(x + (j - 1.5) * w, vals, w, color=RAMP[j], label=lab, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Chloride stations exceeding (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "Figure_5")


def figure_4(bounds):
    """Projected chloride change by 2050, per cent."""
    sp = pd.read_parquet(ROOT / "scenario_projection.parquet")
    cl = sp[sp.ion == "cl"].copy()
    cl["pct"] = (cl["value_2050_trend"] / cl["base_level"] - 1) * 100
    pts = to_points(cl).sort_values("pct", key=lambda s: s.abs())
    vlim = float(np.nanpercentile(np.abs(pts["pct"]), 95))
    vlim = max(10.0, round(vlim / 5) * 5)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    map_axes(ax, bounds, extent_pts=pts)
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts["pct"], cmap=DIVERGE, norm=norm,
                    s=7, linewidths=0, alpha=0.85, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, extend="both")
    cb.set_label("Projected chloride change by 2050 (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.text(0.01, 0.02, f"n = {len(pts):,} stations", transform=ax.transAxes,
            fontsize=8, color=BLACK)
    save(fig, "Figure_7", png_dpi=800, tif_dpi=650)


def graphical_abstract(bounds):
    """Optional graphical abstract, landscape, no title on the artwork."""
    tr = pd.read_parquet(ROOT / "station_trends.parquet")
    cl = tr[tr.ion == "cl"].copy()
    cl["pct_yr"] = (np.exp(cl["slope_log_yr"]) - 1) * 100
    pts = to_points(cl).sort_values("pct_yr", key=lambda s: s.abs())
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    res = json.loads((ROOT / "scenario_results.json").read_text())["chloride_exceedance"]

    fig = plt.figure(figsize=(5.9, 2.36))        # 2.5:1, matches the 531 x 1328 px ratio
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.05)
    axm = fig.add_subplot(gs[0]); map_axes(axm, bounds, extent_pts=pts)
    axm.scatter(pts.geometry.x, pts.geometry.y, c=pts["pct_yr"], cmap=DIVERGE, norm=norm,
                s=2.2, linewidths=0, alpha=0.85, zorder=2)
    axm.text(0.5, -0.04, "Chloride trend, 1990-2026", transform=axm.transAxes,
             ha="center", va="top", fontsize=7, color=BLACK)

    axb = fig.add_subplot(gs[1])
    thr = ["120", "230", "860"]; x = np.arange(len(thr)); w = 0.38
    axb.bar(x - w / 2, [res["recent"][t] for t in thr], w, color=GREY, label="Recent")
    axb.bar(x + w / 2, [res["2050_trend"][t] for t in thr], w, color=NAVY, label="2050")
    axb.set_xticks(x); axb.set_xticklabels(["120", "230", "860"], fontsize=6.5)
    axb.set_xlabel("Chloride threshold (mg L$^{-1}$)", fontsize=6.5)
    axb.set_ylabel("Stations\nexceeding (%)", fontsize=6.5)
    axb.tick_params(labelsize=6)
    axb.legend(frameon=False, fontsize=6, loc="upper right")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    fig.savefig(OUT / "GraphicalAbstract.png", dpi=340, facecolor="white")
    fig.savefig(OUT / "GraphicalAbstract.tif", dpi=340, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("  wrote GraphicalAbstract.png and .tif")


def main():
    print("Rendering Journal of Hydrology figures (no titles on artwork):")
    bounds = load_boundaries()
    figure_1(bounds)
    figure_2()
    figure_3()
    figure_4(bounds)
    graphical_abstract(bounds)
    # report pixel widths so the Elsevier minimums can be checked
    from PIL import Image
    print("\nWidth check (Elsevier: single col >=1063 px, full page >=2244 px):")
    for p in sorted(OUT.glob("*.png")):
        with Image.open(p) as im:
            print(f"  {p.name:26s} {im.size[0]} x {im.size[1]} px")
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
