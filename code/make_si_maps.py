"""Supplementary per-ion maps: observed trend and projected 2050 change for all six ions.

Two six-panel figures, one for the observed trend (per cent per year) and one for the
projected 2050 change (per cent). Panels are meant to be compared, so they are combined
in one file, which Elsevier allows for comparison figures. No figure title is drawn; a
short panel identifier sits in each panel corner.

Run from the project root (or a folder with the phase-5 outputs):
    python make_si_maps.py
Inputs: station_trends.parquet, projection_2050_allion.parquet, na_boundaries.geojson.
Outputs (figures_joh/): Figure_S1 (trend, six ions) and Figure_S2 (2050 change, six ions).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import joh_style as S

warnings.filterwarnings("ignore")

IONS = [("cl", "Chloride"), ("na", "Sodium"), ("ca", "Calcium"),
        ("mg", "Magnesium"), ("so4", "Sulfate"), ("alk", "Alkalinity")]
ABC = "abcdef"


def panel(ax, bounds, pts, values, vlim, label, cblabel):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    S.map_axes(ax, bounds, extent_pts=pts)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    order = np.argsort(np.abs(values))
    sc = ax.scatter(pts.geometry.x.values[order], pts.geometry.y.values[order],
                    c=values[order], cmap=S.DIVERGE, norm=norm, s=4, linewidths=0,
                    alpha=0.85, zorder=2)
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.62, pad=0.01, extend="both")
    cb.ax.tick_params(labelsize=6)
    cb.set_label(cblabel, fontsize=6.5)
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=8.5, va="top",
            ha="left", color=S.BLACK, fontweight="bold")


def make(kind):
    import matplotlib.pyplot as plt
    bounds = S.load_boundaries()
    tr = pd.read_parquet(S.find("station_trends.parquet"))
    pj = pd.read_parquet(S.find("projection_2050_allion.parquet"))
    tr["station_id"] = tr["station_id"].astype(str); pj["station_id"] = pj["station_id"].astype(str)

    fig, axes = plt.subplots(2, 3, figsize=(7.48, 4.7))
    for ax, (ion, name) in zip(axes.ravel(), IONS):
        if kind == "trend":
            d = tr[tr.ion == ion].copy()
            d["val"] = (np.exp(d["slope_log_yr"]) - 1) * 100
            cblabel = "% yr$^{-1}$"
        else:
            d = pj[pj.ion == ion].copy()
            d["val"] = (d["value_2050"] / d["base_level"] - 1) * 100
            cblabel = "% by 2050"
        pts = S.to_points(d)
        v = pts["val"].to_numpy()
        vlim = max(1.0, round(float(np.nanpercentile(np.abs(v), 95)), 1))
        lab = f"({ABC[IONS.index((ion, name))]}) {name}"
        panel(ax, bounds, pts, v, vlim, lab, cblabel)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    S.save(fig, "Figure_S1" if kind == "trend" else "Figure_S2",
           png_dpi=500, tif_dpi=400)


if __name__ == "__main__":
    make("trend")
    make("change")
    print("done SI maps")
