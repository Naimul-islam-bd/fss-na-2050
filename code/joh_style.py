"""Shared plotting style and helpers for the Journal of Hydrology figures.

Import this from the analysis scripts so every figure shares one look: navy as the
house colour, a colour-blind-safe navy-orange diverging scale for maps, no title
drawn on the artwork, and Elsevier resolution. Figures are written as TIFF and PNG.

Data files are found whether the script runs from the project root (06_outputs/phase5)
or from a flat working folder, so the same script runs locally and in a sandbox.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#12355B"
ORANGE = "#B8621B"
BLACK = "#111111"
GREY = "#9AA6B2"
LAND = "#EDEDE7"
LINE = "#BEBEB6"
DIVERGE = LinearSegmentedColormap.from_list("NavyOrange", [ORANGE, "#F3EFE7", NAVY])
RAMP = [GREY, "#3E6DA3", NAVY, "#0A1E38"]
# a categorical palette in the navy family plus orange, for regime typologies
CATEGORICAL = ["#12355B", "#B8621B", "#3E6DA3", "#7A8699", "#C79A3A", "#0A1E38"]
PROJ = "ESRI:102008"                       # North America Albers Equal Area
BBOX = dict(lon=(-141, -52), lat=(24, 62))  # continental clip

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

import os
_HERE = Path(__file__).resolve().parent
_CWD = Path(os.getcwd())
_SEARCH = [_HERE, _HERE / "06_outputs" / "phase5", _HERE / "data",
           _HERE.parent, _HERE.parent / "06_outputs" / "phase5", _HERE.parent / "data",
           _CWD, _CWD / "06_outputs" / "phase5", _CWD / "data"]
_MASTER = [
    _HERE / "master_harmonized.parquet",
    _HERE / "01_data" / "processed" / "master_harmonized.parquet",
    _HERE.parent / "01_data" / "processed" / "master_harmonized.parquet",
    _CWD / "01_data" / "processed" / "master_harmonized.parquet",
    _CWD / "master_harmonized.parquet",
    Path("/mnt/user-data/uploads/Downloads--fss_northamerica_2050/fss_northamerica_2050/"
         "01_data/processed/master_harmonized.parquet"),
]


def find(name: str) -> Path:
    """Locate a data file whether we run flat or under the project tree."""
    for d in _SEARCH:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"could not find {name} in {[str(d) for d in _SEARCH]}")


def find_master() -> Path:
    for p in _MASTER:
        if p.exists():
            return p
    raise FileNotFoundError("master_harmonized.parquet not found")


def outdir() -> Path:
    d = _HERE / "figures_joh"
    d.mkdir(exist_ok=True)
    return d


def save(fig, stem, png_dpi=600, tif_dpi=500):
    d = outdir()
    fig.savefig(d / f"{stem}.png", dpi=png_dpi, facecolor="white")
    fig.savefig(d / f"{stem}.tif", dpi=tif_dpi, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  wrote {stem}.png and {stem}.tif")


def load_boundaries():
    import geopandas as gpd
    g = gpd.read_file(find("na_boundaries.geojson"))
    g = g[g["admin"].isin(["United States of America", "Canada"])].copy()
    g = g[~g["name"].isin(["Hawaii", "Alaska"])]
    return g.to_crs(PROJ)


def to_points(df):
    import geopandas as gpd
    d = df[(df.longitude.between(*BBOX["lon"])) & (df.latitude.between(*BBOX["lat"]))].copy()
    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude),
                           crs="EPSG:4326")
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
