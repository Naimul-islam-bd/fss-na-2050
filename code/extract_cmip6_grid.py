"""Phase 5 climate-change layer, fast version: CMIP6 2050 deltas on a coarse grid.

The mid-century climate change signal is spatially smooth, so sampling daily
GDDP-CMIP6 at all 3,944 stations is unnecessary and slow. This samples the four
delta bands (temperature and precipitation change under SSP2-4.5 and SSP5-8.5,
five-model mean, 2046 to 2055 minus 2005 to 2014) on a coarse grid over the study
area, then interpolates to every station on the client. Same deltas, a fraction
of the time.

Run from the project root:
    python extract_cmip6_grid.py --test           # small grid, to verify
    python extract_cmip6_grid.py                   # full grid, then interpolate
Grid sampling is resume-able (06_outputs/phase5/cmip6_grid.parquet). When the
grid is complete it interpolates to 06_outputs/phase5/cmip6_deltas.parquet for all
stations. If a chunk times out, lower --chunk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATIONS = ROOT / "06_outputs" / "phase5" / "stations_unique.csv"
GRID = ROOT / "06_outputs" / "phase5" / "cmip6_grid.parquet"
OUT = ROOT / "06_outputs" / "phase5" / "cmip6_deltas.parquet"
CKPT = ROOT / "06_outputs" / "phase5" / "cmip6_progress.json"

BASE = ("2005-01-01", "2015-01-01")
FUT = ("2046-01-01", "2056-01-01")
SCENARIOS = ("ssp245", "ssp585")
MODELS = ["ACCESS-CM2", "EC-Earth3", "GFDL-ESM4", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
DELTA_COLS = ["dtemp_ssp245", "dprecip_ssp245", "dtemp_ssp585", "dprecip_ssp585"]
STEP = 3.0          # grid spacing in degrees; the change field is smooth at this scale
NEAR = 2.5          # keep grid points within this many degrees of a station


def build_grid(stations: pd.DataFrame) -> pd.DataFrame:
    lat0, lat1 = stations["latitude"].min() - 1, stations["latitude"].max() + 1
    lon0, lon1 = stations["longitude"].min() - 1, stations["longitude"].max() + 1
    glat = np.arange(np.floor(lat0), np.ceil(lat1) + STEP, STEP)
    glon = np.arange(np.floor(lon0), np.ceil(lon1) + STEP, STEP)
    gg = pd.DataFrame([(la, lo) for la in glat for lo in glon], columns=["latitude", "longitude"])
    # keep only grid points near a station, so we do not sample empty ocean
    slat = stations["latitude"].to_numpy(); slon = stations["longitude"].to_numpy()
    keep = []
    for la, lo in gg.itertuples(index=False):
        d = np.abs(slat - la) + np.abs(slon - lo)
        keep.append(d.min() <= NEAR * 2)
    gg = gg[pd.Series(keep, index=gg.index)].reset_index(drop=True)
    gg["grid_id"] = ["g%04d" % i for i in range(len(gg))]
    return gg


def per_model_delta(ee, model):
    c = ee.ImageCollection("NASA/GDDP-CMIP6").filter(ee.Filter.eq("model", model))

    def mb(scenario, band, window):
        return (c.filter(ee.Filter.eq("scenario", scenario))
                .filterDate(window[0], window[1]).select(band).mean())

    bt, bp = mb("historical", "tas", BASE), mb("historical", "pr", BASE)
    bands = []
    for ssp in SCENARIOS:
        ft, fp = mb(ssp, "tas", FUT), mb(ssp, "pr", FUT)
        bands.append(ft.subtract(bt).rename(f"dtemp_{ssp}"))
        bands.append(fp.subtract(bp).multiply(86400 * 365).rename(f"dprecip_{ssp}"))
    return ee.Image.cat(bands)


def _props(info):
    return pd.DataFrame([f["properties"] for f in info["features"]])


def sample_grid(ee, grid, chunk):
    done, prev = set(), []
    if GRID.exists():
        g = pd.read_parquet(GRID); done = set(g["grid_id"]); prev = [g]
        print(f"resuming grid; {len(done)} points done")
    todo = grid[~grid["grid_id"].isin(done)].reset_index(drop=True)
    print(f"grid points to sample: {len(todo)} ({len(MODELS)} models, chunk {chunk})")
    imgs = {m: per_model_delta(ee, m) for m in MODELS}
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        pts = [ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                          {"grid_id": r.grid_id}) for r in part.itertuples()]
        fc = ee.FeatureCollection(pts)
        frames = []
        for m in MODELS:
            for attempt in range(4):
                try:
                    info = imgs[m].reduceRegions(fc, ee.Reducer.mean(), 27830).getInfo()
                    dfm = _props(info)
                    frames.append(dfm[["grid_id"] + [c for c in DELTA_COLS if c in dfm.columns]].set_index("grid_id"))
                    break
                except Exception as exc:
                    print(f"  {m} failed ({exc!r}); retry {attempt+1}/4"); time.sleep(5 * (attempt + 1))
        if not frames:
            print("no model returned; re-run to retry"); continue
        avg = pd.concat(frames).groupby(level=0).mean()
        avg = part.set_index("grid_id").join(avg).reset_index()
        prev.append(avg)
        pd.concat(prev, ignore_index=True).to_parquet(GRID, index=False)
        n = sum(len(f) for f in prev); el = (time.time() - t0) / 60
        print(f"[{100*min(i+chunk,len(todo))/max(len(todo),1):5.1f}%] {n} grid pts | elapsed {el:.1f} min")
        try:
            CKPT.write_text(json.dumps({"grid_done": int(n), "elapsed_min": round(el, 1)}, indent=2))
        except Exception:
            pass
    return pd.read_parquet(GRID)


def interpolate(grid, stations):
    from scipy.interpolate import griddata
    gpts = grid[["longitude", "latitude"]].to_numpy()
    spts = stations[["longitude", "latitude"]].to_numpy()
    out = stations[["station_id"]].copy()
    for col in DELTA_COLS:
        v = grid[col].to_numpy()
        lin = griddata(gpts, v, spts, method="linear")
        nn = griddata(gpts, v, spts, method="nearest")
        out[col] = np.where(np.isnan(lin), nn, lin)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--test", action="store_true", help="tiny grid subset to verify")
    args = ap.parse_args()
    if not STATIONS.exists():
        print(f"station export not found at {STATIONS}"); return 1
    stations = pd.read_csv(STATIONS); stations["station_id"] = stations["station_id"].astype(str)

    import ee
    try:
        ee.Initialize(project=args.project) if args.project else ee.Initialize()
    except Exception:
        print("Earth Engine not initialized; run earthengine authenticate and set_project first"); raise

    grid = build_grid(stations)
    if args.test:
        grid = grid.iloc[::max(1, len(grid) // 15)].reset_index(drop=True)
        print(f"TEST grid: {len(grid)} points")
    grid = sample_grid(ee, grid, args.chunk)

    full = build_grid(stations)
    if not args.test and len(grid) < len(full):
        print(f"grid incomplete ({len(grid)}/{len(full)}); re-run to finish before interpolation")
        return 0
    print("interpolating grid deltas to all stations")
    out = interpolate(grid, stations)
    out.to_parquet(OUT, index=False)
    print("wrote", OUT, "rows:", len(out))
    print(out[DELTA_COLS].describe().T[["mean", "min", "max"]].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
