"""Phase 5 climate-change layer: CMIP6 2050 deltas per station, through Earth Engine.

For each monitoring station this computes the change in mean annual temperature
and precipitation between a historical baseline (2005 to 2014) and a mid-century
window (2046 to 2055), under SSP2-4.5 and SSP5-8.5, as the mean of a five-model
subset of the NASA NEX-GDDP-CMIP6 downscaled ensemble. These deltas drive the
scenario projection: the climate covariates are shifted to their 2050 values and
the level model is re-evaluated, so the forecast carries a climate range.

The daily GDDP archive is large, so each model is sampled in its own request and
the models are averaged on the client. Keep the chunk small.

Run from the project root, after station_covariates.parquet exists:
    python extract_cmip6_deltas.py --test 50     # verify on 50 stations
    python extract_cmip6_deltas.py               # full 3,944 stations
If a chunk still times out, lower it: --chunk 5

Resume-able. Output: 06_outputs/phase5/cmip6_deltas.parquet with
dtemp_ssp245, dprecip_ssp245, dtemp_ssp585, dprecip_ssp585 per station
(temperature in degrees C, precipitation in mm per year).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STATIONS = ROOT / "06_outputs" / "phase5" / "stations_unique.csv"
OUT = ROOT / "06_outputs" / "phase5" / "cmip6_deltas.parquet"
CKPT = ROOT / "06_outputs" / "phase5" / "cmip6_progress.json"

BASE = ("2005-01-01", "2015-01-01")
FUT = ("2046-01-01", "2056-01-01")
SCENARIOS = ("ssp245", "ssp585")
MODELS = ["ACCESS-CM2", "EC-Earth3", "GFDL-ESM4", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
DELTA_COLS = ["dtemp_ssp245", "dprecip_ssp245", "dtemp_ssp585", "dprecip_ssp585"]


def per_model_delta(ee, model):
    """Delta bands for one model: future minus baseline, per scenario."""
    c = ee.ImageCollection("NASA/GDDP-CMIP6").filter(ee.Filter.eq("model", model))

    def mean_band(scenario, band, window):
        return (c.filter(ee.Filter.eq("scenario", scenario))
                .filterDate(window[0], window[1]).select(band).mean())

    base_t = mean_band("historical", "tas", BASE)
    base_p = mean_band("historical", "pr", BASE)
    bands = []
    for ssp in SCENARIOS:
        ft = mean_band(ssp, "tas", FUT)
        fp = mean_band(ssp, "pr", FUT)
        bands.append(ft.subtract(base_t).rename(f"dtemp_{ssp}"))          # K delta = C delta
        bands.append(fp.subtract(base_p).multiply(86400 * 365).rename(f"dprecip_{ssp}"))
    return ee.Image.cat(bands)


def _props(info):
    return pd.DataFrame([f["properties"] for f in info["features"]])


def extract(project, chunk, test):
    import ee
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        print("Earth Engine did not initialize. Run 'earthengine authenticate' and "
              "'earthengine set_project YOUR_EE_PROJECT' first, then re-run.")
        raise

    stations = pd.read_csv(STATIONS)
    if test:
        stations = stations.head(test)
    done, prev = set(), []
    if OUT.exists():
        p = pd.read_parquet(OUT)
        done = set(p["station_id"].astype(str))
        prev = [p]
        print(f"resuming; {len(done)} stations already done")
    todo = stations[~stations["station_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"stations to extract: {len(todo)} ({len(MODELS)} models, chunk {chunk})")

    imgs = {m: per_model_delta(ee, m) for m in MODELS}
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        pts = [ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                          {"station_id": str(r.station_id)}) for r in part.itertuples()]
        fc = ee.FeatureCollection(pts)

        model_frames = []
        for m in MODELS:
            for attempt in range(4):
                try:
                    info = imgs[m].reduceRegions(fc, ee.Reducer.mean(), 27830).getInfo()
                    dfm = _props(info)
                    dfm = dfm[["station_id"] + [c for c in DELTA_COLS if c in dfm.columns]]
                    model_frames.append(dfm.set_index("station_id"))
                    break
                except Exception as exc:
                    wait = 5 * (attempt + 1)
                    print(f"  model {m} chunk failed ({exc!r}); retry {attempt + 1}/4 in {wait}s")
                    time.sleep(wait)
            else:
                print(f"  model {m} skipped this chunk")

        if not model_frames:
            print("no model returned for this chunk; re-run to retry")
            continue
        avg = pd.concat(model_frames).groupby(level=0).mean().reset_index()
        prev.append(avg)
        pd.concat(prev, ignore_index=True).to_parquet(OUT, index=False)
        ndone = sum(len(f) for f in prev)
        elapsed = (time.time() - t0) / 60
        pct = 100.0 * min(i + chunk, len(todo)) / max(len(todo), 1)
        print(f"[{pct:5.1f}%] {ndone} stations | elapsed {elapsed:.1f} min")
        try:
            CKPT.write_text(json.dumps({"done": int(ndone), "elapsed_min": round(elapsed, 1)}, indent=2))
        except Exception:
            pass

    print("CMIP6 deltas written to", OUT)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--chunk", type=int, default=10, help="stations per request; lower to 5 if timeouts persist")
    ap.add_argument("--test", type=int, default=None)
    args = ap.parse_args()
    if not STATIONS.exists():
        print(f"station export not found at {STATIONS}")
        return 1
    return extract(args.project, args.chunk, args.test)


if __name__ == "__main__":
    sys.exit(main())
