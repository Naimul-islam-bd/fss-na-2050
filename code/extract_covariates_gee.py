"""Phase 5 covariate extraction through Google Earth Engine.

Samples the catchment and climate drivers of salinization at every monitoring
station, server side, so nothing large is downloaded. It reads the station
export written in Phase 5 and writes one covariate row per station.

Setup, once:
    pip install earthengine-api
    earthengine authenticate
    earthengine set_project YOUR_EE_PROJECT
Then run from the project root:
    python extract_covariates_gee.py --test 50      # verify on 50 stations
    python extract_covariates_gee.py                # full 3,944 stations

It is resume-able: stations already written to the output are skipped, so a
dropped connection loses nothing. Re-run the same command to continue.

Sampling design:
- Human-pressure and land-cover covariates are sampled at two scales, a 500 m
  near-stream buffer and a 5 km landscape buffer, because land use influences
  stream chemistry differently at the riparian and the wider catchment scale.
  These columns carry a _500m or _5km suffix.
- Smoothly varying covariates (elevation, slope, climate normals, aridity,
  climate trend, soil properties) are sampled in a 1 km buffer, where scale has
  little effect.

Covariates:
- elevation and slope (MERIT DEM)
- land cover fractions: tree, shrub, grass, cropland, built-up, bare, water,
  wetland (ESA WorldCover 2021), at 500 m and 5 km
- population density (GPW v4.11, 2020) and night-time lights (VIIRS), at 500 m
  and 5 km, as proxies for urban intensity, wastewater, and road salt use
- climate normals: annual mean temperature and precipitation (WorldClim),
  potential evapotranspiration and an aridity index (TerraClimate 1991 to 2020)
- climate trend: temperature and precipitation slope 1990 to 2023 (TerraClimate)
- soil geochemistry: clay, sand, organic carbon, and pH at the surface
  (OpenLandMap), which carry the weathering signal behind calcium, magnesium,
  and alkalinity

Upstream basin attributes (full catchment lithology, soils, and land use) are
joined from HydroATLAS in the next step; those are the true catchment scale.
The soil group is isolated so, if that asset is unavailable, the rest still runs.
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
OUT = ROOT / "06_outputs" / "phase5" / "station_covariates.parquet"
CKPT = ROOT / "06_outputs" / "phase5" / "covariate_progress.json"

SMOOTH_BUFFER = 1000
LOCAL_BUFFERS = {"_500m": 500, "_5km": 5000}

WORLDCOVER_CLASSES = {
    "lc_tree": 10, "lc_shrub": 20, "lc_grass": 30, "lc_crop": 40,
    "lc_builtup": 50, "lc_bare": 60, "lc_snowice": 70, "lc_water": 80,
    "lc_wetland": 90,
}


def build_smooth_image(ee):
    dem = ee.Image("MERIT/DEM/v1_0_3").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    bio = ee.Image("WORLDCLIM/V1/BIO")
    temp = bio.select("bio01").multiply(0.1).rename("temp_c")
    precip = bio.select("bio12").rename("precip_mm")
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate("1991-01-01", "2021-01-01")
    pet = tc.select("pet").mean().multiply(12).multiply(0.1).rename("pet_mm")
    pr_tc = tc.select("pr").mean().multiply(12).rename("pr_tc_mm")
    aridity = pr_tc.divide(pet.max(1)).rename("aridity")
    img = ee.Image.cat([dem, slope, temp, precip, pet, pr_tc, aridity])
    # climate trend, 1990 to 2023
    tc2 = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate("1990-01-01", "2024-01-01")
    years = ee.List.sequence(1990, 2023)

    def annual(y):
        y = ee.Number(y)
        f = tc2.filter(ee.Filter.calendarRange(y, y, "year"))
        t = f.select(["tmmx", "tmmn"]).mean().reduce(ee.Reducer.mean()).multiply(0.1).rename("t")
        p = f.select("pr").sum().rename("p")
        yr = ee.Image.constant(y).toFloat().rename("yr")
        return yr.addBands(t).addBands(p)

    coll = ee.ImageCollection(years.map(annual))
    t_slope = coll.select(["yr", "t"]).reduce(ee.Reducer.linearFit()).select("scale").rename("temp_trend_c_yr")
    p_slope = coll.select(["yr", "p"]).reduce(ee.Reducer.linearFit()).select("scale").rename("precip_trend_mm_yr")
    return img.addBands(t_slope).addBands(p_slope)


def build_soil_image(ee):
    """Surface soil geochemistry from OpenLandMap (band b0 is 0 cm)."""
    def sol(asset, name):
        return ee.Image(asset).select("b0").rename(name)
    clay = sol("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02", "soil_clay")
    sand = sol("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02", "soil_sand")
    soc = sol("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", "soil_soc")
    ph = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select("b0").multiply(0.1).rename("soil_ph")
    return ee.Image.cat([clay, sand, soc, ph])


def build_local_image(ee):
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    frac = [wc.eq(v).rename(k).multiply(100) for k, v in WORLDCOVER_CLASSES.items()]
    pop = (ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
           .filterDate("2020-01-01", "2020-12-31").first()
           .select("population_density").rename("pop_density"))
    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate("2022-01-01", "2023-01-01").select("avg_rad").mean()
             .rename("night_lights"))
    return ee.Image.cat(frac + [pop, viirs])


def _props(info, suffix=""):
    df = pd.DataFrame([f["properties"] for f in info["features"]])
    if suffix and len(df):
        df = df.rename(columns={c: (c + suffix) for c in df.columns if c != "station_id"})
    return df


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
    done, prev_frames = set(), []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(prev["station_id"].astype(str))
        prev_frames = [prev]
        print(f"resuming; {len(done)} stations already done")
    todo = stations[~stations["station_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"stations to extract: {len(todo)} (buffers 500 m and 5 km, chunk {chunk})")

    smooth = build_smooth_image(ee)
    local = build_local_image(ee)
    try:
        soil = build_soil_image(ee)
        use_soil = True
    except Exception as exc:
        print(f"soil layer unavailable, continuing without it: {exc!r}")
        soil, use_soil = None, False

    t0 = time.time()
    soil_warned = False
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        pts = [ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                          {"station_id": str(r.station_id)}) for r in part.itertuples()]
        fc = ee.FeatureCollection(pts)

        def buffered(radius):
            return fc.map(lambda f: f.setGeometry(f.geometry().buffer(radius)))

        for attempt in range(4):
            try:
                sm = _props(smooth.reduceRegions(buffered(SMOOTH_BUFFER), ee.Reducer.mean(), 500).getInfo())
                merged = sm
                for suffix, radius in LOCAL_BUFFERS.items():
                    scale = 30 if radius <= 1000 else 100
                    li = _props(local.reduceRegions(buffered(radius), ee.Reducer.mean(), scale).getInfo(), suffix)
                    merged = merged.merge(li, on="station_id", how="outer")
                if use_soil:
                    try:
                        so = _props(soil.reduceRegions(buffered(SMOOTH_BUFFER), ee.Reducer.mean(), 250).getInfo())
                        merged = merged.merge(so, on="station_id", how="outer")
                    except Exception as exc:
                        if not soil_warned:
                            print(f"soil sampling failed, leaving those columns out: {exc!r}")
                            soil_warned = True
                prev_frames.append(merged)
                pd.concat(prev_frames, ignore_index=True).to_parquet(OUT, index=False)
                ndone = sum(len(f) for f in prev_frames)
                elapsed = (time.time() - t0) / 60
                pct = 100.0 * min(i + chunk, len(todo)) / max(len(todo), 1)
                print(f"[{pct:5.1f}%] {ndone} stations | elapsed {elapsed:.1f} min")
                try:
                    CKPT.write_text(json.dumps({"done": int(ndone), "elapsed_min": round(elapsed, 1)}, indent=2))
                except Exception:
                    pass
                break
            except Exception as exc:
                wait = 5 * (attempt + 1)
                print(f"chunk failed ({exc!r}); retry {attempt + 1}/4 in {wait}s "
                      f"(lower --chunk if this repeats)")
                time.sleep(wait)
        else:
            print("giving up this chunk this run; re-run to retry the rest")

    print("covariates written to", OUT)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="Earth Engine cloud project id; omit if set with set_project")
    ap.add_argument("--chunk", type=int, default=150, help="stations per request; lower it if requests time out")
    ap.add_argument("--test", type=int, default=None, help="only the first N stations, to verify")
    args = ap.parse_args()
    if not STATIONS.exists():
        print(f"station export not found at {STATIONS}")
        return 1
    return extract(args.project, args.chunk, args.test)


if __name__ == "__main__":
    sys.exit(main())
