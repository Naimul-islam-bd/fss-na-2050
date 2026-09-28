"""End to end verification slice on real data.

Run this from the project root:

    python run_verify_slice.py

It pulls small live slices from each verified source, runs them through the
source mappers and the harmonizer, and writes a merged sample plus a summary to
06_outputs/verify_slice/. The point is to confirm the ingest and harmonize path
works on real data before the full continental pull, not to gather data.

It needs internet and dataretrieval>=1.3. Each source is wrapped so one failure
does not stop the others.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import traceback
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "02_code"
OUT = ROOT / "06_outputs" / "verify_slice"
OUT.mkdir(parents=True, exist_ok=True)
PKG = "fsscode"


# The code folder is named 02_code, which is not a valid module name, so build
# an import package tree by hand and load the modules into it. This lets their
# relative imports (from ..utils ...) resolve without renaming the folder.
def _pkg(name: str, path: Path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    m.__package__ = name
    sys.modules[name] = m
    return m


def _mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = name.rpartition(".")[0]
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_pkg(PKG, CODE)
_pkg(f"{PKG}.utils", CODE / "utils")
_pkg(f"{PKG}.ingestion", CODE / "ingestion")
_pkg(f"{PKG}.preprocessing", CODE / "preprocessing")
_mod(f"{PKG}.utils.logging_utils", CODE / "utils" / "logging_utils.py")
# config_loader is not loaded here on purpose: it needs pyyaml, and the slice
# does not read the YAML config. The ingestion modules import it lazily.
wqp = _mod(f"{PKG}.ingestion.wqp_ingest", CODE / "ingestion" / "wqp_ingest.py")
eccc = _mod(f"{PKG}.ingestion.eccc_ingest", CODE / "ingestion" / "eccc_ingest.py")
H = _mod(f"{PKG}.preprocessing.harmonize", CODE / "preprocessing" / "harmonize.py")

# Small slice settings. New York, one quarter, two ions.
US_STATE = "US:36"
CHARS = ["Chloride", "Sodium"]
LO, HI = "2020-01-01", "2020-03-31"
ON_BASE = "https://files.ontario.ca/moe_mapping/downloads/2Water/PWQMN/"
BC_URL = ("https://pub.data.gov.bc.ca/datasets/949f2233-9612-4b06-92a9-903e817da659/"
          "ems_sample_results_current_expanded.csv")


def slice_wqp():
    frames = []
    for char in CHARS:
        df = wqp.inspect_wqp_header(US_STATE, char, LO, HI)
        if len(df):
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return H.standardize_wqx3(raw, "EPA_WQP") if len(raw) else raw


def slice_usgs():
    frames = []
    for char in CHARS:
        df = wqp.inspect_usgs_header(US_STATE, char, LO, HI)
        if len(df):
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return H.standardize_wqx3(raw, "USGS_SAMPLES") if len(raw) else raw


def slice_eccc():
    sites_raw = eccc.http_get(eccc.api_file_url(eccc.ROOT_PATH + "Water-Qual-Eau-Sites-National.csv"))
    sites = pd.read_csv(io.BytesIO(sites_raw), encoding="utf-8", dtype={"SITE_NO": "string"})
    region = "saint-lawrence-river-basin-long-term-water-quality-monitoring-data"
    res = eccc.inspect_eccc_header(region, nbytes=500_000)
    return H.standardize_eccc(res, sites)


def slice_ontario():
    res = pd.read_csv(ON_BASE + "PWQMN_2025.csv", encoding="cp1252", low_memory=False)
    st = pd.read_csv(ON_BASE + "PWQMN-Stations.csv", encoding="cp1252")
    return H.standardize_ontario(res, st)


def slice_bc():
    # Bounded byte fetch so a large remote CSV cannot hang the slice; the last
    # partial line is dropped by on_bad_lines.
    import urllib.request
    req = urllib.request.Request(BC_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read(6_000_000)
    df = pd.read_csv(io.BytesIO(data), on_bad_lines="skip", low_memory=False)
    return H.standardize_bc_ems(df)


def main() -> int:
    sources = [("EPA_WQP", slice_wqp), ("USGS_SAMPLES", slice_usgs),
               ("ECCC", slice_eccc), ("ON_PWQMN", slice_ontario), ("BC_EMS", slice_bc)]
    harmonized = []
    summary = []
    for name, fn in sources:
        print("=" * 70)
        print("slice:", name)
        try:
            std = fn()
        except Exception:
            print(traceback.format_exc())
            summary.append({"source": name, "status": "FAILED", "rows": 0})
            continue
        if std is None or len(std) == 0:
            print("  no rows in this slice")
            summary.append({"source": name, "status": "empty", "rows": 0})
            continue
        # Convert units per ion, then keep the standard long rows.
        parts = []
        for ion in H.ION_KEYS:
            sub = std[std["ion"] == ion]
            if len(sub):
                parts.append(H.harmonize_units(sub, ion, "value_source", "unit_source"))
        if not parts:
            summary.append({"source": name, "status": "no_target_ions", "rows": 0})
            continue
        got = pd.concat(parts, ignore_index=True)
        got = H.handle_below_detection(got, "value_mg_L", "detection_limit",
                                       "below_flag", strategy="half_dl")
        harmonized.append(got)
        by_ion = got["ion"].value_counts().to_dict()
        print(f"  rows: {len(got)}  ions: {by_ion}")
        summary.append({"source": name, "status": "ok", "rows": int(len(got)),
                        "by_ion": by_ion})

    if harmonized:
        keep = ["source", "station_id", "latitude", "longitude", "state_prov",
                "site_type", "datetime", "ion", "value_mg_L", "below_detection",
                "fraction", "unit_source", "characteristic_source"]
        merged = pd.concat(harmonized, ignore_index=True)
        merged = merged[[c for c in keep if c in merged.columns]]
        merged.to_csv(OUT / "harmonized_sample.csv", index=False)
        print("\n" + "=" * 70)
        print("MERGED HARMONIZED SAMPLE")
        print("=" * 70)
        print("total rows:", len(merged))
        print("by source:", merged["source"].value_counts().to_dict())
        print("by ion:", merged["ion"].value_counts().to_dict())
        print("columns:", list(merged.columns))
        print("\nfirst rows:")
        with pd.option_context("display.max_columns", 20, "display.width", 200):
            print(merged.head(8).to_string())
        print(f"\nsaved: {OUT / 'harmonized_sample.csv'}")

    pd.DataFrame(summary).to_json(OUT / "slice_summary.json", orient="records", indent=2)
    print(f"summary: {OUT / 'slice_summary.json'}")
    ok = [s for s in summary if s["status"] == "ok"]
    print(f"\n{len(ok)} of {len(sources)} sources produced harmonized rows.")
    print("Send me slice_summary.json and the first rows above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
