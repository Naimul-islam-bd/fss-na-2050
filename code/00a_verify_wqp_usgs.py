"""Phase 2 verification, step 1: EPA Water Quality Portal and USGS Samples.

Purpose
-------
Confirm the real schema of the two primary United States water quality
sources before any bulk download or feature code is written. Nothing here
is used downstream. Each source is pulled on a small time and space slice,
the true column names, dtypes, units, row count, and missingness are printed
to the screen, and a paste-ready summary is written for schema_verified.md.

Why this exists
---------------
The data_source_registry lists the columns we expect. Expected is not
verified. This script replaces every assumed column name with the header
that the service actually returns today.

Package note
------------
This uses dataretrieval 1.3.x. In that version the old NWIS water quality
functions (nwis.get_qwdata and the earlier nwis.get_samples) are removed.
Sample chemistry now comes from the USGS Samples service through
dataretrieval.waterdata.get_samples. EPA WQP still comes from
dataretrieval.wqp.get_results. If you are on an older dataretrieval the
USGS block will tell you to upgrade.

How to run
----------
From the project root, in the environment that has dataretrieval installed:

    python 03_notebooks/00a_verify_wqp_usgs.py

It needs internet. It writes to 01_data/raw/_verification/ and prints a
block you can paste into 01_data/schema_verified.md. It does not touch any
other file.

If dataretrieval is missing:

    pip install "dataretrieval>=1.3,<2"
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Keep the console readable when frames are wide.
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 180)

# Resolve project paths relative to this file so the script works from any cwd.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
OUT_DIR = PROJECT_ROOT / "01_data" / "raw" / "_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Small, salinization-relevant slice. Dutchess County, New York (FIPS 36:027)
# sits in the northeast road-salt belt and has long records, so a three month
# winter window returns real chloride rows without a large download.
STATE_FIPS = "36"        # New York
COUNTY_FIPS = "027"      # Dutchess County
STATE_WQP = f"US:{STATE_FIPS}"
COUNTY_WQP = f"US:{STATE_FIPS}:{COUNTY_FIPS}"
# The two services want different date formats. The USGS Samples service takes
# ISO (YYYY-MM-DD). The Water Quality Portal wants MM-DD-YYYY and returns
# HTTP 400 on ISO. Both are kept so each call passes the format it expects.
DATE_LOW = "2019-01-01"       # ISO, USGS Samples service
DATE_HIGH = "2019-03-31"
DATE_LOW_WQP = "01-01-2019"   # MM-DD-YYYY, Water Quality Portal
DATE_HIGH_WQP = "03-31-2019"

# The six target ions plus the two proxies named in the charter.
TARGET_CHARACTERISTICS = [
    "Chloride", "Sodium", "Calcium", "Magnesium",
    "Sulfate", "Alkalinity", "Specific conductance", "pH",
]

# USGS parameter codes from config/data_sources.yaml, for reference in output.
USGS_PCODES = {
    "Chloride": "00940", "Sodium": "00930", "Calcium": "00915",
    "Magnesium": "00925", "Sulfate": "00945", "Alkalinity": "00410",
    "Specific conductance": "00095",
}


def profile_frame(df: pd.DataFrame, key_columns: list[str]) -> dict:
    """Return a compact schema profile of a dataframe."""
    profile = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    # Missingness on the columns we actually care about, when present.
    miss = {}
    for col in key_columns:
        if col in df.columns:
            miss[col] = round(float(df[col].isna().mean()), 4)
    profile["missingness_key_columns"] = miss
    return profile


def show_unit_and_value_columns(df: pd.DataFrame, unit_hints: list[str], value_hints: list[str]) -> None:
    """Print the distinct units and a value sample from likely columns.

    Column names differ between services, so match on substrings rather than
    hardcoding a name.
    """
    unit_cols = [c for c in df.columns if any(h.lower() in c.lower() for h in unit_hints)]
    value_cols = [c for c in df.columns if any(h.lower() in c.lower() for h in value_hints)]
    print(f"  likely unit columns : {unit_cols}")
    for c in unit_cols:
        vals = df[c].dropna().unique()[:10]
        print(f"    {c} distinct units (first 10): {list(vals)}")
    print(f"  likely value columns: {value_cols}")


def verify_wqp() -> dict:
    """Pull a small EPA Water Quality Portal chloride slice and profile it."""
    print("=" * 78)
    print("EPA WATER QUALITY PORTAL  (dataretrieval.wqp.get_results, WQX3.0)")
    print("=" * 78)
    from dataretrieval import wqp

    # WQX3.0 (legacy=False) is used on purpose. The legacy WQX profile has not
    # refreshed USGS data since March 2024 and rejects ISO dates with HTTP 400.
    # WQX3.0 also returns the same column schema as the USGS Samples service,
    # which removes one harmonization step later. basicPhysChem is the compact
    # profile; fullPhysChem returns far more columns than we need for a check.
    # Dates here must be MM-DD-YYYY.
    df, meta = wqp.get_results(
        legacy=False,
        dataProfile="basicPhysChem",
        statecode=STATE_WQP,
        countycode=COUNTY_WQP,
        characteristicName="Chloride",
        startDateLo=DATE_LOW_WQP,
        startDateHi=DATE_HIGH_WQP,
    )
    print(f"request url: {getattr(meta, 'url', 'n/a')}")
    print(f"returned rows: {len(df)}   columns: {df.shape[1]}")
    if len(df) == 0:
        print("  no rows for this slice. Widen the county or dates and rerun.")
    print("\nfull column list:")
    for c in df.columns:
        print(f"  - {c}")
    print("\nunits and values:")
    show_unit_and_value_columns(
        df,
        unit_hints=["Result_MeasureUnit", "MeasureUnit", "Unit"],
        value_hints=["Result_Measure", "result", "value", "measure"],
    )
    key_cols = [
        "Activity_StartDate", "Location_Identifier", "Result_Characteristic",
        "Result_Measure", "Result_MeasureUnit", "Result_ResultDetectionCondition",
    ]
    print("\nfirst 5 rows (selected columns when present):")
    present = [c for c in key_cols if c in df.columns]
    with pd.option_context("display.max_colwidth", 28):
        print(df[present].head().to_string() if present else df.head().to_string())

    sample_path = OUT_DIR / "wqp_chloride_ny_dutchess_2019q1.csv"
    df.head(200).to_csv(sample_path, index=False)
    print(f"\nsaved sample: {sample_path}")

    profile = profile_frame(df, key_cols)
    profile["source"] = "EPA Water Quality Portal (WQX3.0 basicPhysChem)"
    profile["slice"] = f"{COUNTY_WQP} Chloride {DATE_LOW_WQP}..{DATE_HIGH_WQP}"
    profile["sample_file"] = str(sample_path.relative_to(PROJECT_ROOT))
    return profile


def verify_usgs() -> dict:
    """Pull a small USGS Samples service chloride slice and profile it."""
    print("\n" + "=" * 78)
    print("USGS SAMPLES SERVICE  (dataretrieval.waterdata.get_samples)")
    print("=" * 78)
    try:
        from dataretrieval import waterdata
    except Exception as exc:  # older dataretrieval without the samples service
        raise RuntimeError(
            "dataretrieval.waterdata is missing. Upgrade with "
            '"pip install \\"dataretrieval>=1.3,<2\\"" and rerun.'
        ) from exc

    # The Samples service accepts characteristic name and FIPS style codes.
    # If the code format below is not accepted, the error is printed and logged,
    # which is itself a verification result to record.
    df, meta = waterdata.get_samples(
        characteristic="Chloride",
        state_code=f"US:{STATE_FIPS}",
        county_code=COUNTY_WQP,
        activity_start_date_lower=DATE_LOW,
        activity_start_date_upper=DATE_HIGH,
    )
    print(f"returned rows: {len(df)}   columns: {df.shape[1]}")
    print("\nfull column list:")
    for c in df.columns:
        print(f"  - {c}")
    print("\nunits and values:")
    show_unit_and_value_columns(
        df,
        unit_hints=["unit", "Unit"],
        value_hints=["result", "value", "measure"],
    )
    print("\nfirst 5 rows:")
    with pd.option_context("display.max_colwidth", 24):
        print(df.head().to_string())

    sample_path = OUT_DIR / "usgs_samples_chloride_ny_dutchess_2019q1.csv"
    df.head(200).to_csv(sample_path, index=False)
    print(f"\nsaved sample: {sample_path}")

    profile = profile_frame(df, list(df.columns)[:12])
    profile["source"] = "USGS Samples service (waterdata.get_samples)"
    profile["slice"] = f"US:{STATE_FIPS} / {COUNTY_WQP} Chloride {DATE_LOW}..{DATE_HIGH}"
    profile["sample_file"] = str(sample_path.relative_to(PROJECT_ROOT))
    profile["pcode_reference"] = USGS_PCODES
    return profile


def write_markdown_block(results: dict) -> Path:
    """Write a paste-ready markdown summary for schema_verified.md."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"<!-- generated by 00a_verify_wqp_usgs.py on {stamp} UTC -->", ""]
    for name, prof in results.items():
        lines.append(f"### Verified: {prof.get('source', name)}")
        if "error" in prof:
            lines.append("")
            lines.append(f"Status: FAILED on this slice. Error recorded below.")
            lines.append("")
            lines.append("```")
            lines.append(prof["error"].strip())
            lines.append("```")
            lines.append("")
            continue
        lines.append("")
        lines.append(f"- Status: verified")
        lines.append(f"- Access: dataretrieval Python package")
        lines.append(f"- Slice inspected: {prof.get('slice','n/a')}")
        lines.append(f"- Row count on slice: {prof.get('row_count','n/a')}")
        lines.append(f"- Column count: {prof.get('column_count','n/a')}")
        lines.append(f"- Sample saved: {prof.get('sample_file','n/a')}")
        lines.append(f"- Real columns:")
        for c in prof.get("columns", []):
            lines.append(f"  - `{c}`")
        if prof.get("missingness_key_columns"):
            lines.append(f"- Missingness on key columns: {prof['missingness_key_columns']}")
        lines.append("")
    out = OUT_DIR / "schema_block_wqp_usgs.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    results: dict = {}

    for name, fn in [("wqp", verify_wqp), ("usgs", verify_usgs)]:
        try:
            results[name] = fn()
        except Exception:
            err = traceback.format_exc()
            print(f"\n[{name}] FAILED. Full error captured for the log:\n{err}")
            results[name] = {"source": name, "error": err}

    # Machine readable profile for later diffing.
    json_path = OUT_DIR / "schema_profile_wqp_usgs.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path = write_markdown_block(results)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"JSON profile     : {json_path}")
    print(f"Markdown block   : {md_path}")
    print("Next: paste the markdown block into 01_data/schema_verified.md under")
    print("sections 1 (EPA WQP) and 2 (USGS), then send me the JSON profile.")

    # Non zero exit if either source failed, so a wrapper can notice.
    return 0 if all("error" not in v for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
