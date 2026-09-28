"""Harmonize United States and Canadian water quality records.

Harmonization is the highest risk step in this project. Every decision made
here is documented in `08_docs/harmonization_notes.md` so reviewers can audit
it. The functions below are pure, deterministic, and idempotent.

The harmonized long record has one row per (station, datetime, ion) with these
columns:
    source                provider tag: EPA_WQP, USGS_SAMPLES, ECCC
    station_id            namespaced id, unique across sources
    station_id_source     the raw id at the source
    latitude, longitude   decimal degrees
    datum                 horizontal datum string
    country               US or CA
    state_prov            state or province code
    site_type             raw site type string
    datetime              tz naive pandas timestamp
    ion                   standard ion key (cl, na, ca, mg, so4, alk)
    value_mg_L            value converted to mg/L (alkalinity as CaCO3)
    below_detection       bool
    detection_limit       numeric detection limit in source units, when given
    fraction              total, dissolved, filtered, or unknown
    characteristic_source raw characteristic string
    unit_source           raw unit string

Design rules followed here:
- Units are never silently coerced. Unknown units are dropped and their count
  is logged and recorded in the returned frame's attrs.
- Detection limit handling is explicit and selected by the pipeline config.
- Provenance columns sit alongside the harmonized columns, never replace them.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# Standard wide column names, kept for the feature build step.
ION_STANDARD_NAMES = {
    "chloride": "cl_mg_L",
    "sodium": "na_mg_L",
    "calcium": "ca_mg_L",
    "magnesium": "mg_mg_L",
    "sulfate": "so4_mg_L",
    "alkalinity": "alk_mg_L_as_CaCO3",
}

# Short ion keys used in the harmonized long record.
ION_KEYS = ("cl", "na", "ca", "mg", "so4", "alk")

# Molar mass (g/mol) and charge magnitude for unit conversion. Alkalinity is
# handled as calcium carbonate equivalent, equivalent weight 50.04 g/eq.
ION_CHEMISTRY = {
    "cl":  {"molar_mass": 35.45,  "valence": 1, "equiv_weight": 35.45},
    "na":  {"molar_mass": 22.99,  "valence": 1, "equiv_weight": 22.99},
    "ca":  {"molar_mass": 40.078, "valence": 2, "equiv_weight": 20.039},
    "mg":  {"molar_mass": 24.305, "valence": 2, "equiv_weight": 12.1525},
    "so4": {"molar_mass": 96.06,  "valence": 2, "equiv_weight": 48.03},
    "alk": {"molar_mass": 100.09, "valence": 2, "equiv_weight": 50.04},
}


# --- characteristic name mapping -------------------------------------------

def to_standard_ion(name: str) -> Optional[str]:
    """Map a raw characteristic string from any source to a standard ion key.

    Returns one of ION_KEYS, or None when the string is not a target ion.
    Matching is case insensitive and tolerant of the different phrasings used
    by EPA WQP, the USGS Samples service, and ECCC.
    """
    if not isinstance(name, str):
        return None
    n = name.strip().lower()
    # Order matters: check alkalinity and sulfate before the bare cation names.
    if "alkalinity" in n:
        return "alk"
    if n.startswith("sulfate") or n.startswith("sulphate") or n in ("so4", "sulfate as so4"):
        return "so4"
    if n.startswith("chloride") or n == "cl":
        return "cl"
    # Sodium, calcium, magnesium: match the element word, exclude compound names
    # such as "sodium adsorption ratio" or "calcium carbonate".
    if re.match(r"^sodium\b", n) and "adsorption" not in n:
        return "na"
    if re.match(r"^calcium\b", n) and "carbonate" not in n:
        return "ca"
    if re.match(r"^magnesium\b", n):
        return "mg"
    return None


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and byte-order marks from column names.

    Some source CSVs are UTF-8 with a BOM; read under another encoding the first
    column name arrives as '﻿Name' or 'ï»¿Name'. This normalizes them so
    column lookups are stable regardless of the encoding used to read the file.
    """
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip()
                  for c in df.columns]
    return df


def classify_fraction(name: str, fallback: str = "unknown") -> str:
    """Infer the sample fraction from a characteristic or fraction string."""
    if not isinstance(name, str):
        return fallback
    n = name.lower()
    if "dissolved" in n or "filtered" in n or "filterable" in n:
        return "dissolved"
    if "total" in n or "unfiltered" in n or "recoverable" in n:
        return "total"
    return fallback


# --- unit conversion --------------------------------------------------------

_UNIT_ALIASES = {
    "mg/l": "mg/L", "mg/l ": "mg/L", "milligram per liter": "mg/L", "ppm": "mg/L",
    "ug/l": "ug/L", "µg/l": "ug/L", " microgram per liter": "ug/L", "ppb": "ug/L",
    "meq/l": "meq/L",
    "mmol/l": "mmol/L", "mmol/l ": "mmol/L",
    "mol/m3": "mol/m3", "mol/m^3": "mol/m3",
}


def _norm_unit(u: str) -> str:
    if not isinstance(u, str):
        return ""
    key = u.strip().lower()
    return _UNIT_ALIASES.get(key, u.strip())


def harmonize_units(
    df: pd.DataFrame,
    ion: str,
    value_column: str,
    unit_column: str,
    out_column: str = "value_mg_L",
) -> pd.DataFrame:
    """Convert every record for one ion to mg/L (alkalinity as CaCO3).

    Supported source units: mg/L, ug/L, meq/L, mmol/L, mol/m3. Rows in an
    unsupported unit are dropped; their count and the offending units are
    recorded in the returned frame's attrs under 'dropped_units'.
    """
    if ion not in ION_CHEMISTRY:
        raise ValueError(f"unknown ion {ion!r}")
    chem = ION_CHEMISTRY[ion]
    out = df.copy()
    units = out[unit_column].map(_norm_unit)
    val = pd.to_numeric(out[value_column], errors="coerce")

    factor = pd.Series(np.nan, index=out.index, dtype="float64")
    factor[units == "mg/L"] = 1.0
    factor[units == "ug/L"] = 1e-3
    factor[units == "meq/L"] = chem["equiv_weight"]
    factor[units == "mmol/L"] = chem["molar_mass"]
    factor[units == "mol/m3"] = chem["molar_mass"]  # 1 mol/m3 == 1 mmol/L

    unsupported = units[factor.isna() & units.ne("")].value_counts().to_dict()
    out[out_column] = val * factor
    keep = factor.notna() & val.notna()
    dropped = out.loc[~keep]
    out = out.loc[keep].copy()
    out.attrs["dropped_units"] = unsupported
    out.attrs["dropped_row_count"] = int(len(dropped))
    return out


# --- detection limits -------------------------------------------------------

def handle_below_detection(
    df: pd.DataFrame,
    value_column: str,
    detection_limit_column: Optional[str] = None,
    below_flag_column: Optional[str] = None,
    strategy: str = "half_dl",
) -> pd.DataFrame:
    """Impute or flag values reported below the detection limit.

    A row counts as below detection when below_flag_column is truthy (for
    example the ECCC '<' marker) or when the value is missing but a detection
    limit is present. Strategies: half_dl, dl, zero, drop. A boolean
    'below_detection' column is always added.
    """
    if strategy not in ("half_dl", "dl", "zero", "drop"):
        raise ValueError(f"unknown strategy {strategy!r}")
    out = df.copy()
    val = pd.to_numeric(out[value_column], errors="coerce")
    dl = (pd.to_numeric(out[detection_limit_column], errors="coerce")
          if detection_limit_column and detection_limit_column in out else
          pd.Series(np.nan, index=out.index))

    flag = pd.Series(False, index=out.index)
    if below_flag_column and below_flag_column in out:
        raw = out[below_flag_column].astype("string").fillna("").str.strip()
        flag = raw.isin(["<", "ND", "BDL", "U", "<="])
    flag = flag | (val.isna() & dl.notna())

    if strategy == "half_dl":
        val = val.where(~flag, dl / 2.0)
    elif strategy == "dl":
        val = val.where(~flag, dl)
    elif strategy == "zero":
        val = val.where(~flag, 0.0)
    elif strategy == "drop":
        out = out.loc[~flag].copy()
        val = val.loc[~flag]
        flag = flag.loc[~flag]

    out[value_column] = val
    out["below_detection"] = flag.reindex(out.index).fillna(False)
    return out


# --- cross border deduplication --------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def deduplicate_border_stations(
    df: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    station_id_column: str,
    threshold_km: float = 1.0,
) -> pd.DataFrame:
    """Cluster stations from different sources within threshold_km.

    Adds a 'crossborder_cluster_id' column; downstream code decides which
    cluster member to keep. Uses a latitude band sweep plus union find, so it
    needs no spatial index library. Stations, not observations, are clustered.
    """
    out = df.copy()
    stations = (out[[station_id_column, latitude_column, longitude_column]]
                .dropna(subset=[latitude_column, longitude_column])
                .drop_duplicates(subset=[station_id_column])
                .reset_index(drop=True))
    n = len(stations)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    lat = stations[latitude_column].to_numpy(dtype=float)
    lon = stations[longitude_column].to_numpy(dtype=float)
    order = np.argsort(lat)
    lat_window = threshold_km / 111.0 + 1e-9  # degrees latitude for the band
    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            if lat[j] - lat[i] > lat_window:
                break
            if _haversine_km(lat[i], lon[i], lat[j], lon[j]) <= threshold_km:
                union(i, j)

    cluster = {stations.loc[k, station_id_column]: find(k) for k in range(n)}
    out["crossborder_cluster_id"] = out[station_id_column].map(cluster)
    return out


# --- quality filters --------------------------------------------------------

def apply_quality_filters(
    df: pd.DataFrame,
    min_record_years: int,
    min_observations_per_year: int,
    station_column: str = "station_id",
    datetime_column: str = "datetime",
    ion_column: str = "ion",
) -> pd.DataFrame:
    """Keep station and ion series meeting the minimum record length and density.

    A (station, ion) series is kept when it has at least min_record_years
    distinct years in which the count of observations reaches
    min_observations_per_year.
    """
    out = df.copy()
    dt = pd.to_datetime(out[datetime_column], errors="coerce")
    out = out.loc[dt.notna()].copy()
    out["_year"] = dt.loc[out.index].dt.year

    per_year = (out.groupby([station_column, ion_column, "_year"])
                .size().rename("n").reset_index())
    dense = per_year.loc[per_year["n"] >= min_observations_per_year]
    good_years = (dense.groupby([station_column, ion_column])["_year"]
                  .nunique().rename("n_years").reset_index())
    keep_keys = good_years.loc[good_years["n_years"] >= min_record_years,
                               [station_column, ion_column]]

    merged = out.merge(keep_keys, on=[station_column, ion_column], how="inner")
    return merged.drop(columns=["_year"])


# --- source specific mappers ------------------------------------------------

# WQX 3.0 core columns shared by EPA WQP and the USGS Samples service.
def standardize_wqx3(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a WQX 3.0 frame (EPA WQP or USGS Samples) to the harmonized schema."""
    df = clean_columns(df)
    ion = df["Result_Characteristic"].map(to_standard_ion)
    dt = pd.to_datetime(df.get("Activity_StartDateTime",
                               df.get("Activity_StartDate")), errors="coerce", utc=True)
    dt = dt.dt.tz_localize(None)
    frac = df.get("Result_SampleFraction",
                  pd.Series("unknown", index=df.index)).map(classify_fraction)
    sid = df["Location_Identifier"].astype("string")
    detcond = df.get("Result_ResultDetectionCondition",
                     pd.Series(np.nan, index=df.index))
    out = pd.DataFrame({
        "source": source,
        "station_id": sid,
        "station_id_source": sid,
        "latitude": pd.to_numeric(df.get("Location_Latitude"), errors="coerce"),
        "longitude": pd.to_numeric(df.get("Location_Longitude"), errors="coerce"),
        "datum": df.get("Location_HorzCoordReferenceSystemDatum"),
        "country": "US",
        "state_prov": df.get("Location_State"),
        "site_type": df.get("Location_Type"),
        "datetime": dt,
        "ion": ion,
        "value_source": pd.to_numeric(df.get("Result_Measure"), errors="coerce"),
        "unit_source": df.get("Result_MeasureUnit"),
        "below_flag": detcond.notna().map({True: "<", False: ""}),
        "detection_limit": pd.to_numeric(df.get("DetectionLimit_MeasureA"), errors="coerce"),
        "fraction": frac,
        "characteristic_source": df["Result_Characteristic"],
    })
    return out.loc[out["ion"].notna()].reset_index(drop=True)


def standardize_eccc(results: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Map an ECCC region results frame plus the sites table to the schema."""
    results = clean_columns(results)
    sites = clean_columns(sites)
    ion = results["VARIABLE"].map(to_standard_ion)
    dt = pd.to_datetime(results["DATE_TIME_HEURE"], errors="coerce")
    site_cols = sites.set_index("SITE_NO")
    sid = results["SITE_NO"].astype("string")
    out = pd.DataFrame({
        "source": "ECCC",
        "station_id": "ECCC-" + sid,
        "station_id_source": sid,
        "latitude": pd.to_numeric(sid.map(site_cols.get("LATITUDE", {})), errors="coerce")
        if "LATITUDE" in sites else np.nan,
        "longitude": pd.to_numeric(sid.map(site_cols.get("LONGITUDE", {})), errors="coerce")
        if "LONGITUDE" in sites else np.nan,
        "datum": sid.map(site_cols["DATUM"]) if "DATUM" in sites else None,
        "country": "CA",
        "state_prov": sid.map(site_cols["PROV_TERR"]) if "PROV_TERR" in sites else None,
        "site_type": sid.map(site_cols["SITE_TYPE"]) if "SITE_TYPE" in sites else None,
        "datetime": dt,
        "ion": ion,
        "value_source": pd.to_numeric(results["VALUE_VALEUR"], errors="coerce"),
        "unit_source": results["UNIT_UNITÉ"] if "UNIT_UNITÉ" in results else results.get("UNIT_UNIT"),
        "below_flag": results.get("FLAG_MARQUEUR", pd.Series("", index=results.index)),
        "detection_limit": pd.to_numeric(results.get("MDL_LDM"), errors="coerce"),
        "fraction": results["VARIABLE"].map(classify_fraction),
        "characteristic_source": results["VARIABLE"],
    })
    return out.loc[out["ion"].notna()].reset_index(drop=True)


def _pick(df: pd.DataFrame, *aliases: str) -> Optional[str]:
    """Return the first alias that is an actual column of df, else None."""
    cols = set(df.columns)
    for a in aliases:
        if a in cols:
            return a
    return None


def standardize_ontario(results: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Map an Ontario PWQMN results frame plus the stations table to the schema.

    Ontario has published PWQMN results under two layouts over the years, and
    both are handled here by resolving column names against a set of aliases:
      newer (2021 onward): Collection Site, Analyte, Collection Date,
        Collection Time, Result, Units, Result Call, Value Qualifier,
        Detection Limit
      older (1990 to 2018): STATION, PARM_DESCRIPTION, DATE_YYYYMMDD,
        TIME_HH:MM, RESULT, VALUQUALIFI, UNITS, REMARK_CODE
    Read the source CSVs with encoding='cp1252'. A station id may carry a leading
    apostrophe text guard, stripped here. Below detection is inferred from the
    value qualifier or result call markers; the older files carry no detection
    limit column, so that field is left null for them.
    """
    results = clean_columns(results)
    stations = clean_columns(stations)

    site_col = _pick(results, "Collection Site", "STATION", "Station")
    analyte_col = _pick(results, "Analyte", "PARM_DESCRIPTION", "Parameter")
    date_col = _pick(results, "Collection Date", "DATE_YYYYMMDD", "Date")
    result_col = _pick(results, "Result", "RESULT", "Value")
    if not all((site_col, analyte_col, date_col, result_col)):
        raise KeyError(
            "Ontario results layout not recognized; columns present: "
            + ", ".join(map(str, results.columns))
        )
    time_col = _pick(results, "Collection Time", "TIME_HH:MM", "Time")
    units_col = _pick(results, "Units", "UNITS")
    qual_col = _pick(results, "Value Qualifier", "VALUQUALIFI")
    call_col = _pick(results, "Result Call", "REMARK_CODE", "REMARK_CODE_1")
    dl_col = _pick(results, "Detection Limit", "MDL")

    ion = results[analyte_col].map(to_standard_ion)
    sid = (results[site_col].astype("string")
           .str.replace("'", "", regex=False).str.strip())

    ds = results[date_col].astype("string").str.strip()
    ts = (results[time_col].astype("string").fillna("").str.strip()
          if time_col else pd.Series("", index=results.index))
    dt = pd.to_datetime((ds + " " + ts).str.strip(), errors="coerce")
    # Older files store the date as a compact YYYYMMDD the free parser can miss.
    na = dt.isna() & ds.notna()
    if na.any():
        dt = dt.fillna(pd.to_datetime(ds.where(na), format="%Y%m%d", errors="coerce"))

    st = stations.copy()
    st_key = _pick(st, "STATION", "Station", "Collection Site")
    st[st_key] = st[st_key].astype("string").str.replace("'", "", regex=False).str.strip()
    st = st.drop_duplicates(subset=[st_key]).set_index(st_key)

    qual = (results[qual_col].astype("string").fillna("")
            if qual_col else pd.Series("", index=results.index))
    call = (results[call_col].astype("string").fillna("")
            if call_col else pd.Series("", index=results.index))
    below = (call.str.contains("not detect", case=False, na=False)
             | call.str.contains("<", na=False) | qual.str.contains("<", na=False))

    out = pd.DataFrame({
        "source": "ON_PWQMN",
        "station_id": "ONPWQMN-" + sid,
        "station_id_source": sid,
        "latitude": pd.to_numeric(sid.map(st["LATITUDE"]) if "LATITUDE" in st else np.nan, errors="coerce"),
        "longitude": pd.to_numeric(sid.map(st["LONGITUDE"]) if "LONGITUDE" in st else np.nan, errors="coerce"),
        "datum": "NAD83",
        "country": "CA",
        "state_prov": "ON",
        "site_type": sid.map(st["NAME"]) if "NAME" in st else None,
        "datetime": dt,
        "ion": ion,
        "value_source": pd.to_numeric(results[result_col], errors="coerce"),
        "unit_source": results[units_col] if units_col else None,
        "below_flag": below.map({True: "<", False: ""}),
        "detection_limit": pd.to_numeric(results[dl_col], errors="coerce") if dl_col else np.nan,
        "fraction": results[analyte_col].map(classify_fraction),
        "characteristic_source": results[analyte_col],
    })
    return out.loc[out["ion"].notna()].reset_index(drop=True)


def standardize_bc_ems(df: pd.DataFrame) -> pd.DataFrame:
    """Map a BC EMS expanded results frame to the harmonized schema.

    The expanded CSV carries coordinates and location type inline, so no station
    join is needed. COLLECTION_START is a YYYYMMDDHHMMSS timestamp. Below
    detection is flagged by RESULT_LETTER '<'. Confirm the exact PARAMETER
    strings for the six ions on the first pull.
    """
    df = clean_columns(df)
    ion = df["PARAMETER"].map(to_standard_ion)
    dt = pd.to_datetime(df["COLLECTION_START"].astype("string").str.strip(),
                        format="%Y%m%d%H%M%S", errors="coerce")
    sid = df["EMS_ID"].astype("string").str.strip()
    letter = df.get("RESULT_LETTER", pd.Series("", index=df.index)).astype("string").fillna("")
    out = pd.DataFrame({
        "source": "BC_EMS",
        "station_id": "BCEMS-" + sid,
        "station_id_source": sid,
        "latitude": pd.to_numeric(df.get("LATITUDE"), errors="coerce"),
        "longitude": pd.to_numeric(df.get("LONGITUDE"), errors="coerce"),
        "datum": None,
        "country": "CA",
        "state_prov": "BC",
        "site_type": df.get("LOCATION_TYPE"),
        "datetime": dt,
        "ion": ion,
        "value_source": pd.to_numeric(df.get("RESULT"), errors="coerce"),
        "unit_source": df.get("UNIT"),
        "below_flag": letter.where(letter.isin(["<"]), ""),
        "detection_limit": pd.to_numeric(df.get("METHOD_DETECTION_LIMIT"), errors="coerce"),
        "fraction": df["PARAMETER"].map(classify_fraction),
        "characteristic_source": df["PARAMETER"],
    })
    return out.loc[out["ion"].notna()].reset_index(drop=True)
