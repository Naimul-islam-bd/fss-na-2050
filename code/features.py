"""Feature engineering for the FSS-NA-2050 salinization forecast.

Input is the harmonized long record (one row per station, datetime, ion) written
by the pull pipeline at 01_data/processed/master_harmonized.parquet. This module
reduces it to an annual station-ion modelling table and derives the temporal and
static predictors used by the baseline model.

Design choices, all documented so reviewers can audit them:
- Grain is one row per station, ion, and calendar year, using the annual mean of
  the harmonized mg/L value. Annual means damp sub-annual sampling noise and give
  a regular yearly series for trend work.
- The target is log1p of the annual mean. Concentrations are right skewed over
  orders of magnitude, so a log target stabilizes variance. Non-positive values
  are physically impossible for these ions and are dropped before the transform.
- Predictors use only information available up to and including the target year,
  so the table can be used for backward looking validation without leakage:
  lagged annual means (t-1, t-2, t-3), trailing 3 and 5 year means, a trailing
  linear slope over the observed history, the count of years observed so far, and
  the within-year sample count. Static predictors are location, provider, region,
  site type, and ion identity.

The functions are deterministic and side effect free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ION_KEYS = ("cl", "na", "ca", "mg", "so4", "alk")


def annual_means(long_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the harmonized long record to station-ion-year annual means."""
    df = long_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()]
    df["year"] = df["datetime"].dt.year
    df = df[df["value_mg_L"].notna() & (df["value_mg_L"] > 0)]

    # First static attribute per station, carried onto every station-year row.
    static_cols = ["latitude", "longitude", "source", "country", "state_prov", "site_type"]
    static = (df.sort_values("datetime")
                .groupby("station_id")[static_cols].first().reset_index())

    grp = df.groupby(["station_id", "ion", "year"])
    ann = grp["value_mg_L"].agg(["mean", "count"]).reset_index()
    ann = ann.rename(columns={"mean": "value", "count": "n_obs_year"})
    ann = ann.merge(static, on="station_id", how="left")
    return ann


def _trailing_slope(years: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Slope of log value on year using only the strictly earlier points.

    Returns one value per row (NaN until at least three earlier points exist).
    Computed in a single pass with running sums so it is cheap on long series.
    """
    n = len(years)
    out = np.full(n, np.nan)
    sx = sy = sxx = sxy = 0.0
    k = 0
    for i in range(n):
        if k >= 3:
            denom = k * sxx - sx * sx
            if denom != 0:
                out[i] = (k * sxy - sx * sy) / denom
        x = float(years[i]); y = float(vals[i])
        sx += x; sy += y; sxx += x * x; sxy += x * y; k += 1
    return out


def build_features(ann: pd.DataFrame) -> pd.DataFrame:
    """Add the log target, lagged levels, trailing means, and trailing slope."""
    ann = ann.sort_values(["station_id", "ion", "year"]).copy()
    ann["log_value"] = np.log1p(ann["value"])

    g = ann.groupby(["station_id", "ion"], sort=False)
    for k in (1, 2, 3):
        ann[f"lag{k}"] = g["log_value"].shift(k)
    ann["roll3"] = g["log_value"].shift(1).rolling(3, min_periods=1).mean().reset_index(drop=True)
    ann["roll5"] = g["log_value"].shift(1).rolling(5, min_periods=1).mean().reset_index(drop=True)
    ann["years_seen"] = g.cumcount()
    ann["year_c"] = ann["year"] - 2000

    slopes = np.concatenate([
        _trailing_slope(sub["year"].to_numpy(), sub["log_value"].to_numpy())
        for _, sub in g
    ])
    ann["trail_slope"] = slopes
    return ann


def load_and_build(master_path: str | Path) -> pd.DataFrame:
    """Full path from the harmonized parquet to the modelling table."""
    long_df = pd.read_parquet(master_path)
    return build_features(annual_means(long_df))


if __name__ == "__main__":
    import sys
    out = load_and_build(sys.argv[1])
    print("modelling table:", out.shape)
    print(out[["station_id", "ion", "year", "value", "log_value",
               "lag1", "roll3", "trail_slope"]].head(12).to_string())
