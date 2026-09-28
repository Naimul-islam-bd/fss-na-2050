"""United States surface water ion ingestion: EPA WQP and USGS Samples.

Verified against the live services on 2026-09-21 (see 01_data/schema_verified.md
sections 1 and 2). Two things drove the design:

- The EPA Water Quality Portal legacy WQX profile is stale for USGS data since
  March 2024 and rejects ISO dates, so this module uses WQX 3.0
  (dataretrieval.wqp.get_results with legacy=False). WQX 3.0 dates are
  MM-DD-YYYY.
- The old NWIS water quality functions were removed in dataretrieval 1.3, so
  USGS chemistry comes from the Samples service
  (dataretrieval.waterdata.get_samples), whose dates are ISO YYYY-MM-DD.

Both services return the same WQX 3.0 column vocabulary, so the harmonizer can
treat them with one mapper. USGS records appear in both feeds, so deduplicate
after loading.

Requires dataretrieval>=1.3,<2.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger("wqp_ingest")

RAW_WQP = Path(__file__).resolve().parents[2] / "01_data" / "raw" / "wqp"
RAW_USGS = Path(__file__).resolve().parents[2] / "01_data" / "raw" / "usgs_samples"

TARGET_CHARACTERISTICS = [
    "Chloride", "Sodium", "Calcium", "Magnesium",
    "Sulfate", "Alkalinity", "Specific conductance", "pH",
]

# Columns kept from the wide WQX 3.0 / Samples response. Everything the
# harmonizer needs plus provenance; the rest is dropped to keep files small.
WQX3_CORE_COLUMNS = [
    "Org_Identifier", "ProviderName",
    "Location_Identifier", "Location_Name", "Location_Type",
    "Location_State", "Location_CountyName",
    "Location_Latitude", "Location_Longitude",
    "Location_HorzCoordReferenceSystemDatum",
    "Location_HUCEightDigitCode", "Location_HUCTwelveDigitCode",
    "Activity_StartDate", "Activity_StartDateTime",
    "Result_Characteristic", "Result_CharacteristicUserSupplied",
    "Result_SampleFraction",
    "Result_Measure", "Result_MeasureUnit",
    "Result_ResultDetectionCondition", "Result_MeasureStatusIdentifier",
    "DetectionLimit_MeasureA", "DetectionLimit_MeasureUnitA",
    "USGSpcode",
]


def _select_core(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the core columns that are present, preserving order."""
    present = [c for c in WQX3_CORE_COLUMNS if c in df.columns]
    return df[present].copy()


def _retry(fn, *, tries: int = 4, base_sleep: float = 3.0):
    """Call fn with retries; the WQX 3.0 service can fail intermittently."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - service errors are broad
            last = exc
            wait = base_sleep * (attempt + 1)
            logger.warning("attempt %d/%d failed: %r; sleeping %.0fs",
                           attempt + 1, tries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"all {tries} attempts failed: {last!r}")


def _year_windows(start: str, end: str):
    """Yield (lo, hi) ISO date pairs, one per calendar year, inclusive."""
    y0 = int(start[:4])
    y1 = int(end[:4])
    for y in range(y0, y1 + 1):
        lo = f"{max(y, y0)}-01-01" if y != y0 else start
        hi = f"{min(y, y1)}-12-31" if y != y1 else end
        yield lo, hi


def _iso_to_wqp(d: str) -> str:
    """YYYY-MM-DD to the MM-DD-YYYY the Water Quality Portal expects."""
    dt = datetime.strptime(d[:10], "%Y-%m-%d")
    return dt.strftime("%m-%d-%Y")


# --- EPA Water Quality Portal (WQX 3.0) ------------------------------------

def inspect_wqp_header(state_code: str, characteristic: str,
                       start_date: str, end_date: str) -> pd.DataFrame:
    """Small WQX 3.0 pull for header inspection. Dates are ISO here."""
    from dataretrieval import wqp
    df, _ = wqp.get_results(
        legacy=False, dataProfile="basicPhysChem",
        statecode=state_code, characteristicName=characteristic,
        startDateLo=_iso_to_wqp(start_date), startDateHi=_iso_to_wqp(end_date),
    )
    logger.info("WQP %s %s: %d rows, %d cols", state_code, characteristic,
                len(df), df.shape[1])
    return df


def download_wqp_bulk(
    states: List[str],
    start_date: str,
    end_date: str,
    characteristics: Iterable[str] = tuple(TARGET_CHARACTERISTICS),
    out_dir: Path = RAW_WQP,
) -> Path:
    """Bulk WQX 3.0 pull, chunked by state, characteristic, and year.

    Writes one parquet per state into out_dir and a manifest. state codes are
    the WQX form, for example 'US:36'. Only run after schema verification.
    """
    from dataretrieval import wqp
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for state in states:
        frames = []
        for char in characteristics:
            for lo, hi in _year_windows(start_date, end_date):
                def _pull(char=char, lo=lo, hi=hi):
                    df, _ = wqp.get_results(
                        legacy=False, dataProfile="basicPhysChem",
                        statecode=state, characteristicName=char,
                        startDateLo=_iso_to_wqp(lo), startDateHi=_iso_to_wqp(hi),
                    )
                    return df
                try:
                    df = _retry(_pull)
                except RuntimeError as exc:
                    logger.error("WQP giving up on %s %s %s-%s: %r",
                                 state, char, lo, hi, exc)
                    continue
                if len(df):
                    frames.append(_select_core(df))
        if frames:
            state_df = pd.concat(frames, ignore_index=True)
            safe = state.replace(":", "_")
            path = out_dir / f"wqp_{safe}.parquet"
            state_df.to_parquet(path, index=False)
            manifest.append({"state": state, "rows": len(state_df), "file": path.name})
            logger.info("wrote %s (%d rows)", path.name, len(state_df))
    pd.DataFrame(manifest).to_json(out_dir / "manifest.json", orient="records", indent=2)
    return out_dir


# --- USGS Samples service ---------------------------------------------------

def inspect_usgs_header(state_code: str, characteristic: str,
                        start_date: str, end_date: str) -> pd.DataFrame:
    """Small Samples service pull for header inspection. Dates are ISO."""
    from dataretrieval import waterdata
    df, _ = waterdata.get_samples(
        characteristic=characteristic, state_code=state_code,
        activity_start_date_lower=start_date, activity_start_date_upper=end_date,
    )
    logger.info("USGS %s %s: %d rows, %d cols", state_code, characteristic,
                len(df), df.shape[1])
    return df


def download_usgs_samples_bulk(
    states: List[str],
    start_date: str,
    end_date: str,
    characteristics: Iterable[str] = tuple(TARGET_CHARACTERISTICS),
    out_dir: Path = RAW_USGS,
) -> Path:
    """Bulk USGS Samples pull, chunked by state, characteristic, and year.

    state codes are the FIPS form, for example 'US:36'. Dates are ISO.
    """
    from dataretrieval import waterdata
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for state in states:
        frames = []
        for char in characteristics:
            for lo, hi in _year_windows(start_date, end_date):
                def _pull(char=char, lo=lo, hi=hi):
                    df, _ = waterdata.get_samples(
                        characteristic=char, state_code=state,
                        activity_start_date_lower=lo, activity_start_date_upper=hi,
                    )
                    return df
                try:
                    df = _retry(_pull)
                except RuntimeError as exc:
                    logger.error("USGS giving up on %s %s %s-%s: %r",
                                 state, char, lo, hi, exc)
                    continue
                if len(df):
                    frames.append(_select_core(df))
        if frames:
            state_df = pd.concat(frames, ignore_index=True)
            safe = state.replace(":", "_")
            path = out_dir / f"usgs_{safe}.parquet"
            state_df.to_parquet(path, index=False)
            manifest.append({"state": state, "rows": len(state_df), "file": path.name})
            logger.info("wrote %s (%d rows)", path.name, len(state_df))
    pd.DataFrame(manifest).to_json(out_dir / "manifest.json", orient="records", indent=2)
    return out_dir


def load_config():
    """Return the WQP subsection of the data sources config.

    Imported lazily so the ingestion functions do not require pyyaml at import
    time; only this helper needs it.
    """
    from ..utils.config_loader import load_data_sources_config
    return load_data_sources_config()["water_quality"]["epa_wqp"]
