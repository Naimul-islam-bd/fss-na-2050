"""Full continental pull and harmonize for the FSS-NA-2050 project.

Run from the project root:

    python run_full_pull.py                # everything, resume-able
    python run_full_pull.py --sources us   # only EPA WQP + USGS
    python run_full_pull.py --sources canada
    python run_full_pull.py --states US:36 US:34   # a subset, for a trial

It pulls each verified source, harmonizes to the standard schema, saves a
checkpoint parquet per state or region, and finally builds one master table
(unit conversion, below detection, domain filter, deduplication, quality
filter). It is resume-able: a state or region whose checkpoint already exists
is skipped, so a crash or a dropped connection loses nothing. Re-run the same
command to continue.

Progress: after each state or region it prints a line with percent done,
elapsed minutes, and a rough ETA, and it rewrites 06_outputs/pull_progress.json
so progress can be checked without watching the console.

Needs internet, dataretrieval>=1.3, pyyaml, pyarrow. Runs entirely on this
machine; it does not depend on any remote link once started.

Scope is the locked charter scope: United States lower 48 plus DC and southern
Canada, latitude 24 to 60 N, 1990 to 2025, six ions. Quebec and Alberta are not
included yet because their schemas are not verified; add them once they are.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "02_code"
INTERIM = ROOT / "01_data" / "interim"
PROCESSED = ROOT / "01_data" / "processed"
OUTDIR = ROOT / "06_outputs"
for d in (INTERIM, PROCESSED, OUTDIR):
    d.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = OUTDIR / "pull_progress.json"
PKG = "fsscode"


def _pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [str(path)]; m.__package__ = name
    sys.modules[name] = m; return m


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); m.__package__ = name.rpartition(".")[0]
    sys.modules[name] = m; spec.loader.exec_module(m); return m


_pkg(PKG, CODE)
_pkg(f"{PKG}.utils", CODE / "utils")
_pkg(f"{PKG}.ingestion", CODE / "ingestion")
_pkg(f"{PKG}.preprocessing", CODE / "preprocessing")
_mod(f"{PKG}.utils.logging_utils", CODE / "utils" / "logging_utils.py")
wqp = _mod(f"{PKG}.ingestion.wqp_ingest", CODE / "ingestion" / "wqp_ingest.py")
eccc = _mod(f"{PKG}.ingestion.eccc_ingest", CODE / "ingestion" / "eccc_ingest.py")
H = _mod(f"{PKG}.preprocessing.harmonize", CODE / "preprocessing" / "harmonize.py")

# --- scope -----------------------------------------------------------------
US_STATES = [f"US:{s}" for s in [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41",
    "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"]]
CHARS = ["Chloride", "Sodium", "Calcium", "Magnesium", "Sulfate", "Alkalinity"]
WINDOWS = [("1990-01-01", "1999-12-31"), ("2000-01-01", "2009-12-31"),
           ("2010-01-01", "2019-12-31"), ("2020-01-01", "2025-12-31")]
LAT_MIN, LAT_MAX = 24.0, 60.0
DATE_MIN = "1990-01-01"

ON_BASE = "https://files.ontario.ca/moe_mapping/downloads/2Water/PWQMN/"
ON_HIST = "https://files.ontario.ca/moe_mapping/downloads/2Water/PWQMN_historical/"
ON_RESULT_FILES = [ON_BASE + f for f in [
    "PWQMN_2025.csv", "PWQMN_March2021-2024.csv", "PWQMN_2019-2021Marn.csv",
    "PWQMN-2010_2018.csv", "PWQMN-2000_2009.csv"]] + [ON_HIST + f for f in [
    "PWQMN_OpenData_1995-1999n.csv", "PWQMN_OpenData_1990-1994n.csv"]]
ON_STATIONS = ON_BASE + "PWQMN-Stations.csv"
BC_BASE = "https://pub.data.gov.bc.ca/datasets/949f2233-9612-4b06-92a9-903e817da659/"
BC_FILES = ["ems_sample_results_current_expanded.csv",
            "ems_sample_results_historic_expanded.csv"]


class Progress:
    """Track completed units, print percent and ETA, and write a status file."""

    def __init__(self, total: int, log):
        self.total = max(total, 1)
        self.log = log
        self.done = 0
        self.newly = 0          # units actually pulled this run, for ETA timing
        self.t0 = time.time()

    def tick(self, label: str, cached: bool):
        self.done += 1
        if not cached:
            self.newly += 1
        elapsed = time.time() - self.t0
        rate = (elapsed / self.newly) if self.newly else 0.0
        remaining = self.total - self.done
        eta_min = (rate * remaining / 60.0) if rate else None
        pct = 100.0 * self.done / self.total
        eta_txt = f"~{eta_min:.0f} min" if eta_min is not None else "n/a (resuming)"
        self.log.info("[%d/%d %.1f%%] %s | elapsed %.1f min | eta %s",
                      self.done, self.total, pct, label, elapsed / 60.0, eta_txt)
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "done": self.done, "total": self.total, "percent": round(pct, 1),
                "elapsed_min": round(elapsed / 60.0, 1),
                "eta_min": round(eta_min, 1) if eta_min is not None else None,
                "last_unit": label,
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _done_marker(path: Path) -> Path:
    return path.with_suffix(".done")


def _parquet_rows(path: Path) -> int:
    """Row count of a checkpoint parquet, cheaply from its footer. -1 if unreadable."""
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except Exception:
        try:
            return len(pd.read_parquet(path))
        except Exception:
            return -1


def _is_done(path: Path) -> bool:
    """A unit counts as complete only if it has real rows on disk, or it was
    explicitly marked done (a genuine zero-row unit). An empty checkpoint with
    no marker is a network-caused gap and must be re-pulled on the next run."""
    if _done_marker(path).exists():
        return True
    return path.exists() and _parquet_rows(path) > 0


def _commit(path: Path, std, net_fail: bool, prog, prefix: str, unit: str):
    """Decide what to do with a unit's result.

    rows found          -> save parquet, mark done.
    zero rows, no error -> genuinely empty; save empty parquet, mark done.
    zero rows + failure -> leave uncommitted so a re-run retries it.
    """
    if std is not None and len(std):
        _save(std, path)
        _done_marker(path).touch()
        prog.tick(f"{prefix} {unit} ({len(std)} rows)", False)
    elif net_fail:
        prog.tick(f"{prefix} {unit} (network gap; will retry on re-run)", False)
    else:
        _save(std if std is not None else pd.DataFrame(), path)
        _done_marker(path).touch()
        prog.tick(f"{prefix} {unit} (0 rows)", False)


# --- United States ---------------------------------------------------------

def pull_us(log, prog, with_usgs=False):
    (INTERIM / "wqp").mkdir(parents=True, exist_ok=True)
    from dataretrieval import wqp as _w
    for state in US_STATES:
        safe = state.replace(":", "_")
        wpath = INTERIM / "wqp" / f"{safe}.parquet"
        if _is_done(wpath):
            prog.tick(f"WQP {state} (cached)", True)
            continue
        frames = []
        net_fail = False
        # All six ions in one request per window: httpx encodes the list as
        # repeated characteristicName params, which the portal reads as OR.
        # That is 4 requests per state instead of 24.
        for lo, hi in WINDOWS:
            def _c(lo=lo, hi=hi):
                df, _ = _w.get_results(legacy=False, dataProfile="basicPhysChem",
                                       statecode=state, characteristicName=CHARS,
                                       startDateLo=wqp._iso_to_wqp(lo),
                                       startDateHi=wqp._iso_to_wqp(hi))
                return df
            try:
                df = wqp._retry(_c, tries=5)
                if len(df):
                    frames.append(wqp._select_core(df))
            except Exception as exc:
                net_fail = True
                log.warning("WQP %s %s-%s gave up: %r", state, lo, hi, exc)
        std = H.standardize_wqx3(pd.concat(frames, ignore_index=True), "EPA_WQP") if frames else pd.DataFrame()
        _commit(wpath, std, net_fail, prog, "WQP", state)

    if not with_usgs:
        return
    # Optional: the USGS Samples service is a subset of what the portal already
    # returns and it times out often, so it is off by default. Enable with
    # --with-usgs only if you specifically want the Samples service records.
    from dataretrieval import waterdata as _wd
    (INTERIM / "usgs").mkdir(parents=True, exist_ok=True)
    for state in US_STATES:
        safe = state.replace(":", "_")
        upath = INTERIM / "usgs" / f"{safe}.parquet"
        if _is_done(upath):
            prog.tick(f"USGS {state} (cached)", True)
            continue
        frames = []
        net_fail = False
        for char in CHARS:
            for lo, hi in WINDOWS:
                def _c(char=char, lo=lo, hi=hi):
                    df, _ = _wd.get_samples(characteristic=char, state_code=state,
                                            activity_start_date_lower=lo,
                                            activity_start_date_upper=hi)
                    return df
                try:
                    df = wqp._retry(_c, tries=2)
                    if len(df):
                        frames.append(wqp._select_core(df))
                except Exception as exc:
                    net_fail = True
                    log.warning("USGS %s %s %s-%s gave up: %r", state, char, lo, hi, exc)
        std = H.standardize_wqx3(pd.concat(frames, ignore_index=True), "USGS_SAMPLES") if frames else pd.DataFrame()
        _commit(upath, std, net_fail, prog, "USGS", state)


# --- ECCC ------------------------------------------------------------------

def pull_eccc(log, prog, regions):
    (INTERIM / "eccc").mkdir(parents=True, exist_ok=True)
    # The national sites file gives station coordinates and is needed to
    # standardize every region. If it cannot be fetched (network gap), skip the
    # whole ECCC phase for this run rather than crashing; a re-run retries it.
    try:
        sites_raw = eccc.http_get(eccc.api_file_url(eccc.ROOT_PATH + "Water-Qual-Eau-Sites-National.csv"))
        sites = eccc.read_csv_bytes(sites_raw, dtype={"SITE_NO": "string"})
    except Exception:
        log.warning("ECCC sites file unreachable; skipping ECCC this run (will retry on re-run):\n%s",
                    traceback.format_exc())
        return
    for region in regions:
        safe = "".join(c if c.isalnum() else "_" for c in region)[:60]
        rpath = INTERIM / "eccc" / f"{safe}.parquet"
        if _is_done(rpath):
            prog.tick(f"ECCC {region[:30]} (cached)", True)
            continue
        net_fail = False
        try:
            files = [e["name"] for e in eccc.list_region_files(region) if e["name"].lower().endswith(".csv")]
            frames = []
            for name in files:
                raw = eccc.http_get(eccc.api_file_url(eccc.ROOT_PATH + region + "/" + name))
                res = eccc.read_csv_bytes(raw, dtype={"SITE_NO": "string"}, low_memory=False)
                frames.append(H.standardize_eccc(res, sites))
            std = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        except Exception:
            log.warning("ECCC region %s failed:\n%s", region, traceback.format_exc())
            std = pd.DataFrame()
            net_fail = True
        _commit(rpath, std, net_fail, prog, "ECCC", region[:30])


# --- Ontario ---------------------------------------------------------------

def pull_ontario(log, prog):
    (INTERIM / "ontario").mkdir(parents=True, exist_ok=True)
    try:
        st = pd.read_csv(ON_STATIONS, encoding="cp1252")
    except Exception:
        log.warning("Ontario stations file unreachable; skipping Ontario this run (will retry):\n%s",
                    traceback.format_exc())
        return
    for url in ON_RESULT_FILES:
        fname = url.rsplit("/", 1)[-1]
        opath = INTERIM / "ontario" / f"{fname.replace('.csv', '')}.parquet"
        if _is_done(opath):
            prog.tick(f"Ontario {fname} (cached)", True)
            continue
        net_fail = False
        try:
            res = pd.read_csv(url, encoding="cp1252", low_memory=False)
            std = H.standardize_ontario(res, st)
        except Exception:
            log.warning("Ontario %s failed:\n%s", fname, traceback.format_exc())
            std = pd.DataFrame()
            net_fail = True
        _commit(opath, std, net_fail, prog, "Ontario", fname)


# --- British Columbia ------------------------------------------------------

def _download_resumable(url, dest, log, block_mb=8, max_tries=8):
    """Stream a large file to dest, resuming from a partial file if the
    connection drops. The BC historic export is about 8 GB and a single read
    was cut off part way, so this keeps the bytes already on disk and asks the
    server for the remainder with a Range request. Returns True when complete.
    """
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    base_headers = {"User-Agent": "Mozilla/5.0"}
    total = 0
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD", headers=base_headers),
                timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
    except Exception:
        total = 0
    for attempt in range(max_tries):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            return True
        headers = dict(base_headers)
        mode = "wb"
        if have:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=180) as r:
                # Server ignored the Range request: start clean to avoid a
                # corrupt append.
                if have and getattr(r, "status", 200) == 200:
                    have, mode = 0, "wb"
                if not total:
                    total = int(r.headers.get("Content-Length", 0)) + have
                with open(dest, mode) as f:
                    while True:
                        block = r.read(block_mb * 1024 * 1024)
                        if not block:
                            break
                        f.write(block)
            have = dest.stat().st_size
            if not total or have >= total:
                return True
            log.warning("BC download incomplete (%d/%d bytes); resuming (try %d/%d)",
                        have, total, attempt + 1, max_tries)
        except Exception as exc:
            got = dest.stat().st_size if dest.exists() else 0
            log.warning("BC download interrupted at %d bytes (%r); resuming (try %d/%d)",
                        got, exc, attempt + 1, max_tries)
        time.sleep(3 * (attempt + 1))
    return bool(total) and dest.exists() and dest.stat().st_size >= total


def pull_bc(log, prog):
    (INTERIM / "bc").mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "01_data" / "raw" / "bc"
    for fname in BC_FILES:
        bpath = INTERIM / "bc" / f"{fname.replace('.csv', '')}.parquet"
        if _is_done(bpath):
            prog.tick(f"BC {fname} (cached)", True)
            continue
        url = BC_BASE + fname
        net_fail = False
        try:
            # Download to disk first, resuming across drops, then parse locally.
            # A direct streamed read of the 8 GB historic file is cut off by any
            # connection blip; a partial file on disk lets a re-run continue it.
            raw_csv = raw_dir / fname
            if not _download_resumable(url, raw_csv, log):
                raise RuntimeError(f"download did not complete for {fname}")
            parts = []
            for chunk in pd.read_csv(raw_csv, chunksize=200_000, low_memory=False):
                std = H.standardize_bc_ems(chunk)
                if len(std):
                    parts.append(std)
            std = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        except Exception:
            log.warning("BC %s failed:\n%s", fname, traceback.format_exc())
            std = pd.DataFrame()
            net_fail = True
        _commit(bpath, std, net_fail, prog, "BC", fname)


# --- master build ----------------------------------------------------------

def build_master(log):
    frames = []
    for sub in ("wqp", "usgs", "eccc", "ontario", "bc"):
        d = INTERIM / sub
        if d.exists():
            for p in sorted(d.glob("*.parquet")):
                df = pd.read_parquet(p)
                if len(df):
                    frames.append(df)
    if not frames:
        log.warning("no interim data found; run a pull first")
        return
    allrows = pd.concat(frames, ignore_index=True)
    log.info("combined standardized rows: %d", len(allrows))

    parts = []
    for ion in H.ION_KEYS:
        sub = allrows[allrows["ion"] == ion]
        if len(sub):
            parts.append(H.harmonize_units(sub, ion, "value_source", "unit_source"))
    merged = pd.concat(parts, ignore_index=True)
    merged = H.handle_below_detection(merged, "value_mg_L", "detection_limit",
                                      "below_flag", strategy="half_dl")

    lat = pd.to_numeric(merged["latitude"], errors="coerce")
    merged = merged[(lat >= LAT_MIN) & (lat <= LAT_MAX)]
    dt = pd.to_datetime(merged["datetime"], errors="coerce")
    merged = merged[dt >= pd.Timestamp(DATE_MIN)]
    merged = merged[merged["value_mg_L"].notna()]

    order = {"USGS_SAMPLES": 0, "EPA_WQP": 1, "ECCC": 2, "ON_PWQMN": 3, "BC_EMS": 4}
    merged["_pref"] = merged["source"].map(order).fillna(9)
    merged = (merged.sort_values("_pref")
              .drop_duplicates(subset=["station_id", "datetime", "ion"])
              .drop(columns="_pref"))

    merged = H.deduplicate_border_stations(merged, "latitude", "longitude", "station_id")
    merged = H.apply_quality_filters(merged, min_record_years=10, min_observations_per_year=4)

    master = PROCESSED / "master_harmonized.parquet"
    merged.to_parquet(master, index=False)
    merged.head(2000).to_csv(OUTDIR / "master_sample.csv", index=False)
    summary = {
        "total_rows": int(len(merged)),
        "by_source": merged["source"].value_counts().to_dict(),
        "by_ion": merged["ion"].value_counts().to_dict(),
        "stations": int(merged["station_id"].nunique()),
        "date_min": str(pd.to_datetime(merged["datetime"]).min()),
        "date_max": str(pd.to_datetime(merged["datetime"]).max()),
    }
    (OUTDIR / "master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("MASTER: %d rows, %d stations, %d ions",
             len(merged), merged["station_id"].nunique(), merged["ion"].nunique())
    print("\nMASTER SUMMARY:", summary)
    print("saved:", master)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="all", choices=["all", "us", "canada"])
    ap.add_argument("--states", nargs="*", help="subset of WQX state codes, e.g. US:36")
    ap.add_argument("--with-usgs", action="store_true",
                    help="also pull the USGS Samples service (subset of WQP, slow); off by default")
    ap.add_argument("--skip-master", action="store_true")
    args = ap.parse_args()
    log = wqp.logger

    global US_STATES
    if args.states:
        US_STATES = args.states

    # Work out the total unit count so percent and ETA are meaningful.
    total = 0
    if args.sources in ("all", "us"):
        total += len(US_STATES) * (2 if args.with_usgs else 1)
    eccc_regions = []
    if args.sources in ("all", "canada"):
        try:
            eccc_regions = eccc.list_region_folders()
        except Exception:
            log.warning("could not list ECCC regions up front; estimating 22")
            eccc_regions = [f"region_{i}" for i in range(22)]
        total += len(eccc_regions) + len(ON_RESULT_FILES) + len(BC_FILES)
    prog = Progress(total, log)
    log.info("planned units: %d  (check %s or ask any time)", total, PROGRESS_FILE.name)

    def phase(name, fn, *a):
        # One source failing must never abort the run or lose the master build.
        try:
            log.info("=== %s ===", name); fn(*a)
        except Exception:
            log.warning("%s phase aborted (will resume on re-run):\n%s", name, traceback.format_exc())

    t0 = time.time()
    if args.sources in ("all", "us"):
        phase("United States pull", pull_us, log, prog, args.with_usgs)
    if args.sources in ("all", "canada"):
        phase("ECCC pull", pull_eccc, log, prog, eccc_regions)
        phase("Ontario pull", pull_ontario, log, prog)
        phase("BC pull", pull_bc, log, prog)
    if not args.skip_master:
        phase("building master table", build_master, log)
    log.info("done in %.1f min", (time.time() - t0) / 60)
    log.info("If any unit showed a network gap, just run this again; it skips "
             "what already has data and retries only the gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
