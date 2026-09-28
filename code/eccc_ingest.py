"""Environment and Climate Change Canada long term water quality ingestion.

Verified against the live catalogue on 2026-09-21 (see 01_data/schema_verified.md
section 3). The federal record redirects to a JavaScript catalogue app at
data-donnees.az.ec.gc.ca that exposes two plain endpoints:
    listing: GET /api/path_contents?path=<url-encoded path>   -> JSON
    file:    GET /api/file?path=<url-encoded path>            -> raw bytes

Under the dataset root are three national metadata CSVs (station list, variable
dictionary, table descriptions) and about 22 drainage region subfolders, each
holding one results CSV named Water-Qual-Eau-<region>-2000-present.csv. Records
are long format, comma delimited, UTF-8, with one row per variable per sample.

The host serves a Government of Canada certificate chain the default trust
store rejects, and the file endpoint answered 400 and 503 to header-less
clients, so requests send a browser-like User-Agent and Referer, fall back to
unverified TLS, and retry on transient errors.
"""

from __future__ import annotations

import io
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger("eccc_ingest")

RAW_DIR = Path(__file__).resolve().parents[2] / "01_data" / "raw" / "eccc"

BASE = "https://data-donnees.az.ec.gc.ca"
ROOT_PATH = "/substances/monitor/national-long-term-water-quality-monitoring-data/"
NATIONAL_METADATA = [
    "Water-Qual-Eau-Sites-National.csv",
    "Water-Qual-Eau-VariableInfo.csv",
    "Water-Qual-Eau-TableDescriptions.csv",
]
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/json,application/octet-stream,*/*",
    "Referer": BASE + "/data" + ROOT_PATH,
}


def _ssl_contexts():
    yield ssl.create_default_context()
    unv = ssl.create_default_context()
    unv.check_hostname = False
    unv.verify_mode = ssl.CERT_NONE
    yield unv


def http_get(url: str, max_bytes: Optional[int] = None, tries: int = 4,
             timeout: int = 300) -> bytes:
    """GET with header spoofing, TLS fallback, and retry on transient errors."""
    last = None
    for attempt in range(tries):
        for ctx in _ssl_contexts():
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    return resp.read(max_bytes) if max_bytes else resp.read()
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code in (429, 500, 502, 503, 504):
                    break
            except Exception as exc:  # noqa: BLE001
                last = exc
        wait = 3 * (attempt + 1)
        logger.warning("GET %s failed (%r), retry %d/%d in %ds",
                       url, last, attempt + 1, tries, wait)
        time.sleep(wait)
    raise RuntimeError(f"GET failed for {url}: {last!r}")


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_csv_bytes(raw: bytes, **kw) -> pd.DataFrame:
    """Read a CSV from raw bytes, trying several encodings in turn.

    Most ECCC region files are UTF-8, but several French basin files (Ottawa,
    Pacific Coastal, Gaspe and others) carry accented station and basin names in
    cp1252, which makes a strict UTF-8 read raise UnicodeDecodeError. latin-1
    decodes any byte, so it is the final guard and this never fails on encoding.
    """
    last = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, **kw)
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # only reachable if every encoding failed, which latin-1 should not


def api_list(path: str) -> list[dict]:
    """Return the path_contents entries for a folder path (leading slash)."""
    url = f"{BASE}/api/path_contents?path={quote(path, safe='')}"
    return json.loads(http_get(url).decode("utf-8")).get("path_contents", [])


def api_file_url(inner_path: str) -> str:
    return f"{BASE}/api/file?path={quote('/' + inner_path.lstrip('/'), safe='')}"


def download_file(inner_path: str, out_path: Path, max_bytes: Optional[int] = None) -> Path:
    """Download one file by its catalogue path to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(http_get(api_file_url(inner_path), max_bytes=max_bytes))
    logger.info("saved %s (%d bytes)", out_path.name, out_path.stat().st_size)
    return out_path


def list_region_folders() -> list[str]:
    """Return the drainage region subfolder names under the dataset root."""
    return [e["name"] for e in api_list(ROOT_PATH) if e.get("is_directory")]


def list_region_files(region: str) -> list[dict]:
    """Return the file entries inside one region folder."""
    return [e for e in api_list(ROOT_PATH + region + "/") if not e.get("is_directory")]


def inspect_eccc_header(region: Optional[str] = None, nbytes: int = 200_000) -> pd.DataFrame:
    """Inspect a header: the sites table by default, or a region result CSV.

    Reads only the first nbytes so a 20 MiB region file is not fully pulled.
    """
    if region is None:
        raw = http_get(api_file_url(ROOT_PATH + "Water-Qual-Eau-Sites-National.csv"))
        return read_csv_bytes(raw, nrows=200)
    files = [e["name"] for e in list_region_files(region) if e["name"].lower().endswith(".csv")]
    if not files:
        raise RuntimeError(f"no CSV in region {region}")
    raw = http_get(api_file_url(ROOT_PATH + region + "/" + files[0]),
                   max_bytes=nbytes)
    return read_csv_bytes(raw, nrows=200)


def download_eccc_bulk(out_dir: Path = RAW_DIR, regions: Optional[list[str]] = None) -> Path:
    """Download the metadata CSVs and each region results CSV.

    regions defaults to every drainage region; pass a subset (the folder
    names) to limit to the domain, for example the southern Canada basins.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in NATIONAL_METADATA:
        download_file(ROOT_PATH + name, out_dir / name)
    if regions is None:
        regions = list_region_folders()
    manifest = []
    for region in regions:
        for entry in list_region_files(region):
            if entry["name"].lower().endswith(".csv"):
                dest = out_dir / region / entry["name"]
                download_file(ROOT_PATH + region + "/" + entry["name"], dest)
                manifest.append({"region": region, "file": entry["name"],
                                 "size": entry.get("content_length")})
    pd.DataFrame(manifest).to_json(out_dir / "manifest.json", orient="records", indent=2)
    return out_dir


def load_sites(path: Path) -> pd.DataFrame:
    """Read the national sites table (UTF-8, comma delimited)."""
    return pd.read_csv(path, encoding="utf-8", dtype={"SITE_NO": "string"})


def load_variable_info(path: Path) -> pd.DataFrame:
    """Read the variable dictionary (UTF-8, comma delimited)."""
    return pd.read_csv(path, encoding="utf-8")


def load_region_results(path: Path) -> pd.DataFrame:
    """Read one region results CSV (UTF-8, comma delimited, long format)."""
    return pd.read_csv(path, encoding="utf-8", dtype={"SITE_NO": "string"},
                       low_memory=False)
