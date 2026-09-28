"""Phase 2 verification, step 2: ECCC National Long Term Water Quality.

Purpose
-------
Confirm the real schema of the Environment and Climate Change Canada federal
long term water quality data, and compare its columns against the United States
WQX 3.0 schema verified in step 1.

Data mart structure (confirmed by inspecting the live catalogue)
----------------------------------------------------------------
The federal open data record points at a single page host that redirects to a
JavaScript catalogue app at data-donnees.az.ec.gc.ca. That app talks to two
plain endpoints, which this script uses directly:
  - listing:  GET /api/path_contents?path=<url-encoded path>   -> JSON
  - file:     GET /api/file?path=<url-encoded path>            -> raw bytes

Under the dataset folder there are three national metadata CSVs
(Water-Qual-Eau-Sites-National.csv, Water-Qual-Eau-VariableInfo.csv,
Water-Qual-Eau-TableDescriptions.csv) and about 22 drainage region subfolders.
Each region folder holds one large results CSV named
Water-Qual-Eau-<region>-2000-present.csv (for example the Saint Lawrence file
is roughly 22 MiB). So the measurement records are split by drainage region,
not delivered as one national table.

Notes carried over from earlier runs
-------------------------------------
- The host presents a Government of Canada certificate chain the default trust
  store rejects, so downloads try verified TLS first, then fall back to
  unverified TLS for this public endpoint, with a printed warning.
- The file endpoint answered 503 and 400 to header-less clients, so requests
  send a browser-like User-Agent and a Referer, and retry on transient errors.
- Government CSVs are often UTF-16 or Latin-1 and may use tab or semicolon
  separators, so the reader tries several encodings and separators.

How to run
----------
    python 03_notebooks/00b_verify_eccc.py

Needs internet. Writes to 01_data/raw/_verification/.
"""

from __future__ import annotations

import io
import json
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pandas as pd

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
OUT_DIR = PROJECT_ROOT / "01_data" / "raw" / "_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://data-donnees.az.ec.gc.ca"
ROOT_PATH = "/substances/monitor/national-long-term-water-quality-monitoring-data/"
# One region to header-inspect for the measurement schema.
SAMPLE_REGION = "saint-lawrence-river-basin-long-term-water-quality-monitoring-data"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/json,application/octet-stream,*/*",
    "Referer": BASE + "/data" + ROOT_PATH,
}

ENCODINGS = ("utf-8-sig", "utf-16", "cp1252", "latin-1")
SEPS = (",", ";", "\t")
NROWS = 200

US_CORE = [
    "Location_Identifier", "Location_Latitude", "Location_Longitude",
    "Activity_StartDate", "Result_Characteristic", "Result_Measure",
    "Result_MeasureUnit",
]


def _ssl_contexts():
    yield ssl.create_default_context()
    unv = ssl.create_default_context()
    unv.check_hostname = False
    unv.verify_mode = ssl.CERT_NONE
    yield unv


def http_get(url: str, max_bytes: int | None = None, tries: int = 4,
             timeout: int = 240) -> bytes:
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
                    break  # transient, wait then retry both contexts
                # 400/403 etc. are not TLS related; try the other context anyway
            except Exception as exc:
                last = exc
        wait = 2 * (attempt + 1)
        print(f"    retry {attempt + 1}/{tries} after {last!r} (sleep {wait}s)")
        time.sleep(wait)
    raise RuntimeError(f"GET failed for {url}: {last!r}")


def api_list(path: str) -> list[dict]:
    url = f"{BASE}/api/path_contents?path={quote(path, safe='')}"
    data = http_get(url)
    return json.loads(data.decode("utf-8")).get("path_contents", [])


def api_file(inner_path: str, max_bytes: int | None = None) -> bytes:
    url = f"{BASE}/api/file?path={quote('/' + inner_path.lstrip('/'), safe='')}"
    return http_get(url, max_bytes=max_bytes)


def parse_csv(data: bytes) -> pd.DataFrame:
    last = None
    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(io.BytesIO(data), nrows=NROWS, encoding=enc,
                                 sep=sep, engine="python", on_bad_lines="skip")
                if df.shape[1] > 1:
                    df.attrs["encoding_used"] = enc
                    df.attrs["sep_used"] = sep
                    return df
            except Exception as exc:
                last = exc
    raise RuntimeError(f"could not parse CSV bytes: {last!r}")


def show(df: pd.DataFrame, note: str) -> dict:
    print(f"{note}: encoding {df.attrs.get('encoding_used')}  "
          f"sep {df.attrs.get('sep_used')!r}  cols {df.shape[1]}")
    print("columns:", list(df.columns))
    with pd.option_context("display.max_colwidth", 26):
        print(df.head(3).to_string())
    return {"columns": list(df.columns), "column_count": int(df.shape[1]),
            "encoding_used": df.attrs.get("encoding_used"),
            "sep_used": df.attrs.get("sep_used")}


def main() -> int:
    results: dict = {"root_listing": [], "metadata": {}, "region_sample": {}}

    print("=" * 78)
    print("ECCC NATIONAL LONG TERM WATER QUALITY  (az data catalogue API)")
    print("=" * 78)

    # 1. Root listing.
    try:
        root = api_list(ROOT_PATH)
    except Exception:
        print("root listing failed:\n" + traceback.format_exc())
        return 1
    files = [e for e in root if not e["is_directory"]]
    dirs = [e for e in root if e["is_directory"]]
    print(f"\nroot: {len(files)} files, {len(dirs)} region folders")
    for e in files:
        print(f"  FILE  {e['name']}  ({e['content_length']}, mod {e['last_modified']})")
    for e in dirs:
        print(f"  DIR   {e['name']}")
    results["root_listing"] = [{"name": e["name"], "is_dir": e["is_directory"],
                                "size": e.get("content_length"),
                                "modified": e.get("last_modified")} for e in root]

    # 2. National metadata CSVs.
    meta_targets = [e["name"] for e in files if e["name"].lower().endswith(".csv")]
    for name in meta_targets:
        print("\n" + "-" * 78)
        print(f"metadata CSV: {name}")
        print("-" * 78)
        try:
            raw = api_file(ROOT_PATH + name, max_bytes=1_500_000)
            df = parse_csv(raw)
            prof = show(df, name)
            (OUT_DIR / f"eccc_{name}").write_bytes(raw)
            results["metadata"][name] = prof
        except Exception:
            print("failed:\n" + traceback.format_exc())
            results["metadata"][name] = {"error": "read_failed"}

    # 3. One region result CSV (header only, capped download).
    print("\n" + "=" * 78)
    print(f"REGION SAMPLE: {SAMPLE_REGION}")
    print("=" * 78)
    try:
        region = api_list(ROOT_PATH + SAMPLE_REGION + "/")
        region_csv = [e for e in region if e["name"].lower().endswith(".csv")]
        for e in region_csv:
            print(f"  result file: {e['name']}  ({e['content_length']})")
        if region_csv:
            name = region_csv[0]["name"]
            raw = api_file(ROOT_PATH + SAMPLE_REGION + "/" + name, max_bytes=600_000)
            df = parse_csv(raw)
            prof = show(df, name)
            (OUT_DIR / f"eccc_region_{name}").write_bytes(raw)
            results["region_sample"] = {"region": SAMPLE_REGION, "file": name, **prof}

            print("\nQUICK MAP HINT vs United States WQX 3.0 core:")
            low = {c.lower(): c for c in df.columns}
            for tgt in US_CORE:
                hits = [orig for lc, orig in low.items()
                        if any(tok in lc for tok in tgt.lower().split("_"))]
                print(f"  {tgt:24} -> {hits[:5]}")
    except Exception:
        print("region sample failed:\n" + traceback.format_exc())

    json_path = OUT_DIR / "schema_profile_eccc.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDONE. JSON profile: {json_path}")
    print("Send me schema_profile_eccc.json and I will lock section 3 and map")
    print("the Canadian columns to the United States WQX 3.0 schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
