"""Print only the header row of each Ontario PWQMN file that failed to parse.

Reads just the first line over HTTP (no full download), so it is fast even for
the large historical files. Run from a new terminal:

    python check_ontario_headers.py

Copy the whole output back so the Ontario mapper can be matched to the real
column names in each file era.
"""
import urllib.request

BASE = "https://files.ontario.ca/moe_mapping/downloads/2Water/PWQMN/"
HIST = "https://files.ontario.ca/moe_mapping/downloads/2Water/PWQMN_historical/"
FILES = [
    BASE + "PWQMN_2019-2021Marn.csv",
    BASE + "PWQMN-2010_2018.csv",
    BASE + "PWQMN-2000_2009.csv",
    HIST + "PWQMN_OpenData_1995-1999n.csv",
    HIST + "PWQMN_OpenData_1990-1994n.csv",
]
for url in FILES:
    name = url.rsplit("/", 1)[-1]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            header = r.readline().decode("cp1252", "replace").strip()
        print(name, "->", header)
    except Exception as exc:
        print(name, "-> ERROR", repr(exc))
