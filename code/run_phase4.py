"""Phase 4 runner: build the modelling table and the baseline model.

Run from the project root, after a master table exists:

    pip install lightgbm            # once, if not already installed
    python run_phase4.py

It reads 01_data/processed/master_harmonized.parquet, writes the station-ion-year
modelling table to 01_data/processed/model_table.parquet, trains the multitask
LightGBM baseline with a temporal split, and writes metrics to
06_outputs/phase4/baseline_metrics.json. Re-run it whenever the master table is
rebuilt; it always reflects the current data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "02_code"
MASTER = ROOT / "01_data" / "processed" / "master_harmonized.parquet"
OUT = ROOT / "06_outputs" / "phase4"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    if not MASTER.exists():
        print(f"master table not found at {MASTER}; run the pull first")
        return 1
    feats = _load("annual_features", CODE / "features" / "annual_features.py")
    model = _load("baseline_lgbm", CODE / "models" / "baseline_lgbm.py")

    OUT.mkdir(parents=True, exist_ok=True)
    print("building modelling table from", MASTER.name)
    feat = feats.load_and_build(MASTER)
    feat.to_parquet(ROOT / "01_data" / "processed" / "model_table.parquet", index=False)
    print("modelling table:", feat.shape)

    print("training baseline")
    summary = model.run(feat, OUT)
    print("\nOVERALL (log scale):")
    print(json.dumps(summary["overall"], indent=2))
    print("\nper-ion and feature importance:", OUT / "baseline_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
