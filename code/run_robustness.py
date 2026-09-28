"""Phase 5 robustness checks (reviewer-proofing).

Confirms that the salinity-level result does not depend on the model choice or the
land-cover buffer scale. It compares the boosted model against a random forest, and
the two-buffer feature set against each buffer alone, all under the same
spatial-block cross validation used in the main analysis.

Run from the project root, after run_phase5.py has produced the projection table:
    python run_robustness.py
Inputs (06_outputs/phase5/): projection_2050_allion.parquet (or projection_2050.parquet)
and station_covariates.parquet. Output: robustness_results.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PH5 = ROOT / "06_outputs" / "phase5"


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.nansum((y - p) ** 2); st = np.nansum((y - np.nanmean(y)) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")


def main() -> int:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    proj = PH5 / "projection_2050_allion.parquet"
    if not proj.exists():
        proj = PH5 / "projection_2050.parquet"
    pj = pd.read_parquet(proj)
    cov = pd.read_parquet(PH5 / "station_covariates.parquet")
    pj["station_id"] = pj["station_id"].astype(str); cov["station_id"] = cov["station_id"].astype(str)
    d = pj.merge(cov, on="station_id", how="inner")
    if "base_level" not in d:
        raise SystemExit("projection table lacks base_level; run run_phase5.py first")
    d["log_level"] = np.log1p(d["base_level"])
    d["block"] = (np.floor(d["latitude"] / 2).astype(int).astype(str) + "_"
                  + np.floor(d["longitude"] / 2).astype(int).astype(str))
    d = pd.concat([d, pd.get_dummies(d["ion"], prefix="ion")], axis=1)
    ion_cols = [c for c in d.columns if c.startswith("ion_")]
    covcols = [c for c in cov.columns if c != "station_id"]
    base = covcols + ["latitude", "longitude"] + ion_cols

    def cv(feats, kind):
        oof = np.full(len(d), np.nan)
        for tri, tei in GroupKFold(5).split(d, groups=d["block"]):
            if kind == "lgb":
                m = lgb.LGBMRegressor(objective="regression", learning_rate=0.05, num_leaves=63,
                                      min_child_samples=40, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, n_estimators=400, random_state=0,
                                      n_jobs=-1, verbose=-1)
            else:
                m = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=0)
            m.fit(d.iloc[tri][feats].astype(float).fillna(-999), d.iloc[tri]["log_level"])
            oof[tei] = m.predict(d.iloc[tei][feats].astype(float).fillna(-999))
        return round(_r2(d["log_level"], oof), 3)

    res = {
        "LightGBM_both": cv(base, "lgb"),
        "RandomForest_both": cv(base, "rf"),
        "LightGBM_500m_only": cv([c for c in base if not c.endswith("_5km")], "lgb"),
        "LightGBM_5km_only": cv([c for c in base if not c.endswith("_500m")], "lgb"),
    }
    (PH5 / "robustness_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
