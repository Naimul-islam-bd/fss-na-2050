"""Baseline salinization model for FSS-NA-2050.

A single multitask gradient boosted model (LightGBM) predicts the log annual
mean concentration for all six ions, with ion identity passed as a categorical
feature. The split is temporal: the model trains on early years and is tested on
held out later years, which is the honest test for a forecasting task. A naive
persistence forecast (this year equals the last observed year) is reported
alongside so the model is judged against the obvious benchmark rather than
against zero.

Metrics are reported per ion on the log scale (MAE, RMSE, R2) and on the native
mg/L scale (median absolute error), because a good log fit can still hide large
absolute error on the most concentrated sites.

This is the predictability baseline. Projection to 2050 uses the per station
trend and static catchment covariates and is built in the next phase; the
trailing slope feature here is the basis for that step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["lag1", "lag2", "lag3", "roll3", "roll5", "trail_slope",
            "years_seen", "year_c", "n_obs_year", "latitude", "longitude",
            "ion", "source", "country", "state_prov", "site_type"]
CATEGORICAL = ["ion", "source", "country", "state_prov", "site_type"]
TARGET = "log_value"
ION_KEYS = ("cl", "na", "ca", "mg", "so4", "alk")


def _metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def temporal_split(feat: pd.DataFrame, train_end=2013, valid_end=2017):
    """Rows need a lag1 so the persistence benchmark is defined everywhere."""
    d = feat[feat["lag1"].notna()].copy()
    for c in CATEGORICAL:
        d[c] = d[c].astype("category")
    train = d[d["year"] <= train_end]
    valid = d[(d["year"] > train_end) & (d["year"] <= valid_end)]
    test = d[d["year"] > valid_end]
    return train, valid, test


def run(feat: pd.DataFrame, outdir: Path) -> dict:
    import lightgbm as lgb
    outdir.mkdir(parents=True, exist_ok=True)
    train, valid, test = temporal_split(feat)

    params = dict(objective="regression", metric="l2", learning_rate=0.05,
                  num_leaves=63, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, n_estimators=1500,
                  random_state=0, n_jobs=-1, verbose=-1)
    model = lgb.LGBMRegressor(**params)
    model.fit(train[FEATURES], train[TARGET],
              eval_set=[(valid[FEATURES], valid[TARGET])],
              eval_metric="l2",
              callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
              categorical_feature=CATEGORICAL)

    test = test.copy()
    test["pred"] = model.predict(test[FEATURES])
    test["persist"] = test["lag1"]  # naive benchmark on the log scale

    summary = {"n_train": int(len(train)), "n_valid": int(len(valid)),
               "n_test": int(len(test)), "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
               "per_ion": {}, "overall": {}}

    for ion in ION_KEYS:
        s = test[test["ion"] == ion]
        if len(s) < 50:
            continue
        m_mae, m_rmse, m_r2 = _metrics(s[TARGET].to_numpy(), s["pred"].to_numpy())
        p_mae, p_rmse, p_r2 = _metrics(s[TARGET].to_numpy(), s["persist"].to_numpy())
        native_mae = float(np.median(np.abs(np.expm1(s["pred"]) - np.expm1(s[TARGET]))))
        summary["per_ion"][ion] = {
            "n_test": int(len(s)),
            "model": {"MAE_log": round(m_mae, 4), "RMSE_log": round(m_rmse, 4), "R2_log": round(m_r2, 4)},
            "persistence": {"MAE_log": round(p_mae, 4), "RMSE_log": round(p_rmse, 4), "R2_log": round(p_r2, 4)},
            "model_median_abs_err_mgL": round(native_mae, 3),
        }

    mm = _metrics(test[TARGET].to_numpy(), test["pred"].to_numpy())
    pm = _metrics(test[TARGET].to_numpy(), test["persist"].to_numpy())
    summary["overall"] = {
        "model": {"MAE_log": round(mm[0], 4), "RMSE_log": round(mm[1], 4), "R2_log": round(mm[2], 4)},
        "persistence": {"MAE_log": round(pm[0], 4), "RMSE_log": round(pm[1], 4), "R2_log": round(pm[2], 4)},
    }

    imp = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False)
    summary["feature_importance"] = imp.assign(
        gain_pct=(imp["gain"] / imp["gain"].sum() * 100).round(1)
    )[["feature", "gain_pct"]].to_dict("records")

    (outdir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    feat = pd.read_parquet(sys.argv[1])
    s = run(feat, Path(sys.argv[2] if len(sys.argv) > 2 else "."))
    print(json.dumps(s, indent=2))
