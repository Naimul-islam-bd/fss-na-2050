"""Management counterfactual: how much 2050 chloride exceedance is avoidable.

The climate scenarios show that mid-century chloride is set by management, not by
climate. This script turns that into a policy-relevant number. It isolates the
stations that are driving the increase, the ones with a significant rising chloride
trend, and asks what their 2050 outcome would be under two management counterfactuals
applied from a policy start year, against the no-action trend:

  Trend (no action):  each station keeps its measured slope to 2050.
  Stabilization:      the rising stations hold at their policy-year level (slope 0).
  Halved application: the rising stations continue at half their measured slope.

Stations that are not significantly rising keep their own trajectory in every case, so
the difference between scenarios is purely the avoidable, management-driven part. This
is an illustrative counterfactual, not a prediction of policy.

Run from the project root (or a folder with the phase-5 outputs):
    python run_management_scenario.py
Inputs: projection_2050_allion.parquet, station_trends.parquet.
Outputs (06_outputs/phase5/): management_scenario.parquet, management_summary.json;
figure Figure_9 (chloride exceedance under management counterfactuals).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import joh_style as S

warnings.filterwarnings("ignore")

POLICY_START = 2025
TARGET = 2050
THRESHOLDS = [120, 230, 860]
SIG = 0.05


def project(base_level, slope, base_year, policy_slope_after):
    """Concentration in 2050: measured slope to the policy year, then policy slope after."""
    yr0 = np.minimum(base_year, POLICY_START)
    pre = slope * (POLICY_START - yr0)                       # measured slope up to policy start
    post = policy_slope_after * (TARGET - POLICY_START)      # policy slope after
    return base_level * np.exp(pre + post)


def main():
    import matplotlib.pyplot as plt

    pj = pd.read_parquet(S.find("projection_2050_allion.parquet"))
    tr = pd.read_parquet(S.find("station_trends.parquet"))[["station_id", "ion", "mk_p"]]
    for d in (pj, tr):
        d["station_id"] = d["station_id"].astype(str)
    cl = pj[pj.ion == "cl"].merge(tr[tr.ion == "cl"].drop(columns="ion"), on="station_id", how="left")

    rising = (cl["slope_log"] > 0) & (cl["mk_p"] < SIG)     # the manageable, significantly rising set
    cl["rising"] = rising

    def scen(policy_factor):
        # rising stations get the policy slope; everyone else keeps their measured slope
        slope_after = np.where(rising, cl["slope_log"] * policy_factor, cl["slope_log"])
        return project(cl["base_level"].values, cl["slope_log"].values,
                       cl["base_year"].values, slope_after)

    cl["cl_2050_trend"] = scen(1.0)          # no action
    cl["cl_2050_halved"] = scen(0.5)         # half application on rising stations
    cl["cl_2050_stabilized"] = scen(0.0)     # hold rising stations at policy-year level

    def exc(col):
        return {str(t): round(float((cl[col] > t).mean() * 100), 1) for t in THRESHOLDS}
    res = {
        "n_chloride_stations": int(len(cl)),
        "n_rising_significant": int(rising.sum()),
        "share_rising_pct": round(float(rising.mean() * 100), 1),
        "policy_start": POLICY_START,
        "exceedance": {
            "recent": exc("base_level"),
            "2050_trend": exc("cl_2050_trend"),
            "2050_halved": exc("cl_2050_halved"),
            "2050_stabilized": exc("cl_2050_stabilized"),
        },
    }
    # avoided exceedance at the EPA chronic threshold (230)
    t = "230"
    res["avoided_at_230_pp"] = {
        "halved_vs_trend": round(res["exceedance"]["2050_trend"][t] - res["exceedance"]["2050_halved"][t], 1),
        "stabilized_vs_trend": round(res["exceedance"]["2050_trend"][t] - res["exceedance"]["2050_stabilized"][t], 1),
    }
    cl[["station_id", "latitude", "longitude", "base_level", "rising",
        "cl_2050_trend", "cl_2050_halved", "cl_2050_stabilized"]].to_parquet(
        S.find("projection_2050_allion.parquet").parent / "management_scenario.parquet", index=False)
    (S.find("projection_2050_allion.parquet").parent / "management_summary.json").write_text(
        json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # Figure 9: exceedance under the counterfactuals
    labels = ["Aquatic-life\nchronic (120)", "EPA chronic\n(230)", "EPA acute\n(860)"]
    series = [("Recent", "recent"), ("2050 no action", "2050_trend"),
              ("2050 halved", "2050_halved"), ("2050 stabilized", "2050_stabilized")]
    x = np.arange(len(THRESHOLDS)); w = 0.2
    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    for j, (lab, key) in enumerate(series):
        vals = [res["exceedance"][key][str(t)] for t in THRESHOLDS]
        ax.bar(x + (j - 1.5) * w, vals, w, color=S.RAMP[j], label=lab,
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Chloride stations exceeding (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    S.save(fig, "Figure_6")
    print("done management scenario")


if __name__ == "__main__":
    main()
