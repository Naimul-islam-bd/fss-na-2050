# Continental freshwater salinization forecast for the United States and Canada to 2050

Code and reproducible workflow for a bi-national, continental analysis of freshwater
salinization across the United States and Canada. The study harmonizes five long-term
water-quality networks, quantifies multi-decadal trends in the six major salinization
ions, attributes their drivers with interpretable machine learning under spatial
cross-validation, fingerprints salt sources with ion ratios, classifies stations into
salinization regimes, and projects salinization and threshold exceedance to 2050 under
trend, climate, and management scenarios.

Author: Naimul Islam, Independent Researcher, Dhaka, Bangladesh
ORCID: https://orcid.org/0009-0002-3442-8980

## Summary of the study

- 2,512,644 major-ion records at 3,944 long-term stations, 1990 to 2026, for chloride,
  sodium, calcium, magnesium, sulfate, and alkalinity.
- Chloride rises significantly at 25.9 percent of stations and falls at 11.0 percent;
  sodium mirrors it, and sulfate declines with reduced atmospheric deposition.
- Salinity level is well predicted from catchment and climate drivers (spatial
  cross-validation R2 0.58), but its rate of change is not (R2 about 0.05), which points
  to local de-icing management as the control on the trend.
- The sodium-to-chloride ratio marks a halite road-salt signature across the cold,
  developed northeast; a five-regime typology summarizes the continental structure.
- By 2050 the median chloride station gains about 11 percent and the share exceeding the
  United States chronic aquatic-life criterion rises from 8.1 to 12.1 percent. Climate
  scenarios move this little; stabilizing the rising stations would avoid about three
  quarters of the increase.

## Data sources

Raw records are public and were retrieved from the United States Water Quality Portal,
the United States Geological Survey, Environment and Climate Change Canada, the Ontario
Provincial Water Quality Monitoring Network, and the British Columbia Environmental
Monitoring System. Geospatial covariates were sampled in Google Earth Engine from MERIT
DEM, ESA WorldCover, WorldClim, TerraClimate, the Gridded Population of the World, VIIRS
night-time lights, and OpenLandMap soils. Climate deltas are from NASA NEX-GDDP-CMIP6.

The harmonized dataset and the derived station-level tables are archived on Zenodo (DOI
to be added on publication). Large raw and intermediate files are not stored in this
repository; see `data/README.md`.

## Repository layout

    code/          all analysis and pipeline scripts (see below)
    data/          na_boundaries.geojson and a pointer to the Zenodo data archive
    results/       small result summaries (JSON and CSV) that back the figures and tables
    requirements.txt
    CITATION.cff
    LICENSE

## Code

Data pull and harmonization:

    code/run_full_pull.py            resume-able pull of all five networks
    code/wqp_ingest.py               Water Quality Portal and USGS ingestion
    code/eccc_ingest.py              Environment and Climate Change Canada ingestion
    code/harmonize.py                harmonization to a common long schema
    code/features.py                 annual aggregation and feature construction

Modeling and projection:

    code/model_baseline.py           baseline concentration model
    code/run_phase4.py               baseline run
    code/run_phase5.py               trends, level and trend models, SHAP, projection
    code/extract_covariates_gee.py   Earth Engine covariate extraction
    code/extract_cmip6_grid.py       CMIP6 2050 deltas on a coarse grid, interpolated
    code/run_scenarios.py            climate-scenario projection
    code/run_robustness.py           algorithm and buffer-scale robustness checks

Additional analyses (ion ratios, seasonality, management):

    code/joh_style.py                shared plotting style and file finder
    code/analysis_ratios_typology.py ion-ratio fingerprinting and regime typology
    code/analysis_seasonality.py     winter versus summer chloride signal
    code/run_management_scenario.py  avoidable-exceedance management counterfactual

Figures and manuscript:

    code/make_figures_joh.py         main figures 1, 2, 5, 7 and graphical abstract
    code/make_si_maps.py             per-ion supplementary maps S1 and S2
    code/build_manuscript.py         builds the Word manuscript, highlights, supplement

## Reproducing the analysis

1. Install dependencies:

       pip install -r requirements.txt

2. Obtain the data. Download the harmonized dataset from the Zenodo archive and place it
   as `01_data/processed/master_harmonized.parquet`, or re-run the pull with
   `code/run_full_pull.py` (needs internet and, for covariates, a Google Earth Engine
   account). Keep `na_boundaries.geojson` in the working directory.

3. Run, from the project root, in this order:

       python code/run_phase5.py
       python code/extract_cmip6_grid.py
       python code/run_scenarios.py
       python code/run_robustness.py
       python code/analysis_ratios_typology.py
       python code/analysis_seasonality.py
       python code/run_management_scenario.py
       python code/make_figures_joh.py
       python code/make_si_maps.py
       python code/build_manuscript.py

The scripts locate their inputs in `06_outputs/phase5/` and `01_data/processed/`
automatically. Figures are written to `figures_joh/` and the manuscript to the working
directory.

## Methods in brief

Trends use the Theil-Sen slope of the log annual mean with the Mann-Kendall test.
Drivers use gradient-boosted trees with ion identity as a categorical feature, validated
with spatial-block cross-validation on two-degree cells, and SHAP for attribution.
Projection continues each station's observed slope to 2050 with a Theil-Sen confidence
interval; a CMIP6 overlay gives a climate response; a management counterfactual reduces
the slope of the significantly rising stations from a policy start year.

## Citation

If you use this code or the derived data, please cite the archived dataset and the
article (see `CITATION.cff`).

## License

Code is released under the MIT License (see `LICENSE`). The derived datasets on Zenodo
are released under CC-BY-4.0. Raw records remain under the terms of their source agencies.

## Contact

Naimul Islam, naimul.islam.bangladesh@gmail.com, https://github.com/Naimul-islam-bd
