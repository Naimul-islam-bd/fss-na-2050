# Data

This folder holds only the small files needed to run the maps. The full dataset is too
large for the repository and is archived separately.

## In this folder

- `na_boundaries.geojson`: United States state and Canadian province boundaries (Natural
  Earth, 1:50m), used as the map backdrop.

## Archived on Zenodo (DOI to be added on publication)

- `master_harmonized.parquet`: 2,512,644 harmonized major-ion observations at 3,944
  stations, 1990 to 2026, with station, ion, datetime, value in mg/L, coordinates, and
  source. Place it at `01_data/processed/master_harmonized.parquet` to reproduce the
  seasonality analysis.
- `station_trends.parquet`, `projection_2050_allion.parquet`,
  `station_covariates.parquet`, `scenario_projection.parquet`, and the CMIP6 deltas:
  the derived station-level tables the analysis and figure scripts read from
  `06_outputs/phase5/`.

## Raw sources (public)

- United States Water Quality Portal, https://www.waterqualitydata.us
- United States Geological Survey, https://www.usgs.gov
- Environment and Climate Change Canada, freshwater quality monitoring data
- Ontario Provincial Water Quality Monitoring Network
- British Columbia Environmental Monitoring System

Geospatial covariates were sampled in Google Earth Engine (MERIT DEM, ESA WorldCover,
WorldClim, TerraClimate, Gridded Population of the World, VIIRS, OpenLandMap). Climate
deltas are from NASA NEX-GDDP-CMIP6.
