# Annual power-plant NOₓ changes from satellite observations

Core analysis code for comparing satellite, fuel, and earlier stack records in estimates of annual emission changes at U.S. power plants.

| Analysis | Entry points |
| --- | --- |
| Hourly stack records and plume-label matching | `prepare_tropomi_cems_ground.py`, `benchmark_nox_reporting.py`, `assess_nox_catalog_sampling.py` |
| Native TROPOMI retrievals and annual comparisons | `fetch_temis_nox_native.py`, `benchmark_nox_annual.py`, `compare_nox_annual_retrievals.py` |
| Sampling controls and coverage diagnostics | `check_selection_assumptions.py`, `audit_nox_recovered_facilities.py` |
| Fuel estimates, satellite correction, and decline rules | `prepare_eia_fuel.py`, `estimate_annual_changes.py`, `prepare_satellite_schedule.py` |
| TEMPO scan aggregation and stack validation | `fetch_tempo_daily_panel.py`, `retrieve_tempo_daily_panel.py`, `score_tempo_daily_panel.py`, `check_tempo_reference.py` |
| Fuel model and paired satellite correction | `read_eia_activity.py`, `paired_change.py`, `satellite_fuel_baselines.py` |
| Decline calls, later year pair, and TEMPO | `validate_decisions.py`, `validate_next_year.py`, `tempo_check.py` |
| Annual and monthly benchmarks | `benchmark_fuel_models.py`, `benchmark_monthly_satellite.py` |
| Annual flux divergence and paired uncertainty | `annual_flux_divergence.py`, `compare_annual_flux.py`, `interval_coverage.py`, `label_sampling_sensitivity.py` |

## Setup

Run `python -m pip install -r requirements.txt` with Python 3.13. Each entry point in `src/` accepts `--help`.

## Input data

- The plant list, observations, plume labels, and city list are from the [TROPOMI archive on Zenodo](https://doi.org/10.5281/zenodo.21466576). Native TROPOMI pixels are from [TEMIS](https://www.temis.nl/airpollution/no2.php).
- Hourly EPA stack records are available from [CAMPD](https://campd.epa.gov/) and [PUDL](https://docs.catalyst.coop/pudl/en/v2026.1.0/data_access.html). Use the study's PUDL snapshot and the 2019 plant inventory from [EPA eGRID](https://www.epa.gov/egrid/detailed-data).
- Native TEMPO data are available from [EPA RSIG](https://www.epa.gov/hesc/web-access-rsig-data), and winds are from [ERA5](https://doi.org/10.24381/cds.adbb2d47).

`prepare_tropomi_cems_ground.py` creates yearly facility-hour Parquet files. Use `--first-utc-hour 0` for full-year references. The `--extended` route of `benchmark_nox_reporting.py` reads the archive CSV under `archive/` and the yearly stack files under `ground/` in the specified work directory. `assess_nox_catalog_sampling.py` uses the matched archive and hourly records to construct the full-year reference and shutdown histories.

## Annual comparison

The prepared schedule in `inputs/annual/` contains `panel.csv`, `visits.parquet`, and `download_schedule.csv`. The `prepare --all-eligible` and `calibrate` stages of `benchmark_nox_annual.py` construct the schedule and screen parameters from the matched archive and separate development data. Pass the local eGRID workbook with `--inventory`.

```bash
python src/fetch_temis_nox_native.py --visits inputs/annual/visits.parquet --destination work/temis
python src/benchmark_nox_annual.py score --provider temis --work work --output inputs/annual --raw inputs/raw --ground inputs/ground
python src/compare_nox_annual_retrievals.py --provider temis --work work --output inputs/annual --ground inputs/ground --persistence-source inputs/archive/full_archive/population.parquet --stop-history inputs/archive/full_archive/continuous_stops.parquet --results results/annual
```

The annual comparison reports coverage and paired errors against complete-year stack records. The all-image method averages signed rates over usable overpasses, sets negative annual means to zero, and subtracts the two annual estimates.

For the stack-only comparison of plume screening, compare `reference_selected_kg_h_change` with `reference_all_hours_kg_h_change` in `plant_annual_changes.csv` at plants with both `all_signed` and `archive_detected_only` estimates. Compute paired errors separately for the Gaussian and cross-sectional retrievals.

## Fuel and satellite estimates

Pass the [EIA-923](https://www.eia.gov/electricity/data/eia923/index.php) Schedule 2–5 archives as `f923_<year>.zip` to `prepare_eia_fuel.py`. The input stack files supply the facility IDs. The training and evaluation plant CSVs use `location` (EPA facility ID) and `State`. To repeat the training-roster sensitivity, use distinct plants in the matched TROPOMI archive through the year before evaluation. The estimator derives state folds unless the evaluation CSV supplies a `fold` column. For the 2024–2025 cohort, use the contiguous U.S. states from the [EPA–EIA crosswalk](https://www.epa.gov/power-sector/power-sector-data-crosswalk) with the 2024 and 2025 [PUDL 2026.10.0](https://docs.catalyst.coop/pudl/en/v2026.10.0/data_access.html) stack files. The evaluation filter requires complete fuel months and 95% valid stack hours in both years.

```bash
python src/prepare_eia_fuel.py --archives inputs/eia --years 2020 2021 2022 2023 2024 2025 \
  --ground inputs/ground/*.parquet --output inputs/fuel
python src/estimate_annual_changes.py --fuel inputs/fuel --ground inputs/ground \
  --training-plants inputs/training_plants.csv --evaluation-plants inputs/annual/panel.csv \
  --years 2023 2024 --reference-ground inputs/ground --output results/fuel
```

For the 2022 TROPOMI fit, run `prepare_satellite_schedule.py --calibration-year 2022` with the matched full-archive population and frozen screen from the annual comparison. Fetch and score the scheduled scenes using the same retrieval commands above. Pass the calibration schedule and scored-block directory to `--earlier-satellite`, and pass evaluation rates through `--target-visits` or `--target-blocks`. To score evaluation imagery without stack records, prepare the schedule with `--image-years 2023 2024` and run the retrieval with `--no-stack`. Omit `--reference-ground` when producing estimates alone. The first-year calibration comparison uses `--same-year-calibration` and release-matched target visits.

## Additional comparisons

For benchmarks using the original fuel model, `read_eia_activity.py` produces the inputs to `paired_change.py` and `validate_next_year.py`. Pass `paired_change_estimates.csv` to `validate_decisions.py --fusion`. The fuel and monthly satellite benchmarks use these estimates as matched comparators.

Run `audit_annual_design.py` for state and region holdouts, coverage and change-threshold tests, and rankings at common false-call allowances.

## Annual flux comparison

Set `NOX_PROJECT_ROOT` to the study input tree. Under this root, the field preparation uses `results/nox_airborne_20261120/annual_retrieval_full_20261121/{scored_visits.parquet,panel.csv}` and `results/nox_airborne_20261120/annual_retrieval/visits.parquet`. Place native orbit cutouts in the `temis/` subdirectories of the corresponding retrieval folders under `work/nox_airborne_20261120/`.

```bash
python src/annual_flux_divergence.py
python src/compare_annual_flux.py --draws 2000
python src/interval_coverage.py results/annual_flux_plant_estimates.csv results/annual_flux_intervals.csv results/annual_flux_interval_coverage.csv
python src/label_sampling_sensitivity.py "$NOX_PROJECT_ROOT/results/nox_airborne_20261120/annual_retrieval_full_20261121/plant_period_estimates.csv" results/annual_label_sampling_sensitivity.csv
```
