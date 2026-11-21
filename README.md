# Annual power-plant NOₓ changes from satellite observations

Core analysis code for comparing and combining satellite and fuel-based estimates of annual NOₓ changes at U.S. power plants.

| Analysis | Entry points |
| --- | --- |
| Hourly stack records and plume-label matching | `prepare_tropomi_cems_ground.py`, `benchmark_nox_reporting.py`, `assess_nox_catalog_sampling.py` |
| Native TROPOMI retrievals and annual comparisons | `fetch_temis_nox_native.py`, `benchmark_nox_annual.py`, `compare_nox_annual_retrievals.py` |
| Sampling controls and coverage diagnostics | `check_selection_assumptions.py`, `audit_nox_recovered_facilities.py` |
| TEMPO scan aggregation and stack validation | `fetch_tempo_daily_panel.py`, `retrieve_tempo_daily_panel.py`, `score_tempo_daily_panel.py`, `check_tempo_reference.py` |
| Fuel model and paired satellite correction | `read_eia_activity.py`, `paired_change.py`, `satellite_fuel_baselines.py` |
| Decline calls, later year pair, and TEMPO | `validate_decisions.py`, `validate_next_year.py`, `tempo_check.py` |
| Annual and monthly benchmarks | `benchmark_fuel_models.py`, `benchmark_monthly_satellite.py` |

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

## Fuel records and paired changes

Place the final [EIA-923 workbooks](https://www.eia.gov/electricity/data/eia923/index.php) in the working directory as `f923_<year>.zip`. Run `read_eia_activity.py --panel <plant-list.csv> --years 2022 2023 2024` for the TROPOMI panel. For the later validation, use the full stack facility list for 2024–2025 and the [EPA–EIA crosswalk](https://www.epa.gov/power-sector/power-sector-data-crosswalk) to verify state codes. The plant list has a `location` column with the facility identifier.

Run `paired_change.py` on the annual retrieval outputs, full-year stack files, and monthly heat inputs. Pass its `paired_change_estimates.csv` to `validate_decisions.py --fusion` with the same inputs. For the 2024–2025 evaluation, pass the two yearly files from [PUDL v2026.10.0](https://docs.catalyst.coop/pudl/en/v2026.10.0/data_access.html) after plant-hour aggregation with `prepare_tropomi_cems_ground.py`.

For `benchmark_fuel_models.py`, use `decisions_2023_2024.csv` and `external_2024_2025_estimates.csv` from the respective validation scripts. An optional TabICLv2 comparison uses `tabicl==2.2.0` and its public [regression checkpoint](https://huggingface.co/jingang/TabICL). Pass the downloaded file via `--foundation-checkpoint`.

For the monthly satellite benchmark, pass `decisions_2023_2024.csv` to `--primary` and `workflow_2023_2024.csv` to `--paired`.

## Geographic and decision checks

Run `audit_annual_design.py` with the annual retrieval outputs, fuel-model evaluation files, later-year estimates, and calendar-year stack files. It writes state-group membership, point-estimate rankings at common false-call allowances, whole-region holdouts, and coverage and change-threshold checks.
