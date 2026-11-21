"""Compare TEMPO estimates to the observed window and to full solar-day stack records."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from score_no2_overpass_benchmark import reference_mass
from score_tempo_daily_panel import annual_changes, check_against_full_year, stack_inputs


METHODS = ("midday_scan", "scan_average", "joint_background")
REFERENCES = ("cems_at_scans", "cems_solar_day")


def match_day_stack(daily, sites, hourly):
    longitude = daily.location.map(sites.longitude)
    end = pd.to_datetime(daily.date, utc=True).dt.normalize() + pd.Timedelta(days=1)
    end -= pd.to_timedelta(longitude / 15, unit="h")
    full_day = [reference_mass(hourly[site], time, 0, 24)
                for site, time in zip(daily.location, end, strict=True)]
    return daily.assign(cems_solar_day=full_day).dropna(subset=[*METHODS, *REFERENCES])


def daily_comparison(daily):
    rng = np.random.default_rng(14368)
    result = []
    for year, data in daily.groupby("year"):
        for reference in REFERENCES:
            absolute = data[list(METHODS)].sub(data[reference], axis=0).abs()
            by_plant = absolute.groupby(data.location).mean()
            baseline = by_plant.midday_scan
            for method in METHODS:
                paired = by_plant[method] - baseline
                draws = rng.choice(paired.to_numpy(), size=(4000, len(paired))).mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
                result.append({
                    "year": year, "reference": reference, "method": method,
                    "plants": len(by_plant), "plant_days": len(data), "mae_kg_h": by_plant[method].mean(),
                    "difference_from_one_scan_kg_h": paired.mean(), "low": low, "high": high,
                })
    return pd.DataFrame(result)


def annual_comparison(daily, monthly, annual, panel):
    records = []
    for reference in REFERENCES:
        period, _ = annual_changes(daily.assign(cems_at_scans=daily[reference]), monthly, panel)
        comparison = check_against_full_year(period, annual).query('method=="cems_at_scans"')
        records.append(comparison.assign(reference=reference))
    return pd.concat(records, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-original", type=Path, required=True)
    parser.add_argument("--ground", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    daily = pd.read_csv(args.annual_original / "tempo_daily/daily_rates.csv", parse_dates=["date"])
    sites = pd.read_csv(args.annual_original / "panel.csv").set_index("location")
    hourly, monthly, full_year = stack_inputs(args.ground, daily.location.unique())
    valid = match_day_stack(daily, sites, hourly)
    comparison = daily_comparison(valid)
    panel = sites.loc[sites.index.isin(valid.location)].reset_index().assign(recovered=True)
    held_dates = annual_comparison(valid, monthly, full_year, panel)
    args.output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output / "tempo_solar_day_errors.csv", index=False)
    held_dates.to_csv(args.output / "tempo_solar_day_annual_changes.csv", index=False)
    print(comparison.round(2).to_string(index=False), flush=True)
    if not held_dates.empty:
        print(held_dates.assign(error=(held_dates.delta - held_dates.cems_delta).abs()).groupby("reference")
              .agg(plants=("location", "size"), annual_mae=("error", "mean")).round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
