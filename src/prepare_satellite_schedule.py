"""Prepare earlier calibration scenes or target scenes selected by imagery alone."""

import argparse
import calendar
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from benchmark_nox_annual import orbit_keys
from benchmark_nox_reporting import ARCHIVE_FIELDS


def sampled_dates(year):
    generator = np.random.default_rng(51803)
    return [f"{year}{month:02d}{day:02d}" for month in range(1, 13)
            for day in sorted(generator.choice(np.arange(1, calendar.monthrange(year, month)[1] + 1),
                                               size=4, replace=False))]


def calibration_schedule(archive, inventory, year):
    fields = ["view_band", "mean_mass_2h", "latitude", "longitude", "wind_u", "wind_v"]
    rows = pd.read_parquet(archive, filters=[("year", "==", year)]).dropna(subset=fields)
    dates = sampled_dates(year)
    known = inventory.index[inventory.annual.notna()]
    keep = orbit_keys(rows.file_path).granule_date.isin(dates) & rows.location.isin(known)
    return rows.loc[keep].sort_values(["location", "utc_time", "file_path"]).copy()


def image_schedule(archive, plants, years, exclude=None):
    columns = ["location", "latitude", "longitude", "State", "Facility_Name"]
    sites = pd.read_csv(plants, usecols=columns).set_index("location", verify_integrity=True)
    rows = pd.read_csv(archive, usecols=ARCHIVE_FIELDS, low_memory=False)
    rows = rows.loc[rows.location.isin(sites.index)].copy()
    rows["utc_time"] = pd.to_datetime(rows.utc_time, format="ISO8601", utc=True)
    rows["year"] = rows.utc_time.dt.year
    rows["wind_m_s"] = np.hypot(rows.wind_u, rows.wind_v)
    rows["view_band"] = pd.cut(rows.sensor_zenith_angle, [-1, 25, 35, 45, 90],
                              labels=["0-25", "25-35", "35-45", "45-90"])
    eligible = (rows.year.isin(years) & rows.wind_m_s.between(2, 15)
                & rows.view_band.notna() & rows.annual_nox_emission.notna())
    rows = rows.loc[eligible].join(sites, on="location")
    if exclude is not None:
        keys = ["location", "utc_time", "file_path"]
        previous = pd.read_parquet(exclude, columns=keys)
        rows = rows.merge(previous, on=keys, how="left", indicator=True, validate="one_to_one")
        if rows._merge.eq("both").sum() != len(previous):
            raise ValueError("Some previously scheduled visits are absent from the archive")
        rows = rows.loc[rows._merge.eq("left_only")].drop(columns="_merge")
    return rows.sort_values(["location", "utc_time"]).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="Matched population Parquet for calibration, raw archive CSV for image-only targets")
    parser.add_argument("--screen", type=Path, required=True, help="Directory containing the frozen inventory and screen")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibration-year", type=int)
    mode.add_argument("--image-years", type=int, nargs="+")
    parser.add_argument("--plants", type=Path, help="Plant CSV with coordinates, name, state, and location")
    parser.add_argument("--exclude-schedule", type=Path, help="Existing schedule when preparing only additional scenes")
    args = parser.parse_args()
    inventory_name = "historical_screening_inventory.csv"
    inventory = pd.read_csv(args.screen / inventory_name).set_index("Facility_ID")
    start = 0
    if args.calibration_year:
        visits = calibration_schedule(args.archive, inventory, args.calibration_year)
    else:
        if args.plants is None:
            parser.error("--image-years requires --plants")
        visits = image_schedule(args.archive, args.plants, args.image_years, args.exclude_schedule)
        if args.exclude_schedule:
            start = int(pd.read_parquet(args.exclude_schedule, columns=["visit_id"]).visit_id.max()) + 1
    if visits.empty or inventory.annual.reindex(visits.location).isna().any():
        raise ValueError("No visits, or a scheduled site has no 2019 source inventory")
    visits["month"] = visits.utc_time.dt.strftime("%Y-%m")
    visits.insert(0, "visit_id", np.arange(start, start + len(visits)))
    blocks = visits.groupby(["location", "month"]).agg(scheduled=("visit_id", "size"),
                                                       latitude=("latitude", "first"), longitude=("longitude", "first"))
    blocks = blocks.reset_index()
    blocks.insert(0, "block_id", np.arange(len(blocks)))
    visits = visits.merge(blocks[["location", "month", "block_id"]], on=["location", "month"], validate="many_to_one")
    args.output.mkdir(parents=True, exist_ok=True)
    visits.to_parquet(args.output / "visits.parquet", index=False)
    blocks.to_csv(args.output / "download_schedule.csv", index=False)
    for name in (inventory_name, "frozen_screen.json"):
        source = args.screen / name
        destination = args.output / name
        if source.resolve() != destination.resolve():
            copyfile(source, destination)
    print(f"Scheduled {len(visits):,} visits at {visits.location.nunique()} plants", flush=True)


if __name__ == "__main__":
    main()
