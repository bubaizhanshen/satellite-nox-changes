#!/usr/bin/env python3
"""Cache native TEMPO footprints and hourly ERA5 winds for a fixed facility panel."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import requests

from compare_tropomi_neighbor_masks import RSIG
from score_no2_overpass_benchmark import NO2


def fetch_swath(site, start, end, destination):
    radius = .35
    dx = radius / np.cos(np.deg2rad(site.latitude))
    bounds = (site.longitude - dx, site.latitude - radius,
              site.longitude + dx, site.latitude + radius)
    params = {
        "SERVICE": "wcs", "VERSION": "1.0.0", "REQUEST": "GetCoverage", "FORMAT": "ascii",
        "COVERAGE": "tempo.l2.no2.vertical_column_troposphere", "MINIMUM_QUALITY": "normal",
        "CORNERS": 1, "ALLOW_NEGATIVE_COUNTS": 1, "MAXIMUM_CLOUD_FRACTION": .15,
        "MAXIMUM_SOLAR_ZENITH_ANGLE": 70, "COMPRESS": 1,
        "BBOX": ",".join(f"{item:.5f}" for item in bounds),
        "TIME": f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z",
    }
    if destination.is_file():
        return destination.stat().st_size, "cached"
    pending = destination.with_suffix(".part")
    error = ""
    for attempt in range(2):
        try:
            with requests.get(RSIG, params=params, stream=True, timeout=(15, 125)) as response:
                response.raise_for_status()
                with pending.open("wb") as sink:
                    for content in response.iter_content(1 << 18):
                        sink.write(content)
            with pending.open("rb") as source:
                compressed = source.read(2) == b"\x1f\x8b"
            opener = gzip.open if compressed else open
            with opener(pending, "rb") as table:
                header = table.readline(1500)
                if NO2.encode() not in header or b"Longitude_SW(deg)" not in header:
                    raise ValueError("RSIG did not return a native NO2 table")
                if not table.readline(1500):
                    raise ValueError("RSIG returned no observations")
            if not compressed:
                pending.write_bytes(gzip.compress(pending.read_bytes()))
            pending.replace(destination)
            return destination.stat().st_size, "downloaded"
        except (requests.RequestException, OSError, ValueError) as exc:
            pending.unlink(missing_ok=True)
            error = str(exc).split(" for url:")[0]
            if attempt == 0:
                sleep(4)
    return 0, error


def fetch_wind(site, years, destination):
    if destination.is_file():
        return destination.stat().st_size, "cached"
    params = {
        "latitude": site.latitude, "longitude": site.longitude, "models": "era5",
        "start_date": f"{min(years) - 1}-12-31", "end_date": f"{max(years) + 1}-01-01",
        "timezone": "UTC", "wind_speed_unit": "ms", "elevation": "nan", "cell_selection": "nearest",
        "hourly": "wind_speed_100m,wind_direction_100m",
    }
    for attempt in range(3):
        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=(12, 45))
            response.raise_for_status()
            result = response.json()
            expected = 24 * (1 + (pd.Timestamp(params["end_date"]) - pd.Timestamp(params["start_date"])).days)
            if len(result.get("hourly", {}).get("time", [])) != expected:
                raise ValueError("ERA5 does not cover the requested hours")
            pending = destination.with_suffix(".part")
            pending.write_text(json.dumps(result, separators=(",", ":")))
            pending.replace(destination)
            return destination.stat().st_size, "downloaded"
        except (requests.RequestException, ValueError) as exc:
            error = str(exc).split(" for url:")[0]
            if attempt < 2:
                sleep(4 * (attempt + 1))
    return 0, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--months-per-request", type=int, choices=[1, 2, 3, 4, 6, 12], default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cohort", choices=["recovered", "other", "all"], default="all")
    parser.add_argument("--wind-only", action="store_true")
    parser.add_argument("--sites", type=int, nargs="+")
    parser.add_argument("--start-months", type=int, nargs="+")
    parser.add_argument("--missing-from", type=Path, help="Repair failed blocks in this acquisition log.")
    args = parser.parse_args()
    panel = pd.read_csv(args.annual / "panel.csv")
    recovery = pd.read_csv(args.annual / "recovery_diagnostic_sites.csv")
    recovered = recovery.loc[recovery.scope.eq("recovered"), "location"].unique()
    panel["recovered"] = panel.location.isin(recovered)
    if args.cohort != "all":
        panel = panel.loc[panel.recovered.eq(args.cohort == "recovered")]
    if args.sites:
        panel = panel.loc[panel.location.isin(args.sites)]
    swaths, wind = args.work / "swaths", args.work / "wind"
    for directory in (swaths, wind):
        directory.mkdir(parents=True, exist_ok=True)

    months = args.start_months or list(range(1, 13, args.months_per_request))
    blocks = [pd.Period(f"{year}-{month:02d}", freq="M") for month in months for year in args.years]
    sites = list(panel.sort_values(["recovered", "location"], ascending=[False, True]).itertuples())
    name = f"{'wind' if args.wind_only else 'swaths'}_{args.cohort}_{args.months_per_request}m"
    if args.missing_from:
        name += "_repair"
    if args.start_months:
        name += "_" + "-".join(map(str, args.start_months))
    results = []

    def download(task):
        site, block = task
        start_clock = perf_counter()
        if block is None:
            first = str(min(args.years) - 1) + "-12-31"
            last = str(max(args.years) + 1) + "-01-01"
            file = wind / f"{site.location}.json"
            size, status = fetch_wind(site, args.years, file)
        else:
            start, end = block.start_time, (block + args.months_per_request - 1).end_time
            first, last = str(start.date()), str(end.date())
            file = swaths / f"{site.location}_{first}_{last}.tsv.gz"
            size, status = fetch_swath(site, start, end, file)
        return {"location": site.location, "first": first, "last": last, "file": file.name,
                "bytes": size, "status": status, "wall_seconds": round(perf_counter() - start_clock, 2)}

    tasks = [(site, None) for site in sites] if args.wind_only else [(site, block) for block in blocks for site in sites]
    if args.missing_from:
        receipt = pd.read_csv(args.missing_from)
        failed = receipt.loc[receipt.bytes.eq(0)].copy()
        failed["first"] = pd.to_datetime(failed["first"]).dt.to_period("M")
        failed["last"] = pd.to_datetime(failed["last"]).dt.to_period("M")
        missed = {key: group for key, group in failed.groupby("location")}
        tasks = [(site, block) for site, block in tasks if site.location in missed
                 and ((missed[site.location]["first"] <= block)
                      & (missed[site.location]["last"] >= block + args.months_per_request - 1)).any()]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(download, task) for task in tasks]):
            row = future.result()
            results.append(row)
            if len(results) % 5 == 0 or len(results) == len(tasks) or row["bytes"] == 0:
                print(f"{len(results)}/{len(tasks)}", row, flush=True)
            pd.DataFrame(results).to_csv(args.work / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
