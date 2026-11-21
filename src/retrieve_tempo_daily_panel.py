#!/usr/bin/env python3
"""Retrieve signed NOx fluxes from independent and jointly fitted TEMPO scans."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_nox_facility_overlap import AGES, plume_kernels, plume_path, wind_vectors
from score_no2_overpass_benchmark import CORNERS, NO2, PIXEL_TIME, polygon_area, xy


LIFETIME = 3.0
DISPERSION = .13
BACKGROUND_SCALE_KM = 9.0
BACKGROUND_PENALTY = .03
SCAN_SEPARATION = pd.Timedelta(minutes=12)
PLUME_WEIGHTS = np.exp(-AGES / LIFETIME) * np.minimum(AGES / .25, 1)
GRID = np.array(np.meshgrid([-18, -6, 6, 18], [-18, -6, 6, 18])).reshape(2, -1).T
LON = "LONGITUDE(deg)"
LAT = "LATITUDE(deg)"


@dataclass
class FittedScan:
    time: pd.Timestamp
    n_pixels: int
    direction: float
    unique_fraction: float
    kernels: np.ndarray
    background: np.ndarray
    observed: np.ndarray
    rates: np.ndarray
    projected_background: np.ndarray
    visual: dict | None


def candidates(location, lon, lat, inventory):
    offsets = xy(inventory.Longitude.to_numpy(), inventory.Latitude.to_numpy(), lon, lat)
    distance = np.linalg.norm(offsets, axis=1)
    select = inventory.Facility_ID.ne(location) & inventory.annual.ge(100) & (distance < 30)
    ids = inventory.loc[select, "Facility_ID"].to_numpy()
    return np.r_[location, ids], np.vstack([[0., 0.], offsets[select]])


def fit_scan(rows, lon, lat, origins, wind, keep_scene=False):
    rows = rows.drop_duplicates([LON, LAT])
    close = np.linalg.norm(xy(rows[LON].to_numpy(), rows[LAT].to_numpy(), lon, lat), axis=1) < 6
    if close.sum() < 4:
        return None
    time = rows.loc[close, PIXEL_TIME].sort_values().iloc[close.sum() // 2]
    hour = time.floor("h")
    section = wind.loc[hour - pd.Timedelta(hours=3):hour + pd.Timedelta(hours=1)]
    if section.shape[0] != 5 or not np.isfinite(section).all(axis=None):
        return None
    path, speed, direction = plume_path(time, section)
    if not 2 <= speed <= 16:
        return None

    corners = xy(rows[[f"Longitude_{c}(deg)" for c in CORNERS]].to_numpy(),
                 rows[[f"Latitude_{c}(deg)" for c in CORNERS]].to_numpy(), lon, lat)
    centers = corners.mean(axis=1)
    observed = rows[NO2].to_numpy() / 1e15
    good = (np.abs(centers).max(axis=1) < 29) & np.isfinite(observed) & (polygon_area(corners) > 0)
    if good.sum() < 100:
        return None
    observed, centers, corners = observed[good], centers[good], corners[good]

    kernel = plume_kernels(corners, origins, path, LIFETIME, DISPERSION)
    target = kernel[:, 0]
    distances = np.linalg.norm(centers[:, None, :] - path[None, :18, :], axis=2)
    if (target > .08 * target.max()).sum() < 12 or (distances.min(axis=0) < 5).mean() < .55:
        return None
    if target.max() < .0002:
        return None

    smooth = np.exp(-np.square(centers[:, None, :] - GRID).sum(axis=2) / (2 * BACKGROUND_SCALE_KM ** 2))
    plane = np.column_stack([np.ones(len(centers)), centers / 29])
    basis = np.linalg.qr(plane)[0]
    response = observed - basis @ (basis.T @ observed)
    design = kernel - basis @ (basis.T @ kernel)
    background = smooth - basis @ (basis.T @ smooth)
    norms = np.linalg.norm(design, axis=0)
    useful = np.r_[True, norms[1:] > norms[0] * .02]
    active = design[:, useful]
    norm = norms[useful]
    if norm[0] <= 0:
        return None
    normalized = active / norm
    neighbors = normalized[:, 1:]
    residual = normalized[:, 0] - neighbors @ np.linalg.lstsq(neighbors, normalized[:, 0], rcond=.02)[0]
    unique = float(np.linalg.norm(residual))
    if unique < .2:
        return None
    # Include the native-image plane and every observable inventory plume in every arm.
    coefficients = np.linalg.lstsq(normalized, np.column_stack([response, background]), rcond=.02)[0]
    rates = coefficients[:, 0] / norm
    image = None
    if keep_scene:
        image = {"corners": corners, "observed": observed, "plume": kernel[:, useful],
                 "smooth": smooth, "plane": plane, "path": path, "origins": origins[useful]}
    return FittedScan(time, len(response), direction, unique, active, background, response,
                      rates, coefficients[:, 1:] / norm[:, None], image)


def shared_background(scans):
    # Eliminate the signed, scan-specific plume coefficients before fitting the
    # shared geographic field. Each scan contributes equal total pixel weight.
    blocks, responses = [], []
    for scan in scans:
        scale = np.sqrt(scan.n_pixels)
        blocks.append((scan.background - scan.kernels @ scan.projected_background) / scale)
        responses.append((scan.observed - scan.kernels @ scan.rates) / scale)
    matrix, observed = np.vstack(blocks), np.concatenate(responses)
    gram = matrix.T @ matrix + np.eye(matrix.shape[1]) * BACKGROUND_PENALTY * len(scans)
    return np.linalg.solve(gram, matrix.T @ observed)


def scans_for_day(rows, site, origins, wind, keep_scene=False):
    close = rows[PIXEL_TIME].sort_values().drop_duplicates()
    groups = pd.Series(close.diff().gt(SCAN_SEPARATION).cumsum().to_numpy(), index=close)
    fits = []
    for _, swath in rows.groupby(rows[PIXEL_TIME].map(groups), sort=True):
        scan = fit_scan(swath, site.longitude, site.latitude, origins, wind, keep_scene)
        if scan is not None:
            fits.append(scan)
    return fits


def result_for_day(day, scans, longitude):
    midday = day + pd.Timedelta(hours=13.5 - longitude / 15)
    scans = [scan for scan in scans if abs(scan.time - midday) <= pd.Timedelta(hours=2)]
    if not scans:
        return [], None
    midday_lag = np.array([abs(scan.time - midday).total_seconds() / 60 for scan in scans])
    nearest = int(midday_lag.argmin())
    all_background = shared_background(scans)
    directions = np.deg2rad([scan.direction for scan in scans])
    rows = []
    for i, scan in enumerate(scans):
        single_background = shared_background([scan])
        rows.append({
            "scan_utc": scan.time, "date": day, "local_solar_hour": 13.5 + (scan.time - midday).total_seconds()/3600,
            "n_pixels": scan.n_pixels, "wind_to": scan.direction, "unique_fraction": scan.unique_fraction,
            "closest_midday": i == nearest and midday_lag[nearest] <= 60,
            "independent_kg_h": scan.rates[0],
            "flexible_single_kg_h": scan.rates[0] - scan.projected_background[0] @ single_background,
            "shared_background_kg_h": scan.rates[0] - scan.projected_background[0] @ all_background,
            "day_direction_spread": 1 - abs(np.exp(1j * directions).mean()),
        })
    return rows, all_background


def date_index(pixels, longitude):
    pixels[PIXEL_TIME] = pd.to_datetime(pixels[PIXEL_TIME], utc=True, format="ISO8601")
    solar = pixels[PIXEL_TIME] + pd.Timedelta(hours=longitude/15)
    minute = solar.dt.hour * 60 + solar.dt.minute
    return pixels.loc[minute.between(11 * 60 + 15, 15 * 60 + 45)].assign(date=solar.dt.normalize())


def retrieve_file(task):
    file, site, inventory, windfile, output = task
    output_file = output / file.name.replace(".tsv.gz", ".parquet")
    if output_file.is_file():
        return {"file": file.name, "scans": len(pd.read_parquet(output_file, columns=["scan_utc"])), "seconds": 0}
    clock = perf_counter()
    wind = wind_vectors(windfile, 100).sort_index()
    ids, origins = candidates(site.location, site.longitude, site.latitude, inventory)
    pixels = date_index(pd.read_csv(file, sep="\t"), site.longitude)
    result = []
    for day, rows in pixels.groupby("date"):
        scans = scans_for_day(rows, site, origins, wind)
        retrieved, _ = result_for_day(day, scans, site.longitude)
        result.extend({"location": site.location, "inventory_neighbors": len(ids)-1, **row} for row in retrieved)
    record = pd.DataFrame(result)
    if len(record) == 0:
        record = pd.DataFrame(columns=["location", "scan_utc", "date"])
    temporary = output_file.with_suffix(".part")
    record.to_parquet(temporary, index=False)
    temporary.replace(output_file)
    return {"file": file.name, "scans": len(record), "seconds": round(perf_counter() - clock, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--site", type=int, nargs="+")
    args = parser.parse_args()
    panel = pd.read_csv(args.annual / "panel.csv")
    if args.site:
        panel = panel.loc[panel.location.isin(args.site)]
    sites = {row["location"]: SimpleNamespace(**row) for row in panel.to_dict("records")}
    inventory = pd.read_csv(args.annual / "historical_screening_inventory.csv").drop_duplicates("Facility_ID")
    output = args.work / "retrieved"
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for file in sorted((args.work / "swaths").glob("*.tsv.gz")):
        location = int(file.name.split("_", 1)[0])
        wind = args.work / "wind" / f"{location}.json"
        if location in sites and wind.is_file():
            jobs.append((file, sites[location], inventory, wind, output))
    cpu_model = Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].split("\n", 1)[0].partition(":")[2].strip()
    print(json.dumps({"host": platform.node(), "processor": cpu_model, "processes": args.workers, "blocks": len(jobs)}), flush=True)
    if args.workers == 1:
        for case in jobs:
            print(retrieve_file(case), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(retrieve_file, jobs):
                print(result, flush=True)


if __name__ == "__main__":
    main()
