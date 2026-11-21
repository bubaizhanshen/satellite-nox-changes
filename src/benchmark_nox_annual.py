#!/usr/bin/env python3
"""Acquire a fixed panel of native satellite overpasses and replay emission screens."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import gzip
from io import BytesIO
import json
from pathlib import Path
import re
from threading import local
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

from benchmark_nox_reporting import pixel_table
from compare_tropomi_neighbor_masks import CORNERS, NO2, RSIG, VARIABLE, swath_evidence, xy
from check_tropomi_footprint_evidence import plume_template_score
from score_no2_overpass_benchmark import PIXEL_TIME
from evaluate_nox_reporting_bias import LIFETIME_HOURS, cross_section_slices, gaussian
from check_tropomi_footprint_evidence import recall_cutoff
from score_no2_overpass_benchmark import NO2 as FLUX_COLUMN, reference_mass


YEARS = (2023, 2024)
INVENTORY_YEAR = 2019
PANEL_SEED = 331170
LAT_RADIUS = 0.95
STAMP = "Timestamp(UTC)"
ORBIT_NAME = re.compile(r"S5P_\w{4}_L2__NO2____(\d{8})T\d{6}_\d{8}T\d{6}_(\d{5})_")


def orbit_keys(paths):
    keys = paths.str.extract(ORBIT_NAME)
    if keys.isna().any(axis=None):
        raise ValueError("An archived record has no identifiable Sentinel-5P orbit")
    return keys.set_axis(["granule_date", "orbit"], axis=1)


def prepare(args, all_eligible=False):
    base = args.previous / "full_archive"
    hourly = pd.read_csv(base / "annual_stack_comparison.csv")
    year_pair = hourly.loc[hourly.year.isin(YEARS)].set_index(["location", "year"]).unstack()
    archive = pd.read_parquet(base / "population.parquet")
    archive = archive.loc[
        archive.year.isin(YEARS) & archive.view_band.notna() & archive.mean_mass_2h.notna()
        & archive.latitude.notna() & archive.longitude.notna() & archive.annual_nox_emission.notna()
    ].copy()
    archive["month"] = archive.utc_time.dt.strftime("%Y-%m")
    use = year_pair.stack_hour_coverage.min(axis=1, skipna=False).ge(.95)
    if all_eligible:
        matched = pd.read_parquet(args.previous / "population.parquet", columns=["location"])
        use &= year_pair.index.isin(matched.location)
    else:
        counts = archive.groupby(["location", "month"]).size().unstack(fill_value=0)
        balanced = counts.min(axis=1).ge(4)
        use &= year_pair.overpasses.min(axis=1, skipna=False).ge(100)
        use &= balanced.reindex(use.index, fill_value=False)
    eligible = year_pair.loc[use].copy()
    all_hour = eligible.all_hours_stack_kg_h
    change = all_hour[2024] - all_hour[2023]
    cohorts = pd.DataFrame({"change_kg_h": change,
                            "cems_2023": all_hour[2023], "cems_2024": all_hour[2024]})
    cohorts["stratum"] = pd.cut(
        change, [-np.inf, -100, -50, -25, 25, 50, 100, np.inf],
        labels=["large_decrease", "decrease", "moderate_decrease", "stable",
                "moderate_increase", "increase", "large_increase"],
    )
    cores = {"large_decrease": (6, 4), "decrease": (7, 3), "moderate_decrease": (5, 2),
             "stable": (10, 4), "moderate_increase": (5, 2), "increase": (8, 3), "large_increase": (4, 3)}
    rng = np.random.default_rng(PANEL_SEED)
    samples = []
    for name, subset in cohorts.groupby("stratum", observed=True):
        varied = subset.sort_index()
        if all_eligible:
            chosen = varied.copy()
            first = len(chosen)
        else:
            size, first = cores[name]
            chosen = varied.iloc[rng.choice(len(varied), min(size, len(varied)), replace=False)].copy()
        chosen["priority_panel"] = np.arange(len(chosen)) < first
        chosen["eligible_plants"] = len(varied)
        chosen["design_weight"] = len(varied) / len(chosen)
        samples.append(chosen)
    panel = pd.concat(samples).join(archive.drop_duplicates("location").set_index("location")[
        ["Facility_Name", "State", "latitude", "longitude"]
    ]).sort_index()
    visits = archive.loc[archive.location.isin(panel.index)].sort_values(["location", "utc_time"]).copy()
    visits.insert(0, "visit_id", np.arange(len(visits)))
    blocks = visits.groupby(["location", "month"]).agg(
        scheduled=("visit_id", "size"), latitude=("latitude", "first"), longitude=("longitude", "first"),
    ).reset_index()
    blocks = blocks.join(panel["priority_panel"], on="location").sort_values(
        ["priority_panel", "location", "month"], ascending=[False, True, True],
    )
    blocks.insert(0, "block_id", np.arange(len(blocks)))
    visits = visits.merge(blocks[["location", "month", "block_id"]], on=["location", "month"], validate="many_to_one")
    panel.to_csv(args.output / "panel.csv")
    blocks.to_csv(args.output / "download_schedule.csv", index=False)
    visits.to_parquet(args.output / "visits.parquet", index=False)
    print("Eligible", len(cohorts), "panel", len(panel), "priority", int(panel.priority_panel.sum()),
          "visits", len(visits), "months", len(blocks), flush=True)


def screening_inventory(raw, workbook, archive):
    names = pd.read_csv(raw / "us-power-plant-list.csv").set_index("Facility_ID")
    annual = pd.read_excel(workbook, sheet_name=f"PLNT{INVENTORY_YEAR % 100:02}", skiprows=1,
                           usecols=["ORISPL", "PLNOXAN"]).set_index("ORISPL").PLNOXAN
    history = pd.read_parquet(archive, columns=["location", "annual_nox_emission"],
                             filters=[("year", "=", INVENTORY_YEAR)])
    backup = history.groupby("location").annual_nox_emission.median()
    plants = names[["Latitude", "Longitude"]].assign(annual=annual.combine_first(backup))
    plants["inventory_source"] = np.select([plants.index.isin(annual.dropna().index), plants.annual.notna()],
                                           ["eGRID2019", "2019_archived_inventory"], default="missing")
    cities = pd.read_csv(raw / "world-cities.csv").query("population >= 200000")
    return plants, cities


def screen(case, available, plants, cities):
    corners, areas, gain, search, hidden, _, _, spread, threshold, _ = swath_evidence(case, available, plants, cities)
    wind = np.array([case.wind_u, case.wind_v]) / case.wind_m_s
    shape = plume_template_score(corners, gain, hidden, wind, spread)
    area = areas[search & ~hidden & (gain > threshold)].sum()
    return shape, area


def calibrate(args):
    panel = pd.read_csv(args.output / "panel.csv")
    plants, cities = screening_inventory(args.raw, args.inventory, args.previous / "full_archive" / "population.parquet")
    plants.to_csv(args.output / "historical_screening_inventory.csv")
    training_schedule = pd.read_csv(args.previous / "overpass_schedule.csv")
    training = training_schedule.loc[training_schedule.year.eq(2022) & ~training_schedule.location.isin(panel.location)
                                     & training_schedule.mean_mass_2h.ge(250)].copy()
    training["utc_time"] = pd.to_datetime(training.utc_time, format="ISO8601", utc=True)
    training["annual_nox_emission"] = training.location.map(plants.annual)
    cached = SimpleNamespace(raw=args.raw, work=args.development_work)
    scores = []
    for case in training.itertuples(index=False):
        row = {"sample_id": case.sample_id, "plume_template_score": np.nan, "score_error": ""}
        try:
            if not np.isfinite(case.annual_nox_emission):
                raise ValueError("No source inventory in the pre-evaluation year")
            source = pixel_table(case, cached, None)
            score, _ = screen(case, pd.read_csv(source, sep="\t"), plants, cities)
            row["plume_template_score"] = score
        except (OSError, ValueError) as error:
            row["score_error"] = str(error)
        scores.append(row)
    full = training.merge(pd.DataFrame(scores), on="sample_id", validate="one_to_one")
    full.to_csv(args.output / "development_calibration.csv", index=False)
    fit = full.loc[full.score_error.eq("")].copy()
    fit["status"] = "active"
    fit["area_score"] = np.where(fit.plume_label, 3., 1.)
    cutoff = recall_cutoff(fit, "plume_template_score")
    if not np.isfinite(cutoff) or fit.empty:
        raise ValueError("The 2022 development sample cannot calibrate the shape screen")
    calibration = {"year": 2022, "inventory_year": INVENTORY_YEAR, "sites": int(fit.location.nunique()),
                   "scheduled_releases": len(training), "scored_releases": len(fit),
                   "archive_recall": float(np.average(fit.plume_label, weights=fit.weight)),
                   "template_recall": float(np.average(fit.plume_template_score.ge(cutoff), weights=fit.weight)),
                   "template_cutoff": float(cutoff), "cems_releases_at_least_kg_h": 250}
    target = args.output / "frozen_screen.json"
    pending = target.with_suffix(".part")
    pending.write_text(json.dumps(calibration, indent=2) + "\n")
    pending.replace(target)
    print(json.dumps(calibration), flush=True)


def block_base(work, row):
    return work / "swaths" / f"{row.location}_{row.month}"


def get_swath(session, path, latitude, longitude, start, stop):
    if path.is_file():
        return path.stat().st_size
    dlon = LAT_RADIUS / np.cos(np.deg2rad(latitude))
    params = {
        "SERVICE": "wcs", "VERSION": "1.0.0", "REQUEST": "GetCoverage", "FORMAT": "ascii",
        "COVERAGE": VARIABLE, "MINIMUM_QUALITY": 75, "CORNERS": 1, "COMPRESS": 1,
        "BBOX": f"{longitude-dlon:.5f},{latitude-LAT_RADIUS:.5f},{longitude+dlon:.5f},{latitude+LAT_RADIUS:.5f}",
        "TIME": f"{start:%Y-%m-%dT00:00:00Z}/{stop:%Y-%m-%dT23:59:59Z}",
    }
    with session.get(RSIG, params=params, timeout=(10, 58)) as response:
        response.raise_for_status()
        body = response.content
        compressed = body.startswith(b"\x1f\x8b")
        with gzip.GzipFile(fileobj=BytesIO(body)) if compressed else BytesIO(body) as stream:
            first = stream.readline(4096)
        if not first.startswith(STAMP.encode()) or NO2.encode() not in first:
            raise ValueError(f"EPA returned no TROPOMI table ({len(body)} bytes)")
        pending = path.with_suffix(".part")
        pending.write_bytes(body if compressed else gzip.compress(body, compresslevel=2))
        pending.replace(path)
    return path.stat().st_size


def chunk_name(base, first, last):
    return base.with_name(f"{base.name}_{first:02}-{last:02}.tsv.gz")


def fetch_month(args, row, session, repair):
    began = perf_counter()
    start = pd.Timestamp(row.month + "-01")
    last = start.days_in_month
    root = block_base(args.work, row)
    month = chunk_name(root, 1, last)
    partial = [(1, 10), (11, 20), (21, last)]
    if repair:
        windows = partial
    elif args.ten_day:
        windows = partial
    else:
        windows = [(1, last)]
    if month.is_file():
        windows = [(1, last)]
    # A failed block stays missing if the provider's response offers no table.
    recorded = []
    for first, end in windows:
        suffix = chunk_name(root, first, end)
        try:
            size = get_swath(session, suffix, row.latitude, row.longitude,
                             start.replace(day=first), start.replace(day=end))
            recorded.append((first, end, size, ""))
        except (OSError, ValueError, EOFError, requests.RequestException) as exc:
            recorded.append((first, end, 0, str(exc)[-220:]))
    return {
        "block_id": row.block_id, "location": row.location, "month": row.month,
        "requested_chunks": len(windows), "available_chunks": sum(x[2] > 0 for x in recorded),
        "received_bytes": sum(x[2] for x in recorded), "wall_seconds": perf_counter() - began,
        "chunks": json.dumps(recorded),
    }


def acquire(args):
    blocks = pd.read_csv(args.output / "download_schedule.csv")
    if args.priority:
        blocks = blocks.loc[blocks.priority_panel]
    blocks = blocks.loc[blocks.block_id.mod(args.shards).eq(args.shard)]
    if args.probe:
        blocks = blocks.iloc[np.linspace(0, len(blocks) - 1, min(len(blocks), args.probe)).astype(int)]
    if args.repair:
        initial = args.output / f"download_{args.shard:02}.csv"
        data = pd.read_csv(initial)
        blocks = blocks.loc[blocks.block_id.isin(data.loc[data.available_chunks.lt(data.requested_chunks), "block_id"])]
    clients = local()

    def run(row):
        if not hasattr(clients, "session"):
            clients.session = requests.Session()
        return fetch_month(args, row, clients.session, args.repair)

    tag = "probe" if args.probe else "repair" if args.repair else "download"
    prefix = args.output / f"{tag}_{args.shard:02}"
    print(len(blocks), "plant–month requests", tag, flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, prefix.with_suffix(".jsonl").open("a") as log:
        tasks = [pool.submit(run, row) for row in blocks.itertuples(index=False)]
        for task in as_completed(tasks):
            report = task.result()
            records.append(report)
            log.write(json.dumps(report) + "\n")
            log.flush()
            if args.probe or len(records) % 24 == 0 or len(records) == len(tasks):
                total = sum(x["received_bytes"] for x in records)
                ok = sum(x["available_chunks"] > 0 for x in records)
                print(f"{len(records)}/{len(tasks)} blocks, {ok} with pixels, {total/1e6:.1f} MB", flush=True)
    pd.DataFrame(records).to_csv(prefix.with_suffix(".csv"), index=False)


def cached_chunks(work, row):
    root = block_base(work, row)
    days = pd.Timestamp(row.month + "-01").days_in_month
    full = chunk_name(root, 1, days)
    if full.is_file():
        return [full]
    return [path for first, stop in [(1, 10), (11, 20), (21, days)]
            if (path := chunk_name(root, first, stop)).is_file()]


def monthly_scenes(files):
    cols = [STAMP, "LONGITUDE(deg)", "LATITUDE(deg)", NO2]
    cols.extend(f"{coordinate}_{corner}(deg)" for coordinate in ("Longitude", "Latitude") for corner in CORNERS)
    month = pd.concat([pd.read_csv(path, sep="\t", usecols=cols) for path in files], ignore_index=True)
    month[STAMP] = pd.to_datetime(month[STAMP], format="ISO8601", utc=True)
    month = month.dropna(subset=[STAMP, NO2]).sort_values(STAMP, kind="stable")
    if month.empty:
        return {}, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    gaps = month[STAMP].diff().gt(pd.Timedelta(minutes=10))
    labels = gaps.cumsum().to_numpy()
    scenes = {number: group for number, group in month.groupby(labels, sort=False)}
    return scenes, month[STAMP].astype("datetime64[ns, UTC]").array.asi8, labels


def scene_for_visit(case, scenes, times, labels):
    if not len(times):
        return pd.DataFrame(), np.nan, np.nan
    when = case.utc_time.value
    insertion = np.searchsorted(times, when)
    neighbors = np.unique(np.clip([insertion-1, insertion], 0, len(times)-1))
    pos = neighbors[np.argmin(np.abs(times[neighbors] - when))]
    difference = abs(times[pos] - when) / 1e9
    if difference > 120:
        return pd.DataFrame(), difference, np.nan
    available = scenes[labels[pos]].drop_duplicates(["LONGITUDE(deg)", "LATITUDE(deg)"])
    distance = np.linalg.norm(xy(available["LONGITUDE(deg)"], available["LATITUDE(deg)"],
                                 case.longitude, case.latitude), axis=1).min()
    return available, difference, float(distance)


@lru_cache(maxsize=96)
def temis_orbit(path):
    data = pd.read_parquet(path)
    return {visit: table for visit, table in data.groupby("visit_id", sort=False)}


def temis_scene(case, path):
    available = temis_orbit(path).get(case.visit_id)
    if available is None:
        return pd.DataFrame(), np.nan, np.nan
    distance = np.linalg.norm(xy(available["LONGITUDE(deg)"], available["LATITUDE(deg)"],
                                 case.longitude, case.latitude), axis=1)
    nearest = distance.argmin()
    lag = abs((available[STAMP].iloc[nearest] - case.utc_time).total_seconds())
    return available, lag, float(distance[nearest])


def measure(case, available, plants, cities, reference):
    began = perf_counter()
    record = {"visit_id": case.visit_id, "native_pixels": len(available),
              "screen_2019_inventory_tons": case.annual_nox_emission}
    area = template = np.nan
    try:
        template, area = screen(case, available, plants, cities)
        record["screen_error"] = ""
    except ValueError as exc:
        record["screen_error"] = str(exc)
    record["template_score"] = template
    record["flagged_area_km2"] = area
    model = available.rename(columns={NO2: FLUX_COLUMN}).assign(**{PIXEL_TIME: case.tropomi_utc})
    _, ground, value = gaussian(case, model, reference)
    record["gaussian_kg_h"] = value
    record["gaussian_cems_kg_h"] = ground
    sections, ranges = cross_section_slices(case, model)
    if np.isfinite(sections).all():
        means = [reference_mass(reference, case.tropomi_utc, young, old) for young, old in ranges]
        record["cross_section_cems_kg_h"] = float(np.mean(means))
        record["cross_section_kg_h"] = float(np.mean(sections))
        midpoint = np.mean(ranges, axis=1)
        for lifetime in (2, 6):
            adjusted = sections * np.exp(midpoint * (1/lifetime - 1/LIFETIME_HOURS))
            record[f"cross_section_{lifetime}h_kg_h"] = float(np.mean(adjusted))
    record["fit_wall_seconds"] = perf_counter() - began
    return record


def stack_reference(work, facilities, year):
    data = pd.read_parquet(work / f"{year}.parquet", columns=["facility_id", "utc", "mass_kg"],
                           filters=[("facility_id", "in", facilities)])
    earlier = work / f"{year-1}.parquet"
    if earlier.is_file():
        first = pd.Timestamp(f"{year-1}-12-31", tz="UTC")
        prev = pd.read_parquet(earlier, columns=["facility_id", "utc", "mass_kg"],
                               filters=[("facility_id", "in", facilities), ("utc", ">=", first)])
        data = pd.concat([prev, data], ignore_index=True)
    return {location: table.set_index("utc").mass_kg.sort_index() for location, table in data.groupby("facility_id")}


def score(args):
    blocks = pd.read_csv(args.output / "download_schedule.csv")
    if args.priority:
        blocks = blocks.loc[blocks.priority_panel]
    blocks = blocks.loc[blocks.block_id.mod(args.shards).eq(args.shard)]
    jobs = []
    scored_dir = args.work / ("scored" if args.provider == "rsig" else f"scored_{args.provider}")
    scored_dir.mkdir(exist_ok=True)
    cases = pd.read_parquet(args.output / "visits.parquet")
    cases["tropomi_utc"] = cases.utc_time
    freeze_file = args.output / "frozen_screen.json"
    frozen = json.loads(freeze_file.read_text())
    if frozen.get("inventory_year") != INVENTORY_YEAR:
        raise ValueError("Run calibrate with the historical inventory before scoring")
    if args.provider != "rsig":
        cases["orbit"] = orbit_keys(cases.file_path)["orbit"]
        files_by_orbit = {path.stem: path for path in (args.work / "temis").glob("*.parquet")}
        orbits_by_block = cases.groupby("block_id").orbit.unique().to_dict()
        covered = cases.orbit.isin(files_by_orbit).groupby(cases.block_id).sum()
        if args.complete_only:
            blocks = blocks.loc[blocks.block_id.map(covered).eq(blocks.scheduled)]
    else:
        files_by_orbit, orbits_by_block, covered = {}, {}, {}
    for row in blocks.itertuples(index=False):
        files = cached_chunks(args.work, row) if args.provider != "temis" else []
        temis = [files_by_orbit[key] for key in orbits_by_block.get(row.block_id, []) if key in files_by_orbit]
        if not files and not temis:
            continue
        path = scored_dir / f"{row.block_id:04}.parquet"
        dependencies = [*files, *temis, freeze_file, args.inventory, args.output / "historical_screening_inventory.csv"]
        if path.is_file() and path.stat().st_mtime_ns >= max(f.stat().st_mtime_ns for f in dependencies):
            if args.provider != "temis":
                continue
            # More orbits can finish downloading between the worker listing the
            # files and writing this cache, even with earlier file timestamps.
            previous = pd.read_parquet(path, columns=["origin"])
            if previous.origin.eq("temis").sum() == covered[row.block_id]:
                continue
        jobs.append((row, files, path))
    if not jobs:
        print("No new plant–months to score", flush=True)
        return
    jobs.sort(key=lambda item: (item[0].month, item[0].location))
    cases = cases.loc[cases.block_id.isin([row.block_id for row, _, _ in jobs])].copy()
    plants = pd.read_csv(args.output / "historical_screening_inventory.csv").set_index("Facility_ID")
    cities = pd.read_csv(args.raw / "world-cities.csv").query("population >= 200000")
    cases["annual_nox_emission"] = cases.location.map(plants.annual)
    if cases.annual_nox_emission.isna().any():
        raise ValueError("A panel plant has no prior-year emissions in the frozen inventory")
    by_block = {number: group for number, group in cases.groupby("block_id", sort=False)}
    by_year = {}
    for year, visits in cases.groupby("year"):
        by_year[year] = stack_reference(args.ground, visits.location.unique().tolist(), year)
    counts = 0
    for index, (row, files, path) in enumerate(jobs, start=1):
        scenes, times, labels = monthly_scenes(files) if files else ({}, np.empty(0), np.empty(0))
        month = by_block[row.block_id]
        records = []
        started = perf_counter()
        for case in month.itertuples(index=False):
            temis = files_by_orbit.get(getattr(case, "orbit", None))
            if temis is not None:
                available, lag, distance = temis_scene(case, temis)
                origin = "temis"
            else:
                available, lag, distance = pd.DataFrame(), np.nan, np.nan
                origin = "pending"
            if files and (available.empty or distance > 20. or lag > 120.):
                available, lag, distance = scene_for_visit(case, scenes, times, labels)
                origin = "rsig"
            if available.empty or distance > 20. or lag > 120.:
                records.append({"visit_id": case.visit_id, "time_offset_s": lag,
                                "nearest_pixel_km": distance, "origin": origin,
                                "pixel_error": "No matched native scene near the plant"})
                continue
            record = measure(case, available, plants, cities, by_year[case.year][case.location])
            records.append({**record, "time_offset_s": lag, "nearest_pixel_km": distance, "origin": origin,
                            "pixel_error": ""})
        dest = path.with_suffix(".part")
        frame = pd.DataFrame(records)
        frame.to_parquet(dest, index=False)
        dest.replace(path)
        counts += len(frame)
        if len(jobs) <= 10 or index % 16 == 0:
            print(f"{row.location} {row.month} {len(frame)} visits | {perf_counter()-started:.2f} s | {counts} processed",
                  flush=True)
    print(len(jobs), "blocks", counts, "visits processed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "calibrate", "fetch", "score"])
    parser.add_argument("--previous", type=Path, default=Path("inputs/archive"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("inputs/raw"))
    parser.add_argument("--ground", type=Path, default=Path("inputs/ground"))
    parser.add_argument("--development-work", type=Path, default=Path("inputs/development"))
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--all-eligible", action="store_true",
                        help="Prepare all matched plants with at least 95%% stack coverage in both years")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--ten-day", action="store_true")
    parser.add_argument("--complete-only", action="store_true",
                        help="Score only plant-months whose native orbit files have all arrived")
    parser.add_argument("--probe", type=int, default=0)
    parser.add_argument("--provider", choices=("rsig", "temis", "hybrid"), default="rsig")
    args = parser.parse_args()
    if args.inventory is None:
        args.inventory = args.work / "egrid2019_data.xlsx"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.work / "swaths").mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args, all_eligible=args.all_eligible)
    elif args.stage == "calibrate":
        calibrate(args)
    elif args.stage == "fetch":
        acquire(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
