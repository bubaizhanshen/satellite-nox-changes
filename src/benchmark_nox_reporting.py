"""Benchmark plume-based emission reports against hourly stack measurements."""

import argparse
import gzip

from io import BytesIO
from pathlib import Path

import numpy as np

import pandas as pd

import requests

from audit_tropomi_plant_shutdowns import add_release_history

from compare_tropomi_neighbor_masks import NO2, RSIG, VARIABLE

YEARS = (2022, 2023, 2024)

KEY = ["location", "utc_time"]

STRATA = ["year", "ground_state", "view_band", "plume_label"]

ARCHIVE_FIELDS = [
    *KEY, "plume_label", "file_path", "wind_u", "wind_v", "annual_nox_emission",
    "sensor_zenith_angle", "cloud_fraction_crb",
]

RNG_SEED = 406592

def archive_frame(args, plants):
    frames = []
    complete_archive = args.work / "archive/us-tropomi-observation-with-variables.csv"
    if args.extended:
        archive = pd.read_csv(complete_archive, usecols=ARCHIVE_FIELDS, low_memory=False)
        archive["utc_time"] = pd.to_datetime(archive.utc_time, format="ISO8601", utc=True)
        grouped = archive.groupby(archive.utc_time.dt.year)
        for year, visits in grouped:
            path = args.work / "ground" / f"{year}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            facilities = visits.location.unique().tolist()
            ground = pd.read_parquet(path, filters=[("facility_id", "in", facilities)])
            ground["utc"] = pd.to_datetime(ground.utc, utc=True)
            # Keep the time index continuous when a stop crosses 1 January.
            previous = path.with_name(f"{year-1}.parquet")
            if previous.is_file():
                start = pd.Timestamp(f"{year}-01-01", tz="UTC") - pd.Timedelta(hours=8)
                tail = pd.read_parquet(previous, filters=[("utc", ">=", start), ("facility_id", "in", facilities)])
                ground = pd.concat([tail, ground], ignore_index=True)
            histories = add_release_history(ground).rename(columns={"facility_id": "location", "utc": "hour"})
            visits = visits.assign(hour=visits.utc_time.dt.floor("h"))
            visits = visits.merge(histories, on=["location", "hour"], validate="many_to_one")
            visits["wind_m_s"] = np.hypot(visits.wind_u, visits.wind_v)
            frames.append(visits.loc[visits.wind_m_s.between(2, 15)])
    else:
        for year in YEARS:
            visits = pd.read_csv(args.results / f"tropomi_{year}/verified_target_labels.csv")
            angle = pd.read_csv(args.work.parent / f"available_tropomi_{year}.csv",
                                usecols=[*KEY, "sensor_zenith_angle"])
            for data in (visits, angle):
                data["utc_time"] = pd.to_datetime(data.utc_time, format="ISO8601", utc=True)
            frames.append(visits.merge(angle.drop_duplicates(KEY), on=KEY, validate="one_to_one"))
    frame = pd.concat(frames, ignore_index=True).join(
        plants[["Latitude", "Longitude", "State", "Facility_Name"]], on="location", rsuffix="_source"
    ).rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate plant–scan records in the population frame")
    frame["year"] = frame.utc_time.dt.year
    frame["view_band"] = pd.cut(frame.sensor_zenith_angle, [-1, 25, 35, 45, 90],
                                labels=["0-25", "25-35", "35-45", "45-90"])
    mass = frame.mean_mass_2h
    frame["ground_state"] = np.select(
        [frame.past_8h_stopped, mass.ge(500), mass.ge(100), mass.gt(0), mass.eq(0)],
        ["stopped_8h", "500_plus", "100_to_500", "under_100", "recent_or_unverified_stop"],
        default="unknown",
    )
    return frame

def prepare(args, plants):
    frame = archive_frame(args, plants)
    keep = [*ARCHIVE_FIELDS, "latitude", "longitude", "State", "Facility_Name",
            "mass_kg", "mean_mass_2h", "wind_m_s", "year", "ground_state", "view_band"]
    frame[keep].to_parquet(args.output / "population.parquet", index=False)
    if args.extended:
        print("Matched visits across the full archive", len(frame), "plants", frame.location.nunique(), flush=True)
        return
    eligible = (
        frame.year.isin(YEARS) & frame.view_band.notna()
        & frame.mean_mass_2h.notna() & frame.latitude.notna()
    )
    rng = np.random.default_rng(RNG_SEED)
    samples = []
    # Oversample positives at independently verified stops. Retain design weights
    # so the study population contains all view directions and both archival decisions.
    for keys, group in frame.loc[eligible, keep].sort_values(KEY).groupby(STRATA, observed=True):
        _, state, _, published_positive = keys
        requested = args.per_stratum * (2 if state == "stopped_8h" and published_positive else 1)
        chosen = group.iloc[rng.choice(len(group), size=min(len(group), requested), replace=False)].copy()
        samples.append(chosen.assign(stratum_population=len(group), weight=len(group) / len(chosen)))
    plan = pd.concat(samples, ignore_index=True).sort_values(KEY)
    plan = plan.iloc[rng.permutation(len(plan))].reset_index(drop=True)
    plan.index.name = "sample_id"
    plan.to_csv(args.output / "overpass_schedule.csv")
    print("Matched visits", len(frame), "eligible for flux sampling", int(eligible.sum()),
          "sampled overpasses", len(plan), "plants", plan.location.nunique(), flush=True)

def pixel_table(case, args, session):
    stamp = case.utc_time.strftime("%Y%m%dT%H%M%S")
    name = f"{case.location}_{stamp}.tsv"
    cached = args.raw / "rsig_pixels" / name
    if cached.is_file():
        return cached
    path = args.work / "tropomi" / f"{name}.gz"
    if path.is_file():
        return path
    if session is None:
        raise FileNotFoundError(name)
    dlon = 1.16 / np.cos(np.deg2rad(case.latitude))
    bbox = (case.longitude - dlon, case.latitude - 1.16,
            case.longitude + dlon, case.latitude + 1.16)
    start = case.utc_time.floor("min") - pd.Timedelta(minutes=12)
    stop = case.utc_time.ceil("min") + pd.Timedelta(minutes=12)
    query = {
        "SERVICE": "wcs", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "FORMAT": "ascii", "COVERAGE": VARIABLE, "MINIMUM_QUALITY": 75, "CORNERS": 1,
        "COMPRESS": 1, "BBOX": ",".join(f"{part:.5f}" for part in bbox),
        "TIME": f"{start:%Y-%m-%dT%H:%M:%SZ}/{stop:%Y-%m-%dT%H:%M:%SZ}",
    }
    for timeout in (38, 68):
        try:
            response = session.get(RSIG, params=query, timeout=(10, timeout))
            response.raise_for_status()
            body = response.content
            gzipped = body.startswith(b"\x1f\x8b")
            with gzip.GzipFile(fileobj=BytesIO(body)) if gzipped else BytesIO(body) as stream:
                header = stream.readline(4096)
            if NO2.encode() not in header:
                raise ValueError("RSIG did not return the requested TROPOMI columns")
            if not body:
                raise ValueError("RSIG returned no bytes")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(body if gzipped else gzip.compress(body, compresslevel=3))
            temporary.replace(path)
            return path
        except (requests.RequestException, ValueError, OSError, EOFError) as error:
            reason = f"{type(error).__name__}"
    raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser(description="Match the TROPOMI archive to hourly stack records.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True,
                        help="Directory containing archive/ and ground/ for the historical match")
    parser.add_argument("--raw", type=Path, required=True,
                        help="Directory containing us-power-plant-list.csv")
    parser.add_argument("--results", type=Path,
                        help="Directory with year-specific verified label tables for the study sample")
    parser.add_argument("--extended", action="store_true",
                        help="Match the full historical archive to the hourly stack records")
    parser.add_argument("--per-stratum", type=int, default=12)
    args = parser.parse_args()
    if not args.extended and args.results is None:
        parser.error("--results is required for the study-sample route")
    args.output.mkdir(parents=True, exist_ok=True)
    plants = pd.read_csv(args.raw / "us-power-plant-list.csv").set_index("Facility_ID")
    prepare(args, plants)


if __name__ == "__main__":
    main()
