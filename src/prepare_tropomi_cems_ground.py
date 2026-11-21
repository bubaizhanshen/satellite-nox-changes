#!/usr/bin/env python3
"""Extract quality-checked EPA facility hours from public PUDL CEMS data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import local

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as fs
import pyarrow.parquet as pq


COLUMNS = ["plant_id_epa", "emissions_unit_id_epa", "operating_datetime_utc",
           "operating_time_hours", "nox_mass_lbs", "nox_mass_measurement_code"]
SOURCE = "pudl.catalyst.coop/stable/core_epacems__hourly_emissions.parquet"


def staged_records(source, year, sites, stage, workers, first_utc_hour=8, endpoint=None):
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", endpoint_override=endpoint,
                         request_timeout=25, connect_timeout=9,
                         retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=3))
    with s3.open_input_file(source) as remote:
        size = remote.size()
        cached_metadata = stage.parent / f"footer_{size}.parquet"
        meta = pq.read_metadata(cached_metadata) if cached_metadata.is_file() else pq.ParquetFile(remote).metadata
        index = meta.schema.names.index("year")
        years = [meta.row_group(i).column(index).statistics for i in range(meta.num_row_groups)]
        groups = [i for i, stats in enumerate(years)
                  if stats is None or stats.min <= year <= stats.max]
    if not groups:
        raise ValueError(f"No row groups for {year}")
    cache = f"{year}_{size}"
    if first_utc_hour != 8:
        cache += f"_utc{first_utc_hour:02}"
    stage = stage / cache
    stage.mkdir(parents=True, exist_ok=True)
    ids = pa.array(sites.astype("int32"))
    readers = local()
    print("source bytes", size, "| row groups", len(groups), flush=True)

    def fetch(group):
        output = stage / f"cems_{group:05}.parquet"
        if output.is_file():
            return output
        if not hasattr(readers, "file"):
            readers.file = s3.open_input_file(source)
            readers.parquet = pq.ParquetFile(readers.file, metadata=meta, pre_buffer=True)
        records = readers.parquet.read_row_group(group, columns=COLUMNS, use_threads=False)
        time = records["operating_datetime_utc"]
        mask = pc.and_(pc.is_in(records["plant_id_epa"], value_set=ids),
                       pc.and_(pc.equal(pc.year(time), year), pc.greater_equal(pc.hour(time), first_utc_hour)))
        tmp = output.with_suffix(".part")
        pq.write_table(records.filter(mask), tmp)
        tmp.rename(output)
        return output

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = []
        for path in pool.map(fetch, groups):
            files.append(path)
            if len(files) % 40 == 0 or len(files) == len(groups):
                print("cached", len(files), "/", len(groups), flush=True)
    return files


def aggregate(raw):
    names = {"plant_id_epa": "facility_id", "emissions_unit_id_epa": "unit_id",
             "operating_datetime_utc": "utc", "operating_time_hours": "operation",
             "nox_mass_lbs": "mass_lb", "nox_mass_measurement_code": "indicator"}
    raw = raw.drop_duplicates().rename(columns=names)
    keys = ["facility_id", "utc"]
    unit_key = [*keys, "unit_id"]
    raw["utc"] = pd.to_datetime(raw.utc, utc=True)
    raw["quarter"] = raw.utc.dt.quarter
    raw["stopped"] = raw.operation.eq(0) & (raw.mass_lb.isna() | raw.mass_lb.eq(0))
    positive = raw.operation.gt(0) & raw.mass_lb.ge(0) & raw.indicator.eq("Measured")
    raw["usable"] = (raw.stopped | positive) & ~raw.duplicated(unit_key, keep=False)
    raw["mass_kg"] = raw.mass_lb.where(~raw.stopped, 0) * 0.45359237
    expected = raw.groupby(["facility_id", "quarter"]).unit_id.nunique().rename("expected")
    hours = raw.groupby([*keys, "quarter"], as_index=False).agg(
        usable=("usable", "all"), stopped=("stopped", "all"), units=("unit_id", "nunique"), mass_kg=("mass_kg", "sum"))
    hours = hours.join(expected, on=["facility_id", "quarter"])
    return hours.loc[hours.usable & hours.units.eq(hours.expected), [*keys, "mass_kg", "stopped", "units"]].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--first-utc-hour", type=int, choices=range(24), default=8)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    plants = pd.read_csv(args.plants, usecols=["Facility_ID"]).Facility_ID
    files = staged_records(args.source, args.year, plants, args.stage, args.workers,
                           first_utc_hour=args.first_utc_hour, endpoint=args.endpoint)
    raw = pq.read_table(files).to_pandas()
    ground = aggregate(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(f".{os.getpid()}.part")
    ground.to_parquet(pending, index=False)
    pending.replace(args.output)
    print("valid hours", len(ground), "| facilities", ground.facility_id.nunique(),
          "| recorded stops", int(ground.stopped.sum()))


if __name__ == "__main__":
    main()
