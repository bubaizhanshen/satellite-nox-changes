#!/usr/bin/env python3
"""Extract native TROPOMI pixels from KNMI's public TEMIS daily orbit archives."""

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import RawIOBase
import json
from pathlib import Path
import tarfile
from threading import local
from time import perf_counter, sleep

import h5py
import numpy as np
import pandas as pd
import requests

from benchmark_nox_annual import CORNERS, LAT_RADIUS, NO2, ORBIT_NAME, STAMP, orbit_keys


TEMIS = "https://d1qb6yzwaaq4he.cloudfront.net/tropomi/no2"
COLUMN = "nitrogendioxide_tropospheric_column"
COORDINATES = ["longitude", "latitude"]
CLIENT = local()
METADATA_BLOCK = 128 * 1024


class Archive:
    def __init__(self, day):
        self.url = f"{TEMIS}/{day[:4]}/{day[4:6]}/tropomi_no2_{day}.tar"

    def read(self, start, size):
        end = start + size - 1
        if not hasattr(CLIENT, "session"):
            CLIENT.session = requests.Session()
            CLIENT.session.trust_env = False
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        for attempt in range(3):
            try:
                with CLIENT.session.get(self.url, headers=headers, stream=True, timeout=(10, 25)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                        raise ValueError(f"Unexpected HTTP byte range: {content_range}")
                    body = response.content
                if len(body) != size:
                    raise IOError("Incomplete TEMIS byte range")
                return body
            except requests.HTTPError as error:
                if error.response.status_code < 500 and error.response.status_code != 429:
                    raise
                if attempt == 2:
                    raise
            except (OSError, requests.RequestException):
                if attempt == 2:
                    raise
            sleep(.5 * 2**attempt)
        raise RuntimeError("Unreachable after the final acquisition attempt")

    def find(self, orbits):
        offset = 0
        remaining = set(orbits)
        for _ in range(32):
            try:
                member = tarfile.TarInfo.frombuf(self.read(offset, 512), "utf-8", "strict")
            except tarfile.EOFHeaderError:
                break
            offset += 512
            found = ORBIT_NAME.search(member.name)
            if found and found[2] in remaining:
                remaining.remove(found[2])
                yield found[2], (offset, member.size, member.name)
            if not remaining:
                break
            offset += (member.size + 511) // 512 * 512


class OrbitFile(RawIOBase):
    """Seek within one member of a remote tar without downloading the orbit."""

    def __init__(self, archive, offset, size):
        self.archive = archive
        self.offset = offset
        self.size = size
        self.position = 0
        self.metadata = OrderedDict()
        self.chunks = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        new = (0, self.position, self.size)[whence] + offset
        if new < 0:
            raise ValueError("Negative orbit file offset")
        self.position = new
        return new

    def readinto(self, buffer):
        begin = self.position
        size = max(0, min(len(buffer), self.size - begin))
        end = begin + size
        if size == 0:
            return 0
        prefetch = self.chunks.get((begin, size))
        if prefetch is not None:
            buffer[:size] = prefetch
        else:
            for i in range(begin // METADATA_BLOCK, (end - 1) // METADATA_BLOCK + 1):
                offset = METADATA_BLOCK * i
                if i not in self.metadata:
                    self.metadata[i] = self.archive.read(self.offset + offset, min(METADATA_BLOCK, self.size - offset))
                    if len(self.metadata) > 80:
                        self.metadata.popitem(last=False)
                first, last = max(begin, offset), min(end, offset + METADATA_BLOCK)
                buffer[first-begin:last-begin] = self.metadata[i][first-offset:last-offset]
        self.position = end
        return size

    def prefetch(self, requests_to_read, pool):
        chunks = set()
        for dataset, sel in requests_to_read:
            for indices in dataset.iter_chunks(sel):
                where = dataset.id.get_chunk_info_by_coord(tuple(part.start for part in indices))
                if where.byte_offset is not None and (where.byte_offset, where.size) not in self.chunks:
                    chunks.add((where.byte_offset, where.size))
        blocks = []
        for start, size in sorted(chunks):
            if blocks and start <= blocks[-1][1] + 64_000:
                blocks[-1][1] = start + size
                blocks[-1][2].append((start, size))
            else:
                blocks.append([start, start + size, [(start, size)]])

        def read(job):
            start, stop, pieces = job
            body = self.archive.read(self.offset + start, stop - start)
            return {key: body[key[0]-start:key[0]-start+key[1]] for key in pieces}

        for block in pool.map(read, blocks):
            self.chunks.update(block)


@dataclass
class Window:
    case: object
    first: int
    stop: int
    near: np.ndarray | None = None


def window_selector(first, stop, left, right, with_bounds=False):
    index = (slice(0, 1), slice(first, stop), slice(left, right))
    return (*index, slice(0, 4)) if with_bounds else index


def extract(archive, expected_orbit, member, visits, pool):
    offset, size, filename = member
    results = []
    with OrbitFile(archive, offset, size) as source, h5py.File(source, "r") as file:
        if int(file.attrs["orbit"][0]) != int(expected_orbit):
            raise ValueError("Orbit metadata does not match the archive schedule")
        data = file["PRODUCT"]
        geolocation = data["SUPPORT_DATA/GEOLOCATIONS"]
        time_base = pd.Timestamp(file.attrs["time_reference"].decode("ascii"))
        ms = data["delta_time"][0]
        when = time_base.value + ms.astype(np.int64) * 1_000_000
        if (np.diff(when) <= 0).any():
            raise ValueError("Native scan times are not strictly increasing")
        rows_needed = []
        columns = data["longitude"].shape[-1]
        for case in visits:
            bounds = np.searchsorted(when, [case.utc_time.value - 90_000_000_000, case.utc_time.value + 90_000_000_000])
            if bounds[0] < bounds[1]:
                rows_needed.append(Window(case, *bounds))
        source.prefetch([(data[name], window_selector(w.first, w.stop, 0, columns))
                         for name in COORDINATES for w in rows_needed], pool)
        science = [(data["qa_value"], False), (data[COLUMN], False),
                   *((geolocation[f"{c}_bounds"], True) for c in COORDINATES)]
        chunks = []
        for w in rows_needed:
            c = w.case
            lon, lat = (data[name][0, w.first:w.stop] for name in COORDINATES)
            dlon = LAT_RADIUS / np.cos(np.deg2rad(c.latitude))
            mask = (np.abs(lon - c.longitude) <= dlon) & (np.abs(lat - c.latitude) <= LAT_RADIUS)
            ir, ic = np.nonzero(mask)
            if len(ir) == 0:
                continue
            rows = slice(w.first + ir.min(), w.first + ir.max() + 1)
            cells = slice(ic.min(), ic.max() + 1)
            first, last = rows.start - w.first, rows.stop - w.first
            w.near = mask[first:last, cells]
            chunks.append((w, rows, cells, lon[first:last, cells], lat[first:last, cells]))
        source.prefetch([(dataset, window_selector(row.start, row.stop, col.start, col.stop, corners))
                         for _, row, col, _, _ in chunks for dataset, corners in science], pool)
        for w, row, col, lon, lat in chunks:
            raw_qa = data["qa_value"][0, row, col]
            vcd = data[COLUMN][0, row, col]
            qa = raw_qa * data["qa_value"].attrs["scale_factor"].item()
            valid = w.near & (qa >= .75 - 1e-7) & (raw_qa <= 100) & (np.abs(vcd) < 1.)
            native = {}
            for coord in COORDINATES:
                vertices = geolocation[f"{coord}_bounds"][0, row, col, :]
                valid &= np.isfinite(vertices).all(axis=2) & (np.abs(vertices) <= 180).all(axis=2)
                native[coord] = vertices
            if not valid.any():
                continue
            frame = pd.DataFrame({"visit_id": w.case.visit_id,
                                  STAMP: pd.to_datetime(when[row][np.nonzero(valid)[0]], utc=True),
                                  "LONGITUDE(deg)": lon[valid], "LATITUDE(deg)": lat[valid],
                                  NO2: vcd[valid].astype(float) * 6.02214076e19})
            for coord, positions in native.items():
                frame[[f"{coord.capitalize()}_{c}(deg)" for c in CORNERS]] = positions[valid]
            results.append(frame)
    columns = ["visit_id", STAMP, "LONGITUDE(deg)", "LATITUDE(deg)", NO2,
               *(f"{c.capitalize()}_{k}(deg)" for c in COORDINATES for k in CORNERS)]
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=columns), filename


def download_day(day, cases, directory):
    started = perf_counter()
    by_orbit = {key: list(group.itertuples(index=False)) for key, group in cases.groupby("orbit")}
    todo = {k for k in by_orbit if not (directory / f"{k}.parquet").is_file()}
    if not todo:
        return {"day": day, "requested": len(by_orbit), "cached": len(by_orbit), "successes": [], "errors": {}}
    fn = directory / f"{day}.index.json"
    current = json.loads(fn.read_text()) if fn.is_file() else {}
    archive, failures, done = Archive(day), {}, []
    try:
        current.update(dict(archive.find(todo - current.keys())) if todo - current.keys() else {})
        temp = fn.with_suffix(".part")
        temp.write_text(json.dumps(current) + "\n")
        temp.replace(fn)
    except (OSError, ValueError, tarfile.HeaderError, requests.RequestException) as error:
        failures["archive_index"] = repr(error)[-280:]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for orbit in sorted(todo):
            try:
                if orbit not in current:
                    raise ValueError("Expected orbit is missing from the TEMIS daily tar index")
                native, name = extract(archive, orbit, current[orbit], by_orbit[orbit], pool)
                target = directory / f"{orbit}.parquet"
                dest = target.with_suffix(".part")
                native.to_parquet(dest, index=False)
                dest.replace(target)
                done.append({"orbit": orbit, "file": name, "visits": native.visit_id.nunique(), "pixels": len(native)})
            except (OSError, ValueError, KeyError, requests.RequestException) as error:
                failures[orbit] = repr(error)[-280:]
    return {"day": day, "requested": len(by_orbit), "cached": len(by_orbit)-len(todo),
            "successes": done, "errors": failures, "wall_s": perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visits", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--skip-dates", nargs="+", default=[])
    parser.add_argument("--calendar-days", type=int, nargs=2, default=(1, 31))
    parser.add_argument("--log-name", default="acquisition.jsonl")
    args = parser.parse_args()
    cases = pd.read_parquet(args.visits, columns=["visit_id", "utc_time", "latitude", "longitude", "file_path"])
    cases = cases.join(orbit_keys(cases.file_path))
    args.destination.mkdir(exist_ok=True, parents=True)
    first, last = args.calendar_days
    days = [(date, block) for date, block in cases.groupby("granule_date", sort=True)
            if first <= int(date[-2:]) <= last and date not in args.skip_dates]
    if args.dates:
        days = [(date, block) for date, block in days if date in args.dates]
    else:
        # Interleave years and months during download to make incremental coverage visible.
        days.sort(key=lambda item: (item[0][-2:], item[0][4:6], item[0][:4]))
    print(f"{len(days)} dates scheduled for native retrieval with {args.workers} workers", flush=True)
    with (args.destination / args.log_name).open("a") as log, ProcessPoolExecutor(args.workers) as pool:
        todo = [pool.submit(download_day, day, rows, args.destination) for day, rows in days]
        successes = failed = 0
        for i, task in enumerate(as_completed(todo), 1):
            result = task.result()
            successes += len(result["successes"])
            failed += len(result["errors"])
            log.write(json.dumps(result) + "\n")
            log.flush()
            if i <= 5 or i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} days | {successes} new orbits | {failed} errors | {result['day']}", flush=True)


if __name__ == "__main__":
    main()
