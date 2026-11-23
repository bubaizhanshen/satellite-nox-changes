"""Build annual NOx flux fields from the native TROPOMI footprints."""

import argparse
from functools import lru_cache
from os import environ
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
PROJECT = Path(environ.get("NOX_PROJECT_ROOT", HERE.parent)).expanduser().resolve()
WORK = PROJECT / "work/nox_airborne_20261120"
DATA = PROJECT / "results/nox_airborne_20261120"

from compare_tropomi_neighbor_masks import CORNERS, NO2, xy
from score_no2_overpass_benchmark import TO_NOX_KG_H


STEP_M = 2500.0
SIDE = 21
CENTERS_M = (np.arange(SIDE) - SIDE // 2) * STEP_M
EAST, NORTH = np.meshgrid(CENTERS_M, CENTERS_M)
EDGE_M = CENTERS_M[0] - STEP_M / 2
RADII_KM = (10, 15, 20)
YEARS = (2023, 2024)
CORNER_LON = [f"Longitude_{corner}(deg)" for corner in CORNERS]
CORNER_LAT = [f"Latitude_{corner}(deg)" for corner in CORNERS]
VARIABLES = ["visit_id", NO2, "LONGITUDE(deg)", "LATITUDE(deg)", *CORNER_LON, *CORNER_LAT]
KEY = ["location", "utc_time", "file_path"]
SAMPLES = (np.arange(6) + 0.5) / 6
P, Q = (a.ravel()[None, :, None] for a in np.meshgrid(SAMPLES, SAMPLES))


def paired_visits():
    full = DATA / "annual_retrieval_full_20261121"
    old = DATA / "annual_retrieval"
    score = pd.read_parquet(full / "scored_visits.parquet")
    required = ["gaussian_kg_h", "cross_section_kg_h",
                "gaussian_cems_kg_h", "cross_section_cems_kg_h"]
    keep = score.screen_available & score[required].notna().all(axis=1)
    score = score.loc[keep].copy()
    older = pd.read_parquet(old / "visits.parquet", columns=[*KEY, "visit_id"]).rename(
        columns={"visit_id": "previous_visit_id"}
    )
    score = score.merge(older, on=KEY, how="left", validate="one_to_one")
    score["former_panel"] = score.previous_visit_id.notna()
    score["native_visit_id"] = score.previous_visit_id.fillna(score.visit_id).astype(int)
    score["orbit"] = score.file_path.str.extract(r"_(\d{5})_\d{2}_\d{6}_")[0]
    if score.orbit.isna().any():
        raise ValueError("A selected overpass lacks an orbit identifier")
    return score


def native_grid(pixels, case):
    # Estimate the background upwind of the control volume.
    points = xy(pixels["LONGITUDE(deg)"], pixels["LATITUDE(deg)"], case.longitude, case.latitude)
    column = pixels[NO2].to_numpy()
    speed = np.hypot(case.wind_u, case.wind_v)
    along = points @ (np.array([case.wind_u, case.wind_v]) / speed)
    upwind = (along <= -15) & (np.linalg.norm(points, axis=1) <= 55)
    if upwind.sum() < 4:
        return None
    background = np.median(column[upwind])

    # Subdivide each native quadrilateral. Its local Jacobian gives the
    # sampled area, so oblique footprints do not get the same area as nadir ones.
    local = (np.abs(points) <= 37.5).all(axis=1)
    if not local.any():
        return None
    corners = xy(pixels.loc[local, CORNER_LON], pixels.loc[local, CORNER_LAT],
                 case.longitude, case.latitude) * 1000
    sw, se, ne, nw = (corners[:, k:k + 1, :] for k in range(4))
    position = (1 - P) * ((1 - Q) * sw + Q * nw) + P * ((1 - Q) * se + Q * ne)
    dp = (1 - Q) * (se - sw) + Q * (ne - nw)
    dq = (1 - P) * (nw - sw) + P * (ne - se)
    area = np.abs(dp[..., 0] * dq[..., 1] - dp[..., 1] * dq[..., 0]) / P.size
    index = np.floor((position - EDGE_M) / STEP_M).astype(int)
    valid = (index >= 0).all(axis=2) & (index < SIDE).all(axis=2)
    bins = index[..., 1][valid] * SIDE + index[..., 0][valid]
    area_sum = np.bincount(bins, weights=area[valid], minlength=SIDE**2)
    abundance = ((column[local, None] - background) * area)[valid]
    column_sum = np.bincount(bins, weights=abundance, minlength=SIDE**2)
    # Cap coverage at one to avoid overweighting visits with overlapping pixels.
    coverage = np.minimum(area_sum / STEP_M**2, 1.0)
    mean = np.divide(column_sum, area_sum, out=np.zeros_like(column_sum), where=area_sum > 0)
    weighted = mean * coverage
    return np.stack((coverage, weighted, case.wind_u * weighted, case.wind_v * weighted))


@lru_cache(maxsize=None)
def circle_area(radius_km):
    subcells = (np.arange(12) + 0.5) / 12 - 0.5
    x, y = np.meshgrid(subcells * STEP_M, subcells * STEP_M)
    return STEP_M**2 * np.mean(
        np.hypot(EAST[..., None] + x.ravel(), NORTH[..., None] + y.ravel()) <= radius_km * 1000,
        axis=-1,
    )


def rates(sums, radius_km=15, lifetime_h=3.0):
    """Return signed transport and loss terms for accumulated flux maps."""
    sums = np.asarray(sums, dtype=np.float64).reshape((*sums.shape[:-1], SIDE, SIDE))
    coverage = sums[..., 0, :, :]
    average = np.divide(sums[..., 1:, :, :], coverage[..., None, :, :],
                        out=np.full_like(sums[..., 1:, :, :], np.nan),
                        where=coverage[..., None, :, :] > 0)
    column, x_flux, y_flux = np.moveaxis(average, -3, 0)
    divergence = (np.gradient(x_flux, STEP_M, axis=-1)
                  + np.gradient(y_flux, STEP_M, axis=-2))
    area = circle_area(radius_km)
    inside = area > 0
    return np.stack((
        TO_NOX_KG_H * np.sum(divergence[..., inside] * area[inside], axis=-1),
        TO_NOX_KG_H * np.sum(column[..., inside] * area[inside], axis=-1) / (3600 * lifetime_h),
    ), axis=-1)


def self_check():
    column = np.full_like(EAST, 1e15)
    constant = np.stack((np.ones_like(EAST), column, 4 * column, -2 * column))
    constant = constant.reshape(4, -1)
    result = rates(constant)
    np.testing.assert_allclose(result[0], 0, atol=1e-8)
    np.testing.assert_allclose(result[1], TO_NOX_KG_H * 1e15 * circle_area(15).sum() / 10800)
    linear = constant.copy()
    linear[2] += (1e11 * EAST).ravel()
    np.testing.assert_allclose(rates(linear)[0], TO_NOX_KG_H * 1e11 * circle_area(15).sum())


def build(shard, shards, pilot_orbits):
    self_check()
    data = paired_visits()
    orbits = sorted(data.orbit.unique())
    if pilot_orbits:
        idx = np.linspace(0, len(orbits) - 1, num=pilot_orbits, dtype=int)
        orbits = np.array(orbits)[idx].tolist()
    orbits = orbits[shard::shards]
    data = data.loc[data.orbit.isin(orbits)]
    older = WORK / "annual_retrieval/temis"
    newer = WORK / "annual_retrieval_full_20261121/temis"
    outputs = []
    visits = []
    missing = []
    start = perf_counter()
    for (former, orbit), cases in data.groupby(["former_panel", "orbit"], sort=True):
        path = (older if former else newer) / f"{orbit}.parquet"
        if not path.exists():
            missing.extend((i, "missing native orbit") for i in cases.visit_id)
            continue
        pixels = pq.read_table(path, columns=VARIABLES, use_threads=False).to_pandas()
        groups = pixels.groupby("visit_id", sort=False)
        for case in cases.itertuples(index=False):
            if case.native_visit_id not in groups.indices:
                missing.append((case.visit_id, "native cutout unavailable"))
                continue
            grid = native_grid(groups.get_group(case.native_visit_id), case)
            if grid is None:
                missing.append((case.visit_id, "insufficient area or upwind background"))
                continue
            visits.append(case.visit_id)
            outputs.append(grid.astype(np.float32))
    prefix = "pilot" if pilot_orbits else f"fields_{shard:02d}"
    target = HERE.parent / "results" / prefix
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".part").open("wb") as stream:
        np.savez_compressed(stream, visit=np.asarray(visits, dtype=int), grid=np.asarray(outputs))
    target.with_suffix(".part").replace(target.with_suffix(".npz"))
    pd.DataFrame(missing, columns=["visit_id", "reason"]).to_csv(
        target.with_name(prefix + "_excluded").with_suffix(".csv"), index=False
    )
    print(f"{prefix}: {len(orbits)} orbits, {len(data)} eligible visits, "
          f"{len(visits)} grids, {len(missing)} excluded, {perf_counter() - start:.1f} seconds",
          flush=True)
    if pilot_orbits and outputs:
        terms = rates(np.sum(outputs, axis=0, dtype=np.float64))
        print(f"Pilot pooled transport and loss terms (kg/h): {terms}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--pilot-orbits", type=int, default=0)
    args = parser.parse_args()
    build(args.shard, args.shards, args.pilot_orbits)
