"""Separate nearby power-plant plumes in native TEMPO data and score against stack records."""

from dataclasses import dataclass

import json

import numpy as np

import pandas as pd

from score_no2_overpass_benchmark import PIXEL_TIME

from score_no2_overpass_benchmark import CORNERS, NO2, NO2_KG_PER_MOL, NOX_TO_NO2, AVOGADRO, polygon_area, xy

AGES = np.arange(.075, 2.75, .075)

COLUMN_PER_KG_KM2 = AVOGADRO / NO2_KG_PER_MOL / NOX_TO_NO2 * 1e-25  # 10^15 molecules cm^-2

@dataclass
class Scene:
    time: pd.Timestamp
    count: int
    design: np.ndarray
    observed: np.ndarray
    speed: float
    direction: float
    coverage: np.ndarray
    blocks: np.ndarray

def wind_vectors(path, height):
    data = json.loads(path.read_text())["hourly"]
    time = pd.to_datetime(data["time"], utc=True)
    speed = np.array(data[f"wind_speed_{height}m"], dtype=float)
    direction = np.deg2rad(np.array(data[f"wind_direction_{height}m"], dtype=float))
    return pd.DataFrame({"u": -speed * np.sin(direction), "v": -speed * np.cos(direction)}, index=time)

def plume_path(time, wind):
    ages = np.linspace(0, AGES[-1], len(AGES) + 1)
    clock = (wind.index - time).total_seconds() / 3600
    vectors = np.column_stack([np.interp(-ages, clock, wind[k]) for k in ("u", "v")])
    speed = np.linalg.norm(vectors, axis=1).mean()
    path = np.cumsum((vectors[1:] + vectors[:-1]) / 2 * (3.6 * (ages[1] - ages[0])), axis=0)
    direction = float(np.rad2deg(np.arctan2(path[-1, 0], path[-1, 1])) % 360)
    return path, speed, direction

def pixel_quadrature(corners):
    lower = (1 - 1 / np.sqrt(3)) / 2
    upper = 1 - lower
    sw, se, ne, nw = corners.transpose((1, 0, 2))
    return np.stack([(1 - y) * ((1 - x) * sw + x * se) + y * ((1 - x) * nw + x * ne)
                     for x in (lower, upper) for y in (lower, upper)], axis=1)

def plume_kernels(corners, origins, positions, lifetime, dispersion):
    sigma = np.hypot(1.5, dispersion * np.linalg.norm(positions, axis=1))
    column = COLUMN_PER_KG_KM2 * .075 * np.exp(-AGES / lifetime) * np.minimum(AGES / .25, 1) / (2 * np.pi * sigma ** 2)
    locations = pixel_quadrature(corners)
    design = []
    for origin in origins:
        dr = locations[:, :, None, :] - (origin + positions)[None, None, :, :]
        exponent = np.square(dr).sum(axis=-1) / (2 * sigma ** 2)
        design.append((np.exp(-exponent) * column).sum(axis=-1).mean(axis=1))
    return np.column_stack(design)

def prepare_scene(pixels, sites, wind, lifetime, dispersion):
    time = pixels[PIXEL_TIME].sort_values().iloc[len(pixels) // 2]
    positions, speed, direction = plume_path(time, wind)
    if not 2 <= speed <= 16:
        return None
    lon, lat = sites.lon.mean(), sites.lat.mean()
    origins = xy(sites.lon.to_numpy(), sites.lat.to_numpy(), lon, lat)
    pixels = pixels.drop_duplicates(["LONGITUDE(deg)", "LATITUDE(deg)"])
    corners = xy(pixels[[f"Longitude_{c}(deg)" for c in CORNERS]].to_numpy(),
                 pixels[[f"Latitude_{c}(deg)" for c in CORNERS]].to_numpy(), lon, lat)
    obs = pixels[NO2].to_numpy() / 1e15
    centers = corners.mean(axis=1)
    near = np.linalg.norm(centers[:, None, :] - origins[None, :, :], axis=2).min(axis=1) < 58
    good = near & np.isfinite(obs) & np.isfinite(corners).all(axis=(1, 2)) & (polygon_area(corners) > 0)
    if good.sum() < 120:
        return None
    obs, centers, corners = obs[good], centers[good], corners[good]
    kernels = plume_kernels(corners, origins, positions, lifetime, dispersion)
    # Count sampled native pixels in each modeled plume, independently of measured NO2 or CEMS.
    coverage = (kernels > .08 * kernels.max(axis=0)).sum(axis=0)
    shaft = origins[:, None, :] + positions[None, :max(1, round(1.4 / .075)), :]
    covered_axis = np.linalg.norm(centers[:, None, None, :] - shaft[None, :, :, :], axis=-1).min(axis=0) < 5
    if (coverage < 12).any() or (covered_axis.mean(axis=1) < .55).any() or (kernels.max(axis=0) < .0002).any():
        return None
    background = np.column_stack([np.ones(len(obs)), centers / 55])
    plane = np.linalg.qr(background, mode="reduced")[0]
    design = kernels - plane @ (plane.T @ kernels)
    observed = obs - plane @ (plane.T @ obs)
    blocks = np.unique(np.floor(centers / 10), return_inverse=True, axis=0)[1]
    return Scene(time, len(obs), design, observed, speed, direction, coverage, blocks)
