"""Propagate satellite label-review decisions to NOx emission reports."""

from types import SimpleNamespace

import numpy as np

import pandas as pd

from evaluate_nox_facility_overlap import AGES, prepare_scene

from score_no2_overpass_benchmark import estimate_slice, pixel_polygons

LIFETIME_HOURS = 3.0

def gaussian_scene(case, pixels):
    sites = pd.DataFrame({"lon": [case.longitude], "lat": [case.latitude]})
    hours = pd.date_range(case.tropomi_utc.floor("h") - pd.Timedelta(hours=4), periods=6, freq="h")
    wind = pd.DataFrame({"u": case.wind_u, "v": case.wind_v}, index=hours)
    return prepare_scene(pixels, sites, wind, LIFETIME_HOURS, .13)

def gaussian(case, pixels, ground):
    scene = gaussian_scene(case, pixels)
    if scene is None:
        return np.nan, np.nan, np.nan
    design = scene.design[:, 0]
    signed = float(design @ scene.observed / (design @ design))
    ages = case.tropomi_utc - pd.to_timedelta(AGES, unit="h")
    weights = np.exp(-AGES / LIFETIME_HOURS) * np.minimum(AGES / .25, 1)
    reference = float(np.average(ground.reindex(ages.floor("h")), weights=weights))
    return max(0., signed), reference, signed

def cross_section_slices(case, pixels):
    location = SimpleNamespace(lon=case.longitude, lat=case.latitude)
    polygons, columns = pixel_polygons(pixels, location)
    flow = np.array([case.wind_u, case.wind_v])
    speed = float(np.linalg.norm(flow))
    vector = flow / speed
    parts = []
    ages = []
    options = SimpleNamespace(coverage=.70, flanks=2, lifetime=LIFETIME_HOURS)
    for start, stop in [(5., 15.), (15., 25.)]:
        lower = start / (speed * 3.6)
        upper = stop / (speed * 3.6)
        section = estimate_slice(polygons, columns, start*vector, stop*vector, lower, upper, options)
        parts.append(section["estimated_kg_h"])
        ages.append((lower, upper))
    return parts, ages
