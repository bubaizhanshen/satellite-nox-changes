"""Compare pooled and time-resolved emission estimates from the same TEMPO scan."""

import numpy as np

import pandas as pd

PIXEL_TIME = "Timestamp(UTC)"

NOX_TO_NO2 = 1.32

NO2_KG_PER_MOL = 46.0055e-3

AVOGADRO = 6.02214076e23

TO_NOX_KG_H = NOX_TO_NO2 * NO2_KG_PER_MOL * 3600 * 1e4 / AVOGADRO

NO2 = "no2_vertical_column_troposphere(molecules/cm2)"

CORNERS = ("SW", "SE", "NE", "NW")

def xy(lon, lat, site_lon, site_lat):
    east = (lon - site_lon) * 111.195 * np.cos(np.deg2rad(site_lat))
    north = (lat - site_lat) * 111.195
    return np.stack((east, north), axis=-1)

def pixel_polygons(pixels, window):
    lon = pixels[[f"Longitude_{s}(deg)" for s in CORNERS]].to_numpy()
    lat = pixels[[f"Latitude_{s}(deg)" for s in CORNERS]].to_numpy()
    coords = xy(lon, lat, window.lon, window.lat)
    valid = np.isfinite(coords).all(axis=(1, 2)) & pixels[NO2].notna().to_numpy()
    valid &= polygon_area(coords) > 0
    return coords[valid], pixels[NO2].to_numpy()[valid]

def polygon_area(points):
    x, y = points[..., 0], points[..., 1]
    return np.abs(np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1)) / 2

def clip_rectangle(points, length, low, high):
    # Sequential half-plane clipping in along- and cross-wind coordinates.
    for axis, edge, sign in ((0, 0.0, 1), (0, length, -1), (1, low, 1), (1, high, -1)):
        output = []
        previous = points[-1]
        prev_inside = sign * (previous[axis] - edge) >= 0
        for point in points:
            inside = sign * (point[axis] - edge) >= 0
            if inside != prev_inside:
                fraction = (edge - previous[axis]) / (point[axis] - previous[axis])
                output.append(previous + fraction * (point - previous))
            if inside:
                output.append(point)
            previous, prev_inside = point, inside
        if len(output) < 3:
            return 0.0
        points = output
    return polygon_area(np.asarray(points))

def area_average(projected_pixels, values, length, low, high):
    along, across = projected_pixels[..., 0], projected_pixels[..., 1]
    inside = (
        (along.max(axis=1) > 0) & (along.min(axis=1) < length)
        & (across.max(axis=1) > low) & (across.min(axis=1) < high)
    )
    overlap = np.array([
        clip_rectangle(points, length, low, high) for points in projected_pixels[inside]
    ])
    sampled = overlap.sum()
    if sampled <= 0:
        return np.nan, 0.0
    return float(np.average(values[inside], weights=overlap)), float(sampled / (length * (high - low)))

def estimate_slice(pixels, values, first, last, younger, older, options):
    delta = last - first
    length = np.linalg.norm(delta)
    age = (younger + older) / 2
    width = 5.5 + 0.045 * np.linalg.norm((first + last) / 2)
    q = {
        "younger_h": younger, "older_h": older, "age_h": age, "length_km": length,
        "width_km": width * 2, "raw_kg_h": np.nan, "estimated_kg_h": np.nan,
        "core_coverage": 0.0, "left_coverage": 0.0, "right_coverage": 0.0,
        "core_no2_molec_cm2": np.nan, "background_no2_molec_cm2": np.nan,
    }
    if length < 1.5:
        return q
    perpendicular = np.array([-delta[1], delta[0]])
    projected = (pixels - first) @ (np.column_stack((delta, perpendicular)) / length)
    core_mean, core_coverage = area_average(projected, values, length, -width, width)
    left_mean, left_coverage = area_average(projected, values, length, width + 4, width + 12)
    right_mean, right_coverage = area_average(projected, values, length, -width - 12, -width - 4)
    q.update(
        core_coverage=core_coverage, left_coverage=left_coverage, right_coverage=right_coverage,
        core_no2_molec_cm2=core_mean,
    )
    flanks = [mean for mean, cover in [(left_mean, left_coverage), (right_mean, right_coverage)] if cover >= .50]
    if core_coverage < options.coverage or len(flanks) < options.flanks:
        return q
    side_background = np.mean(flanks)
    transport_m_s = length * 1000 / ((older - younger) * 3600)
    flux = (core_mean - side_background) * (width * 2000) * transport_m_s * TO_NOX_KG_H
    q.update(
        background_no2_molec_cm2=side_background, raw_kg_h=flux,
        estimated_kg_h=flux * np.exp(age / options.lifetime),
    )
    return q

def reference_mass(ground, scan_time, lower_age, upper_age):
    lower = scan_time - pd.Timedelta(hours=upper_age)
    upper = scan_time - pd.Timedelta(hours=lower_age)
    terms = []
    for hour in pd.date_range(lower.floor("h"), upper.ceil("h"), freq="h", inclusive="left"):
        weight = (min(upper, hour + pd.Timedelta(hours=1)) - max(lower, hour)).total_seconds()
        if weight > 0:
            terms.append((ground.get(hour, np.nan), weight))
    values, weights = np.asarray(terms).T
    return float(np.average(values, weights=weights))
