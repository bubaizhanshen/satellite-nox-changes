"""Compare circular and wind-aware plant masks on native satellite pixels."""

import numpy as np

import pandas as pd

RSIG = "https://ofmpub.epa.gov/rsig/rsigserver"

VARIABLE = "tropomi.offl.no2.nitrogendioxide_tropospheric_column"

NO2 = "nitrogendioxide_tropospheric_column(molecules/cm2)"

CORNERS = ["SW", "SE", "NE", "NW"]

NEIGHBOR_UPWIND_BUFFER_KM = 2.

def xy(lon, lat, center_lon, center_lat):
    x = (np.asarray(lon) - center_lon) * 111.195 * np.cos(np.deg2rad(center_lat))
    y = (np.asarray(lat) - center_lat) * 111.195
    return np.stack([x, y], axis=-1)

def local_sources(case, plants, cities):
    smaller = plants.annual.lt(case.annual_nox_emission)
    others = plants.loc[~smaller & plants.annual.notna() & plants.index.to_series().ne(case.location)]
    origin = case.longitude, case.latitude
    points = xy(others.Longitude, others.Latitude, *origin)
    points = points[np.linalg.norm(points, axis=1) <= 150.]
    close_cities = cities.loc[cities.longitude.sub(case.longitude).abs().le(2.8)
                             & cities.latitude.sub(case.latitude).abs().le(1.5)]
    centers = xy(close_cities.longitude, close_cities.latitude, *origin)
    radii = (9 * np.log10(close_cities.population.to_numpy())).clip(10, 90)
    return points, centers, radii

def masks(points, corners, wind, plants, cities, city_radii):
    urban = np.zeros(len(points), dtype=bool)
    for place, radius in zip(cities, city_radii):
        urban |= np.linalg.norm(points - place, axis=1) <= radius
    disk, directed = urban.copy(), urban.copy()
    for plant in plants:
        within = np.linalg.norm(points - plant, axis=1) <= 20
        disk |= within
        # A footprint can be restored only when even its furthest downwind
        # corner is upstream of the other source by a conservative buffer.
        downwind_edge = ((corners - plant) @ wind).max(axis=1)
        directed |= within & (downwind_edge >= -NEIGHBOR_UPWIND_BUFFER_KM)
    return disk, directed

def plume_search(points, wind):
    r = np.linalg.norm(points, axis=1)
    return ((r <= 20.) & (points @ wind >= r * np.cos(np.deg2rad(25)))) | (r <= 5.)

def pixel_area(corners):
    nxt = np.roll(corners, -1, axis=1)
    return np.abs((corners[:, :, 0] * nxt[:, :, 1] - corners[:, :, 1] * nxt[:, :, 0]).sum(axis=1)) / 2

def swath_evidence(case, path, plants, cities):
    swath = path if isinstance(path, pd.DataFrame) else pd.read_csv(path, sep="\t")
    swath = swath.drop_duplicates(["LONGITUDE(deg)", "LATITUDE(deg)"])
    pos = xy(swath["LONGITUDE(deg)"], swath["LATITUDE(deg)"], case.longitude, case.latitude)
    corners = xy(swath[[f"Longitude_{c}(deg)" for c in CORNERS]], swath[[f"Latitude_{c}(deg)" for c in CORNERS]],
                 case.longitude, case.latitude)
    column = swath[NO2].to_numpy() / 6.02214076e19  # molecules/cm² to mol/m²
    good = np.isfinite(column) & np.isfinite(corners).all(axis=(1, 2))
    pos, corners, column = pos[good], corners[good], column[good]
    area = pixel_area(corners)
    if len(column) < 20:
        raise ValueError("Too few valid pixels")
    wind = np.array([case.wind_u, case.wind_v]) / case.wind_m_s
    original, oriented = masks(pos, corners, wind, *local_sources(case, plants, cities))
    radius = np.linalg.norm(pos, axis=1)
    background_pixels = ~original & (radius >= 10) & (radius <= 100) & (pos @ wind <= -radius * .5)
    if background_pixels.sum() < 8:
        raise ValueError("Too few pixels in the shared upwind background")
    background = np.median(column[background_pixels])
    enhancement = column - background
    nearby = enhancement[radius <= 50]
    if len(nearby) < 6:
        raise ValueError("Too few pixels for the local enhancement threshold")
    median = np.median(nearby)
    spread = 1.4826 * np.median(np.abs(nearby - median))
    if spread <= 0:
        spread = np.std(nearby)
    threshold = max(5e-6, median + 2 * spread)
    search = plume_search(pos, wind)
    return corners, area, enhancement, search, original, oriented, median, spread, threshold, int(background_pixels.sum())
