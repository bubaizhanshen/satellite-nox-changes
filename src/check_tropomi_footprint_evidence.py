"""Measure how one large TROPOMI pixel can pass an area-based plume test."""

from itertools import combinations

import numpy as np

import pandas as pd

from scipy.optimize import linear_sum_assignment

from compare_tropomi_neighbor_masks import swath_evidence

def compare_angles(data):
    output = []
    rng = np.random.default_rng(10923)
    for state, subset in data.groupby("state"):
        contrast = subset.groupby(["location", "month", "angle"]).plume_label.agg(["size", "mean"]).unstack().dropna()
        weight = contrast["size"].near * contrast["size"].mid / contrast["size"].sum(axis=1)
        delta = contrast["mean"].mid - contrast["mean"].near
        plants = pd.DataFrame({"num": weight * delta, "weight": weight}).groupby(level=0).sum().to_numpy()
        bootstrap = plants[rng.integers(0, len(plants), size=(3000, len(plants)))].sum(axis=1)
        lower, upper = np.quantile(bootstrap[:, 0] / bootstrap[:, 1], [.025, .975])
        means = subset.groupby("angle").plume_label.agg(["mean", "size"])
        output.append({"state": state, "near_records": means.at["near", "size"], "mid_records": means.at["mid", "size"],
                       "near_fraction": means.at["near", "mean"], "mid_fraction": means.at["mid", "mean"],
                       "site_months": len(contrast), "plants": len(plants),
                       "matched_difference": np.average(delta, weights=weight), "low": lower, "high": upper})
    return pd.DataFrame(output).set_index("state")

def match_observing_conditions(data):
    fields = ["record", "utc_time", "plume_label", "mean_mass_2h", "wind_u", "wind_v",
              "wind_m_s", "cloud_fraction_crb", "sensor_zenith_angle"]
    matches = []
    for (plant, month, state), group in data.dropna(subset=["cloud_fraction_crb"]).groupby(["location", "month", "state"]):
        near = group.loc[group.angle.eq("near")].reset_index(drop=True)
        mid = group.loc[group.angle.eq("mid")].reset_index(drop=True)
        if near.empty or mid.empty:
            continue
        same = {}
        for name, limit in [("wind_m_s", 2.5), ("cloud_fraction_crb", .15)]:
            diff = abs(near[name].to_numpy()[:, None] - mid[name].to_numpy()[None, :])
            same[name] = diff / limit
        left = near[["wind_u", "wind_v"]].to_numpy() / near.wind_m_s.to_numpy()[:, None]
        right = mid[["wind_u", "wind_v"]].to_numpy() / mid.wind_m_s.to_numpy()[:, None]
        same["wind_angle"] = np.degrees(np.arccos(np.clip(left @ right.T, -1., 1.))) / 60.
        a = near.mean_mass_2h.to_numpy()[:, None]
        b = mid.mean_mass_2h.to_numpy()[None, :]
        same["rate"] = abs(a - b) / np.maximum(1, .175 * (a + b))
        features = np.stack(list(same.values()))
        eligible = np.isfinite(features).all(axis=0) & (features <= 1).all(axis=0)
        # A disallowed assignment must cost more than the cost of all allowed
        # matches combined, so matching first maximizes the number of pairs.
        disallowed = 1 + len(same) * min(len(mid), len(near))
        cost = np.where(eligible, features.sum(axis=0), disallowed)
        i, j = linear_sum_assignment(cost)
        good = eligible[i, j]
        if not np.any(good):
            continue
        a = near.iloc[i[good]][fields].reset_index(drop=True).add_suffix("_near")
        b = mid.iloc[j[good]][fields].reset_index(drop=True).add_suffix("_mid")
        matches.append(pd.concat([a, b], axis=1).assign(location=plant, month=month, state=state))
    return pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()

def shared_edges(corners):
    edges = {}
    # The supplied corner arrays use shared vertices in the native swath.
    # Round to 10 m to absorb small differences in the reported coordinates.
    vertices = np.round(corners * 100).astype(int)
    for i, outline in enumerate(vertices):
        for j in range(4):
            endpoints = [tuple(outline[j]), tuple(outline[(j + 1) % 4])]
            key = tuple(sorted(endpoints))
            edges.setdefault(key, []).append(i)
    return sorted({pair for occupants in edges.values() for pair in combinations(occupants, 2)})

def plume_template_score(corners, gain, excluded, wind, spread):
    centers = corners.mean(axis=1)
    keep = ~excluded & (np.linalg.norm(centers, axis=1) <= 26.)
    plume = corners[keep]
    if len(plume) < 12:
        return -np.inf
    points = (0.1127016654, .5, 0.8872983346)
    weights = (5 / 18, 4 / 9, 5 / 18)
    integral = np.zeros(len(plume))
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            pos = (1 - a) * ((1 - b) * plume[:, 0] + b * plume[:, 1]) + a * ((1 - b) * plume[:, 3] + b * plume[:, 2])
            along, across = pos @ wind, pos @ np.array([-wind[1], wind[0]])
            advancing = np.maximum(along, 0)
            width = 1.7 + .12 * advancing
            kernel = 1.7 / width * np.exp(-.5 * (across / width) ** 2 - advancing / 12)
            integral += weights[i] * weights[j] * kernel * (along >= 1) * (along <= 20)
    background = np.column_stack([np.ones(len(plume)), centers[keep] / 26])
    orthogonal = np.linalg.qr(background, mode="reduced")[0]
    profile = integral - orthogonal @ (orthogonal.T @ integral)
    scale = np.linalg.norm(profile) * spread
    return float(profile @ gain[keep] / scale) if scale > 1e-12 else -np.inf

def score_swath(case, path, plants, cities):
    corners, area, gain, search, original, _, center, spread, threshold, n_background = swath_evidence(case, path, plants, cities)
    eligible = search & ~original & (gain > 5e-6)
    good_area = area[eligible]
    z = (gain[eligible] - center) / max(spread, 1e-12)
    by_score = np.argsort(z)[::-1]
    accumulated = np.cumsum(good_area[by_score])
    area_score = z[by_score[np.searchsorted(accumulated, 25.)]] if accumulated.size and accumulated[-1] >= 25 else -np.inf
    pairs = shared_edges(corners[eligible])
    relational = [min(z[i], z[j]) for i, j in pairs if good_area[i] + good_area[j] >= 25.]
    joint_score = max(relational, default=-np.inf)
    weak = search & ~original & (gain > 0) & (gain > center)
    positions = corners[weak].mean(axis=1)
    weak_z = (gain[weak] - center) / max(spread, 1e-12)
    weak_gain = gain[weak]
    weak_area = area[weak]
    wind = np.array([case.wind_u, case.wind_v]) / case.wind_m_s
    supported = []
    for i, j in shared_edges(corners[weak]):
        step = positions[i] - positions[j]
        if (max(weak_gain[i], weak_gain[j]) > 5e-6 and weak_area[i] + weak_area[j] >= 25
                and abs(step @ wind) >= .4 * np.linalg.norm(step)):
            supported.append(2 / (1 / weak_z[i] + 1 / weak_z[j]))
    flags = search & ~original & (gain > threshold)
    search_areas = area[search & ~original]
    return {"valid_pixels": len(area), "background_pixels": n_background,
            "plume_pixels": int(flags.sum()), "plume_area_km2": area[flags].sum(),
            "eligible_pixel_edges": len(pairs), "max_flagged_pixel_km2": max(area[flags], default=np.nan),
            "median_search_pixel_km2": float(np.median(search_areas)) if len(search_areas) else np.nan,
            "search_pixels_at_least_25_km2": int(np.sum(search_areas >= 25)),
            "area_score": area_score, "adjacent_score": joint_score, "aligned_pair_score": max(supported, default=-np.inf),
            "plume_template_score": plume_template_score(corners, gain, original, wind, spread)}


def recall_cutoff(data, field):
    active = data.loc[data.status.eq("active")].copy()
    total = active.weight.sum()
    required = active.loc[active.area_score.gt(2), "weight"].sum()
    active = active.loc[np.isfinite(active[field])].sort_values(field, ascending=False)
    reached = active.weight.cumsum().ge(required)
    return float(active.loc[reached, field].iloc[0]) if reached.any() and total > 0 else np.nan
