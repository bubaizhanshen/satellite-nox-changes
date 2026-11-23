"""Compare annual flux divergence and per-overpass estimates on paired visits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from annual_flux_divergence import DATA, HERE, RADII_KM, rates, paired_visits


RESULTS = HERE.parent / "results"
PANEL = DATA / "annual_retrieval_full_20261121/panel.csv"
VISIT_METHODS = ("gaussian", "cross_section")
SCALARS = ["gaussian_kg_h", "cross_section_kg_h", "cross_section_cems_kg_h", "mean_mass_2h"]
METHODS = (*VISIT_METHODS, "divergence", "matched_stack", "two_hour_stack")
CUT = 25.0


def sample_blocks(group, grids, draws, rng):
    """Resample one-week groups within each calendar month."""
    blocks = []
    weights = []
    scalars = []
    sizes = []
    for _, month in group.groupby("month", sort=True):
        indices = month.groupby((month.utc_time.dt.day - 1) // 7).indices.values()
        indices = [month.field_index.to_numpy()[i] for i in indices]
        blocks.extend(np.sum(grids[i], axis=0, dtype=np.float64) for i in indices)
        month = month.set_index("field_index")
        scalars.extend(month.loc[i, SCALARS].sum().to_numpy() for i in indices)
        sizes.extend(map(len, indices))
        weights.append(rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=draws))
    counts = np.concatenate(weights, axis=1)
    blocks = np.asarray(blocks)
    # Cells outside the source region and its finite-difference stencil do not enter the rate.
    field_samples = (counts @ blocks.reshape(len(blocks), -1)).reshape(draws, *blocks.shape[1:])
    sampled_rates = (counts @ np.asarray(scalars)) / (counts @ np.asarray(sizes))[:, None]
    divergence = rates(field_samples).sum(axis=-1)
    return np.column_stack([sampled_rates[:, :2].clip(0), divergence.clip(0), sampled_rates[:, 2:]])


def mean_estimates(group, grids):
    summary = group[SCALARS].mean()
    field = grids[group.field_index].sum(axis=0, dtype=np.float64)
    results = dict(zip(VISIT_METHODS, summary.iloc[:2].clip(0)))
    transport, loss = rates(field)
    results.update(
        divergence=max(transport + loss, 0),
        signed_divergence=transport + loss, transport=transport, loss=loss,
        matched_stack=summary.cross_section_cems_kg_h,
        two_hour_stack=summary.mean_mass_2h,
    )
    for radius in RADII_KM:
        if radius != 15:
            results[f"divergence_{radius}km"] = max(rates(field, radius_km=radius).sum(), 0)
    for hours in (2, 6):
        results[f"divergence_{hours}h"] = max(transport + loss * 3 / hours, 0)
    return results


def compute(draws):
    visits, grids = [], []
    for path in sorted(RESULTS.glob("fields_*.npz")):
        with np.load(path) as archive:
            visits.append(archive["visit"])
            if len(archive["visit"]):
                grids.append(archive["grid"])
    ids = np.concatenate(visits)
    grids = np.concatenate(grids)
    metadata = paired_visits().merge(
        pd.DataFrame({"visit_id": ids, "field_index": np.arange(len(ids))}),
        on="visit_id", how="inner", validate="one_to_one",
    )
    if len(metadata) != len(ids):
        raise ValueError("Saved flux fields do not match the paired overpasses")
    panel = pd.read_csv(PANEL).set_index("location")
    estimates, intervals = [], []
    for location, group in metadata.groupby("location"):
        years = [group.loc[group.year.eq(year)] for year in (2023, 2024)]
        if not all(len(year) >= 36 and year.month.nunique() >= 9 for year in years):
            continue
        selected = [year.loc[year.plume_label] for year in years]
        state = panel.loc[location, "State"]
        reference = panel.loc[location, ["cems_2023", "cems_2024"]].to_numpy(dtype=float)
        for policy, periods in [("all", years), ("detected", selected)]:
            if any(year.empty for year in periods):
                continue
            first, second = [mean_estimates(year, grids) for year in periods]
            for method in first:
                estimates.append(dict(
                    location=location, state=state, policy=policy, method=method,
                    rate_2023=first[method], rate_2024=second[method],
                    change=second[method] - first[method], reference=reference[1] - reference[0],
                    reference_2023=reference[0], reference_2024=reference[1],
                    visits_2023=len(periods[0]), visits_2024=len(periods[1]),
                    months_2023=periods[0].month.nunique(), months_2024=periods[1].month.nunique(),
                ))
        if draws:
            rng = np.random.default_rng([5494363, int(location)])
            changes = sample_blocks(years[1], grids, draws, rng) - sample_blocks(years[0], grids, draws, rng)
            for method, column in zip(METHODS, changes.T, strict=True):
                finite = np.isfinite(column)
                lower, upper = (np.quantile(column[finite], [0.025, 0.975])
                                if finite.mean() >= 0.95 else (np.nan, np.nan))
                intervals.append(dict(location=location, method=method, lower=lower, upper=upper,
                                      valid_draw_fraction=finite.mean()))
    table = pd.DataFrame(estimates)
    bounds = pd.DataFrame(intervals)
    table.to_csv(RESULTS / "annual_flux_plant_estimates.csv", index=False)
    bounds.to_csv(RESULTS / "annual_flux_intervals.csv", index=False)
    return table, bounds, panel, len(ids)


def paired_difference(df, baseline, candidate):
    differences = (
        (df[candidate] - df.reference).abs() - (df[baseline] - df.reference).abs()
    )
    totals = differences.groupby(df.state).agg(["sum", "size"])
    rng = np.random.default_rng(331170)
    samples = rng.integers(0, len(totals), size=(2000, len(totals)))
    values = totals.to_numpy()[samples].sum(axis=1)
    return np.array([differences.mean(), *np.quantile(values[:, 0] / values[:, 1], [0.025, 0.975])])


def summarize(table, bounds, panel, count):
    point = table.pivot(index=["location", "state", "reference"], columns=["policy", "method"], values="change")
    all_methods = point["all"].dropna(subset=list(METHODS)).reset_index().rename_axis(columns=None)
    detected_by_id = point["detected"].reindex(columns=list(METHODS)) if "detected" in point else pd.DataFrame()
    detected_sites = detected_by_id.index.get_level_values("location")[detected_by_id.notna().all(axis=1)]
    all_methods["no_change"] = 0.0
    all_methods["detected_available"] = all_methods.location.isin(detected_sites)
    methods = ("no_change", *METHODS)
    policy_comparisons = []
    for method in METHODS:
        pair = point.loc[:, [("all", method), ("detected", method)]].dropna().reset_index()
        pair.columns = ["location", "state", "reference", "all", "detected"]
        error = {name: (pair[name] - pair.reference).abs().mean() for name in ("all", "detected")}
        change, low, high = paired_difference(pair, "detected", "all")
        policy_comparisons.append(dict(method=method, plants=len(pair), all_mae=error["all"],
                                       detected_mae=error["detected"], mae_delta=change, low=low, high=high))
    pd.DataFrame(policy_comparisons).to_csv(RESULTS / "annual_flux_screening.csv", index=False)
    groups = {
        "all": all_methods,
        "large": all_methods.loc[all_methods.reference.abs().gt(CUT)],
        "declines": all_methods.loc[all_methods.reference.lt(-CUT)],
        "newly_estimable": all_methods.loc[~all_methods.detected_available],
        "detected_eligible": all_methods.loc[all_methods.detected_available],
    }
    results = []
    for name, df in groups.items():
        for method in [*methods, *(f"divergence_{suffix}" for suffix in ("10km", "20km", "2h", "6h"))]:
            usable = df.dropna(subset=[method])
            if usable.empty:
                continue
            truth = usable.reference
            errors = usable[method] - truth
            row = dict(group=name, method=method, plants=len(usable), mae=errors.abs().mean(),
                       bias=errors.mean(), same_direction_large=int(
                           ((truth.abs() > CUT) & (usable[method].abs() > CUT)
                            & (truth * usable[method] > 0)).sum()))
            if method in METHODS:
                diff = paired_difference(usable, "cross_section", method)
                row.update(delta_mae=diff[0], ci_low=diff[1], ci_high=diff[2])
            if not bounds.empty and method in METHODS:
                lower_upper = bounds.loc[bounds.method.eq(method)].set_index("location").reindex(usable.location)
                output = np.select([lower_upper.lower.gt(CUT), lower_upper.upper.lt(-CUT)], [1, -1])
                target = np.select([truth.gt(CUT), truth.lt(-CUT)], [1, -1])
                row.update(
                    decisive_correct=int(((output != 0) & (output == target)).sum()),
                    decisive_wrong=int(((output != 0) & (output != target)).sum()),
                    missed_large=int(((output == 0) & (target != 0)).sum()),
                )
            results.append(row)
    summaries = pd.DataFrame(results)
    summaries.to_csv(RESULTS / "annual_flux_comparison.csv", index=False)
    print(f"{count} paired overpasses in the new gridded fields. "
          f"{len(all_methods)} of {len(panel)} plants support all methods.")
    print(summaries.loc[summaries.method.isin(methods)].round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    summarize(*compute(args.draws))
