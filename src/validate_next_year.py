"""Evaluate the frozen fuel model in 2024–2025 at held-out US facilities."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from paired_change import full_stack_months, train_prior
from satellite_fuel_baselines import CHANGE_THRESHOLD, activity_data, hours_weighted, physical_features
from validate_decisions import adequate_ground, fuel_prediction, quantile_bound


def annual_reference(path, year):
    """Retain plants with valid stack measurements for at least 95% of the year."""
    frame = pd.read_parquet(path, columns=["facility_id", "utc", "mass_kg"])
    if frame.duplicated(["facility_id", "utc"]).any():
        raise ValueError(f"Duplicate plant-hour in {path.name}")
    if not frame.utc.dt.year.eq(year).all() or frame.mass_kg.isna().any():
        raise ValueError(f"Unexpected year or missing mass in {path.name}")
    grouped = frame.groupby("facility_id").mass_kg.agg(["mean", "count"])
    hours = (pd.Timestamp(year + 1, 1, 1) - pd.Timestamp(year, 1, 1)).days * 24
    eligible = grouped.loc[grouped["count"].ge(.95 * hours)]
    return eligible.rename_axis("location").rename(columns={"mean": f"rate_{year}", "count": f"hours_{year}"})


def assign_estimates(later, activity, old_activity, past, old_states):
    older_truth = adequate_ground(past, 2022)
    rows = []
    for fold, (_, test_idx) in enumerate(GroupKFold(n_splits=5).split(later, groups=later.State), 1):
        target = later.iloc[test_idx]
        fit_states = old_states.loc[~old_states.isin(target.State)]
        assert set(target.State).isdisjoint(fit_states)
        coeff = train_prior(old_activity, past, fit_states.index, 2023)

        cal_rows = []
        for old_fit, old_hold in GroupKFold(n_splits=4).split(fit_states, groups=fit_states):
            old_prediction = fuel_prediction(old_activity, past, fit_states.index[old_fit],
                                             fit_states.index[old_hold], 2022)
            cal_rows.append(older_truth.reindex(old_prediction.index) - old_prediction)
        residual = pd.concat(cal_rows, verify_integrity=True)
        frame = activity.loc[activity.location.isin(target.index)].copy()
        frame["fuel_rate"] = physical_features(frame) @ coeff
        means = hours_weighted(frame, "fuel_rate", "hours").unstack("year")
        predicted = means[2025] - means[2024]
        out = target.assign(
            fold=fold, fuel=predicted, zero_change=0.,
            upper_90=predicted + quantile_bound(residual, .90),
            upper_95=predicted + quantile_bound(residual, .95),
            used_calibration_plants=int(residual.notna().sum()),
        )
        rows.append(out)
        print("Validation fold", fold, "test plants", len(out), "train states", fit_states.nunique(),
              "calibration plants", int(residual.notna().sum()), flush=True)
    return pd.concat(rows, verify_integrity=True).sort_index()


def metrics(frame):
    output = []
    for sample, retain in {
        "all": frame.reference.notna(),
        "outside_tropomi_panel": ~frame.in_tropomi_panel,
        "in_tropomi_panel": frame.in_tropomi_panel,
    }.items():
        rows = frame.loc[retain]
        truth = rows.reference
        sizeable = truth.abs().gt(CHANGE_THRESHOLD)
        decreases = truth.lt(-CHANGE_THRESHOLD)
        err = rows.fuel.sub(truth).abs()
        delta = err - truth.abs()
        clusters = delta.groupby(rows.State).agg(["sum", "size"]).to_numpy()
        indices = np.random.default_rng(61285).integers(0, len(clusters), size=(10000, len(clusters)))
        sums = clusters[indices].sum(axis=1)
        low, high = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975])
        for column in ("fuel", "upper_90", "upper_95"):
            rule = rows[column]
            flags = rule.lt(-CHANGE_THRESHOLD)
            output.append(dict(sample=sample, n=len(rows), state_count=rows.State.nunique(),
                               large_changes=int(sizeable.sum()), real_declines=int(decreases.sum()),
                               mae_stable=truth.abs().mean(), mae_fuel=err.mean(),
                               difference=delta.mean(), lower=low, upper=high,
                               mae_stable_large=truth[sizeable].abs().mean(), mae_fuel_large=err[sizeable].mean(),
                               mae_stable_declines=truth[decreases].abs().mean(), mae_fuel_declines=err[decreases].mean(),
                               decision_rule=column, true_declines=int((flags & decreases).sum()),
                               false_declines=int((flags & ~decreases).sum()),
                               observed_coverage=truth.le(rule).mean() if column.startswith("upper") else np.nan))
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--expanded-activity", type=Path, required=True)
    parser.add_argument("--historical-activity", type=Path, required=True)
    parser.add_argument("--old-panel", type=Path, required=True)
    parser.add_argument("--historical-ground", type=Path, required=True)
    parser.add_argument("--base-stack", type=Path, required=True)
    parser.add_argument("--new-stack", type=Path, required=True)
    args = parser.parse_args()
    states = pd.read_csv(args.states).set_index("location").State
    assert states.index.is_unique
    states = states.loc[~states.isin(["AK", "HI", "PR", "GU", "VI", "AS", "MP"])]
    old_panel = pd.read_csv(args.old_panel).set_index("location")
    activity = activity_data(args.expanded_activity)
    activity_sites = activity.groupby("location").year.nunique()
    previous = annual_reference(args.base_stack, 2024)
    current = annual_reference(args.new_stack, 2025)
    all_sites = previous.join(current, how="inner").join(states, how="inner")
    all_sites = all_sites.loc[all_sites.index.intersection(activity_sites.index[activity_sites.eq(2)])]
    all_sites["reference"] = all_sites.rate_2025 - all_sites.rate_2024
    all_sites["in_tropomi_panel"] = all_sites.index.isin(old_panel.index)
    if len(all_sites) <= len(old_panel):
        raise ValueError("The external validation failed to include enough additional plants")
    historic = activity_data(args.historical_activity)
    historic = historic.loc[historic.year.le(2023) & historic.location.isin(old_panel.index)]
    old_states = old_panel.State.reindex(historic.location.unique())
    past = full_stack_months(args.historical_ground, old_states.index, (2022, 2023))
    with threadpool_limits(limits=4):
        estimates = assign_estimates(all_sites, activity, historic, past, old_states)
    summary = metrics(estimates)
    estimates.to_csv("external_2024_2025_estimates.csv")
    summary.to_csv("external_2024_2025_metrics.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
