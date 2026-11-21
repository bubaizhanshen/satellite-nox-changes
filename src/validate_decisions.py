"""Validate fuel-based decline decisions on a later year pair and held-out states."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from paired_change import full_stack_months, train_prior
from satellite_fuel_baselines import CHANGE_THRESHOLD, activity_data, hours_weighted, physical_features


def difference(series, year):
    annual = series.unstack("year")
    return annual[year + 1] - annual[year]


def fuel_prediction(activity, stack, training_sites, target_sites, year):
    factors = train_prior(activity, stack, training_sites, year)
    selected = activity.loc[activity.location.isin(target_sites) & activity.year.isin([year, year+1])].copy()
    selected["fuel_prior"] = physical_features(selected) @ factors
    return difference(hours_weighted(selected, "fuel_prior", "hours"), year)


def adequate_ground(stack, year):
    support = stack.groupby(["location", "year"]).stack_hours.sum()
    yearly_days = support.index.get_level_values("year").map(
        lambda value: 366.0 if pd.Timestamp(f"{value}-12-31").is_leap_year else 365.0,
    )
    full_year = hours_weighted(stack, "reference_all_hours_kg_h", "stack_hours")
    full_year = full_year.where(support.ge(.95 * yearly_days * 24))
    return difference(full_year, year)


def quantile_bound(scores, coverage):
    order = np.sort(scores.dropna().to_numpy())
    index = int(np.ceil((len(order) + 1) * coverage))
    return order[index - 1] if 1 <= index <= len(order) else np.inf


def score_methods(covers):
    records = []
    for time_window, frame, policies in covers:
        target = frame["cems"]
        decline = target.lt(-CHANGE_THRESHOLD)
        for name in policies:
            pred = frame[name]
            matched = pred.notna() & target.notna()
            error = (pred - target).abs()
            classify = pred.lt(-CHANGE_THRESHOLD) & matched
            records.append(dict(
                years=time_window, rule=name, plants=int(matched.sum()), measured_declines=int((matched & decline).sum()),
                true_declines=int((classify & decline).sum()), false_declines=int((classify & ~decline).sum()),
                mae=error[matched].mean(),
                upper_bound_coverage=float(target[matched].le(pred[matched]).mean()) if "upper" in name else np.nan,
            ))
    return pd.DataFrame(records)


def evaluate(activities, stack, states):
    earlier_reference = adequate_ground(stack, 2022)
    target_reference = adequate_ground(stack, 2023)
    older, later = [], []
    for fold, (train_rows, test_rows) in enumerate(GroupKFold(n_splits=5).split(states, groups=states), 1):
        training, test = states.iloc[train_rows], states.iloc[test_rows]
        assert set(training).isdisjoint(test)
        earlier = fuel_prediction(activities, stack, training.index, test.index, 2022).rename("fuel")
        older.append(pd.DataFrame({"fuel": earlier, "cems": earlier_reference.reindex(earlier.index), "fold": fold}))

        calibration_predictions = []
        # Refit the older model without the calibration states, and keep all outer test states excluded.
        for rows_fit, rows_cal in GroupKFold(n_splits=4).split(training, groups=training):
            prediction = fuel_prediction(activities, stack, training.index[rows_fit], training.index[rows_cal], 2022)
            calibration_predictions.append(prediction)
        known = pd.concat(calibration_predictions)
        assert known.index.is_unique and not len(known.index.intersection(test.index))
        prior_residuals = earlier_reference.reindex(known.index) - known
        normalized = prior_residuals / (CHANGE_THRESHOLD + known.abs())
        target = fuel_prediction(activities, stack, training.index, test.index, 2023)
        result = pd.DataFrame({"fuel": target, "cems": target_reference.reindex(target.index), "fold": fold})
        for level in (.9, .95):
            key = int(100 * level)
            result[f"upper_{key}"] = target + quantile_bound(prior_residuals, level)
            result[f"scaled_upper_{key}"] = target + (CHANGE_THRESHOLD + target.abs()) * quantile_bound(normalized, level)
        print(f"Fold {fold}: {prior_residuals.notna().sum()} earlier calibration plants, {len(result)} later tests.",
              flush=True)
        later.append(result)
    return pd.concat(older).sort_index(), pd.concat(later).sort_index()


def workflow(second, path):
    columns = ["location", "fuel_change", "paired_satellite_fuel", "all_images", "State",
               "weather_balanced_satellite_fuel"]
    paired = pd.read_csv(path, usecols=columns).set_index("location").reindex(second.index)
    if not np.allclose(paired.fuel_change, second.fuel):
        raise ValueError("The two stages must use the same activity estimate")
    table = second[["cems", "fuel", "upper_90", "upper_95"]].join(paired.drop(columns="fuel_change"))
    substantial = table.fuel.abs().gt(CHANGE_THRESHOLD)
    table["change_with_targeted_satellite"] = table.paired_satellite_fuel.where(substantial, table.fuel)
    direction = np.sign(table.paired_satellite_fuel)
    agreement = direction.eq(np.sign(table.weather_balanced_satellite_fuel))
    table["satellite_disagreement_review"] = (
        ~substantial & agreement & table.paired_satellite_fuel.abs().gt(CHANGE_THRESHOLD)
        & table.weather_balanced_satellite_fuel.abs().gt(CHANGE_THRESHOLD)
    )
    measures = []
    for threshold in ("fuel", "upper_90", "upper_95"):
        selected = table[threshold].lt(-CHANGE_THRESHOLD)
        table[f"decline_flag_{threshold}"] = selected
        corrected = table.paired_satellite_fuel.where(selected, table.fuel)
        table[f"refined_change_{threshold}"] = corrected
        for method in ("all_images", "fuel", "paired_satellite_fuel"):
            error = table.loc[selected, method] - table.loc[selected, "cems"]
            measures.append(dict(rule=threshold, method=method, selected=int(selected.sum()),
                                 verified_declines=int((selected & table.cems.lt(-CHANGE_THRESHOLD)).sum()),
                                 selected_mae=error.abs().mean()))
    table.to_csv("workflow_2023_2024.csv")
    pd.DataFrame(measures).to_csv("workflow_selected_errors.csv", index=False)
    print(pd.DataFrame(measures).round(2).to_string(index=False))
    return table


def full_panel_metrics(results, table):
    panel = pd.read_csv(results / "panel.csv").set_index("location")
    source = pd.read_csv(results / "plant_annual_changes.csv")
    original = source.loc[source.scope.eq("all_valid_overpasses") & source.policy.eq("all_signed")
                          & source.flux_method.isin(["cross_section", "gaussian"])]
    baseline = original.groupby("location").estimated_rate_kg_h_change.mean()
    full = table.reindex(panel.index)
    full["observed_change"] = panel.change_kg_h
    full["State"] = panel.State
    full["satellite_only"] = baseline
    full["activity"] = full.fuel.fillna(baseline)
    full["targeted_satellite"] = full.refined_change_fuel.fillna(baseline)
    full["large_change_fusion"] = full.change_with_targeted_satellite.fillna(baseline)
    columns = ["satellite_only", "activity", "targeted_satellite", "large_change_fusion"]
    if full[columns].isna().any().any():
        raise ValueError("Missing point estimates after applying the satellite fallback")

    truth = full.observed_change
    groups = dict(all=truth.notna(), large=truth.abs().gt(CHANGE_THRESHOLD), declines=truth.lt(-CHANGE_THRESHOLD))
    records = []
    for mode, name in (("targeted_satellite", "paired on predicted declines"),
                       ("large_change_fusion", "paired on predicted changes")):
        for reference in ("satellite_only", "activity"):
            for group, subset in groups.items():
                errors = full[mode].sub(truth).abs().loc[subset]
                difference = errors - full[reference].sub(truth).abs().loc[subset]
                clusters = difference.groupby(full.State[subset]).agg(["sum", "size"]).to_numpy()
                draw = np.random.default_rng(61285).integers(0, len(clusters), size=(10000, len(clusters)))
                draws = clusters[draw].sum(axis=1)
                low, high = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975])
                records.append(dict(method=name, baseline=reference, group=group, n=int(subset.sum()),
                                    mae=errors.mean(), difference=difference.mean(), low=low, high=high))
    full[["observed_change", "State", *columns]].to_csv("full_panel_fallback.csv")
    comparison = pd.DataFrame(records)
    comparison.to_csv("full_panel_fallback_differences.csv", index=False)
    print(comparison.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--ground", required=True, type=Path)
    parser.add_argument("--activity", type=Path, default=Path("eia923_monthly_activity_2022_2024.csv"))
    parser.add_argument("--fusion", type=Path)
    args = parser.parse_args()
    all_activity = activity_data(args.activity)
    recent_count = all_activity.loc[all_activity.year.ge(2023)].groupby("location").year.nunique()
    panel = pd.read_csv(args.results / "panel.csv").set_index("location")
    panel = panel.loc[panel.index.intersection(recent_count.index[recent_count.eq(2)])]
    all_activity = all_activity.loc[all_activity.location.isin(panel.index)]
    stack = full_stack_months(args.ground, panel.index, (2022, 2023, 2024))
    with threadpool_limits(limits=4):
        first, second = evaluate(all_activity, stack, panel.State)
    first = first.dropna(subset=["fuel", "cems"])
    check = panel.change_kg_h.reindex(second.index)
    if not np.allclose(second.cems, check):
        raise ValueError("Later test reference differs from the main panel")
    intervals = pd.read_csv(args.results / "annual_direction_sampling_intervals.csv")
    baseline = intervals.loc[intervals.flux_method.eq("cross_section") & intervals.policy.eq("all_signed")]
    second["satellite_sample_upper"] = baseline.set_index("location").upper_kg_h
    for frame in (first, second):
        frame["stable"] = 0.0
    metrics = score_methods([
        ("2022–2023", first, ["stable", "fuel"]),
        ("2023–2024", second, ["stable", "fuel", "satellite_sample_upper", "upper_90", "upper_95",
                               "scaled_upper_90", "scaled_upper_95"]),
    ])
    metrics.loc[metrics.rule.str.contains("upper"), "mae"] = np.nan
    first.to_csv("decisions_2022_2023.csv")
    second.to_csv("decisions_2023_2024.csv")
    metrics.to_csv("decision_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))
    if args.fusion:
        full_panel_metrics(args.results, workflow(second, args.fusion))


if __name__ == "__main__":
    main()
