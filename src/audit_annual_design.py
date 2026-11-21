"""Audit holdout membership, decline rankings, and sensitivity of annual NOx estimates."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from paired_change import full_stack_months, run_folds, train_prior
from satellite_fuel_baselines import CHANGE_THRESHOLD, activity_data, hours_weighted, physical_features, prepare_inputs
from validate_decisions import adequate_ground


CENSUS_REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IN IL MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
REGION = {state: region for region, states in CENSUS_REGIONS.items() for state in states.split()}


def stratified_error(frame, truth_column, methods, cohort):
    truth = frame[truth_column]
    result = []
    for threshold in (0, 10, 25, 50, 100):
        take = truth.notna() if threshold == 0 else truth.abs().gt(threshold)
        for method in methods:
            errors = (frame.loc[take, method] - truth[take]).abs()
            if errors.isna().any():
                raise ValueError(f"Missing estimates from {method} in {cohort}")
            result.append(dict(cohort=cohort, threshold=threshold, method=method, n=int(take.sum()),
                               mae=errors.mean()))
    return result


def ranking(truth, estimates):
    """Treat tied scores as a single threshold and compare the strongest eligible prefix."""
    candidates = (pd.DataFrame({"score": estimates, "hit": truth.lt(-CHANGE_THRESHOLD)})
                  .groupby("score", sort=True).hit.agg(["sum", "size"]))
    candidates["true_calls"] = candidates["sum"].cumsum()
    candidates["false_calls"] = (candidates["size"] - candidates["sum"]).cumsum()
    path = pd.concat([pd.DataFrame({"true_calls": [0], "false_calls": [0]}),
                      candidates[["true_calls", "false_calls"]]], ignore_index=True)
    return path


def decline_curves(primary, paired, workflow, out):
    predictions = {
        "Satellite (cross section)": paired.cross_section,
        "Satellite (two retrievals)": paired.all_images,
        "Fuel only": primary.fuel,
        "Fuel plus satellite": workflow.change_with_targeted_satellite,
    }
    all_paths, results = [], []
    for name, pred in predictions.items():
        pred = pred.reindex(primary.index)
        if pred.isna().any():
            raise ValueError(f"Unpaired point estimates: {name}")
        truth = primary.cems
        path = ranking(truth, pred)
        all_paths.append(path.assign(method=name))
        flag = pred.lt(-CHANGE_THRESHOLD)
        actual = truth.lt(-CHANGE_THRESHOLD)
        ap = average_precision_score(actual, -pred)
        for max_false in (0, 1, 2, 4, 10):
            possible = path.loc[path.false_calls.le(max_false)]
            results.append(dict(method=name, max_false=max_false, true_calls=int(possible.true_calls.max()),
                                average_precision=ap, point_true=int((flag & actual).sum()),
                                point_false=int((flag & ~actual).sum())))
    pd.concat(all_paths, ignore_index=True).to_csv(out / "decline_point_rankings.csv", index=False)
    return pd.DataFrame(results)


def memberships(primary, external, panel, historic, stack, paired):
    data = []
    for test, is_external in ((primary, False), (external, True)):
        pool = panel.State.reindex(historic.location.unique()) if is_external else primary.State
        prior_reference = adequate_ground(stack, 2022)
        complete_fuel = historic.groupby("location").year.nunique()
        adequate = prior_reference.index[prior_reference.notna()].intersection(
            complete_fuel.index[complete_fuel.eq(2)]
        )
        for fold, holdout in test.groupby("fold"):
            states = sorted(set(holdout.State))
            fitting = pool.loc[~pool.isin(states)]
            assert not set(fitting.index) & set(holdout.index)
            assert set(fitting).isdisjoint(states)
            if not is_external and not paired.loc[holdout.index, "fold"].eq(fold).all():
                raise ValueError("Fuel and satellite folds differ")
            months = historic.loc[historic.location.isin(fitting.index) & historic.year.eq(2023)].merge(
                stack, on=["location", "year", "month"], validate="one_to_one"
            )
            months = months.loc[months.stack_hours.ge(.95 * months.hours)]
            nresidual = len(adequate.intersection(fitting.index))
            if is_external and not holdout.used_calibration_plants.eq(nresidual).all():
                raise ValueError("The computed calibration population differs from the emitted predictions")
            data.append(dict(
                pair="2024–2025" if is_external else "2023–2024", fold=fold, test_states=", ".join(states),
                n_test=len(holdout), n_fitting_states=fitting.nunique(), n_fitting_plants=months.location.nunique(),
                n_fitting_months=len(months), n_calibration=nresidual,
            ))
    return pd.DataFrame(data)


def coverage_check(full, stack):
    good = stack.groupby(["location", "year"]).stack_hours.sum().unstack("year")
    rate = hours_weighted(stack, "reference_all_hours_kg_h", "stack_hours").unstack("year")
    expected = pd.Series({2023: 8760, 2024: 8784})
    assert np.allclose(full.observed_change, (rate[2024] - rate[2023]).reindex(full.index), atol=1e-7)
    coverage = good[expected.index].div(expected).min(axis=1).reindex(full.index)
    out = []
    for minimum in (.95, .98, .99):
        cohort = full.loc[coverage.ge(minimum)]
        for label, sample in (("All", cohort), ("Large", cohort.loc[cohort.observed_change.abs().gt(25)])):
            errors = sample[["satellite_only", "activity", "large_change_fusion"]].sub(sample.observed_change, axis=0)
            out.append(dict(min_coverage=minimum, group=label, n=len(sample), **errors.abs().mean().to_dict()))
    return pd.DataFrame(out)


def region_check(annual, full, expanded, activity, stack, historical, external, pilot):
    visits, _, panel, _ = prepare_inputs(annual, activity.loc[activity.year.ge(2023)], True)
    groups = panel.State.map(REGION)
    matched_activity = activity.loc[activity.location.isin(panel.index)]
    matched_stack = stack.loc[stack.location.isin(panel.index)]
    blend = run_folds(visits, matched_activity, panel, matched_stack, region=groups)
    model = blend.paired_satellite_fuel.where(blend.fuel_change.abs().gt(CHANGE_THRESHOLD), blend.fuel_change)
    joined = full.assign(region=full.State.map(REGION), fuel_region=blend.fuel_change.reindex(full.index),
                         joint_region=model.reindex(full.index))
    joined["fuel_region"] = joined.fuel_region.fillna(joined.satellite_only)
    joined["joint_region"] = joined.joint_region.fillna(joined.satellite_only)
    results = stratified_error(joined, "observed_change", ("satellite_only", "fuel_region", "joint_region"),
                               "TROPOMI, three regions")

    older_states = pilot.State.reindex(historical.location.unique())
    ext_pred = []
    for group, test in external.groupby(external.State.map(REGION)):
        training = older_states.index[older_states.map(REGION).ne(group)]
        coefficients = train_prior(historical, stack, training, 2023)
        two_years = expanded.loc[expanded.location.isin(test.index)].copy()
        two_years["mean_rate"] = physical_features(two_years) @ coefficients
        estimates = hours_weighted(two_years, "mean_rate", "hours").unstack("year")
        one = test.assign(fuel_region=estimates[2025] - estimates[2024], region=group)
        ext_pred.append(one)
    recent = pd.concat(ext_pred, verify_integrity=True).sort_index()
    assert len(recent) == len(external)
    results.extend(stratified_error(recent, "reference", ("zero_change", "fuel_region"),
                                   "Later fuel evaluation, four regions"))
    return joined, recent, pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("annual", "pilot", "experiments", "ground", "output"):
        parser.add_argument(f"--{path}", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    panel = pd.read_csv(args.annual / "panel.csv").set_index("location")
    full = pd.read_csv(args.pilot / "full_panel_fallback.csv").set_index("location")
    primary = pd.read_csv(args.pilot / "decisions_2023_2024.csv").set_index("location")
    primary["State"] = panel.State
    paired = pd.read_csv(args.pilot / "paired_change_estimates.csv").set_index("location")
    workflow = pd.read_csv(args.pilot / "workflow_2023_2024.csv").set_index("location")
    external = pd.read_csv(args.experiments / "external_2024_2025_estimates.csv").set_index("location")
    activity = activity_data(args.pilot / "eia923_monthly_activity_2022_2024.csv")
    expanded = activity_data(args.experiments / "eia923_2024_2025_expanded.csv")
    historical = activity.loc[activity.location.isin(panel.index) & activity.year.le(2023)]
    stack = full_stack_months(args.ground, panel.index, (2022, 2023, 2024))
    assert np.allclose(primary.cems, panel.change_kg_h.reindex(primary.index))
    assert np.allclose(primary.fuel, paired.fuel_change.reindex(primary.index))
    ranks = decline_curves(primary, paired, workflow, args.output)
    splits = memberships(primary, external, panel, historical, stack, paired)
    coverage = coverage_check(full, stack)
    with threadpool_limits(limits=4):
        region_early, region_late, regions = region_check(args.annual, full, expanded, activity, stack,
                                                        historical, external, panel)
    thresholds = pd.DataFrame(stratified_error(
        full, "observed_change", ("satellite_only", "activity", "large_change_fusion"), "2023–2024, state folds")
        + stratified_error(external, "reference", ("zero_change", "fuel"), "2024–2025, state folds"))
    improvements = full.assign(
        improvement=full.activity.sub(full.observed_change).abs() - full.large_change_fusion.sub(full.observed_change).abs()
    )
    states = improvements.groupby("State").improvement.agg(["sum", "size"])
    states["improvement_without_state"] = (states["sum"].sum() - states["sum"]) / (len(full) - states["size"])
    region_early.to_csv(args.output / "tropomi_region_predictions.csv")
    region_late.to_csv(args.output / "later_region_predictions.csv")
    for filename, table in (("holdout_membership", splits), ("fixed_false_budgets", ranks),
                            ("strict_stack_coverage", coverage), ("region_holdout_errors", regions),
                            ("change_threshold_sensitivity", thresholds),
                            ("satellite_gain_leave_one_state_out", states.reset_index())):
        table.to_csv(args.output / f"{filename}.csv", index=False)
    print(splits.to_string(index=False))
    print(ranks.loc[ranks.max_false.isin([1, 4])].round(3).to_string(index=False))
    print(coverage.round(3).to_string(index=False))
    print(regions.loc[regions.threshold.isin([0, 25])].round(3).to_string(index=False))
    print("Satellite gain removing one state at a time (range)",
          states.improvement_without_state.min(), states.improvement_without_state.max())


if __name__ == "__main__":
    main()
