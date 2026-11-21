"""Test a learned monthly satellite correction against held-out annual stack changes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from paired_change import full_stack_months, monthly_rates, train_prior
from satellite_fuel_baselines import CHANGE_THRESHOLD, HEAT_FEATURES, activity_data, hours_weighted, physical_features, prepare_inputs


def estimate(months, stack, states, folds):
    fields = ["satellite", "standard_error", "visits", "fuel_rate", *HEAT_FEATURES, "month"]
    estimates = []
    for _, held_out in folds.groupby("fold"):
        test_sites = held_out.index
        train_sites = states.loc[~states.isin(states.loc[test_sites])].index
        factors = train_prior(months, stack, train_sites, 2023)
        data = months.assign(fuel_rate=physical_features(months) @ factors)
        fit = data.loc[data.location.isin(train_sites) & data.year.eq(2023) & data.visits.ge(2)]
        fit = fit.merge(stack, on=["location", "year", "month"], validate="one_to_one")
        fit = fit.loc[fit.stack_hours.ge(.95 * fit.hours)]
        model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=125, max_leaf_nodes=13, min_samples_leaf=20,
            l2_regularization=4., early_stopping=False, random_state=61285,
        ).fit(fit[fields], fit.reference_all_hours_kg_h - fit.fuel_rate, sample_weight=fit.hours)

        test = data.loc[data.location.isin(test_sites)].copy()
        test["corrected"] = test.fuel_rate
        use = test.visits.ge(2)
        test.loc[use, "corrected"] = (test.loc[use, "fuel_rate"] + model.predict(test.loc[use, fields])).clip(lower=0.)
        annual = hours_weighted(test, "corrected", "hours").unstack("year")
        estimates.append(annual[2024] - annual[2023])
    return pd.concat(estimates, verify_integrity=True).reindex(folds.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True, help="The paired workflow from validate_decisions.py.")
    args = parser.parse_args()
    activity = activity_data(args.activity)
    visits, months, panel, _ = prepare_inputs(args.results, activity.loc[activity.year.ge(2023)], True)
    stats = monthly_rates(visits).rename(columns={"signal": "satellite", "n": "visits"})
    stats["standard_error"] = np.sqrt(stats.variance / stats.visits)
    months = months.merge(stats[["satellite", "standard_error", "visits"]].reset_index(),
                          on=["location", "year", "month"], how="left", validate="one_to_one")
    folds = pd.read_csv(args.primary).set_index("location")
    if set(folds.index) != set(panel.index):
        raise ValueError("The monthly comparison must use the same plants as the primary estimate")
    reference = pd.read_csv(args.paired).set_index("location").reindex(folds.index)
    if not np.allclose(reference.cems, folds.cems):
        raise ValueError("Different full-year stack references in the annual comparisons")
    stack = full_stack_months(args.ground, panel.index, (2023,))
    with threadpool_limits(limits=4):
        annual = estimate(months, stack, panel.State, folds)
    summary = []
    predictions = {"Monthly satellite and fuel learner": annual,
                   "Monthly learner for predicted changes": annual.where(folds.fuel.abs().gt(CHANGE_THRESHOLD),
                                                                        folds.fuel)}
    for label, predicted in predictions.items():
        error = predicted.sub(folds.cems).abs()
        declines = folds.cems.lt(-CHANGE_THRESHOLD)
        flags = predicted.lt(-CHANGE_THRESHOLD)
        reference_error = reference.change_with_targeted_satellite.sub(reference.cems).abs()
        difference = error - reference_error
        clusters = difference.groupby(panel.State).agg(["sum", "size"]).to_numpy()
        draws = np.random.default_rng(61285).integers(0, len(clusters), (10000, len(clusters)))
        grouped = clusters[draws].sum(axis=1)
        low, high = np.quantile(grouped[:, 0] / grouped[:, 1], [.025, .975])
        summary.append(dict(method=label, n=len(error), mae=error.mean(),
                            mae_large=error[folds.cems.abs().gt(CHANGE_THRESHOLD)].mean(),
                            mae_declines=error[declines].mean(), difference=difference.mean(),
                            lower=low, upper=high, true_calls=int((flags & declines).sum()),
                            false_calls=int((flags & ~declines).sum())))
    scores = pd.DataFrame(summary)
    scores.to_csv("satellite_monthly_ml_benchmarks.csv", index=False)
    pd.DataFrame(predictions).to_csv("satellite_monthly_ml_predictions.csv")
    print(scores.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
