"""Apply the paired-change approach to the already measured 2024–2025 TEMPO pilot."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from paired_change import corrected_contrast, full_stack_months, previous_variation, sat_slope, train_prior
from satellite_fuel_baselines import CHANGE_THRESHOLD, activity_data, physical_features


def evaluate(tempo, all_activity, panel, stack):
    companion = pd.read_csv(tempo / "daily_rates.csv").rename(
        columns={"scan_average": "satellite", "cems_at_scans": "matched_stack"},
    )
    companion = companion.loc[companion.location.isin(panel.index)]
    trends = pd.read_csv(tempo / "changes_against_full_year.csv").query("method == 'scan_average'").set_index("location")
    output = []
    for site, evaluation in trends.iterrows():
        target_years = all_activity.loc[all_activity.location.eq(site) & all_activity.year.isin([2024, 2025])].copy()
        if target_years.year.nunique() != 2:
            continue
        other_states = panel.index[panel.State.ne(evaluation.state)]
        training = companion.loc[companion.location.isin(other_states) & companion.year.eq(2024)]
        if training.location.nunique() < 2:
            continue
        response, floor = sat_slope(training)
        factors = train_prior(all_activity, stack, other_states, 2023)
        old_fuel = all_activity.loc[all_activity.year.le(2023)]
        old_fuel = old_fuel.assign(fuel_prior=physical_features(old_fuel) @ factors)
        uncertainty = previous_variation(old_fuel, stack, other_states)
        target_years["fuel_prior"] = physical_features(target_years) @ factors

        monthly = companion.loc[companion.location.eq(site)].groupby(["location", "year", "month"]).agg(
            signal=("satellite", "mean"), variance=("satellite", "var"), n=("satellite", "size"),
        )
        monthly = monthly.join(target_years.set_index(["location", "year", "month"])[["heat_total", "fuel_prior"]])
        prediction = corrected_contrast(
            monthly, target_years, "fuel_prior", "signal", "variance", "n", response, floor, uncertainty,
            years=(2024, 2025),
        ).loc[site]
        output.append(dict(location=site, facility=evaluation.facility, state=evaluation.state,
                           reference=evaluation.cems_delta, tempo_only=evaluation.delta,
                           fuel_only=prediction.fuel_change, paired_fusion=prediction.combined_change,
                           overlap_months=prediction.common_months, calibration_sites=training.location.nunique()))
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--tempo", type=Path, required=True)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel).set_index("location")
    status = activity_data(args.activity)
    stack = full_stack_months(args.ground, panel.index, (2022, 2023))
    with threadpool_limits(limits=4):
        result = evaluate(args.tempo, status, panel, stack)
    print(result.round(2).to_string(index=False))
    true_declines = result.reference.lt(-CHANGE_THRESHOLD)
    print("TEMPO test plants", len(result), "reference declines", true_declines.sum())
    for method in ("tempo_only", "fuel_only", "paired_fusion"):
        difference = result[method] - result.reference
        declares = result[method].lt(-CHANGE_THRESHOLD)
        print(method, "MAE", np.abs(difference).mean().round(2), "fixed TP/FP",
              (declares & true_declines).sum(), (declares & ~true_declines).sum())
    result.to_csv("tempo_2024_2025_transfer.csv", index=False)


if __name__ == "__main__":
    main()
