"""Test whether paired monthly satellite contrasts improve a fuel-based annual change."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from satellite_fuel_baselines import (
    CHANGE, CHANGE_THRESHOLD, RATE_FIELDS, activity_data, annual_raw,
    fit_activity_model, hours_weighted, material_declines_at_false_count, physical_features, prepare_inputs,
)


def full_stack_months(ground, locations, years):
    records = []
    for year in years:
        raw = pd.read_parquet(ground / f"{year}.parquet", columns=["facility_id", "utc", "mass_kg"],
                              filters=[("facility_id", "in", list(locations))])
        raw["month"] = raw.utc.dt.month
        months = raw.groupby(["facility_id", "month"]).mass_kg.agg(["mean", "count"]).reset_index()
        months = months.rename(columns={"facility_id": "location", "mean": "reference_all_hours_kg_h",
                                        "count": "stack_hours"})
        records.append(months.assign(year=year))
    return pd.concat(records, ignore_index=True)


def sat_slope(training_visits):
    """Estimate the instrument response to changing stack rates in the calibration year."""
    train = training_visits.dropna(subset=["satellite", "matched_stack"]).copy()
    response = train[["satellite", "matched_stack"]]
    centered = response - response.groupby(train.location).transform("mean")
    # Equal weight per plant prevents sites with more usable passes dominating the calibration.
    weights = train.groupby("location").location.transform("size").to_numpy()
    weights = np.sqrt(weights.mean() / weights)
    target, x = centered.satellite.to_numpy(), centered.matched_stack.to_numpy()
    robust_fit = least_squares(lambda b: (target - b[0] * x) * weights,
                               [1.0], bounds=(0.05, 5.0), loss="soft_l1", f_scale=50.0)
    slope = robust_fit.x.item()
    raw_residual = train.satellite / slope - train.matched_stack
    monthly = raw_residual.groupby([train.location, train.month]).mean()
    monthly_drift = monthly - monthly.groupby(level=0).transform("mean")
    floor = 1.4826 * np.median(abs(monthly_drift - np.median(monthly_drift)))
    return slope, max(floor, 1.0)


def train_prior(activity, stack, sites, base_year):
    training = activity.loc[activity.location.isin(sites) & activity.year.eq(base_year)].merge(
        stack, on=["location", "year", "month"], validate="one_to_one",
    )
    training = training.loc[training.stack_hours.ge(.95 * training.hours)]
    return fit_activity_model(training)


def previous_variation(full, stack, training_sites):
    prior = hours_weighted(full, "fuel_prior", "hours").unstack("year")
    observed = stack.merge(full[["location", "year", "month", "hours"]], on=["location", "year", "month"],
                           validate="one_to_one")
    observed = observed.loc[observed.location.isin(training_sites) & observed.year.le(2023)]
    full_coverage = observed.groupby(["location", "year"]).stack_hours.sum()
    annual = hours_weighted(observed, "reference_all_hours_kg_h", "stack_hours")
    annual = annual.loc[full_coverage.ge(8760 * .95)].unstack("year")
    residual = (annual[2023] - annual[2022]) - (prior[2023] - prior[2022])
    residual = residual.dropna().to_numpy()
    # Use a second moment, with extreme sites limited by calibration data alone.
    limit = np.quantile(abs(residual), .98)
    return float(np.mean(np.clip(residual, -limit, limit) ** 2))


def historical_factor(full, stack):
    """Optional older plant-specific information, available before the target pair."""
    prior = hours_weighted(full, "fuel_prior", "hours").unstack("year")[2022]
    history = stack.loc[stack.year.eq(2022)]
    count = history.groupby("location").stack_hours.sum()
    history = hours_weighted(history, "reference_all_hours_kg_h", "stack_hours").droplevel("year")
    valid = count.ge(.95 * 8760) & prior.gt(5.0)
    factor = (history / prior).loc[valid.reindex(prior.index, fill_value=False)].clip(lower=0)
    return factor


def monthly_rates(visits):
    visits = visits.assign(satellite=visits[RATE_FIELDS].mean(axis=1))
    return visits.groupby(["location", "year", "month"]).agg(
        signal=("satellite", "mean"), variance=("satellite", "var"), n=("satellite", "size"),
        n_stack=("matched_stack", "count"), exact=("matched_stack", "mean"),
        exact_variance=("matched_stack", "var"),
    )


def matched_environment_rates(visits):
    """Use the same wind and observing-condition distribution in each paired month."""
    fields = ["wind_u", "wind_v", "sensor_zenith_angle", "cloud_fraction_crb"]
    scale = np.array([2.5, 2.5, 15.0, .15])
    scene = visits.dropna(subset=fields).assign(satellite=visits[RATE_FIELDS].mean(axis=1))
    output = []
    for (site, month), entries in scene.groupby(["location", "month"]):
        left = entries.loc[entries.year.eq(2023)]
        right = entries.loc[entries.year.eq(2024)]
        if len(left) < 2 or len(right) < 2:
            continue
        distance = cdist(left[fields].to_numpy() / scale, right[fields].to_numpy() / scale, "sqeuclidean")
        # Subtracting the minimum keeps the relative kernel weights numerically stable.
        mass = np.exp(-.5 * (distance - distance.min()))
        for batch, marginal in ((left, mass.sum(axis=1)), (right, mass.sum(axis=0))):
            weights = marginal / marginal.sum()
            effective_n = 1.0 / np.dot(weights, weights)
            if effective_n <= 1.0 + 1e-8:
                continue
            mean = np.dot(weights, batch.satellite)
            variance = np.dot(weights, (batch.satellite - mean)**2) * effective_n / (effective_n - 1.0)
            output.append(dict(location=site, month=month, year=int(batch.year.iloc[0]),
                               signal=mean, variance=variance, n=effective_n))
    return pd.DataFrame(output).set_index(["location", "year", "month"])


def corrected_contrast(monthly, full, p, signal, variance, counts, slope, floor, prior_variance,
                       years=(2023, 2024)):
    """Estimate an intensity change from shared calendar months, then apply it to annual fuel use."""
    fields = [signal, variance, counts, "heat_total", p]
    paired = monthly[fields].unstack("year")
    left, right = (paired.xs(year, level="year", axis=1) for year in years)
    common = left[signal].notna() & right[signal].notna() & left[counts].ge(2) & right[counts].ge(2)
    left, right = left.loc[common], right.loc[common]
    typical_heat = hours_weighted(full, "heat_total", "hours").unstack("year").mean(axis=1)
    annual_prior = hours_weighted(full, p, "hours").unstack("year")
    annual_prior = annual_prior[years[1]] - annual_prior[years[0]]
    relative_activity = ((left.heat_total + right.heat_total) * .5).div(
        typical_heat.replace(0.0, np.nan), level="location",
    )
    difference = (right[signal] - left[signal]) / slope - (right[p] - left[p])
    noise_var = (left[variance] / left[counts] + right[variance] / right[counts]) / (slope*slope)
    noise_var = noise_var + (2.0 * floor*floor)
    precision = (relative_activity**2 / noise_var).groupby(level="location").sum()
    raw = (relative_activity * difference / noise_var).groupby(level="location").sum()
    weight = prior_variance * precision / (1.0 + prior_variance * precision)
    correction = (prior_variance * raw / (1.0 + prior_variance * precision)).reindex(annual_prior.index).fillna(0.)
    return pd.DataFrame({
        "fuel_change": annual_prior, "combined_change": annual_prior + correction, "change_correction": correction,
        "correction_weight": weight, "common_months": left.groupby(level="location").size(),
        "working_sd": np.sqrt(prior_variance * (1.0 - weight.reindex(annual_prior.index).fillna(0.0))),
    })


def run_folds(visits, full, panel, stack, region=None):
    monthly = monthly_rates(visits)
    balanced = matched_environment_rates(visits)
    outputs = []
    groups = panel.State if region is None else region.reindex(panel.index)
    if groups.isna().any():
        raise ValueError("Every plant must have a geographical group")
    n_splits = 5 if region is None else groups.nunique()
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(panel, groups=groups), 1):
        training_sites, test_sites = panel.index[train], panel.index[test]
        assert set(panel.iloc[train].State).isdisjoint(panel.iloc[test].State)
        fitted = train_prior(full, stack, training_sites, 2023)
        with_prior = full.assign(fuel_prior=physical_features(full) @ fitted)
        prior_var = previous_variation(with_prior, stack, training_sites)
        # An older plant-specific record is optional and has its own baseline.
        past = historical_factor(with_prior, stack)
        test_full = with_prior.loc[with_prior.location.isin(test_sites)].copy()
        test_full["past_prior"] = test_full.fuel_prior * test_full.location.map(past).fillna(1.0)
        test_later = test_full.loc[test_full.year.ge(2023)]
        known = test_later.set_index(["location", "year", "month"])[["heat_total", "fuel_prior", "past_prior"]]
        shared = monthly.join(known, how="inner")
        train_visits = visits.loc[visits.location.isin(training_sites) & visits.year.eq(2023)]
        response, floor = sat_slope(train_visits.assign(satellite=train_visits[RATE_FIELDS].mean(axis=1)))
        model = corrected_contrast(
            shared, test_later, "fuel_prior", "signal", "variance", "n", response, floor, prior_var,
        )
        history_model = corrected_contrast(
            shared, test_later, "past_prior", "signal", "variance", "n", response, floor, prior_var,
        )
        balanced_model = corrected_contrast(
            balanced.join(known, how="inner"), test_later, "fuel_prior", "signal", "variance", "n",
            response, floor, prior_var,
        )
        # Substitute direct rates at the same times as a diagnostic of the observation schedule.
        oracle = corrected_contrast(
            shared, test_later, "fuel_prior", "exact", "exact_variance", "n_stack", 1.0, floor, prior_var,
        )
        result = model.rename(columns={"combined_change": "paired_satellite_fuel"})
        result["historical_fuel"] = history_model.fuel_change
        result["history_paired_satellite"] = history_model.combined_change
        result["weather_balanced_satellite_fuel"] = balanced_model.combined_change
        result["balanced_working_sd"] = balanced_model.working_sd
        result["matched_stack_diagnostic"] = oracle.combined_change
        result["historic_available"] = result.index.isin(past.index)
        result["fold"] = fold
        outputs.append(result)
        print(f"Fold {fold}: estimated satellite response {response:.3g}, scene and background "
              f"error scale {floor:.1f} kg/h, older annual prior residual scale {np.sqrt(prior_var):.1f} kg/h.",
              flush=True)
    return pd.concat(outputs).sort_index()


def score(panel, preds, label):
    actual = panel[CHANGE]
    decline = actual.lt(-CHANGE_THRESHOLD)
    fixed = preds.lt(-CHANGE_THRESHOLD)
    error = (preds - actual).abs()
    weak_fuel = panel["fuel_change"].abs().le(CHANGE_THRESHOLD)
    large = actual.abs().gt(CHANGE_THRESHOLD)
    hits, fp = material_declines_at_false_count(actual, preds)
    return {
        "method": label, "n": len(actual), "mae": error.mean(), "mae_large": error[large].mean(),
        "n_large_weak_activity": int((weak_fuel & large).sum()),
        "mae_large_weak_activity": error[weak_fuel & large].mean(), "mae_declines": error[decline].mean(),
        "true_declines_at_fixed_rate": int((decline & fixed).sum()),
        "false_declines_at_fixed_rate": int((~decline & fixed).sum()), "declines_at_4_false": hits, "false": fp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--ground", required=True, type=Path)
    parser.add_argument("--activity", type=Path, default=Path("eia923_monthly_activity_2022_2024.csv"))
    parser.add_argument("--extra-visits", type=Path, help="Use satellite-only eligibility and include unpaired scenes.")
    args = parser.parse_args()
    all_activity = activity_data(args.activity)
    visits, activity, panel, _ = prepare_inputs(
        args.results, all_activity.loc[all_activity.year.ge(2023)], not args.extra_visits, extra=args.extra_visits,
    )
    print(f"Evaluation panel: {len(panel)} plants, {len(visits):,} visits, {panel.State.nunique()} states.", flush=True)
    stack = full_stack_months(args.ground, panel.index, [2022, 2023, 2024])
    reference = hours_weighted(stack, "reference_all_hours_kg_h", "stack_hours").unstack("year")
    if not np.allclose((reference[2024] - reference[2023]).reindex(panel.index), panel[CHANGE]):
        raise ValueError("Independent full-year reference does not reproduce the manuscript panel")
    with threadpool_limits(limits=4):
        scores = run_folds(visits, all_activity.loc[all_activity.location.isin(panel.index)], panel, stack)
    raw = annual_raw(visits, activity).unstack("year")
    raw = raw.xs(2024, level="year", axis=1) - raw.xs(2023, level="year", axis=1)
    table = panel.join(scores).join(raw)
    methods = [
        "all_images", "nondetections_zero", "fuel_change", "paired_satellite_fuel", "weather_balanced_satellite_fuel",
        "matched_stack_diagnostic", "historical_fuel", "history_paired_satellite",
    ]
    metrics = pd.DataFrame([score(table, table[method], method) for method in methods])
    print(metrics.round(2).to_string(index=False))
    prefix = "satellite_eligibility_change" if args.extra_visits else "paired_change"
    table.to_csv(f"{prefix}_estimates.csv")
    metrics.to_csv(f"{prefix}_metrics.csv", index=False)


if __name__ == "__main__":
    main()
