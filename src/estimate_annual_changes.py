"""Estimate two-year NOx changes from fuel, earlier stack data, and optional satellite rates."""

import argparse
import calendar
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

from paired_change import full_stack_months as stack_months
from satellite_fuel_baselines import CHANGE_THRESHOLD, hours_weighted as weighted_levels


KEYS = ["location", "year", "month"]
FUELS = ["coal", "gas", "other"]
TECHNOLOGY = ["coal", "gas_steam", "gas_combined_cycle", "gas_simple_turbine", "gas_other", "other"]
MEASURED = "reference_all_hours_kg_h"


def monthly_fuel(directory, years):
    raw = pd.concat([pd.read_parquet(directory / f"plant_heat_{year}.parquet") for year in years], ignore_index=True)
    if not raw.fuel.isin(FUELS).all() or not np.isfinite(raw.heat_mmbtu).all() or raw.heat_mmbtu.lt(0).any():
        raise ValueError("Fuel records must contain nonnegative MMBtu for coal, gas, or other fuel")
    mover = raw.prime_mover
    gas_kind = np.select([mover.eq("ST"), mover.isin(["CA", "CC", "CS", "CT"]), mover.isin(["GT", "IC"])],
                         ["steam", "combined_cycle", "simple_turbine"], default="other")
    raw["component"] = raw.fuel.mask(raw.fuel.eq("gas"), "gas_" + gas_kind)
    detail = raw.groupby(KEYS + ["component"]).heat_mmbtu.sum().unstack(fill_value=0)
    total = raw.groupby(KEYS + ["fuel"]).heat_mmbtu.sum().unstack(fill_value=0)
    result = detail.reindex(columns=TECHNOLOGY, fill_value=0).join(total.reindex(columns=["gas"], fill_value=0))
    result = result.reset_index()
    result["hours"] = pd.PeriodIndex.from_fields(year=result.year, month=result.month, freq="M").days_in_month * 24
    result["ozone"] = result.month.between(5, 9)
    result["heat_mmbtu"] = result[FUELS].sum(axis=1)
    return result


def complete_fuel(monthly, years):
    counts = monthly.loc[monthly.year.isin(years)].groupby(["location", "year"]).month.nunique()
    full = counts.unstack().reindex(columns=years).eq(12).all(axis=1)
    return full.index[full]


def measured_levels(stack):
    rates = weighted_levels(stack, MEASURED, "stack_hours")
    hours = stack.groupby(["location", "year"]).stack_hours.sum()
    expected = hours.index.get_level_values("year").map(lambda year: (365 + calendar.isleap(year)) * 24)
    return rates.where(hours.ge(.95 * expected)).unstack("year")


def features(monthly, columns, separate_conversion=False):
    if separate_conversion:
        heat_rate = monthly[columns].div(monthly.hours, axis=0).div(1000.0).to_numpy()
    else:
        heat_rate = monthly[columns].to_numpy() / (1000 * monthly.hours.to_numpy()[:, None])
    ozone = monthly.ozone.to_numpy()[:, None]
    return np.concatenate([heat_rate * ozone, heat_rate * ~ozone], axis=1)


def fit_factors(training, columns, analytic_jacobian=True):
    if training.empty:
        raise ValueError("No monthly stack rates in the calibration states")
    x = features(training, columns, separate_conversion=not analytic_jacobian)
    y = training[MEASURED].to_numpy()
    weights = np.sqrt(training.hours / training.hours.mean()).to_numpy()
    jacobian = (lambda _: x * weights[:, None]) if analytic_jacobian else "2-point"
    result = least_squares(lambda factors: (x @ factors - y) * weights, np.ones(x.shape[1]),
                           jac=jacobian, loss="soft_l1", f_scale=CHANGE_THRESHOLD, bounds=(0, np.inf))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def training_rows(monthly, stack, sites, year):
    fuel = monthly.loc[monthly.location.isin(sites) & monthly.year.eq(year)]
    train = fuel.merge(stack.loc[stack.year.eq(year)], on=KEYS, validate="one_to_one")
    return train.loc[train.stack_hours.ge(.95 * train.hours)]


def local_factor(fuel, stack, by_season=False):
    indices = ["location", "ozone"] if by_season else ["location"]
    paired = fuel.merge(stack, on=KEYS, validate="one_to_one")
    total = paired.assign(observed=paired[MEASURED] * paired.stack_hours,
                          modeled=paired.prediction * paired.stack_hours)
    sums = total.groupby(indices)[["observed", "modeled", "stack_hours"]].sum()
    expected = fuel.groupby(indices).hours.sum().reindex(sums.index)
    valid = sums.stack_hours.ge(.95 * expected) & sums.modeled.div(sums.stack_hours).gt(5.0)
    return sums.observed.div(sums.modeled).loc[valid].rename("scale")


def estimate_fuel(test, monthly, stack, states, years, designs):
    year = years[0] - 1
    training = training_rows(monthly, stack, states.index[~states.isin(test.State)], year)
    test_fuel = monthly.loc[monthly.location.isin(test.index) & monthly.year.between(year, years[-1])]
    earlier_stack = stack.loc[stack.year.eq(year)]
    result = test.copy()
    all_months = []
    for model, columns in designs.items():
        values = test_fuel.assign(prediction=features(test_fuel, columns) @ fit_factors(training, columns))
        prior = values.loc[values.year.eq(year) & values.location.isin(complete_fuel(values, [year]))]
        future = values.loc[values.year.isin(years)].copy()
        annual = local_factor(prior, earlier_stack)
        seasonal = local_factor(prior, earlier_stack, by_season=True)
        future["plant"] = future.prediction * future.location.map(annual).fillna(1.0)
        future = future.join(seasonal, on=["location", "ozone"])
        future["seasonal"] = future.prediction * future.scale.fillna(future.location.map(annual)).fillna(1.0)
        for name in ("prediction", "plant", "seasonal"):
            rates = weighted_levels(future, name, "hours").unstack("year")
            result[f"{model}_{name}"] = rates[years[-1]] - rates[years[0]]
        result[f"{model}_prior_available"] = result.index.isin(annual.index)
        fields = KEYS + ["hours", "heat_mmbtu", "prediction", "plant", "seasonal"]
        all_months.append(future[fields].assign(model=model))
    return result, pd.concat(all_months, ignore_index=True)


def state_folds(plants, n_splits):
    result = plants.copy()
    for fold, (_, indices) in enumerate(GroupKFold(n_splits=n_splits).split(plants, groups=plants.State), 1):
        result.loc[plants.index[indices], "fold"] = fold
    result["fold"] = result.fold.astype(int)
    return result


def calibration_residuals(monthly, stack, states, holdout, years, measured, designs):
    changes = (measured[years[-1]] - measured[years[0]]).dropna().rename("reference")
    known = states.loc[~states.isin(holdout)]
    pool = known.to_frame().join(changes, how="inner")
    pool = pool.loc[pool.index.isin(complete_fuel(monthly, years))]
    folded = state_folds(pool, 4)
    rows = [estimate_fuel(test, monthly, stack, known, years, designs)[0]
            for _, test in folded.groupby("fold")]
    return pd.concat(rows, verify_integrity=True).sort_index()


def upper_offset(values, level):
    rank = math.ceil((len(values) + 1) * level)
    return np.sort(values)[rank - 1] if rank <= len(values) else np.inf


def add_decline_bounds(predictions, calibration):
    for model in ("fuel_prediction", "fuel_plant"):
        residual = calibration.reference - calibration[model]
        for percent in (90, 95):
            if model == "fuel_plant":
                offsets = residual.groupby(calibration.fuel_prior_available).apply(upper_offset, level=percent / 100)
                offset = predictions.fuel_prior_available.map(offsets).fillna(np.inf)
            else:
                offset = upper_offset(residual, percent / 100)
            column = f"{model}_upper_{percent}"
            predictions[column] = predictions[model] + offset
            predictions[f"{column}_flag"] = predictions[column].lt(-CHANGE_THRESHOLD)
    return predictions


def read_plants(path):
    data = pd.read_csv(path, usecols=lambda field: field in ("location", "State", "fold")).set_index("location")
    if not data.index.is_unique or not data.State.str.fullmatch("[A-Z]{2}").fillna(False).all():
        raise ValueError(f"{path.name} must have one record and a two-letter state per facility")
    if "fold" in data and (data.fold.isna().any() or data.groupby("State").fold.nunique().gt(1).any()):
        raise ValueError(f"Every state in {path.name} must belong to exactly one fold")
    return data.sort_index()


def paired_interval(error, state):
    clusters = error.groupby(state).agg(["sum", "count"]).to_numpy()
    random = np.random.default_rng(5679128)
    selected = random.integers(len(clusters), size=(10_000, len(clusters)))
    total = clusters[selected].sum(axis=1)
    return np.quantile(total[:, 0] / total[:, 1], [.025, .975])


def recovered_at_false_limit(values, truth, allowance):
    counts = pd.DataFrame({"prediction": values, "decline": truth}).groupby("prediction").decline.agg(["sum", "size"])
    return int(counts.loc[(counts["size"] - counts["sum"]).cumsum().le(allowance), "sum"].sum())


def assess(result, methods):
    truth = result.reference
    large = truth.abs().gt(CHANGE_THRESHOLD)
    declined = truth.lt(-CHANGE_THRESHOLD)
    groups = {"all": truth.notna(), "large": large, "declines": declined,
              "missed_by_fuel": large & result.fuel_prediction.abs().le(CHANGE_THRESHOLD),
              "with_earlier_stack": truth.notna() & result.fuel_prior_available,
              "without_earlier_stack": truth.notna() & ~result.fuel_prior_available}
    errors = result[methods].sub(truth, axis=0).abs()
    rows = []
    for name, mask in groups.items():
        if not mask.any():
            continue
        for method in methods:
            predicted = result.loc[mask, method]
            absolute = errors.loc[mask, method]
            baseline = method.split("_satellite", 1)[0] if "_satellite" in method else "fuel_prediction"
            difference = absolute - errors.loc[mask, baseline]
            low, high = paired_interval(difference, result.loc[mask, "State"])
            recovered = (predicted * np.sign(truth[mask])).gt(CHANGE_THRESHOLD) & large[mask]
            rows.append(dict(cohort=name, method=method, n=int(mask.sum()), mae=absolute.mean(),
                             comparator=baseline, paired_difference=difference.mean(), lower=low, upper=high,
                             correct_large_changes=int(recovered.sum()),
                             declines_at_four_false=recovered_at_false_limit(predicted, declined[mask], 4)))
    decisions = []
    for column in result.filter(regex=r"_upper_\d+$"):
        flag = result[column].lt(-CHANGE_THRESHOLD) & truth.notna()
        correct = flag & declined
        decisions.append(dict(method=column, n=int(truth.notna().sum()), actual_declines=int(declined.sum()),
                              correct_declines=int(correct.sum()), false_declines=int((flag & ~declined).sum()),
                              coverage=truth.dropna().le(result.loc[truth.notna(), column]).mean(),
                              fraction_decline_mass=truth[correct].sum() / truth[declined].sum() if declined.any() else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuel", type=Path, required=True, help="Output directory of prepare_eia_fuel.py")
    parser.add_argument("--ground", type=Path, required=True, help="Earlier facility-hour Parquet files")
    parser.add_argument("--training-plants", type=Path, required=True, help="CSV with location and State")
    parser.add_argument("--evaluation-plants", type=Path, required=True, help="CSV with location, State, and optional fold")
    parser.add_argument("--years", nargs=2, required=True, type=int)
    parser.add_argument("--reference-ground", type=Path, help="Score facilities with 95%% stack coverage in both target years")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-visits", type=Path, help="scored_visits.parquet from compare_nox_annual_retrievals.py")
    target.add_argument("--target-blocks", nargs=2, type=Path, metavar=("SCHEDULE", "SCORES"),
                        help="Evaluation schedule and directory of native-scored blocks")
    parser.add_argument("--extra-satellite", nargs=2, type=Path, metavar=("SCHEDULE", "SCORES"),
                        help="Image-only schedule and directory of native-scored blocks")
    parser.add_argument("--earlier-satellite", nargs=2, type=Path, metavar=("SCHEDULE", "SCORES"),
                        help="Earlier schedule and directory of stack-matched native-scored blocks")
    parser.add_argument("--same-year-calibration", action="store_true",
                        help="Also calibrate on first-year stacks in other states")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    years = tuple(args.years)
    if years[1] != years[0] + 1:
        parser.error("Use consecutive comparison years")
    target = args.target_visits or args.target_blocks
    if (args.earlier_satellite or args.extra_satellite or args.same_year_calibration) and not target:
        parser.error("Satellite settings need --target-visits or --target-blocks")
    if target and not (args.earlier_satellite or args.same_year_calibration):
        parser.error("Satellite correction needs an earlier or first-year calibration")
    history = (years[0] - 2, years[0] - 1)
    monthly = monthly_fuel(args.fuel, range(history[0] - 1, years[1] + 1))
    sites = read_plants(args.evaluation_plants)
    states = read_plants(args.training_plants).State
    common = sites.index.intersection(states.index)
    if not sites.loc[common, "State"].eq(states.loc[common]).all():
        raise ValueError("State assignments differ between training and evaluation registries")
    sites = sites.loc[sites.index.isin(complete_fuel(monthly, years))]
    satellite = None
    if target:
        from paired_satellite import SatelliteCorrection, read_blocks, read_visits

        if args.target_visits:
            visits = read_visits(args.target_visits, match_stacks=args.same_year_calibration)
        else:
            visits = read_blocks(*args.target_blocks)
        satellite = SatelliteCorrection(visits, args.extra_satellite, args.earlier_satellite, monthly,
                                        years, args.same_year_calibration)
        sites = sites.loc[sites.index.isin(satellite.sites)]
    reference = None
    if args.reference_ground:
        observed = stack_months(args.reference_ground, sites.index, years)
        rates = measured_levels(observed).reindex(columns=years).dropna()
        sites = sites.loc[sites.index.isin(rates.index)]
        reference = rates[years[1]] - rates[years[0]]
    if sites.empty:
        raise ValueError("No evaluation facilities meet the fuel and image coverage rules")
    if "fold" not in sites:
        sites = state_folds(sites, 5)

    # Reference coverage defines the scoring cohort. Its measured rates enter only after prediction.
    required_sites = states.index.union(sites.index)
    if satellite is not None:
        required_sites = required_sites.union(satellite.sites)
    historical = stack_months(args.ground, required_sites, range(history[0] - 1, years[0]))
    measured = measured_levels(historical)
    first_year = stack_months(args.ground, satellite.sites, [years[0]]) if args.same_year_calibration else None
    all_changes, all_months, diagnostics = [], [], []
    designs = {"fuel": FUELS, "technology": TECHNOLOGY}
    calibration_designs = designs if args.earlier_satellite else {"fuel": FUELS}
    for fold, test in sites.groupby("fold"):
        result, month = estimate_fuel(test, monthly, historical, states, years, designs)
        residuals = calibration_residuals(monthly, historical, states, set(test.State), history, measured, calibration_designs)
        result = add_decline_bounds(result, residuals).assign(calibration_n=len(residuals))
        if satellite is not None:
            estimates, params = satellite.predict(test, month, historical, first_year, residuals)
            result = result.join(estimates)
            diagnostics.append({"fold": fold, **params})
        all_changes.append(result)
        all_months.append(month.assign(fold=fold))
    estimates = pd.concat(all_changes, verify_integrity=True).sort_index()
    months = pd.concat(all_months, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if reference is not None:
        estimates["reference"] = reference
        models = [column for column in estimates.columns
                  if column in {"fuel_prediction", "fuel_plant", "fuel_seasonal", "technology_prediction",
                                "technology_plant", "technology_seasonal", "satellite_only"}
                  or "_satellite" in column]
        scores, flags = assess(estimates, models)
        scores.to_csv(args.output / "errors.csv", index=False)
        flags.to_csv(args.output / "decline_rules.csv", index=False)
    estimates.to_csv(args.output / "changes.csv")
    months.to_parquet(args.output / "monthly_fuel_rates.parquet", index=False)
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(args.output / "satellite_calibration.csv", index=False)
    print(f"{len(estimates)} evaluation plants in {estimates.State.nunique()} states", flush=True)


if __name__ == "__main__":
    main()
