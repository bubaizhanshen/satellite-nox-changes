"""Test calendar and fuel adjustments on held-out states in the TROPOMI panel."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


RATE_FIELDS = ["cross_section_kg_h", "gaussian_kg_h"]
MATCHED_FIELDS = ["cross_section_cems_kg_h", "gaussian_cems_kg_h"]
HEAT_FIELDS = ["heat_coal", "heat_gas", "heat_other"]
RETRIEVAL_FEATURES = RATE_FIELDS + [
    "wind_m_s", "sensor_zenith_angle", "cloud_fraction_crb", "month_sin", "month_cos", "latitude",
]
HEAT_FEATURES = [f"log_{name}" for name in HEAT_FIELDS]
CHANGE_THRESHOLD = 25.0
CHANGE = "observed_change"
BASELINE = "all_images"
CANDIDATE = "activity_adjusted_calibrated"


def hours_weighted(frame, field, weights):
    """Aggregate only months with observations, using the supplied hours or counts."""
    keys = ["location", "year"]
    weighted = (frame[field] * frame[weights]).groupby([frame[k] for k in keys]).sum()
    return weighted / frame.groupby(keys)[weights].sum()


def activity_data(path):
    data = pd.read_csv(path)
    if data.duplicated(["location", "year", "month"]).any():
        raise ValueError("Duplicate plant-month records")
    if not data.groupby(["location", "year"]).size().eq(12).all():
        raise ValueError("The activity data must contain 12 months per included plant and year")
    data["hours"] = pd.PeriodIndex.from_fields(year=data.year, month=data.month, freq="M").days_in_month * 24
    data[HEAT_FIELDS] = data[HEAT_FIELDS].div(data.hours, axis=0).div(1000.0)
    data["heat_total"] = data[HEAT_FIELDS].sum(axis=1)
    data[HEAT_FEATURES] = np.log1p(data[HEAT_FIELDS].to_numpy())
    return data


def extra_visits(directory, columns):
    archive_columns = [col for col in columns if col not in RATE_FIELDS + MATCHED_FIELDS + ["screen_available"]]
    archive = pd.read_parquet(directory / "inputs/visits.parquet", columns=archive_columns)
    pieces = [pd.read_parquet(path) for path in sorted((directory / "scored_temis").glob("*.parquet"))]
    scores = pd.concat(pieces, ignore_index=True)
    scores["screen_available"] = scores.screen_error.eq("") & scores.template_score.notna()
    fields = ["visit_id", *RATE_FIELDS, *MATCHED_FIELDS, "screen_available"]
    return archive.merge(scores[fields], on="visit_id", how="left", validate="one_to_one")


def prepare_inputs(results, activity, paired, extra=None):
    columns = list(dict.fromkeys(
        ["visit_id", "location", "year", "utc_time", "screen_available", "plume_label", "wind_u", "wind_v"]
        + RATE_FIELDS + MATCHED_FIELDS
        + [field for field in RETRIEVAL_FEATURES if field not in ("month_sin", "month_cos")],
    ))
    visits = pd.read_parquet(results / "scored_visits.parquet", columns=columns)
    if extra:
        if paired:
            raise ValueError("Additional unpaired visits require satellite-only eligibility")
        visits = pd.concat([visits, extra_visits(extra, columns)], ignore_index=True)
        if not visits.visit_id.is_unique:
            raise ValueError("Duplicate scheduled visits")
    visits = visits.loc[visits.screen_available].dropna(subset=RATE_FIELDS)
    if paired:
        visits = visits.dropna(subset=MATCHED_FIELDS)
    visits = visits.assign(month=visits.utc_time.dt.month,
                           matched_stack=visits[MATCHED_FIELDS].mean(axis=1, skipna=False))
    days = visits.month * 2 * np.pi / 12
    visits = visits.assign(month_sin=np.sin(days), month_cos=np.cos(days))
    visits = visits.merge(activity, how="inner", on=["location", "year", "month"], validate="many_to_one")
    year_counts = visits.groupby(["location", "year"]).month.agg(["nunique", "size"])
    usable = year_counts["nunique"].ge(9) & year_counts["size"].ge(36)
    complete = usable.unstack("year").reindex(columns=[2023, 2024]).eq(True).all(axis=1)
    sites = complete.index[complete]
    visits = visits.loc[visits.location.isin(sites)].copy()
    activity = activity.loc[activity.location.isin(sites)].copy()

    panel = pd.read_csv(results / "panel.csv", usecols=["location", "State", "change_kg_h"])
    panel = panel.set_index("location").loc[sites].rename(columns={"change_kg_h": CHANGE})
    annual = pd.read_csv(results / "plant_annual_changes.csv",
                         usecols=["location", "scope", "policy", "flux_method",
                                  "estimated_rate_kg_h_change"])
    both_periods = annual.loc[annual.scope.eq("all_valid_overpasses") & annual.policy.eq("archive_detected_only")
                              & annual.flux_method.isin(("cross_section", "gaussian"))]
    covered = both_periods.groupby("location").estimated_rate_kg_h_change.count().eq(2)
    panel["previously_covered"] = covered.reindex(panel.index, fill_value=False)

    columns = ["location", "scale", "period", "reference_all_hours_kg_h", "stack_coverage"]
    reference = pd.read_csv(results / "plant_period_estimates.csv", usecols=columns)
    reference = reference.loc[reference.scale.eq("month") & reference.stack_coverage.ge(.95)]
    reference = reference[["location", "period", "reference_all_hours_kg_h"]].drop_duplicates()
    reference = reference.assign(year=reference.period.str[:4].astype(int),
                                 month=reference.period.str[5:].astype(int))
    return visits, activity, panel, reference.drop(columns="period")


def annual_raw(visits, activity):
    yearly = visits.groupby(["location", "year"])[RATE_FIELDS].mean()
    heat_full = hours_weighted(activity, "heat_total", "hours")
    heat_sampled = visits.groupby(["location", "year"]).heat_total.mean()
    heat_ratio = heat_full / heat_sampled
    heat_ratio = heat_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    zero = visits[RATE_FIELDS].clip(lower=0).mul(visits.plume_label, axis=0)
    zero_rates = zero.groupby([visits.location, visits.year]).mean()

    monthly = visits.groupby(["location", "year", "month"]).agg(
        **{c: (c, "mean") for c in RATE_FIELDS}, hours=("hours", "first"),
    ).reset_index()
    monthly = pd.concat([hours_weighted(monthly, field, "hours") for field in RATE_FIELDS], axis=1)
    return pd.DataFrame({
        BASELINE: yearly.clip(lower=0).mean(axis=1),
        "cross_section": yearly.cross_section_kg_h.clip(lower=0),
        "gaussian": yearly.gaussian_kg_h.clip(lower=0),
        "nondetections_zero": zero_rates.mean(axis=1),
        "calendar_weighting": monthly.clip(lower=0).mean(axis=1),
        "fuel_ratio": yearly.mul(heat_ratio, axis=0).clip(lower=0).mean(axis=1),
    })


def physical_features(data):
    heat = data[HEAT_FIELDS].to_numpy()
    summer = data.month.between(5, 9).to_numpy()[:, None]
    return np.hstack([heat * summer, heat * ~summer])


def fit_activity_model(by_month):
    """Fit nonnegative fuel-specific factors within and outside ozone season."""
    design = physical_features(by_month)
    response = by_month.reference_all_hours_kg_h.to_numpy()
    weights = np.sqrt(by_month.hours.to_numpy() / by_month.hours.mean())
    fit = least_squares(
        lambda coefficients: (design @ coefficients - response) * weights,
        np.ones(design.shape[1]),
        bounds=(0.0, np.inf),
        loss="soft_l1",
        f_scale=CHANGE_THRESHOLD,
    )
    if not fit.success:
        raise RuntimeError(f"Cannot fit training factors: {fit.message}")
    return fit.x


def calibrated_rates(train, test, features):
    usable = train.loc[train[MATCHED_FIELDS].notna().all(axis=1)].copy()
    weights = 1.0 / usable.groupby("location").location.transform("size")
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=61285,
    )
    return model.fit(usable[features], usable.matched_stack, sample_weight=weights).predict(test[features])


def annual(value_table, columns):
    return value_table.groupby(["location", "year"])[columns].mean()


def state_holdout(visits, activity, panel, reference, n_folds, train_through):
    outputs = []
    for fold, (train_ids, test_ids) in enumerate(
        GroupKFold(n_splits=n_folds).split(panel, groups=panel.State), 1
    ):
        train_sites, test_sites = panel.index[train_ids], panel.index[test_ids]
        train = visits.loc[visits.location.isin(train_sites) & visits.year.le(train_through)]
        test = visits.loc[visits.location.isin(test_sites)].copy()
        train_states = set(panel.loc[train_sites, "State"])
        test_states = set(panel.loc[test_sites, "State"])
        assert not train_states & test_states
        train_months = activity.loc[activity.location.isin(train_sites) & activity.year.le(train_through)].merge(
            reference, on=["location", "year", "month"], validate="one_to_one")
        factors = fit_activity_model(train_months)
        full_activity = activity.loc[activity.location.isin(test_sites)].copy()
        full_activity["fuel_prior"] = physical_features(full_activity) @ factors
        full_year_prior = hours_weighted(full_activity, "fuel_prior", "hours")
        test["fuel_prior"] = physical_features(test) @ factors
        test["calibrated_satellite"] = calibrated_rates(train, test, RETRIEVAL_FEATURES)
        test["calibrated_satellite_and_fuel"] = calibrated_rates(
            train, test, RETRIEVAL_FEATURES + HEAT_FEATURES,
        )
        means = annual(test, ["fuel_prior", "calibrated_satellite", "calibrated_satellite_and_fuel"])
        correction = full_year_prior - means.fuel_prior
        result = annual_raw(test, full_activity)
        result["fuel_only"] = full_year_prior
        result["activity_adjusted_raw"] = (result[BASELINE] + correction).clip(lower=0)
        result["calibrated_satellite"] = means.calibrated_satellite
        result["calibrated_satellite_and_fuel"] = means.calibrated_satellite_and_fuel
        result[CANDIDATE] = (means.calibrated_satellite_and_fuel + correction).clip(lower=0)
        result["fold"] = fold
        outputs.append(result)
        print(f"Fold {fold}, {len(test_sites)} plants in {', '.join(sorted(test_states))}", flush=True)
    annuals = pd.concat(outputs).sort_index()
    assert annuals.index.is_unique
    return annuals


def change_from_annual(annuals):
    values = annuals.drop(columns="fold").unstack("year")
    change = values.xs(2024, level="year", axis=1) - values.xs(2023, level="year", axis=1)
    change["no_change"] = 0.0
    return change


def material_declines_at_false_count(target, scores, allowed=4):
    """An evaluation curve point, with the cut swept over held-out scores."""
    order = pd.DataFrame({"truth": target.lt(-CHANGE_THRESHOLD), "estimate": scores})
    order = order.loc[order.estimate.lt(0)]
    cut = order.groupby("estimate").truth.agg(["sum", "count"]).sort_index()
    hits = cut["sum"].cumsum()
    false = (cut["count"] - cut["sum"]).cumsum()
    eligible = hits.loc[false.le(allowed)]
    return (int(eligible.iloc[-1]), int(false.loc[eligible.index[-1]])) if len(eligible) else (0, 0)


def assessment(panel, changes, original_score):
    truth = panel[CHANGE]
    large = truth.abs().gt(CHANGE_THRESHOLD)
    decline = truth.lt(-CHANGE_THRESHOLD)
    recovered = ~panel.previously_covered
    metrics = []
    for method in changes:
        pred = changes[method]
        error = pred.sub(truth).abs()
        found, false = material_declines_at_false_count(truth, pred)
        metrics.append({"method": method, "plants": len(truth), "large_changes": int(large.sum()),
                        "declines": int(decline.sum()), "new_plants": int(recovered.sum()),
                        "mae": error.mean(), "mae_large": error[large].mean(),
                        "mae_declines": error[decline].mean(), "mae_new_large": error[large & recovered].mean(),
                        "decline_average_precision": average_precision_score(decline, -pred),
                        "declines_at_4_false": found, "false_decline_alerts": false})
    reference = original_score.reindex(truth.index)
    found, false = material_declines_at_false_count(truth[reference.notna()], reference.dropna())
    print("Original cross-section upper interval, evaluated at the same allowance:", found, false)
    return pd.DataFrame(metrics).set_index("method")


def grouped_uncertainty(panel, changes):
    errors = changes.sub(panel[CHANGE], axis=0).abs()
    scopes = {
        "all": panel[CHANGE].notna(),
        "large_changes": panel[CHANGE].abs().gt(CHANGE_THRESHOLD),
        "declines": panel[CHANGE].lt(-CHANGE_THRESHOLD),
    }
    output = []
    for baseline in (BASELINE, "nondetections_zero", "fuel_only", "calibrated_satellite", "no_change"):
        for scope, mask in scopes.items():
            diff = (errors[CANDIDATE] - errors[baseline]).loc[mask]
            clusters = diff.groupby(panel.State[mask]).agg(["sum", "size"]).to_numpy()
            draws = np.random.default_rng(61285).choice(len(clusters), (10000, len(clusters)), replace=True)
            sums = clusters[draws].sum(axis=1)
            lo, hi = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975])
            output.append(dict(baseline=baseline, scope=scope, n=int(mask.sum()), difference=diff.mean(),
                               low=lo, high=hi))
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--activity", type=Path, default=Path("eia923_monthly_activity.csv"))
    parser.add_argument("--satellite-only", action="store_true", help="Image eligibility uses satellite fields only.")
    parser.add_argument("--extra-visits", type=Path, help="Previously scored imagery without a paired stack rate.")
    parser.add_argument("--train-through", type=int, choices=[2023, 2024], default=2024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    activity = activity_data(args.activity)
    visits, activity, panel, reference = prepare_inputs(
        args.results, activity, not args.satellite_only, extra=args.extra_visits,
    )
    print(f"Evaluation panel: {len(panel)} plants, {len(visits):,} visits, {panel.State.nunique()} states.", flush=True)
    intervals = pd.read_csv(args.results / "annual_direction_sampling_intervals.csv")
    original_score = intervals.loc[intervals.flux_method.eq("cross_section") & intervals.policy.eq("all_signed")]
    with threadpool_limits(limits=args.threads):
        annuals = state_holdout(visits, activity, panel, reference, n_folds=5, train_through=args.train_through)
    changes = change_from_annual(annuals)
    scores = assessment(panel, changes, original_score.set_index("location").upper_kg_h)
    differences = grouped_uncertainty(panel, changes)
    print(scores.round(2).to_string())
    print(differences.round(2).to_string(index=False))
    prefix = "satellite_eligibility" if args.satellite_only else "paired_eligibility"
    if args.extra_visits:
        prefix += "_with_extra"
    if args.train_through == 2023:
        prefix += "_train_2023"
    annuals.to_csv(f"{prefix}_annual.csv")
    panel.join(changes).to_csv(f"{prefix}_changes.csv")
    scores.to_csv(f"{prefix}_metrics.csv")
    differences.to_csv(f"{prefix}_differences.csv", index=False)


if __name__ == "__main__":
    main()
