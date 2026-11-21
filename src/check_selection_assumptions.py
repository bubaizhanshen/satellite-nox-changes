"""Controls for annual change estimates from satellite overpass samples."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = (2023, 2024)
FIELDS = {
    "stack_2h": "mean_mass_2h",
    "cross_section": "cross_section_kg_h",
    "gaussian": "gaussian_kg_h",
}
PAIRED_FIELDS = list(FIELDS.values())[1:] + ["cross_section_cems_kg_h", "gaussian_cems_kg_h"]
RANDOM_DRAWS = 512


def paired_years(frame, minimum_visits=36, minimum_months=9, years=YEARS):
    counts = frame.groupby(["location", "year"]).agg(
        visits=("utc_time", "size"), months=("month", "nunique"),
    )
    ready = (counts.visits.ge(minimum_visits) & counts.months.ge(minimum_months)).unstack().fillna(False)
    ready = ready.reindex(columns=years, fill_value=False).all(axis=1)
    return frame.loc[frame.location.isin(ready[ready].index)]


def change(frame, column, detections=False, clip="none", years=YEARS):
    selected = frame.loc[frame.plume_label] if detections else frame
    rate = selected[column].clip(lower=0) if clip == "overpass" else selected[column]
    yearly = selected.assign(rate=rate).groupby(["location", "year"]).rate.mean().unstack()
    yearly = yearly.reindex(columns=years)
    if clip == "annual":
        yearly = yearly.clip(lower=0)
    return yearly[years[1]] - yearly[years[0]]


def calendar_weighted_change(frame, column, clip=False):
    monthly = frame.groupby(["location", "year", "month"])[column].mean().reset_index(name="mean_rate")
    month_days = pd.PeriodIndex(monthly.month, freq="M").days_in_month
    monthly["days"] = month_days
    monthly["mass"] = monthly.mean_rate * month_days
    annual = monthly.groupby(["location", "year"])[["mass", "days"]].sum()
    means = (annual.mass / annual.days).unstack().reindex(columns=YEARS)
    if clip:
        means = means.clip(lower=0)
    return means[YEARS[1]] - means[YEARS[0]]


def errors_against_reference(estimate, actual):
    return (estimate - actual).abs().dropna()


def uncertainty(differences, panel):
    if len(differences) < 2:
        return np.nan, np.nan
    sampling = panel.reindex(differences.index)
    rng = np.random.default_rng(61285)
    groups = sampling.groupby("stratum", sort=True, observed=True).indices.values()
    choices = np.concatenate([rng.choice(group, (4000, len(group)), replace=True) for group in groups], axis=1)
    weights = sampling.design_weight.to_numpy()[choices]
    draws = (differences.to_numpy()[choices] * weights).sum(axis=1) / weights.sum(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def contrast(before, after, actual, panel, test, method):
    paired = pd.concat([before.rename("before"), after.rename("after"), actual.rename("actual")], axis=1).dropna()
    row = dict(test=test, method=method, total=len(actual), before_coverage=before.notna().sum(),
               after_coverage=after.notna().sum(), paired=len(paired))
    if paired.empty:
        return row
    err_before = errors_against_reference(paired.before, paired.actual)
    err_after = errors_against_reference(paired.after, paired.actual)
    w = panel.design_weight.loc[paired.index]
    big = paired.actual.abs().gt(25)
    diff = err_after - err_before
    lo, hi = uncertainty(diff, panel)
    return dict(row, before_mae=np.average(err_before, weights=w), after_mae=np.average(err_after, weights=w),
                delta_mae=np.average(diff, weights=w), low=lo, high=hi, large_changes=int(big.sum()),
                before_reversals=int((big & (paired.before * paired.actual < 0)).sum()),
                after_reversals=int((big & (paired.after * paired.actual < 0)).sum()),
                improved=int(diff.lt(0).sum()))


def random_matched_rates(year, rng, match_month, fields):
    """Keep the detection counts constant, with or without matching each month."""
    parts = [year] if not match_month else [rows for _, rows in year.groupby("month", sort=True)]
    total = np.zeros((RANDOM_DRAWS, len(fields)))
    count = 0
    for part in parts:
        detections = int(part.plume_label.sum())
        if not detections:
            continue
        values = part[list(fields.values())].clip(lower=0).to_numpy()
        if detections == len(values):
            total += values.sum(axis=0)
        else:
            pick = np.argpartition(rng.random((RANDOM_DRAWS, len(values))),
                                   detections - 1, axis=1)[:, :detections]
            total += values[pick].sum(axis=1)
        count += detections
    return total / count


def equal_count_control(usable, actual, panel, years=YEARS, fields=FIELDS):
    usable = usable.loc[usable.mean_mass_2h.notna()]
    rng = np.random.default_rng(24962026)
    outputs = []
    for match_month in (False, True):
        random_errors, observed_errors, sites = [], [], []
        for location, rows in usable.groupby("location", sort=True):
            by_year = {year: group for year, group in rows.groupby("year")}
            if location not in actual or set(by_year) != set(years):
                continue
            if not all(frame.plume_label.any() for frame in by_year.values()):
                continue
            drawn = [random_matched_rates(by_year[year], rng, match_month, fields) for year in years]
            detected = [by_year[year].loc[lambda x: x.plume_label, list(fields.values())].clip(lower=0).mean().to_numpy()
                        for year in years]
            target = actual.loc[location]
            random_errors.append(np.abs(drawn[1] - drawn[0] - target))
            observed_errors.append(np.abs(detected[1] - detected[0] - target))
            sites.append(location)
        if not sites:
            continue
        weights = panel.design_weight.loc[sites].to_numpy()
        sampled = np.average(np.stack(random_errors), axis=0, weights=weights)
        detected = np.average(np.stack(observed_errors), axis=0, weights=weights)
        for j, method in enumerate(fields):
            draws = sampled[:, j]
            outputs.append({
                "test": "equal_count_and_month" if match_month else "equal_count",
                "method": method, "plants": len(sites), "draws": RANDOM_DRAWS,
                "detected_mae": detected[j], "random_mean_mae": draws.mean(),
                "random_p05_mae": np.quantile(draws, .05), "random_p95_mae": np.quantile(draws, .95),
                "fraction_random_with_smaller_error": float(np.mean(draws < detected[j])),
            })
    return pd.DataFrame(outputs)


def historical_controls(archive):
    visits = pd.read_parquet(archive / "population.parquet")
    visits = visits.loc[visits[["view_band", "mean_mass_2h", "latitude", "longitude", "annual_nox_emission"]]
                        .notna().all(axis=1)].copy()
    visits["month"] = visits.utc_time.dt.strftime("%Y-%m")
    reference = pd.read_csv(archive / "annual_stack_comparison.csv")
    reference = reference.loc[reference.stack_hour_coverage.ge(.95)]
    out = []
    matched_controls = []
    for end in range(2020, 2025):
        years = (end - 1, end)
        ref = reference.loc[reference.year.isin(years)].pivot(
            index="location", columns="year", values="all_hours_stack_kg_h",
        ).dropna()
        if set(ref.columns) != set(years) or ref.empty:
            continue
        span = visits.loc[visits.year.isin(years) & visits.location.isin(ref.index)]
        paired_visits = paired_years(span, years=years)
        ref = ref.loc[ref.index.isin(paired_visits.location)]
        actual = ref[years[1]] - ref[years[0]]
        strata = pd.cut(actual, [-np.inf, -100, -50, -25, 25, 50, 100, np.inf])
        panel = pd.DataFrame({"design_weight": 1., "stratum": strata}, index=actual.index)
        out.append(contrast(
            change(paired_visits, "mean_mass_2h", detections=True, years=years),
            change(paired_visits, "mean_mass_2h", years=years),
            actual, panel, f"{years[0]}_{years[1]}", "stack_2h",
        ))
        controls = equal_count_control(paired_visits, actual, panel, years, {"stack_2h": "mean_mass_2h"})
        matched_controls.append(controls.assign(period=f"{years[0]}_{years[1]}"))
    return pd.DataFrame(out), pd.concat(matched_controls, ignore_index=True)


def read_extra(work):
    inputs = work / "imagery_only" / "inputs"
    paths = sorted((work / "imagery_only" / "scored_temis").glob("*.parquet"))
    extra = pd.read_parquet(inputs / "visits.parquet")
    scores = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    values = extra.merge(scores, on="visit_id", validate="one_to_one", how="left")
    values["screen_available"] = values.screen_error.eq("") & values.template_score.notna()
    return values


def full_checks(annual_output, run, destination, include_extra, archive):
    panel = pd.read_csv(annual_output / "panel.csv").set_index("location")
    actual = panel.change_kg_h
    stack_coverage = pd.read_csv(archive / "annual_stack_comparison.csv")
    complete_stack = (stack_coverage.loc[stack_coverage.year.isin(YEARS)]
                      .pivot(index="location", columns="year", values="stack_hour_coverage")
                      .reindex(actual.index)[list(YEARS)].ge(.99).all(axis=1))
    actual_complete = actual.loc[complete_stack]
    data = pd.read_parquet(annual_output / "scored_visits.parquet")
    raw = data.loc[data.screen_available]
    paired_flux = raw.loc[raw[PAIRED_FIELDS].notna().all(axis=1)]
    primary = paired_years(paired_flux)
    report = []
    for method, field in FIELDS.items():
        if method == "stack_2h":
            report.append(contrast(change(primary, field, detections=True), change(primary, field),
                                   actual, panel, "stack_rates_at_selected_overpasses", method))
            continue
        reference = f"{method}_cems_kg_h"
        report.append(contrast(change(primary, reference, detections=True), change(primary, reference),
                               actual, panel, "same_release_windows", method))
        base = change(primary, field, detections=True, clip="overpass")
        all_signed = change(primary, field, clip="annual")
        all_clipped = change(primary, field, clip="overpass")
        detected_signed = change(primary, field, detections=True, clip="annual")
        zero_rates = primary[field].clip(lower=0).where(primary.plume_label, 0.)
        zero_nondetections = change(primary.assign(zero_rates=zero_rates), "zero_rates")
        for label, target in (
            ("nondetections_as_zero", actual),
            ("nondetections_as_zero_on_large_changes", actual.loc[actual.abs().gt(25)]),
        ):
            report.append(contrast(zero_nondetections.reindex(target.index), all_signed.reindex(target.index),
                                   target, panel, label, method))
        report.append(contrast(base.reindex(actual_complete.index), all_signed.reindex(actual_complete.index),
                               actual_complete, panel, "99_percent_stack_coverage", method))
        report.append(contrast(all_signed, calendar_weighted_change(primary, field, clip=True), actual,
                               panel, "weight_by_calendar_days", method))
        for n, months in ((24, 6), (48, 10), (60, 12)):
            alternative = paired_years(paired_flux, minimum_visits=n, minimum_months=months)
            report.append(contrast(change(alternative, field, detections=True, clip="overpass"),
                                   change(alternative, field, clip="annual"), actual, panel,
                                   f"{n}_visits_{months}_months", method))
        for policy, b, a in (
            ("standard_comparison", base, all_signed),
            ("same_overpass_clipping", base, all_clipped),
            ("same_annual_clipping", detected_signed, all_signed),
            ("change_clipping_only_detected", base, detected_signed),
            ("change_clipping_only_all", all_clipped, all_signed),
        ):
            report.append(contrast(b, a, actual, panel, policy, method))

    if include_extra:
        extra = read_extra(run)
        data = pd.concat([data, extra], ignore_index=True)
        if not data.visit_id.is_unique:
            raise ValueError("Imagery-only overpasses collide with the primary input IDs")
    joint = paired_years(data.loc[data[list(FIELDS.values())[1:]].notna().all(axis=1)])
    for method in ("cross_section", "gaussian"):
        field = FIELDS[method]
        independent = paired_years(data.loc[data[field].notna()])
        for label, sample in (("imagery_only_both_retrievals", joint),
                              ("imagery_only_single_retrieval", independent)):
            report.append(contrast(change(sample, field, detections=True, clip="overpass"),
                                   change(sample, field, clip="annual"), actual, panel,
                                   label, method))
    pd.DataFrame(report).to_csv(destination / "selection_controls.csv", index=False)
    equal_count_control(primary, actual, panel).to_csv(destination / "equal_count_controls.csv", index=False)
    print(f"Selection controls written for {len(panel)} candidate plants", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--annual-output", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--extra-overpasses", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    history, matched_history = historical_controls(args.archive)
    history.to_csv(args.destination / "historical_stack_sampling.csv", index=False)
    matched_history.to_csv(args.destination / "historical_equal_count_controls.csv", index=False)
    print(history[["test", "total", "paired", "before_mae", "after_mae"]].to_string(index=False), flush=True)
    if args.annual_output:
        full_checks(args.annual_output, args.run, args.destination, args.extra_overpasses, args.archive)


if __name__ == "__main__":
    main()
