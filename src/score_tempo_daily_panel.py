#!/usr/bin/env python3
"""Validate annual TEMPO changes on common sampling days against stack records."""

import argparse
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd

from retrieve_tempo_daily_panel import AGES, PLUME_WEIGHTS


YEARS = (2024, 2025)
ESTIMATES = ("midday_scan", "midday_flexible", "scan_average", "joint_background", "cems_at_scans")


def stack_inputs(ground_files, sites):
    hours = pd.concat([pd.read_parquet(path, columns=["facility_id", "utc", "mass_kg"],
                                      filters=[("facility_id", "in", list(sites))]) for path in ground_files])
    hours["utc"] = pd.to_datetime(hours.utc, utc=True)
    assert not hours.duplicated(["facility_id", "utc"]).any()
    monthly = hours.assign(year=hours.utc.dt.year, month=hours.utc.dt.month).groupby(
        ["facility_id", "year", "month"]).mass_kg.agg(["mean", "count"])
    full = {key: float(row["mean"]) for key, row in monthly.iterrows() if row["count"] >= .95 * 24 * monthrange(*key[1:])[1]}
    annual = {}
    for (plant, year), group in hours.groupby(["facility_id", hours.utc.dt.year]):
        available = sum(monthrange(year, month)[1] for month in range(1, 13))*24
        if len(group) >= .95*available:
            annual[plant, year] = group.mass_kg.mean()
    return {key: part.set_index("utc").mass_kg for key, part in hours.groupby("facility_id")}, full, annual


def matched_days(scans, stack):
    blocks = []
    for location, group in scans.groupby("location"):
        truth = stack.get(location, pd.Series(dtype=float))
        times = pd.DatetimeIndex(group.scan_utc)
        aligned = np.column_stack([truth.reindex((times - pd.Timedelta(hours=float(age))).floor("h")).to_numpy()
                                   for age in AGES])
        releases = np.column_stack([truth.reindex(times.floor("h") - pd.Timedelta(hours=h)).to_numpy()
                                    for h in range(1, 9)])
        group = group.assign(reference=np.average(aligned, weights=PLUME_WEIGHTS, axis=1),
                             stopped_8h=(releases == 0).all(axis=1) & (aligned == 0).all(axis=1))
        blocks.append(group)
    align = pd.concat(blocks, ignore_index=True)
    rows = []
    for (location, day), group in align.groupby(["location", "date"]):
        closest = group.loc[group.closest_midday]
        if len(closest) != 1 or len(group) < 2 or group.reference.isna().any():
            continue
        primary = closest.iloc[0]
        common = dict(location=location, date=day, year=day.year, month=day.month,
                      n_scans=len(group), direction_spread=primary.day_direction_spread,
                      midday_utc=primary.scan_utc, stopped_8h=group.stopped_8h.all())
        rows.append({**common, "midday_scan": primary.independent_kg_h, "midday_flexible": primary.flexible_single_kg_h,
                     "scan_average": group.independent_kg_h.mean(), "joint_background": group.shared_background_kg_h.mean(),
                     "cems_at_scans": group.reference.mean(), "cems_at_midday": primary.reference})
    return align, pd.DataFrame(rows)


def resample_month(data, rng, draws):
    value = data[list(ESTIMATES)].to_numpy()
    weeks = (data.date.dt.day - 1) // 7
    _, inverse = np.unique(weeks, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=y) for y in value.T])
    weight = rng.multinomial(len(counts), np.repeat(1 / len(counts), len(counts)), size=draws)
    resampled = (weight @ sums) / (weight @ counts)[:, None]
    return np.concatenate([value.mean(axis=0, keepdims=True), resampled], axis=0)


def annual_changes(daily, full, panel):
    trend, coverage = [], []
    for site in panel.itertuples(index=False):
        site_days = daily.loc[daily.location.eq(site.location)]
        counts = site_days.groupby(["year", "month"]).size()
        months = [m for m in range(1, 13) if all(counts.get((y, m), 0) >= 2 and (site.location, y, m) in full
                                                for y in YEARS)]
        days = {y: sum(counts.get((y, m), 0) for m in months) for y in YEARS}
        support = {f"matched_days_{y}": days[y] for y in YEARS}
        coverage.append({"location": site.location, "recovered": site.recovered, **support,
                         "matched_months": len(months), "months": ",".join(map(str, months)),
                         **{f"days_{y}_{m:02d}": counts.get((y, m), 0) for y in YEARS for m in range(1, 13)}})
        if len(months) < 3 or min(days.values()) < 10:
            continue
        scope = "annual" if len(months) == 12 and min(days.values()) >= 36 else "matched_months"
        rng = np.random.default_rng(1451 + site.location)
        period, truth = {}, {}
        for y in YEARS:
            rates = [resample_month(site_days.loc[site_days.year.eq(y) & site_days.month.eq(m)], rng, 4000) for m in months]
            weights = [monthrange(y, m)[1] for m in months]
            period[y] = np.maximum(0., np.average(rates, weights=weights, axis=0))
            truth[y] = float(np.average([full[site.location, y, m] for m in months], weights=weights))
        delta = period[2025] - period[2024]
        lower, upper = np.quantile(delta[1:], [.025, .975], axis=0)
        for i, name in enumerate(ESTIMATES):
            trend.append({"location": site.location, "facility": site.Facility_Name, "state": site.State,
                          "recovered": site.recovered, **support, "matched_months": len(months), "scope": scope,
                          "method": name, "rate_2024": period[2024][0, i], "rate_2025": period[2025][0, i],
                          "delta": delta[0, i], "lower": lower[i], "upper": upper[i],
                          "cems_2024": truth[2024], "cems_2025": truth[2025], "cems_delta": truth[2025]-truth[2024]})
    return pd.DataFrame(trend), pd.DataFrame(coverage)


def curve(scores, actual):
    ranked = pd.DataFrame({"score": scores, "positive": actual})
    counts = ranked.groupby("score").positive.agg(["sum", "count"]).sort_index(ascending=False)
    positives = np.r_[0, counts["sum"].to_numpy().cumsum()]
    negatives = np.r_[0, (counts["count"] - counts["sum"]).to_numpy().cumsum()]
    return pd.DataFrame({"threshold": np.r_[np.inf, counts.index], "true_decreases": positives,
                         "false_decreases": negatives, "n_true": int(actual.sum()), "n_other": int((~actual).sum())})


def evaluate(trends):
    summary, roc = [], []
    if trends.empty:
        return pd.DataFrame(), pd.DataFrame()
    for scope, support in trends.groupby("scope"):
        for cohort, cases in [("all", support), ("recovered", support.loc[support.recovered])]:
            for name, group in cases.groupby("method"):
                for threshold in [25, 50, 100]:
                    actual = group.cems_delta.to_numpy() < -threshold
                    score = -group.upper.to_numpy() - threshold
                    called = score > 0
                    material = group.cems_delta.abs() > threshold
                    opposite = material & (np.sign(group.delta) != np.sign(group.cems_delta))
                    upward = group.lower > threshold
                    actual_upward = group.cems_delta > threshold
                    summary.append({"scope": scope, "cohort": cohort, "threshold": threshold, "method": name,
                                    "n": len(group), "n_true_changes": int(material.sum()), "n_true_decreases": int(actual.sum()),
                                    "direction_errors": int(opposite.sum()), "confirmed_decreases": int((called & actual).sum()),
                                    "false_decreases": int((called & ~actual).sum()), "confirmed_increases": int((upward & actual_upward).sum()),
                                    "false_increases": int((upward & ~actual_upward).sum()),
                                    "mae": (group.delta - group.cems_delta).abs().mean(), "zero_change_mae": group.cems_delta.abs().mean()})
                    roc.append(curve(score, actual).assign(scope=scope, cohort=cohort, method=name, minimum_change=threshold))
    return pd.DataFrame(summary), pd.concat(roc, ignore_index=True)


def check_against_full_year(trends, full_year):
    if trends.empty:
        return trends
    candidates = trends.loc[trends.matched_months.ge(9) & trends.matched_days_2024.ge(36)
                            & trends.matched_days_2025.ge(36)].copy()
    covered = {site for site in candidates.location.unique() if all((site, year) in full_year for year in YEARS)}
    candidates = candidates.loc[candidates.location.isin(covered)].copy()
    for year in YEARS:
        rates = {site: full_year[site, year] for site in covered}
        candidates[f"cems_{year}"] = candidates.location.map(rates)
    candidates["cems_delta"] = candidates.cems_2025 - candidates.cems_2024
    candidates["scope"] = "full_year_reference"
    return candidates


def daily_summary(daily, panel):
    rows = []
    data = daily.join(panel.set_index("location").recovered, on="location")
    for year, frame in data.groupby("year"):
        for cohort, group in [("all", frame), ("recovered", frame.loc[frame.recovered])]:
            for subset, part in [("all", group), ("eight_hours_off", group.loc[group.stopped_8h])]:
                if part.empty:
                    continue
                for method in ESTIMATES[:-1]:
                    same_hours = part.cems_at_midday if method.startswith("midday") else part.cems_at_scans
                    rows.append({"year": year, "cohort": cohort, "subset": subset, "method": method,
                                 "facility_days": len(part), "facilities": part.location.nunique(),
                                 "mae_same_hours": (part[method] - same_hours).abs().mean(),
                                 "mae_same_day_time_window": (part[method] - part.cems_at_scans).abs().mean(),
                                 "reported_100_or_more": (part[method] >= 100).sum()})
    return pd.DataFrame(rows)


def paired_daily_errors(daily):
    rows = []
    rng = np.random.default_rng(7582)
    for year, data in daily.groupby("year"):
        absolute = data[list(ESTIMATES[:-1])].subtract(data.cems_at_scans, axis=0).abs()
        by_plant = absolute.groupby(data.location).mean()
        for comparison in ["midday_scan", "midday_flexible", "scan_average"]:
            difference = by_plant.joint_background - by_plant[comparison]
            resampled = rng.choice(difference.to_numpy(), size=(4000, len(difference))).mean(axis=1)
            rows.append({"year": year, "comparison": comparison, "plants": len(difference), "days": len(data),
                         "baseline_plant_average_mae": by_plant[comparison].mean(),
                         "joint_plant_average_mae": by_plant.joint_background.mean(),
                         "mae_difference": difference.mean(), "lower": np.quantile(resampled, .025),
                         "upper": np.quantile(resampled, .975), "plants_with_lower_error": int((difference < 0).sum())})
    return pd.DataFrame(rows)


def equal_false_calls(roc):
    if roc.empty:
        return pd.DataFrame()
    rows = []
    for key, group in roc.groupby(["scope", "cohort", "method", "minimum_change"]):
        false = group.n_other.iloc[0]
        for budget in range(false + 1):
            hit = group.loc[group.false_decreases.le(budget), "true_decreases"].max()
            rows.append(dict(zip(["scope", "cohort", "method", "minimum_change"], key),
                             max_false_decreases=budget, true_decreases=int(hit),
                             n_true=group.n_true.iloc[0], n_other=false))
    return pd.DataFrame(rows)


def daily_detection(daily):
    applicable = daily.loc[daily.stopped_8h | daily.cems_at_scans.ge(250)]
    rows = []
    for year, group in [("pooled", applicable), *applicable.groupby("year")]:
        emitting = group.cems_at_scans.ge(250).to_numpy()
        for method in ESTIMATES[:-1]:
            calls = group[method].ge(100).to_numpy()
            rows.append({"year": year, "method": method, "emitting": int(emitting.sum()),
                         "stopped": int((~emitting).sum()), "emitting_called": int((calls & emitting).sum()),
                         "stopped_called": int((calls & ~emitting).sum())})
    return pd.DataFrame(rows)


def pair_with_tropomi(daily, annual):
    fields = ["location", "utc_time", "year", "cross_section_kg_h", "cross_section_cems_kg_h", "plume_label"]
    tropomi = pd.read_parquet(annual / "scored_visits.parquet", columns=fields, filters=[("year", "=", 2024)])
    tropomi = tropomi.dropna(subset=["cross_section_kg_h", "cross_section_cems_kg_h"])
    tropomi["date"] = tropomi.utc_time.dt.normalize()
    options = daily.loc[daily.year.eq(2024)].merge(tropomi.drop(columns="year"), on=["location", "date"])
    options["offset_minutes"] = (options.utc_time - options.midday_utc).abs().dt.total_seconds() / 60
    pairs = (options.loc[options.offset_minutes.le(60)].sort_values("offset_minutes")
             .drop_duplicates(["location", "date"]).sort_values(["location", "date"]))
    rows = []
    for method, reference in [("cross_section_kg_h", "cross_section_cems_kg_h"),
                              ("midday_scan", "cems_at_midday"), ("scan_average", "cems_at_scans"),
                              ("joint_background", "cems_at_scans")]:
        error = pairs[method] - pairs[reference]
        by_plant = error.abs().groupby(pairs.location).mean()
        rows.append({"method": method, "aligned_cems": reference, "facilities": len(by_plant), "plant_days": len(pairs),
                     "plant_average_mae": by_plant.mean(), "signed_mean_error": error.mean()})
    return pairs, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--ground", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.annual / "panel.csv")
    recovery = pd.read_csv(args.annual / "recovery_diagnostic_sites.csv")
    panel["recovered"] = panel.location.isin(recovery.loc[recovery.scope.eq("recovered"), "location"])
    stack, full, full_year = stack_inputs(args.ground, panel.location)
    lists = [pd.read_parquet(path) for path in sorted((args.work / "retrieved").glob("*.parquet"))]
    scans = pd.concat([table for table in lists if len(table)], ignore_index=True)
    if scans.duplicated(["location", "scan_utc"]).any():
        raise ValueError("Input subsets overlap. Use one retrieval per site and time.")
    aligned, daily = matched_days(scans, stack)
    trend, coverage = annual_changes(daily, full, panel)
    summary, roc = evaluate(trend)
    annual = check_against_full_year(trend, full_year)
    annual_summary, annual_roc = evaluate(annual)
    tropomi_pairs, instrument_summary = pair_with_tropomi(daily, args.annual)
    for key, data in [("daily_rates", daily), ("matched_period_changes", trend), ("coverage", coverage),
                      ("change_summary", summary), ("false_call_tradeoff", roc), ("equal_false_calls", equal_false_calls(roc)),
                      ("changes_against_full_year", annual), ("full_year_summary", annual_summary),
                      ("full_year_false_calls", equal_false_calls(annual_roc)),
                      ("daily_summary", daily_summary(daily, panel)), ("paired_daily_errors", paired_daily_errors(daily)),
                      ("daily_detection_at_100", daily_detection(daily)),
                      ("tropomi_2024_pairs", tropomi_pairs), ("cross_instrument_daily_summary", instrument_summary)]:
        data.to_csv(args.output / (key + ".csv"), index=False)
    aligned.to_parquet(args.output / "scan_rates.parquet", index=False)
    print(len(scans), "qualified scans", len(daily), "common site-days", daily.location.nunique(), "sites")
    print(coverage.loc[coverage.recovered, ["location", "matched_months", "matched_days_2024", "matched_days_2025"]].to_string(index=False))
    print(summary.loc[summary.threshold.eq(25)].to_string(index=False) if not summary.empty else "Insufficient matched calendar coverage")


if __name__ == "__main__":
    main()
