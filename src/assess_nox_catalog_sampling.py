#!/usr/bin/env python3
"""Quantify NOx sampling bias using measured emissions on archived overpasses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_tropomi_footprint_evidence import compare_angles, match_observing_conditions


def load_frame(path):
    frame = pd.read_parquet(path)
    frame["utc_time"] = pd.to_datetime(frame.utc_time, utc=True)
    frame["month"] = frame.utc_time.dt.month
    frame["month_day"] = 100 * frame.month + frame.utc_time.dt.day
    frame = frame.loc[frame.mean_mass_2h.notna() & frame.view_band.notna()].copy()
    if frame.duplicated(["location", "utc_time"]).any():
        raise ValueError("A facility overpass occurs more than once")
    frame["true_rate"] = frame.mean_mass_2h
    frame["true_rate_detected"] = frame.true_rate.where(frame.plume_label, 0)
    return frame


def totals(frame):
    groups = [("year",), ("year", "view_band"), ("year", "ground_state")]
    summaries = []
    for keys in groups:
        table = frame.groupby(list(keys), observed=True).agg(
            visits=("true_rate", "size"), detections=("plume_label", "sum"),
            plants=("location", "nunique"), rate_sum=("true_rate", "sum"),
            rate_on_detections=("true_rate_detected", "sum"),
        ).reset_index()
        table["all_mean_kg_h"] = table.rate_sum / table.visits
        table["detected_mean_kg_h"] = table.rate_on_detections / table.detections.replace(0, np.nan)
        table["zero_filled_mean_kg_h"] = table.rate_on_detections / table.visits
        table["detected_mass_fraction"] = table.rate_on_detections / table.rate_sum.replace(0, np.nan)
        summaries.append(table.assign(grouping=" by ".join(keys)))
    return pd.concat(summaries, ignore_index=True)


def plant_changes(frame, early, late):
    years = frame.loc[frame.year.isin([early, late]) & frame.month_day.le(1222)].copy()
    month = ["location", "month"]
    months = years.groupby([*month, "year"]).size().unstack()
    months = months.dropna()
    months["common_weight"] = 2 * months[early] * months[late] / (months[early] + months[late])
    paired = years.join(months["common_weight"], on=month).dropna(subset="common_weight")
    paired["weight"] = paired.common_weight / paired.groupby([*month, "year"]).year.transform("size")
    paired["true_weighted"] = paired.weight * paired.true_rate
    paired["detected_weighted"] = paired.weight * paired.true_rate_detected
    paired["detection_weight"] = paired.weight * paired.plume_label
    plant = paired.groupby(["location", "year"]).agg(
        common_months=("month", "nunique"), visits=("true_rate", "size"), detections=("plume_label", "sum"),
        all_weight=("weight", "sum"), detected_weight=("detection_weight", "sum"),
        true_rate_weighted_sum=("true_weighted", "sum"), true_rate_detected_weighted_sum=("detected_weighted", "sum"),
    )
    plant["all_kg_h"] = plant.true_rate_weighted_sum / plant.all_weight
    plant["detected_only_kg_h"] = plant.true_rate_detected_weighted_sum / plant.detected_weight.replace(0, np.nan)
    plant["zero_filled_kg_h"] = plant.true_rate_detected_weighted_sum / plant.all_weight
    paired = plant.unstack("year")
    change = paired["all_kg_h"][late] - paired["all_kg_h"][early]
    paired[("measured_change_kg_h", late)] = change
    for method in ["detected_only", "zero_filled"]:
        selected = paired[f"{method}_kg_h"][late] - paired[f"{method}_kg_h"][early]
        paired[(f"{method}_change_kg_h", late)] = selected
        paired[(f"{method}_direction_reversed", late)] = (selected * change).lt(0).where(selected.notna())
    paired.columns = [f"{variable}_{year}" for variable, year in paired.columns]
    return paired.join(frame.drop_duplicates("location").set_index("location")[["Facility_Name", "State"]])


def change_summary(plant, early, late):
    results = []
    for minimum_visits in [1, 30, 60]:
        enough_visits = plant[[f"visits_{early}", f"visits_{late}"]].min(axis=1).ge(minimum_visits)
        for minimum_detections in [1, 5, 10]:
            enough_detections = plant[[f"detections_{early}", f"detections_{late}"]].min(axis=1).ge(minimum_detections)
            for minimum_difference in [0, 25, 50, 100]:
                true_change = plant[f"measured_change_kg_h_{late}"]
                meaningful = true_change.abs().gt(minimum_difference)
                covered = enough_visits & meaningful & enough_detections
                observed = true_change.loc[covered]
                count = len(observed)
                for method in ["detected_only", "zero_filled"]:
                    selected = plant.loc[covered, f"{method}_change_kg_h_{late}"]
                    results.append({
                        "period": f"{early}-{late}", "method": method,
                        "min_visits_each_year": minimum_visits, "min_detections_each_year": minimum_detections,
                        "abs_measured_change_gt_kg_h": minimum_difference, "eligible_plants": int((enough_visits & meaningful).sum()),
                        "compared_plants": count, "direction_reversals": int((observed * selected).lt(0).sum()),
                        "direction_ties": int(selected.eq(0).sum()),
                        "change_mae_kg_h": (observed - selected).abs().mean(),
                    })
    return pd.DataFrame(results)


def sustained_shutdowns(frame, ground_dir):
    visits = []
    for year, overpasses in frame.groupby("year"):
        path = ground_dir / f"{year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        facilities = overpasses.location.unique().tolist()
        stack = pd.read_parquet(path, filters=[("facility_id", "in", facilities)])
        previous = path.with_name(f"{year-1}.parquet")
        if previous.is_file():
            begin = pd.Timestamp(f"{year}-01-01", tz="UTC") - pd.Timedelta(days=30)
            tail = pd.read_parquet(previous, filters=[("utc", ">=", begin), ("facility_id", "in", facilities)])
            stack = pd.concat([tail, stack], ignore_index=True)
        stack["utc"] = pd.to_datetime(stack.utc, utc=True)
        stack = stack.sort_values(["facility_id", "utc"])
        starts = stack.facility_id.ne(stack.facility_id.shift()) | stack.utc.diff().ne(pd.Timedelta(hours=1))
        runs = (starts | stack.stopped.ne(stack.stopped.shift())).cumsum()
        stack["stop_age_hours"] = stack.groupby(runs).cumcount().add(1).where(stack.stopped, 0)
        # The first hour of a run may have a different facility or a gap.
        # Group starts require a new, explicitly measured stop before counting age.
        age = stack[["facility_id", "utc", "stop_age_hours"]].rename(
            columns={"facility_id": "location", "utc": "hour"}
        )
        visits.append(overpasses.assign(hour=overpasses.utc_time.dt.floor("h")).merge(
            age, on=["location", "hour"], validate="many_to_one",
        ))
    linked = pd.concat(visits, ignore_index=True)
    off = linked.loc[linked.stop_age_hours.ge(8)].copy()
    if off.true_rate.abs().gt(1e-8).any():
        raise ValueError("A continuous stop has a nonzero two-hour stack rate")
    off["elapsed_stop_hours"] = off.stop_age_hours - 1 + (off.utc_time - off.hour).dt.total_seconds() / 3600
    off["continuous_stop_age"] = pd.cut(off.elapsed_stop_hours, [7, 24, 168, 720, np.inf], right=False,
                                       labels=["8-23 h", "1-6 days", "7-29 days", "30+ days"])
    result = off.groupby(["year", "continuous_stop_age", "view_band"], observed=True).agg(
        visits=("plume_label", "size"), detections=("plume_label", "sum"), plants=("location", "nunique"),
    ).reset_index()
    result["positive_fraction"] = result.detections / result.visits
    return off, result


def matched_shutdown_angles(off):
    sample = off.loc[off.view_band.isin(["0-25", "35-45"])].copy()
    sample["angle"] = sample.view_band.map({"0-25": "near", "35-45": "mid"})
    sample["month"] = sample.utc_time.dt.month
    sample["record"] = np.arange(len(sample))
    rows, weather = [], []
    for year, group in sample.groupby("year"):
        for hours in [24, 168, 720]:
            lengthy = group.loc[group.elapsed_stop_hours.ge(hours)].copy().assign(state="verified_idle")
            matched = compare_angles(lengthy).reset_index()
            rows.append(matched.assign(year=year, minimum_idle_hours=hours))
            pairs = match_observing_conditions(lengthy)
            if pairs.empty:
                continue
            pairs["difference"] = pairs.plume_label_mid.astype(int) - pairs.plume_label_near.astype(int)
            blocks = pairs.groupby("location").difference.agg(["sum", "size"]).to_numpy()
            rng = np.random.default_rng(168720)
            draw = blocks[rng.integers(0, len(blocks), size=(4000, len(blocks)))].sum(axis=1)
            low, high = np.quantile(draw[:, 0] / draw[:, 1], [.025, .975])
            weather.append({
                "year": year, "minimum_idle_hours": hours, "pairs": len(pairs), "plants": len(blocks),
                "near_fraction": pairs.plume_label_near.mean(), "mid_fraction": pairs.plume_label_mid.mean(),
                "difference": pairs.difference.mean(), "low": low, "high": high,
            })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(weather)


def annual_stack_comparison(frame, ground_dir):
    years = []
    for year, overpasses in frame.groupby("year"):
        path = ground_dir / f"{year}.parquet"
        ids = overpasses.location.unique().tolist()
        recorded = pd.read_parquet(path, columns=["facility_id", "mass_kg"], filters=[("facility_id", "in", ids)])
        calendar = recorded.groupby("facility_id").mass_kg.agg(["size", "mean"]).rename(
            columns={"size": "stack_hours", "mean": "all_hours_stack_kg_h"}
        ).rename_axis("location")
        calendar_hours = (pd.Timestamp(f"{year + 1}-01-01") - pd.Timestamp(f"{year}-01-01")).days * 24
        calendar["stack_hour_coverage"] = calendar.stack_hours / calendar_hours
        observed = overpasses.groupby("location").agg(
            overpasses=("true_rate", "size"), positives=("plume_label", "sum"),
            all_overpass_stack_kg_h=("true_rate", "mean"), detected_mass_rate_sum=("true_rate_detected", "sum"),
        )
        annual = calendar.join(observed, how="inner")
        annual["positive_only_stack_kg_h"] = annual.detected_mass_rate_sum / annual.positives.replace(0, np.nan)
        annual["zero_filled_stack_kg_h"] = annual.detected_mass_rate_sum / annual.overpasses
        years.append(annual.assign(year=year).reset_index())
    table = pd.concat(years, ignore_index=True)
    methods = ["all_overpass_stack", "positive_only_stack", "zero_filled_stack"]
    true = "all_hours_stack_kg_h"
    trends = []
    years = sorted(table.year.unique())
    for early, late in zip(years[:-1], years[1:]):
        if late - early != 1:
            continue
        frame_years = table.loc[table.year.isin([early, late])].set_index(["location", "year"]).unstack("year")
        availability = frame_years.stack_hour_coverage.min(axis=1, skipna=False)
        change = frame_years[true][late] - frame_years[true][early]
        for coverage in [0.90, 0.95, 0.99]:
            for minimum_change in [0, 25, 50, 100]:
                included = availability.ge(coverage) & change.abs().gt(minimum_change)
                for method in methods:
                    estimate = frame_years[f"{method}_kg_h"][late] - frame_years[f"{method}_kg_h"][early]
                    scored = included & estimate.notna()
                    err = estimate[scored] - change[scored]
                    trends.append({
                        "from": early, "to": late, "approach": method, "coverage_threshold": coverage,
                        "minimum_change_kg_h": minimum_change, "included_plants": int(included.sum()),
                        "estimable_plants": int(scored.sum()), "direction_reversals": int((estimate[scored] * change[scored]).lt(0).sum()),
                        "direction_ties": int(estimate[scored].eq(0).sum()), "difference_mae_kg_h": err.abs().mean(),
                    })
    return table, pd.DataFrame(trends)


def annual_common_support(table):
    rows = []
    methods = ["all_overpass_stack_kg_h", "positive_only_stack_kg_h", "zero_filled_stack_kg_h"]
    calendar_years = sorted(table.year.unique())
    for first, second in zip(calendar_years[:-1], calendar_years[1:]):
        if second - first != 1:
            continue
        years = table.loc[table.year.isin([first, first + 1])].set_index(["location", "year"]).unstack("year")
        coverage = years.stack_hour_coverage.min(axis=1, skipna=False)
        reference = years.all_hours_stack_kg_h[first + 1] - years.all_hours_stack_kg_h[first]
        has_two = years.positive_only_stack_kg_h.notna().all(axis=1)
        for size in (25, 50):
            qualified = coverage.ge(.95) & reference.abs().gt(size)
            common = qualified & has_two
            for method in methods:
                delta = years[method][first + 1] - years[method][first]
                rows.append({
                    "from": first, "to": first + 1, "approach": method, "min_change_kg_h": size,
                    "plants_with_verifiable_change": int(qualified.sum()), "common_plants": int(common.sum()),
                    "reversals": int((delta[common] * reference[common]).lt(0).sum()),
                    "ties": int(delta[common].eq(0).sum()), "mae_kg_h": (delta[common] - reference[common]).abs().mean(),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-dir", type=Path)
    parser.add_argument("--years", nargs="+", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.frame)
    if args.years:
        frame = frame.loc[frame.year.isin(args.years)].copy()
    values = totals(frame)
    values.to_csv(args.output / "catalog_sampling.csv", index=False)
    print(values.loc[values.grouping.eq("year")].drop(columns=["view_band", "ground_state"]).round(3).to_string(index=False), flush=True)
    comparisons = []
    years = set(frame.year)
    earlier = min(years)
    for year in sorted(years):
        if (year == earlier) or (year - 1 not in years):
            continue
        plant = plant_changes(frame, year-1, year)
        plant.to_csv(args.output / f"plant_changes_{year-1}_{year}.csv")
        comparisons.append(change_summary(plant, year-1, year))
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(args.output / "plant_change_summary.csv", index=False)
    if args.ground_dir:
        off, result = sustained_shutdowns(frame, args.ground_dir)
        result.to_csv(args.output / "sustained_stop_rates.csv", index=False)
        angle, weather = matched_shutdown_angles(off)
        angle.to_csv(args.output / "sustained_stop_matched_angles.csv", index=False)
        weather.to_csv(args.output / "sustained_stop_weather_pairs.csv", index=False)
        annual, trends = annual_stack_comparison(frame, args.ground_dir)
        annual.to_csv(args.output / "annual_stack_comparison.csv", index=False)
        if not trends.empty:
            trends.to_csv(args.output / "annual_stack_trends.csv", index=False)
            annual_common_support(annual).to_csv(args.output / "annual_stack_common_support.csv", index=False)
        off[["location", "Facility_Name", "State", "year", "utc_time", "hour", "stop_age_hours", "elapsed_stop_hours",
             "continuous_stop_age", "plume_label", "view_band", "sensor_zenith_angle",
             "wind_u", "wind_v", "wind_m_s", "file_path", "latitude", "longitude"]].to_parquet(
                 args.output / "continuous_stops.parquet", index=False,
             )


if __name__ == "__main__":
    main()
