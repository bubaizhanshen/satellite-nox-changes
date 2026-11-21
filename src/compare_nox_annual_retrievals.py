#!/usr/bin/env python3
"""Compare satellite screening and annual flux decisions with independent stack records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_nox_annual import YEARS


METHODS = {
    "gaussian": "gaussian",
    "cross_section": "cross_section",
    "cross_section_2h": "cross_section",
    "cross_section_6h": "cross_section",
}
PRIMARY_METHODS = ("gaussian", "cross_section")
TARGETS = {"coincident_cems": "reference_common_kg_h", "all_hours_cems": "reference_all_hours_kg_h"}
PAIR_POLICIES = (
    ("all_signed", "archive_detected_only"),
    ("all_signed", "archive_persistence"),
    ("all_signed", "archive_nondetect_zero"),
    ("template_detected_only", "archive_detected_only"),
    ("template_nondetect_zero", "archive_nondetect_zero"),
    ("reconstructed_area_detected_only", "archive_detected_only"),
    ("all_clipped", "all_signed"),
)
GATES = {"archive": "plume_label", "template": "template_positive", "reconstructed_area": "area_positive"}


def read_inputs(args):
    visits = pd.read_parquet(args.output / "visits.parquet")
    panel = pd.read_csv(args.output / "panel.csv").set_index("location")
    directory = "scored" if args.provider == "rsig" else f"scored_{args.provider}"
    block_paths = sorted((args.work / directory).glob("*.parquet"))
    if not block_paths:
        raise FileNotFoundError("The satellite retrieval has no scored blocks")
    pieces = [pd.read_parquet(path) for path in block_paths]
    scores = pd.concat(pieces, ignore_index=True)
    if scores.visit_id.duplicated().any():
        raise ValueError("More than one satellite estimate per scheduled visit")
    counts = visits.groupby("block_id").size()
    present = visits.loc[visits.visit_id.isin(scores.visit_id)].groupby("block_id").size()
    if not present.eq(counts.reindex(present.index)).all():
        raise ValueError("A finished block omitted scheduled visits")
    data = visits.merge(scores, on="visit_id", how="left", validate="one_to_one")
    data["score_attempted"] = data.visit_id.isin(scores.visit_id)
    if "origin" in data:
        data["score_attempted"] &= data.origin.ne("pending")
    data["native_available"] = data.native_pixels.notna()
    rule = json.loads((args.output / "frozen_screen.json").read_text())
    assert rule["year"] < min(YEARS)
    data["template_positive"] = data.template_score.ge(rule["template_cutoff"])
    data["area_positive"] = data.flagged_area_km2.ge(25.)
    data["screen_available"] = data.screen_error.eq("") & data.template_score.notna()
    if args.exact_granules:
        manifest = pd.read_csv(args.output / "native_orbit_manifest.csv")
        exact = manifest.temis_file.str.replace("_reduced.nc", ".nc", regex=False).eq(manifest.archive_file)
        available_names = manifest.loc[exact, "archive_file"]
        data["screen_available"] &= data.file_path.str.rsplit("/", n=1).str[-1].isin(available_names)
    return data, panel


def independent_hours(ground, panel):
    rows = []
    for year in YEARS:
        table = pd.read_parquet(ground / f"{year}.parquet", columns=["facility_id", "utc", "mass_kg"],
                                filters=[("facility_id", "in", panel.index.tolist())])
        if table.duplicated(["facility_id", "utc"]).any():
            raise ValueError("Duplicate independent stack hours")
        table["month"] = table.utc.dt.strftime("%Y-%m")
        for scale, keys in (("month", ["facility_id", "month"]), ("year", "facility_id")):
            result = table.groupby(keys).mass_kg.agg(["mean", "size"]).reset_index()
            result = result.rename(columns={"facility_id": "location", "mean": "reference_all_hours_kg_h",
                                            "size": "usable_stack_hours"})
            if scale == "year":
                result["period"] = str(year)
                expected = (pd.Timestamp(f"{year+1}-01-01") - pd.Timestamp(f"{year}-01-01")).days * 24
            else:
                result = result.rename(columns={"month": "period"})
                expected = pd.PeriodIndex(result.period, freq="M").days_in_month * 24
            result["stack_coverage"] = result.usable_stack_hours / expected
            rows.append(result.assign(scale=scale))
    return pd.concat(rows, ignore_index=True)


def support_table(data):
    records = []
    for (location, year, month), group in data.groupby(["location", "year", "month"]):
        reference = group[[f"{m}_cems_kg_h" for m in PRIMARY_METHODS]].notna().all(axis=1)
        quality = group[[f"{m}_kg_h" for m in PRIMARY_METHODS]].notna().all(axis=1)
        common = group.screen_available & reference & quality
        counts = {
            "location": location, "year": year, "month": month, "expected": len(group),
            "attempted": int(group.score_attempted.sum()), "native_available": int(group.native_available.sum()),
            "usable_screens": int(group.screen_available.sum()), "common_flux": int(common.sum()),
            "archive_positive_on_common": int((common & group.plume_label).sum()),
            "template_positive_on_common": int((common & group.template_positive).sum()),
            "median_fit_wall_s": group.fit_wall_seconds.median(),
            "matched_native_median_lag_s": group.loc[group.native_available, "time_offset_s"].median(),
        }
        records.append(counts)
    return pd.DataFrame(records)


def historical_persistence(path, held_out):
    archive = pd.read_parquet(path, columns=["location", "plume_label"], filters=[("year", "=", 2022)])
    training = archive.loc[~archive.location.isin(held_out)].groupby("location").plume_label.agg(["sum", "size"])
    training = training.loc[training["sum"].gt(0) & training["size"].ge(10)].assign(
        detected_fraction=lambda rows: rows["sum"] / rows["size"],
    )
    histogram, edges = np.histogram(training.detected_fraction, bins=np.linspace(0, 1, 21))
    points = (np.arange(1000) + .5) / 1000
    log_prior = np.log((histogram + .5)[np.searchsorted(edges, points, side="right") - 1])
    return (points, log_prior), training


def mean_persistence(detected, available, prior):
    prob, log_prior = prior
    log_mass = log_prior + detected * np.log(prob) + (available - detected) * np.log1p(-prob)
    return np.average(prob, weights=np.exp(log_mass - np.max(log_mass)))


def summaries_for_period(group, support, flux, reference, prior):
    matching = group.loc[support].copy()
    signed = matching[flux]
    stack = matching[reference]
    rows = []
    common = stack.mean()
    scheduled = group.mean_mass_2h.mean()
    native = len(matching)
    # The first policy clips once after aggregating. The alternative clips at
    # every overpass, which changes the mean when retrieval noise is symmetric.
    rows.append(("all_signed", max(signed.mean(), 0), common, native))
    rows.append(("all_clipped", signed.clip(lower=0).mean(), common, native))
    for name, column in GATES.items():
        gate = matching[column]
        selected = int(gate.sum())
        rows.append((f"{name}_detected_only", signed[gate].clip(lower=0).mean(), stack[gate].mean(), selected))
        rows.append((f"{name}_nondetect_zero", signed.clip(lower=0).where(gate, 0.).mean(),
                     stack.where(gate, 0.).mean(), selected))
        if name == "archive":
            persistence = mean_persistence(selected, native, prior)
            rows.append(("archive_persistence", persistence * signed[gate].clip(lower=0).mean(),
                         persistence * stack[gate].mean(), selected))
    return [
        {
            "policy": policy, "estimated_rate_kg_h": rate, "reference_selected_kg_h": truth,
            "reference_common_kg_h": common, "reference_scheduled_kg_h": scheduled,
            "reference_common_2h_kg_h": matching.mean_mass_2h.mean(),
            "valid_overpasses": native, "detected_overpasses": detections,
            "valid_months": int(matching.month.nunique()),
            "scheduled_overpasses": len(group), "attempted_overpasses": int(group.score_attempted.sum()),
        }
        for policy, rate, truth, detections in rows
    ]


def period_estimates(data, panel, independent, prior):
    out = []
    scopes = [("month", "month", 2, 1), ("year", "year", 36, 9)]
    core_fields = [f"{method}_{end}" for method in PRIMARY_METHODS for end in ("kg_h", "cems_kg_h")]
    for scale, key, n_required, minimum_months in scopes:
        for (location, period), group in data.groupby(["location", key]):
            support = group.screen_available & group[core_fields].notna().all(axis=1)
            for method, reference_method in METHODS.items():
                if scale == "month" and method not in PRIMARY_METHODS:
                    continue
                flux, truth = f"{method}_kg_h", f"{reference_method}_cems_kg_h"
                valid = support & group[flux].notna()
                rows = summaries_for_period(group, valid, flux, truth, prior)
                for row in rows:
                    enough = row["valid_overpasses"] >= n_required and row["valid_months"] >= minimum_months
                    out.append({
                        "location": location, "scale": scale, "period": str(period), "flux_method": method,
                        "sufficient_common_support": enough, **row,
                    })
    entries = pd.DataFrame(out).merge(independent, on=["location", "scale", "period"], validate="many_to_one")
    entries.loc[~entries.sufficient_common_support, "estimated_rate_kg_h"] = np.nan
    entries.loc[entries.stack_coverage.lt(.90), "reference_all_hours_kg_h"] = np.nan
    entries = entries.join(panel[["stratum", "design_weight", "Facility_Name", "State", "priority_panel"]], on="location")
    entries["within_report_error_kg_h"] = entries.estimated_rate_kg_h - entries.reference_selected_kg_h
    entries["screening_error_kg_h"] = entries.reference_selected_kg_h - entries.reference_common_kg_h
    entries["transport_window_error_kg_h"] = entries.reference_common_kg_h - entries.reference_common_2h_kg_h
    entries["observation_support_error_kg_h"] = entries.reference_common_2h_kg_h - entries.reference_scheduled_kg_h
    entries["overpass_sampling_error_kg_h"] = entries.reference_scheduled_kg_h - entries.reference_all_hours_kg_h
    total = entries.estimated_rate_kg_h - entries.reference_all_hours_kg_h
    terms = entries[["within_report_error_kg_h", "screening_error_kg_h",
                     "transport_window_error_kg_h", "observation_support_error_kg_h",
                     "overpass_sampling_error_kg_h"]].sum(axis=1, min_count=5)
    assert np.allclose(total.to_numpy(), terms.to_numpy(), equal_nan=True)
    return entries


def calendar_standardized(entries):
    """
    Average the calendar months present in both years for the original screen
    and for the signed workflow. Use those exact months for every policy.
    """
    values = ["estimated_rate_kg_h", *TARGETS.values()]
    monthly = entries.loc[entries.scale.eq("month") & entries.estimated_rate_kg_h.notna()].copy()
    monthly["year"] = monthly.period.str[:4]
    monthly["calendar_month"] = monthly.period.str[-2:]
    paired = monthly.set_index(["location", "flux_method", "policy", "calendar_month", "year"])
    paired = paired[values].unstack("year").dropna(how="any")
    anchored = paired.reset_index().loc[lambda frame: frame.policy.isin(["archive_detected_only", "all_signed"])]
    common_months = (anchored.groupby(["location", "flux_method", "calendar_month"], dropna=False)
                     .size().loc[lambda count: count.eq(2)])
    by_plant = {(plant, model): support.index.get_level_values("calendar_month")
                for (plant, model), support in common_months.groupby(level=[0, 1])}
    records = []
    for (location, method, policy), matched in paired.groupby(level=[0, 1, 2]):
        months = by_plant.get((location, method), [])
        matched = matched.loc[matched.index.get_level_values("calendar_month").isin(months)]
        if len(months) < 9 or len(matched) != len(months):
            continue
        masses = {f"{name}_{year}": matched[(name, str(year))].mean()
                  for name in values for year in YEARS}
        records.append({"location": location, "flux_method": method, "policy": policy,
                        "same_calendar_months": len(matched), **masses})
    return pd.DataFrame(records)


def annual_trends(entries, panel, calendar):
    cols = ["estimated_rate_kg_h", *TARGETS.values(), "valid_overpasses", "valid_months", "detected_overpasses"]
    cols += ["attempted_overpasses", "scheduled_overpasses"]
    annual = entries.loc[entries.scale.eq("year")].set_index(["location", "flux_method", "policy", "period"])
    wide = annual[cols].unstack("period")
    wide.columns = ["_".join(key) for key in wide.columns]
    unadjusted = wide.reset_index().assign(scope="all_valid_overpasses")
    alternate = calendar.assign(scope="common_calendar_months")
    thorough = unadjusted.loc[
        unadjusted.valid_months_2023.eq(12) & unadjusted.valid_months_2024.eq(12)
        & unadjusted.attempted_overpasses_2023.ge(.9 * unadjusted.scheduled_overpasses_2023)
        & unadjusted.attempted_overpasses_2024.ge(.9 * unadjusted.scheduled_overpasses_2024)
        & (~(unadjusted.policy.str.endswith("_detected_only") | unadjusted.policy.eq("archive_persistence"))
           | (unadjusted.detected_overpasses_2023.ge(5) & unadjusted.detected_overpasses_2024.ge(5)))
    ].assign(scope="high_coverage")
    rows = pd.concat([unadjusted, alternate, thorough], ignore_index=True)
    for item in ("estimated_rate_kg_h", *TARGETS.values()):
        rows[f"{item}_change"] = rows[f"{item}_2024"] - rows[f"{item}_2023"]
    return rows.join(panel[["State", "Facility_Name", "stratum", "design_weight", "priority_panel",
                            "change_kg_h"]], on="location")


def mean(x, weights):
    known = np.isfinite(x)
    return float(np.average(x[known], weights=weights[known])) if known.any() else np.nan


def differences_by_site(frame, value, draws=2000):
    if frame.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(60933)
    g = (frame.assign(w=frame.design_weight * frame[value])
         .groupby(["stratum", "location"])[["w", "design_weight"]].sum())
    arr = g.to_numpy()
    b = np.zeros((draws, 2))
    for _, supported in g.groupby(level="stratum"):
        sampled = supported.to_numpy()
        b += sampled[rng.integers(0, len(sampled), size=(draws, len(sampled)))].sum(axis=1)
    low, high = np.quantile(b[:, 0] / b[:, 1], [.025, .975])
    return arr[:, 0].sum() / arr[:, 1].sum(), float(low), float(high)


def uncertainty_by_state(frame, value, draws=2000):
    grouped = (frame.assign(w=frame.design_weight * frame[value])
               .groupby("State")[["w", "design_weight"]].sum().to_numpy())
    rng = np.random.default_rng(12385)
    selected = grouped[rng.integers(0, len(grouped), size=(draws, len(grouped)))].sum(axis=1)
    return np.quantile(selected[:, 0] / selected[:, 1], [.025, .975])


def compare_table(frame, rate, reference_cols, keys, panel_size):
    summaries, changes = [], []
    for group_name, same in frame.groupby(keys, dropna=False):
        base = dict(zip(keys, group_name))
        setting = base["scope"]
        for reference_name, truth in reference_cols.items():
            for policy, part in same.groupby("policy"):
                valid = part.loc[part[rate].notna() & part[truth].notna()].copy()
                error = valid[rate] - valid[truth]
                weights = valid.design_weight.to_numpy()
                summary = {
                    **base, "reference": reference_name, "policy": policy, "panel_plants": panel_size,
                    "valid_plants": valid.location.nunique(), "valid_records": len(valid),
                    "bias_kg_h": mean(error.to_numpy(), weights),
                    "mae_kg_h": mean(abs(error.to_numpy()), weights),
                }
                if setting != "level":
                    summary["no_change_mae_kg_h"] = mean(abs(valid[truth].to_numpy()), weights)
                    for minimum in (25, 50, 100):
                        assessed = valid.loc[valid[truth].abs().gt(minimum)]
                        suffix = "" if minimum == 25 else f"_gt_{minimum}"
                        summary |= {
                            f"real_change_gt_{minimum}": len(assessed),
                            f"mae_on_real_changes_gt_{minimum}_kg_h": mean(
                                abs(assessed[rate] - assessed[truth]).to_numpy(), assessed.design_weight.to_numpy()),
                            f"no_change_mae_on_real_changes_gt_{minimum}_kg_h": mean(
                                abs(assessed[truth]).to_numpy(), assessed.design_weight.to_numpy()),
                            f"direction_reversed{suffix}": int((assessed[rate] * assessed[truth]).lt(0).sum()),
                            f"direction_tied{suffix}": int(assessed[rate].eq(0).sum()),
                            f"false_stable_changes_gt_{minimum}": int((valid[rate].abs().gt(minimum) & valid[truth].abs().le(minimum)).sum()),
                            f"false_large_decline_gt_{minimum}": int((valid[rate].lt(-minimum) & valid[truth].ge(-minimum)).sum()),
                            f"missed_large_decline_gt_{minimum}": int((valid[rate].ge(-minimum) & valid[truth].lt(-minimum)).sum()),
                        }
                summaries.append(summary)

            for candidate, baseline in PAIR_POLICIES:
                a = same.loc[same.policy.eq(candidate)]
                b = same.loc[same.policy.eq(baseline)]
                dims = ["location", "period"] if "period" in same else ["location"]
                pair = a[[*dims, truth, rate, "design_weight", "stratum", "State"]].merge(
                    b[[*dims, truth, rate]], on=dims, suffixes=("_a", "_b"), validate="one_to_one",
                ).dropna()
                if pair.empty:
                    continue
                if not np.allclose(pair[f"{truth}_a"], pair[f"{truth}_b"]):
                    raise ValueError("A paired comparison changed the reference")
                err_a = abs(pair[f"{rate}_a"] - pair[f"{truth}_a"])
                err_b = abs(pair[f"{rate}_b"] - pair[f"{truth}_b"])
                pair["change"] = err_a - err_b
                value, lo, hi = differences_by_site(pair, "change")
                state_low, state_high = uncertainty_by_state(pair, "change")
                comparison = {
                    **base, "reference": reference_name, "policy": candidate, "baseline": baseline,
                    "paired_records": len(pair), "paired_plants": pair.location.nunique(),
                    "baseline_mae_kg_h": mean(err_b.to_numpy(), pair.design_weight.to_numpy()),
                    "policy_mae_kg_h": mean(err_a.to_numpy(), pair.design_weight.to_numpy()),
                    "closer_to_reference": int((err_a < err_b).sum()),
                    "mae_change_kg_h": value, "low": lo, "high": hi,
                    "state_low": state_low, "state_high": state_high,
                }
                if setting != "level":
                    valid = pair.loc[pair[f"{truth}_a"].abs().gt(25)]
                    comparison.update({
                        "paired_changes_gt_25": len(valid),
                        "baseline_direction_reversed_gt_25": int((valid[f"{rate}_b"] * valid[f"{truth}_a"]).lt(0).sum()),
                        "policy_direction_reversed_gt_25": int((valid[f"{rate}_a"] * valid[f"{truth}_a"]).lt(0).sum()),
                    })
                changes.append(comparison)
    return pd.DataFrame(summaries), pd.DataFrame(changes)


def matched_fleet_totals(trend):
    """Sum each panel member's annual estimates over identical reporting plants."""
    hours = {year: (pd.Timestamp(f"{year+1}-01-01") - pd.Timestamp(f"{year}-01-01")).days * 24
             for year in YEARS}
    raw = trend.loc[trend.scope.eq("all_valid_overpasses") & trend.flux_method.isin(PRIMARY_METHODS)]
    output = []
    for method, group in raw.groupby("flux_method"):
        widths = [f"estimated_rate_kg_h_{year}" for year in YEARS]
        actual = [f"reference_all_hours_kg_h_{year}" for year in YEARS]
        for candidate, baseline in PAIR_POLICIES:
            a = group.loc[group.policy.eq(candidate)]
            b = group.loc[group.policy.eq(baseline)]
            rows = a[["location", *widths, *actual]].merge(b[["location", *widths]],
                     on="location", suffixes=("_a", "_b"), validate="one_to_one").dropna()
            if rows.empty:
                continue
            estimates = {policy: [rows[f"{key}_{suffix}"].sum() * hours[year]/1000
                                  for year, key in zip(YEARS, widths)]
                         for policy, suffix in ((candidate, "a"), (baseline, "b"))}
            estimates["cems"] = [rows[key].sum() * hours[year]/1000 for year, key in zip(YEARS, actual)]
            for policy, values in estimates.items():
                output.append({"flux_method": method, "candidate": candidate, "baseline": baseline,
                               "policy": policy, "common_plants": len(rows),
                               "nox_2023_tonnes": values[0], "nox_2024_tonnes": values[1],
                               "change_tonnes": values[1] - values[0]})
    return pd.DataFrame(output)


def screen_effects(data, panel, stop_history):
    history = pd.read_parquet(stop_history, columns=["location", "utc_time", "elapsed_stop_hours"])
    screen = (data.loc[data.screen_available].merge(history, how="left", on=["location", "utc_time"], validate="one_to_one")
              .join(panel[["design_weight", "stratum"]], on="location"))
    groups = {
        "stopped_ge_8h": screen.ground_state.eq("stopped_8h"),
        "stopped_ge_30d": screen.elapsed_stop_hours.ge(720),
        "emitting_ge_250": screen.mean_mass_2h.ge(250),
    }
    records = []
    for state, relevant in groups.items():
        full = screen.loc[relevant]
        for year, yearly in full.groupby("year"):
            for band, cases in [("all", yearly), *list(yearly.groupby("view_band", observed=True))]:
                for rule, field in GATES.items():
                    selected = cases[field].to_numpy(dtype=bool)
                    base = cases.plume_label.to_numpy(dtype=bool)
                    weights = cases.design_weight.to_numpy()
                    descriptor = {"year": year, "state": state, "view_band": band, "screen": rule,
                                  "usable_scenes": len(cases), "plants": cases.location.nunique(),
                                  "positive_scenes": int(selected.sum()), "positive_share": mean(selected, weights)}
                    paired = cases.assign(change=selected.astype(float) - base.astype(float))
                    delta, lo, hi = differences_by_site(paired, "change")
                    descriptor.update({"positive_share_delta_archive": delta, "low": lo, "high": hi})
                    for method in PRIMARY_METHODS:
                        values = cases[f"{method}_kg_h"].to_numpy()
                        result = np.where(np.isfinite(values), np.where(selected, np.maximum(values, 0), 0.), np.nan)
                        descriptor[f"{method}_reported_per_usable_scene_kg_h"] = mean(result, weights)
                        descriptor[f"{method}_scenes_with_flux"] = int(np.isfinite(values).sum())
                    records.append(descriptor)
    return pd.DataFrame(records)


def annual_direction_intervals(data, trend, prior, draws=2000, block_days=1):
    """Repeat the year's sampling by resampling whole overpass days within month."""
    policies = {
        "all_signed": None,
        "archive_detected_only": "plume_label",
        "archive_persistence": "plume_label",
        "template_detected_only": "template_positive",
    }
    reference = "reference_all_hours_kg_h_change"
    estimated = "estimated_rate_kg_h_change"
    eligible = trend.loc[trend.scope.eq("all_valid_overpasses") & trend.flux_method.isin(PRIMARY_METHODS)
                         & trend.policy.isin(policies) & trend[estimated].notna() & trend[reference].notna()]
    core_fields = [f"{m}_{end}" for m in PRIMARY_METHODS for end in ("kg_h", "cems_kg_h")]
    usable = data.loc[data.screen_available & data[core_fields].notna().all(axis=1)].copy()
    usable["time_block"] = (usable.utc_time.dt.day - 1) // block_days
    groups = {loc: rows for loc, rows in usable.groupby("location")}
    mean_by_support = {}
    intervals = []
    for row in eligible.itertuples(index=False):
        rng = np.random.default_rng([5494363, int(row.location)])
        row_fields = row._asdict()
        observations = groups[row.location]
        gate = policies[row.policy]
        column = f"{row.flux_method}_kg_h"
        rates = []
        for year in YEARS:
            totals = np.zeros((draws, 3))
            for _, month in observations.loc[observations.year.eq(year)].groupby("month"):
                values = month[column] if gate is None else month[column].clip(lower=0).where(month[gate])
                daily = month.assign(flux=values).groupby("time_block").flux.agg(["sum", "count", "size"])
                sampled = rng.integers(0, len(daily), size=(draws, len(daily)))
                totals += daily.to_numpy()[sampled].sum(axis=1)
            average = np.divide(totals[:, 0], totals[:, 1], out=np.full(draws, np.nan), where=totals[:, 1] > 0).clip(0)
            if row.policy == "archive_persistence":
                for i, (count, available) in enumerate(totals[:, 1:].astype(int)):
                    key = (count, available)
                    if key not in mean_by_support:
                        mean_by_support[key] = mean_persistence(count, available, prior)
                    average[i] *= mean_by_support[key]
            rates.append(average)
        sample_changes = rates[1] - rates[0]
        finite = np.isfinite(sample_changes)
        lo, hi = np.quantile(sample_changes[finite], [.025, .975]) if finite.mean() >= .95 else (np.nan, np.nan)
        intervals.append({"location": row.location, "flux_method": row.flux_method, "policy": row.policy,
                          estimated: row_fields[estimated], reference: row_fields[reference],
                          "valid_bootstrap_fraction": finite.mean(), "lower_kg_h": lo, "upper_kg_h": hi})
    output = pd.DataFrame(intervals)
    summaries = []
    if not output.empty:
        for (method, policy), group in output.groupby(["flux_method", "policy"]):
            truth = group[reference]
            for cut in (25, 50, 100):
                decision = np.select([group.lower_kg_h.gt(cut), group.upper_kg_h.lt(-cut)], [1, -1], default=0)
                actual = np.select([truth.gt(cut), truth.lt(-cut)], [1, -1], default=0)
                summaries.append({"flux_method": method, "policy": policy, "change_threshold_kg_h": cut,
                                  "panel": data.location.nunique(), "plants_with_estimate": len(group),
                                  "decisive_signs": int((decision != 0).sum()),
                                  "decisive_correct": int(((decision != 0) & (actual == decision)).sum()),
                                  "decisive_wrong": int(((decision != 0) & (actual != decision)).sum()),
                                  "actual_large_change_no_decision": int(((actual != 0) & (decision == 0)).sum())})
    return output, pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground", type=Path, default=Path("inputs/ground"))
    parser.add_argument("--provider", choices=("rsig", "temis", "hybrid"), default="rsig")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--stop-history", type=Path, default=Path("inputs/archive/full_archive/continuous_stops.parquet"))
    parser.add_argument("--persistence-source", type=Path, default=Path("inputs/archive/full_archive/population.parquet"))
    parser.add_argument("--exact-granules", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    data, panel = read_inputs(args)
    support = support_table(data)
    if args.preview:
        scored_fraction = support.assign(attempted_block=support.attempted.gt(0)).groupby("location")["attempted_block"].sum()
        enough = scored_fraction.index[scored_fraction.ge(19)]
        if len(enough) == 0:
            print("Preview has no facilities with 19 downloaded months yet.", flush=True)
            return
        panel = panel.loc[panel.index.isin(enough)]
        data = data.loc[data.location.isin(enough)]
    output = args.results or args.output
    if args.preview:
        output = output / "preview"
    output.mkdir(exist_ok=True, parents=True)
    independent = independent_hours(args.ground, panel)
    all_panel = pd.read_csv(args.output / "panel.csv").location
    prior, training = historical_persistence(args.persistence_source, all_panel)
    training.to_csv(output / "persistence_prior_2022.csv")
    estimates = period_estimates(data, panel, independent, prior)
    calendar = calendar_standardized(estimates)
    trend = annual_trends(estimates, panel, calendar)
    levels = estimates.assign(scope="level")
    baseline, level_pair = compare_table(levels, "estimated_rate_kg_h", TARGETS, ["scope", "scale", "flux_method"], len(panel))
    trends, trend_pair = compare_table(
        trend, "estimated_rate_kg_h_change", {k: v + "_change" for k, v in TARGETS.items()},
        ["scope", "flux_method"], len(panel),
    )
    intervals, decisions = annual_direction_intervals(data, trend, prior)
    for filename, result in {
        "coverage.csv": support, "plant_period_estimates.csv": estimates, "plant_annual_changes.csv": trend,
        "error_summary.csv": pd.concat([baseline, trends], ignore_index=True),
        "paired_differences.csv": pd.concat([level_pair, trend_pair], ignore_index=True),
        "matched_fleet_totals.csv": matched_fleet_totals(trend),
        "screen_decisions.csv": screen_effects(data, panel, args.stop_history),
        "annual_direction_sampling_intervals.csv": intervals, "annual_direction_decisions.csv": decisions,
    }.items():
        result.to_csv(output / filename, index=False)
    data.to_parquet(output / "scored_visits.parquet", index=False)
    print(f"{len(panel)} plants | {int(data.native_available.sum())}/{len(data)} native overpasses", flush=True)
    print(trends.loc[trends.scope.eq("all_valid_overpasses")
                     & trends.reference.eq("all_hours_cems") & trends.flux_method.isin(PRIMARY_METHODS),
                     ["flux_method", "policy", "valid_plants", "mae_kg_h", "direction_reversed"]]
          .round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
