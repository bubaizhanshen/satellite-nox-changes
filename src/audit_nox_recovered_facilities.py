#!/usr/bin/env python3
"""Assess which recovered annual trends are limited by sampling or flux retrieval."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_nox_annual_retrievals import PRIMARY_METHODS, YEARS
from evaluate_nox_reporting_bias import LIFETIME_HOURS
from score_no2_overpass_benchmark import NOX_TO_NO2


OUTCOMES = (
    "gaussian",
    "cross_section",
    "cross_section_local_chemistry",
    "cems_at_gaussian_overpasses",
    "cems_at_cross_section_overpasses",
)
REFERENCE = "reference_all_hours_kg_h_change"
RATE = "estimated_rate_kg_h_change"
HOURS = dict(zip(YEARS, (8760, 8784)))


def reweight_chemistry(visits):
    """Use Kuhlmann et al. Eq. 4 on the original two cross-section windows."""
    speed = np.hypot(visits.wind_u, visits.wind_v).to_numpy()
    age = np.array([10., 20.]) / (3.6 * speed[:, None])
    shift_to_two = np.exp(age * (1 / 2 - 1 / LIFETIME_HOURS))
    combined = 2 * visits.cross_section_kg_h.to_numpy()
    shifted = 2 * visits.cross_section_2h_kg_h.to_numpy()
    first = (shifted - combined * shift_to_two[:, 1]) / (shift_to_two[:, 0] - shift_to_two[:, 1])
    sections = np.column_stack((first, combined - first))
    known = np.mean(sections * np.exp(age * (1 / 6 - 1 / LIFETIME_HOURS)), axis=1)
    np.testing.assert_allclose(known, visits.cross_section_6h_kg_h.to_numpy(), rtol=1e-9, atol=1e-9)
    # NOx:NO2 ratio as a function of transport age and a 2.5 h lifetime.
    conversion = 1.6 * np.exp(-age * 3600 / 1638) + 1.31
    return np.mean(sections * conversion / NOX_TO_NO2 * np.exp(age * (1 / 2.5 - 1 / LIFETIME_HOURS)), axis=1)


def sampled_annual_estimates(group, facility, draws=2000):
    random = np.random.default_rng([5494363, int(facility)])
    original, samples = [], []
    for year in YEARS:
        visits = group.loc[group.year.eq(year)]
        original.append(visits[list(OUTCOMES)].mean().to_numpy().clip(0))
        totals = np.zeros((draws, len(OUTCOMES) + 1))
        for _, month in visits.groupby("month"):
            blocks = month.groupby("time_block")[list(OUTCOMES)].sum()
            blocks["count"] = month.groupby("time_block").size()
            positions = random.integers(0, len(blocks), size=(draws, len(blocks)))
            totals += blocks.to_numpy()[positions].sum(axis=1)
        samples.append((totals[:, :-1] / totals[:, -1:]).clip(0))
    ci = np.quantile(samples[1] - samples[0], [.025, .975], axis=0)
    return pd.DataFrame({"rate_2023_kg_h": original[0], "rate_2024_kg_h": original[1],
                         RATE: original[1] - original[0], "lower_kg_h": ci[0], "upper_kg_h": ci[1]},
                        index=pd.Index(OUTCOMES, name="outcome")).reset_index()


def measure(annual):
    summaries = []
    for (scope, method), group in annual.groupby(["scope", "outcome"]):
        ref = group[REFERENCE].to_numpy()
        changes = group[RATE].to_numpy()
        for threshold in (25., 50., 100.):
            actual = np.select([ref < -threshold, ref > threshold], [-1, 1], default=0)
            inferred = np.select([changes < -threshold, changes > threshold], [-1, 1], default=0)
            clear = np.select([group.upper_kg_h < -threshold, group.lower_kg_h > threshold], [-1, 1], default=0)
            material = actual != 0
            delta = (group.rate_2024_kg_h * HOURS[2024] - group.rate_2023_kg_h * HOURS[2023]).sum() / 1000
            ref_total = (group.cems_2024 * HOURS[2024] - group.cems_2023 * HOURS[2023]).sum() / 1000
            summaries.append({"scope": scope, "outcome": method, "minimum_kg_h": threshold,
                              "facilities": len(group), "cems_material_changes": material.sum(),
                              "cems_material_decreases": (actual < 0).sum(),
                              "point_correct_direction": (material & (np.sign(changes) == actual)).sum(),
                              "point_correct_material_change": (material & (inferred == actual)).sum(),
                              "interval_correct_material_change": (material & (clear == actual)).sum(),
                              "interval_correct_material_decrease": ((actual < 0) & (clear == actual)).sum(),
                              "interval_false_decisions": ((clear != 0) & (clear != actual)).sum(),
                              "average_absolute_error_kg_h": abs(changes - ref).mean(),
                              "cohort_change_tonnes": delta, "cems_cohort_change_tonnes": ref_total})
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--intervals", type=Path)
    parser.add_argument("--block-days", type=int, default=1, choices=(1, 7))
    args = parser.parse_args()
    trends = pd.read_csv(args.results / "plant_annual_changes.csv")
    published = pd.read_csv(args.intervals or args.results / "annual_direction_sampling_intervals.csv")
    panel = pd.read_csv(args.results / "panel.csv")
    keep = trends.scope.eq("all_valid_overpasses") & trends.flux_method.eq("cross_section")
    available = trends.loc[keep & trends.policy.eq("all_signed") & trends[RATE].notna(), "location"]
    panel = panel.loc[panel.location.isin(available)].copy()
    old = trends.loc[keep & trends.policy.eq("archive_detected_only")].set_index("location")[RATE]
    panel["scope"] = np.where(panel.location.map(old).isna(), "recovered", "previously_reportable")
    scene = pd.read_parquet(args.results / "scored_visits.parquet")
    fields = [f"{method}_{suffix}" for method in PRIMARY_METHODS for suffix in ("kg_h", "cems_kg_h")]
    common = scene.loc[scene.screen_available & scene[fields].notna().all(axis=1)
                       & scene.location.isin(available)].copy()
    common["cross_section_local_chemistry"] = reweight_chemistry(common)
    common = common.rename(columns={"gaussian_kg_h": "gaussian", "cross_section_kg_h": "cross_section",
                                    "gaussian_cems_kg_h": "cems_at_gaussian_overpasses",
                                    "cross_section_cems_kg_h": "cems_at_cross_section_overpasses"})
    common["time_block"] = (common.utc_time.dt.day - 1) // args.block_days
    computed = pd.concat([sampled_annual_estimates(group, location).assign(location=location)
                          for location, group in common.groupby("location")], ignore_index=True)
    computed = computed.merge(panel[["location", "Facility_Name", "State", "scope", "cems_2023", "cems_2024"]],
                              on="location", validate="many_to_one")
    computed[REFERENCE] = computed.cems_2024 - computed.cems_2023
    unchanged = published.loc[published.policy.eq("all_signed")].rename(columns={"flux_method": "outcome"})
    verification = computed.merge(unchanged[["location", "outcome", RATE, "lower_kg_h", "upper_kg_h", REFERENCE]],
                                  on=["location", "outcome"], suffixes=("", "_archived"), validate="one_to_one")
    assert len(verification) == 2 * len(panel)
    for key in (RATE, "lower_kg_h", "upper_kg_h", REFERENCE):
        np.testing.assert_allclose(verification[key], verification[f"{key}_archived"], rtol=1e-8, atol=1e-8)
    coincident = {"gaussian": "cems_at_gaussian_overpasses", "cross_section": "cems_at_cross_section_overpasses"}
    earlier = trends.loc[trends.scope.eq("all_valid_overpasses") & trends.policy.eq("all_signed")
                         & trends.flux_method.isin(PRIMARY_METHODS)].copy()
    earlier["outcome"] = earlier.flux_method.map(coincident)
    check = computed.merge(earlier[["location", "outcome", "reference_common_kg_h_change"]],
                           on=["location", "outcome"], validate="one_to_one")
    assert len(check) == 2 * len(panel)
    np.testing.assert_allclose(check[RATE], check.reference_common_kg_h_change, rtol=1e-8, atol=1e-8)
    summary = measure(computed)
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    computed.to_csv(output / "recovery_diagnostic_sites.csv", index=False)
    summary.to_csv(output / "recovery_diagnostic_summary.csv", index=False)
    print(summary.loc[summary.minimum_kg_h.eq(25)].round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
