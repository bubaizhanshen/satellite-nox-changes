"""Compare the annual sampling error of alternative plume screens on common plants."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("gaussian", "cross_section")
RULES = ("all_signed", "archive_detected_only", "reconstructed_area_detected_only",
         "template_detected_only")
YEARS = ("2023", "2024")
COLUMNS = ("location", "State", "scale", "period", "flux_method", "policy",
           "estimated_rate_kg_h", "reference_selected_kg_h", "reference_all_hours_kg_h",
           "stack_coverage")


def measure_sampling_error(periods):
    year = periods.period.astype(str)
    eligible = periods.loc[
        periods.scale.eq("year") & year.isin(YEARS) & periods.flux_method.isin(METHODS)
        & periods.policy.isin(RULES) & periods.stack_coverage.ge(.95)
        & periods.estimated_rate_kg_h.notna()
    ].copy()
    eligible["period"] = eligible.period.astype(str)
    if not eligible.groupby(["location", "period"]).reference_all_hours_kg_h.nunique().eq(1).all():
        raise ValueError("Every policy must use the same full-year reference")
    index = ["location", "flux_method", "policy"]
    selected_minus_year = eligible.reference_selected_kg_h - eligible.reference_all_hours_kg_h
    end_points = eligible.assign(error=selected_minus_year).pivot(
        index=index, columns="period", values="error"
    ).reindex(columns=YEARS)
    errors = end_points["2024"].sub(end_points["2023"]).abs().unstack(index[1:])
    combinations = pd.MultiIndex.from_product([METHODS, RULES], names=index[1:])
    errors = errors.reindex(columns=combinations).dropna()
    if errors.empty:
        raise ValueError("No plants have estimates for both years and all screen rules")
    errors = errors.T.groupby(level="policy").mean().T.reindex(columns=RULES)
    states = eligible[["location", "State"]].drop_duplicates().set_index("location")
    return errors.join(states, validate="one_to_one")


def summarize(errors):
    groups = errors.groupby("State")
    totals = groups[list(RULES)].sum().to_numpy()
    counts = groups.size().to_numpy()
    draws = np.random.default_rng(942300).integers(len(counts), size=(10_000, len(counts)))
    boot_mae = totals[draws].sum(axis=1) / counts[draws].sum(axis=1, keepdims=True)
    ci = np.quantile(boot_mae - boot_mae[:, [0]], (.025, .975), axis=0)
    means = errors[list(RULES)].mean()
    return pd.DataFrame({
        "rule": RULES, "plants": len(errors), "states": len(counts),
        "mae_kg_h": means.to_numpy(),
        "increase_kg_h": means.sub(means.iloc[0]).to_numpy(),
        "low_kg_h": ci[0], "high_kg_h": ci[1],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("period_estimates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(measure_sampling_error(pd.read_csv(args.period_estimates, usecols=COLUMNS)))
    summary.to_csv(args.output, index=False)
