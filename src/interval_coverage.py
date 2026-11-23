"""Compare date-resampling intervals with the full-year stack reference."""

import argparse
from pathlib import Path

import pandas as pd


METHODS = ("gaussian", "cross_section", "divergence", "matched_stack")
LARGE_CHANGE = 25.0


def coverage(estimates, intervals):
    selected = estimates.loc[estimates.policy.eq("all") & estimates.method.isin(METHODS)]
    limits = intervals.loc[intervals.method.isin(METHODS)]
    paired = selected.merge(limits, on=["location", "method"], how="outer",
                            validate="one_to_one", indicator=True)
    if (not paired._merge.eq("both").all()
            or paired[["reference", "lower", "upper"]].isna().any().any()
            or (paired.lower > paired.upper).any()):
        raise ValueError("Each common-overpass estimate needs a valid pair of interval limits")
    if (not paired.groupby("location").size().eq(len(METHODS)).all()
            or not paired.groupby("location").reference.nunique().eq(1).all()):
        raise ValueError("The methods must have identical plants and annual references")

    paired["covered"] = paired.reference.between(paired.lower, paired.upper)
    paired["width"] = paired.upper - paired.lower
    large = paired.reference.abs().gt(LARGE_CHANGE)
    groups = {"all": paired, "small": paired.loc[~large], "large": paired.loc[large]}
    summaries = []
    for name, group in groups.items():
        summary = group.groupby("method").agg(
            plants=("covered", "size"), covered=("covered", "sum"),
            median_width_kg_h=("width", "median"),
        ).reindex(METHODS)
        summary["coverage_pct"] = 100 * summary.covered / summary.plants
        summary["group"] = name
        summaries.append(summary.reset_index())
    return pd.concat(summaries, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("estimates", "intervals", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = coverage(pd.read_csv(args.estimates), pd.read_csv(args.intervals))
    result.to_csv(args.output, index=False)
