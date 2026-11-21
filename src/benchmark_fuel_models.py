"""Compare annual ML regressions and monthly boosting with the fuel-factor model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from paired_change import full_stack_months, train_prior
from satellite_fuel_baselines import CHANGE_THRESHOLD, HEAT_FIELDS, HEAT_FEATURES, activity_data, hours_weighted, physical_features
from validate_decisions import adequate_ground

POINT_MODELS = [
    "Fuel factors",
    "Ridge annual regression",
    "Histogram boosting annual regression",
    "XGBoost annual regression",
    "Two-hidden-layer MLP",
    "Histogram boosting monthly rates",
    "LightGBM monthly rates",
    "Boosted fuel factors",
    "TabICLv2 annual regression",
]


def annual_features(activities, year):
    inputs = activities.copy()
    names = [f"heat_{i}" for i in range(6)]
    inputs[names] = physical_features(inputs)
    values = pd.DataFrame({name: hours_weighted(inputs, name, "hours") for name in names}).unstack("year")
    earlier = values.xs(year, level=1, axis=1)
    later = values.xs(year + 1, level=1, axis=1)
    features = (later - earlier).add_prefix("change_")
    levels = np.log1p((later + earlier) / 2.).add_prefix("level_")
    return features.join(levels, how="inner").dropna()


def monthly_rates(activity, train, test, year, coefficients):
    from lightgbm import LGBMRegressor

    x = [*HEAT_FIELDS, "month"]
    rows = activity.loc[activity.location.isin(test) & activity.year.isin([year, year + 1])].copy()
    labels = train.reference_all_hours_kg_h
    weights = train.hours / train.hours.mean()
    models = {
        "Histogram boosting monthly rates": HistGradientBoostingRegressor(
            loss="poisson", max_iter=125, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=2., early_stopping=False, random_state=61285,
        ),
        "LightGBM monthly rates": LGBMRegressor(
            objective="poisson", n_estimators=125, num_leaves=15, min_child_samples=20,
            learning_rate=.05, reg_lambda=2., n_jobs=4, verbosity=-1, random_state=61285,
        ),
    }
    for name, model in models.items():
        rows[name] = model.fit(train[x], labels, sample_weight=weights).predict(rows[x])

    # Learn a small correction to the emission factor while preserving zero emissions at zero fuel.
    positive = train.heat_total.gt(0.)
    baseline = physical_features(train) @ coefficients
    ratio = np.log((labels + 1.) / (baseline + 1.))
    intensity_fields = [*HEAT_FEATURES, "month"]
    correction = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=100, max_leaf_nodes=7, min_samples_leaf=35,
        learning_rate=.05, l2_regularization=5., early_stopping=False, random_state=61285,
    )
    correction.fit(train.loc[positive, intensity_fields], ratio.loc[positive], sample_weight=weights.loc[positive])
    factor = np.exp(correction.predict(rows[intensity_fields]))
    rows["Boosted fuel factors"] = (physical_features(rows) @ coefficients) * factor
    annual = pd.DataFrame({name: hours_weighted(rows, name, "hours")
                           for name in [*models, "Boosted fuel factors"]}).unstack("year")
    return annual.xs(year + 1, level=1, axis=1) - annual.xs(year, level=1, axis=1)


def annual_models(checkpoint):
    if checkpoint is not None:
        from tabicl import TabICLRegressor

        return {"TabICLv2 annual regression": TabICLRegressor(
            model_path=checkpoint, allow_auto_download=False, device="cpu", n_jobs=4,
            n_estimators=8, random_state=61285,
        )}

    from xgboost import XGBRegressor

    return {
        "Ridge annual regression": make_pipeline(StandardScaler(), Ridge(alpha=5.)),
        "Histogram boosting annual regression": HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=125, max_leaf_nodes=9, min_samples_leaf=10,
            l2_regularization=2, early_stopping=False, random_state=61285,
        ),
        "XGBoost annual regression": XGBRegressor(
            objective="reg:absoluteerror", n_estimators=180, max_depth=3, min_child_weight=10,
            learning_rate=.04, subsample=.85, colsample_bytree=.85, reg_lambda=4.,
            tree_method="hist", n_jobs=4, random_state=61285,
        ),
        "Two-hidden-layer MLP": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(24, 12), activation="tanh", solver="lbfgs", alpha=2.,
                max_iter=700, random_state=61285,
            )),
            transformer=StandardScaler(),
        ),
    }


def evaluate(folds, old, future, stack, old_states, train_year, checkpoint):
    prior_features = annual_features(old, 2022)
    target_features = annual_features(future, train_year)
    prior_truth = adequate_ground(stack, 2022)
    by_fold = []
    for fold, tests in folds.groupby("fold"):
        states = old_states.loc[old_states.notna() & ~old_states.isin(tests.State)]
        assert set(states).isdisjoint(tests.State)
        labels = prior_truth.reindex(states.index).rename("response")
        train = prior_features.join(labels, how="inner").dropna()
        test = target_features.reindex(tests.index)
        if test.isna().any().any():
            raise ValueError("A test plant is missing fuel features")
        x, y = train[prior_features.columns], train.response
        outputs = tests.copy()
        for name, model in annual_models(checkpoint).items():
            outputs[name] = model.fit(x, y).predict(test)
        if checkpoint is None:
            train_months = old.loc[old.location.isin(states.index) & old.year.eq(2023)].merge(
                stack, on=["location", "year", "month"], validate="one_to_one")
            train_months = train_months.loc[train_months.stack_hours.ge(.95 * train_months.hours)]
            coefficients = train_prior(old, stack, states.index, 2023)
            outputs = outputs.join(monthly_rates(future, train_months, tests.index, train_year, coefficients))
        print(train_year, "fold", fold, "training contrasts", len(train), "test plants", len(test), flush=True)
        by_fold.append(outputs)
    return pd.concat(by_fold, verify_integrity=True).sort_index()


def summaries(rows, year):
    outcome = rows.reference
    large = outcome.abs().gt(CHANGE_THRESHOLD)
    declines = outcome.lt(-CHANGE_THRESHOLD)
    result = []
    for method in (name for name in POINT_MODELS if name in rows):
        pred = rows[method]
        error = pred.sub(outcome).abs()
        reported = pred.lt(-CHANGE_THRESHOLD)
        differences = (error - rows["Fuel factors"].sub(outcome).abs()).groupby(rows.State)
        blocks = differences.agg(["sum", "size"]).to_numpy()
        ids = np.random.default_rng(61285).integers(0, len(blocks), size=(10000, len(blocks)))
        sampled = blocks[ids].sum(axis=1)
        low, high = np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975])
        result.append(dict(year_pair=year, method=method, n=len(rows), mae=error.mean(),
                           mae_large=error[large].mean(), mae_declines=error[declines].mean(),
                           correct_declines=int((reported & declines).sum()),
                           false_declines=int((reported & ~declines).sum()),
                           mae_difference_low=low, mae_difference_high=high))
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-panel", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--historical-activity", type=Path, required=True)
    parser.add_argument("--external-activity", type=Path, required=True)
    parser.add_argument("--ground", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint", type=Path,
                        help="Evaluate TabICLv2 using the supplied public regression checkpoint.")
    args = parser.parse_args()
    panel = pd.read_csv(args.old_panel).set_index("location")
    primary = pd.read_csv(args.primary).set_index("location").rename(columns={"cems": "reference", "fuel": "Fuel factors"})
    primary["State"] = panel.State
    external = pd.read_csv(args.external).set_index("location").rename(columns={"fuel":"Fuel factors"})
    cols = ["fold", "State", "reference", "Fuel factors"]
    base = activity_data(args.historical_activity)
    old_states = panel.State.reindex(base.location.unique())
    stack = full_stack_months(args.ground, old_states.index, (2022, 2023))
    with threadpool_limits(limits=4):
        a = evaluate(primary[cols], base, base, stack, old_states, 2023, args.foundation_checkpoint)
        b = evaluate(external[cols], base, activity_data(args.external_activity), stack, old_states, 2024,
                     args.foundation_checkpoint)
    score = pd.concat([summaries(a, "2023–2024"), summaries(b, "2024–2025")], ignore_index=True)
    prefix = "fuel_foundation" if args.foundation_checkpoint is not None else "fuel_model"
    score.to_csv(f"{prefix}_benchmarks.csv", index=False)
    a.to_csv(f"{prefix}_primary_predictions.csv")
    b.to_csv(f"{prefix}_external_predictions.csv")
    print(score.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
