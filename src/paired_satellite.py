"""Calibrate and apply the satellite correction for paired calendar months."""

import numpy as np
import pandas as pd

from estimate_annual_changes import (
    CHANGE_THRESHOLD, FUELS, KEYS, complete_fuel, features, fit_factors, measured_levels,
    training_rows, weighted_levels,
)
from paired_change import corrected_contrast, sat_slope
from satellite_fuel_baselines import MATCHED_FIELDS as STACK, RATE_FIELDS as RATES


VISITS = ["visit_id", "location", "State", "year", "utc_time"]


def usable(frame):
    return frame.loc[frame.screen_available.eq(True)].dropna(subset=RATES).assign(month=lambda x: x.utc_time.dt.month)


def read_visits(path, match_stacks):
    image_fields = VISITS + ["screen_available", *RATES]
    columns = image_fields + STACK if match_stacks else image_fields
    return usable(pd.read_parquet(path, columns=columns).reindex(columns=image_fields + STACK))


def read_blocks(schedule, scores):
    planned = pd.read_parquet(schedule, columns=VISITS)
    paths = sorted(scores.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No native-scored blocks in {scores}")
    matched = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    matched["screen_available"] = matched.screen_error.eq("") & matched.template_score.notna()
    columns = ["visit_id", "screen_available", *RATES, *STACK]
    table = planned.merge(matched.reindex(columns=columns), on="visit_id", validate="one_to_one")
    return usable(table)


def coverage(flights, years):
    observations = flights.groupby(["location", "year"]).month.agg(["nunique", "size"])
    usable_year = observations["nunique"].ge(9) & observations["size"].ge(36)
    complete = usable_year.unstack().reindex(columns=years).eq(True).all(axis=1)
    return complete.index[complete]


def calibrate_response(visits):
    return sat_slope(visits.assign(satellite=visits[RATES].mean(axis=1),
                                  matched_stack=visits[STACK].mean(axis=1, skipna=False)))


def residual_variance(residual):
    values = residual.dropna().to_numpy()
    limit = np.quantile(np.abs(values), .98)
    return float(np.mean(np.clip(values, -limit, limit)**2))


def summarize_months(visits):
    entries = visits.assign(rate=visits[RATES].mean(axis=1))
    return entries.groupby(KEYS).rate.agg(signal="mean", variance="var", n="size")


def apply_satellite(monthly_rates, full, parameters, years, model="fuel", fuel_rate="plant"):
    prior = full.loc[full.model.eq(model)].copy()
    prior["fuel_rate"] = prior[fuel_rate]
    prior["heat_total"] = prior.heat_mmbtu / (1000 * prior.hours)
    paired = monthly_rates.join(prior.set_index(KEYS)[["fuel_rate", "heat_total"]], how="inner")
    result = corrected_contrast(paired, prior, "fuel_rate", "signal", "variance", "n", *parameters, years=years)
    targeted = result.combined_change.where(result.fuel_change.abs().gt(CHANGE_THRESHOLD), result.fuel_change)
    return result.combined_change, targeted


class SatelliteCorrection:
    def __init__(self, visits, extra, earlier, fuel, years, same_year):
        self.years = years
        self.same_year = same_year
        visits = visits.loc[visits.year.isin(years)]
        if extra:
            added = read_blocks(*extra)
            target = pd.concat([visits, added.loc[added.year.isin(years)]], ignore_index=True, sort=False)
            if not target.visit_id.is_unique:
                raise ValueError("Evaluation schedule contains duplicate visit IDs")
        else:
            target = visits
        self.images = target.loc[target.location.isin(coverage(target, years))]
        common = visits.dropna(subset=STACK)
        self.paired = common.loc[common.location.isin(coverage(common, years))]
        basis = self.paired if same_year else self.images
        self.sites = coverage(basis, years).intersection(complete_fuel(fuel, years))
        self.image_months = summarize_months(self.images)
        self.stack_matched_months = summarize_months(self.paired)
        self.fuel = fuel.loc[fuel.location.isin(self.sites)]
        self.earlier = read_blocks(*earlier).dropna(subset=STACK) if earlier else None
        if self.earlier is not None and not self.earlier.year.eq(years[0] - 1).all():
            raise ValueError("Earlier satellite calibration must precede the two evaluation years")
        annual = basis.groupby(["location", "year"])[RATES].mean().clip(lower=0).mean(axis=1).unstack("year")
        annual = annual.reindex(columns=years)
        self.satellite_only = annual[years[1]] - annual[years[0]]

    def current_parameters(self, heldout, earlier, first_year):
        year = self.years[0]
        data = self.paired.loc[self.paired.location.isin(self.sites) & self.paired.year.eq(year)]
        data = data.loc[~data.State.isin(heldout)]
        training_sites = data.location.unique()
        train = training_rows(self.fuel, first_year, training_sites, year)
        factors = fit_factors(train, FUELS, analytic_jacobian=False)
        fuel = self.fuel.loc[self.fuel.location.isin(training_sites) & self.fuel.year.between(year - 1, year)].copy()
        fuel["rate"] = features(fuel, FUELS, separate_conversion=True) @ factors
        predicted = weighted_levels(fuel, "rate", "hours").unstack("year")
        stacks = pd.concat([earlier.loc[earlier.year.eq(year - 1)], first_year], ignore_index=True)
        stacks = stacks.merge(fuel[KEYS], on=KEYS, validate="one_to_one")
        observed = measured_levels(stacks)
        error = (observed[year] - observed[year - 1]) - (predicted[year] - predicted[year - 1])
        response, floor = calibrate_response(data)
        return response, floor, residual_variance(error)

    def predict(self, test, monthly, earlier_stack, first_year, earlier_calibration):
        result = self.satellite_only.reindex(test.index).rename("satellite_only").to_frame()
        info = {}
        heldout = set(test.State)
        if self.same_year:
            current = self.current_parameters(heldout, earlier_stack, first_year)
            result["fuel_plant_satellite_ungated"], result["fuel_plant_satellite"] = apply_satellite(
                self.stack_matched_months, monthly, current, self.years,
            )
            info.update(slope_first_year=current[0], floor_first_year=current[1], variance_first_year=current[2])
        if self.earlier is not None:
            history = self.earlier.loc[~self.earlier.State.isin(heldout)]
            counts = history.groupby("location").month.agg(["size", "nunique"])
            enough = counts.index[counts["size"].ge(4) & counts["nunique"].ge(3)]
            if len(enough) < 10:
                raise ValueError("Fewer than ten earlier calibration sites meet the visit and month criteria")
            history = history.loc[history.location.isin(enough)]
            response, floor = calibrate_response(history)
            residual = earlier_calibration.reference - earlier_calibration.fuel_plant
            parameters = (response, floor, max(residual_variance(residual), 1.0))
            result["fuel_plant_satellite_earlier_ungated"], result["fuel_plant_satellite_earlier_image_only"] = apply_satellite(
                self.image_months, monthly, parameters, self.years,
            )
            if self.same_year:
                open_change, selected_change = apply_satellite(self.stack_matched_months, monthly, parameters, self.years)
                result["fuel_plant_satellite_earlier_stack_matched_ungated"] = open_change
                result["fuel_plant_satellite_earlier_stack_matched"] = selected_change
                _, result["fuel_plant_satellite_first_year_image_only"] = apply_satellite(
                    self.image_months, monthly, current, self.years,
                )
            for model, rate in (("fuel", "seasonal"), ("technology", "plant"), ("technology", "seasonal")):
                name = f"{model}_{rate}"
                error = earlier_calibration.reference - earlier_calibration[name]
                variance = max(residual_variance(error), 1.0)
                _, result[f"{name}_satellite_earlier_image_only"] = apply_satellite(
                    self.image_months, monthly, (response, floor, variance), self.years, model, rate,
                )
            info.update(slope_earlier=response, floor_earlier=floor, variance_earlier=parameters[2],
                        sites_earlier=len(enough), visits_earlier=len(history))
        return result, info
