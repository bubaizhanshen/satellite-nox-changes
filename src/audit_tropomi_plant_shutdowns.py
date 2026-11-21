"""Audit published TROPOMI source detections against independent shutdown reports."""

import pandas as pd

def add_release_history(ground):
    keys = ["facility_id", "utc"]
    ground = ground[keys + ["mass_kg", "stopped"]].copy()
    history = ground.copy()
    for lag in range(1, 8):
        shifted = history.copy()
        shifted["utc"] += pd.Timedelta(hours=lag)
        shifted = shifted.rename(columns={"mass_kg": f"mass_lag{lag}", "stopped": f"stop_lag{lag}"})
        if lag > 1:
            shifted = shifted.drop(columns=f"mass_lag{lag}")
        ground = ground.merge(shifted, on=keys, how="left", validate="one_to_one")
    for hours in (4, 8):
        states = ["stopped", *(f"stop_lag{lag}" for lag in range(1, hours))]
        ground[f"past_{hours}h_stopped"] = ground[states].eq(True).all(axis=1)
    ground["mean_mass_2h"] = (ground.mass_kg + ground.mass_lag1) / 2
    return ground
