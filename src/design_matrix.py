"""
Builds the regression matrix for the tyre degradation model from a FastF1 laps DataFrame.

The model we're fitting is:

    lap_time = base_pace[compound] + deg_rate[compound] * tyre_life
               + fuel_effect * lap_number + driver_offset[driver]

Each compound gets its own base pace and its own degradation rate.
Fuel burns the same no matter which tyre you're on, so there's one shared fuel term.
Each driver gets an offset relative to a reference driver (the first alphabetically),
so differences between cars don't leak into the tyre numbers.
"""

import warnings

import numpy as np
import pandas as pd

# The order we want compounds to appear in, so columns always come out the same way.
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def build_design_matrix(laps: pd.DataFrame):
    """
    Turn FastF1's laps DataFrame into X (the design matrix) and y (lap times in seconds).

    Returns:
        X            - numpy array, one row per clean lap
        y            - numpy array of lap times in seconds
        column_names - list of what each column of X means
        clean_laps   - the filtered DataFrame, so you can trace any row back to a real lap
    """

    # Work on a copy so we never accidentally change the original DataFrame.
    laps = laps.copy()

    # ------------------------------------------------------------------
    # Step 1: Throw away laps that would mislead the model
    # ------------------------------------------------------------------

    # IsAccurate is FastF1's own quality check. It's False for in-laps, out-laps,
    # and laps where the timing didn't sync properly. Pit laps disappear here.
    keep = laps["IsAccurate"] == True

    # TrackStatus is a string of codes. "1" on its own means the whole lap was green.
    # Anything else (yellow flags, safety car, VSC, red flag) gets dropped.
    keep &= laps["TrackStatus"] == "1"

    # Laps deleted for track limits aren't representative pace, so drop them.
    # Deleted can be missing (NaN) for some sessions, so treat missing as "not deleted".
    if "Deleted" in laps.columns:
        keep &= laps["Deleted"].fillna(False) != True

    # We can't use a lap if we don't know its time or how old the tyre was.
    keep &= laps["LapTime"].notna()
    keep &= laps["TyreLife"].notna()

    # Only dry compounds for now. Intermediates and wets behave completely differently.
    keep &= laps["Compound"].isin(DRY_COMPOUNDS)

    clean_laps = laps[keep].reset_index(drop=True)

    if clean_laps.empty:
        raise ValueError("No clean laps left after filtering. Was this a wet or very disrupted race?")

    # ------------------------------------------------------------------
    # Step 2: Work out which compounds were actually used in this race
    # ------------------------------------------------------------------

    # Most races only use two of the three dry compounds. If a compound never appears,
    # its columns would be all zeros, which would wreck the rank. So only include
    # compounds that really show up, kept in the SOFT -> MEDIUM -> HARD order.
    compounds_used = [c for c in DRY_COMPOUNDS if c in clean_laps["Compound"].unique()]

    # ------------------------------------------------------------------
    # Step 3: Build the columns, in the same order as our hand-made matrix
    # ------------------------------------------------------------------

    columns = []
    column_names = []

    # First the "is this compound?" columns. Each one is 1 on that compound's laps, 0 otherwise.
    # Its coefficient becomes that compound's base pace.
    for compound in compounds_used:
        is_compound = (clean_laps["Compound"] == compound).astype(float)
        columns.append(is_compound)
        column_names.append(f"is_{compound.lower()}")

    # Then the tyre life columns. Each one holds the tyre age on that compound's laps, and 0 elsewhere.
    # Its coefficient becomes that compound's degradation rate (seconds lost per lap of tyre age).
    for compound in compounds_used:
        is_compound = (clean_laps["Compound"] == compound).astype(float)
        tyre_life_on_this_compound = clean_laps["TyreLife"].astype(float) * is_compound
        columns.append(tyre_life_on_this_compound)
        column_names.append(f"{compound.lower()}_tyre_life")

    # Driver columns: one per driver, except the first (the reference driver).
    # Leaving one out stops the driver columns adding up to the same thing as
    # the compound columns, which would drop the rank by 1.
    # Each coefficient is that driver's gap to the reference driver, in seconds per lap.
    drivers = sorted(clean_laps["Driver"].unique())
    for driver in drivers[1:]:  # [1:] skips the reference driver
        columns.append((clean_laps["Driver"] == driver).astype(float))
        column_names.append(f"is_{driver.lower()}")

    # Finally the shared lap number column. Its coefficient is the fuel effect:
    # how much faster the car gets each lap as it burns fuel (should come out negative).
    columns.append(clean_laps["LapNumber"].astype(float))
    column_names.append("lap_number")

    # Stack the columns side by side into one matrix.
    X = np.column_stack(columns)

    # y is the thing we're trying to predict: lap time, converted to plain seconds.
    y = clean_laps["LapTime"].dt.total_seconds().to_numpy()

    # ------------------------------------------------------------------
    # Step 4: Check the model can actually separate tyres from fuel
    # ------------------------------------------------------------------

    # This is the rank check we worked through by hand. If the rank is less than the number
    # of columns, some column is a combination of others, so the coefficients can't be trusted.
    # This happens if, for example, you only pass in one driver's laps.
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        warnings.warn(
            f"Design matrix has rank {rank} but {X.shape[1]} columns. "
            "Tyre wear, fuel and driver pace can't be separated. Try including more drivers "
            "who pitted on different laps."
        )

    return X, y, column_names, clean_laps


# ----------------------------------------------------------------------
# Quick test with a made-up race, so you can check it works without downloading data
# ----------------------------------------------------------------------

def _make_fake_race():
    """Three drivers with different pace, with lap times generated from known numbers."""

    # The "true" answers we hope the model recovers.
    base = {"MEDIUM": 90.0, "HARD": 90.5}
    deg = {"MEDIUM": 0.10, "HARD": 0.05}
    fuel = -0.06

    # (driver, list of (compound, number of laps)) for each driver's stints
    strategies = [
        ("AAA", [("MEDIUM", 3), ("HARD", 3)]),   # pits after lap 3
        ("BBB", [("MEDIUM", 4), ("HARD", 2)]),   # pits after lap 4
        ("CCC", [("MEDIUM", 2), ("HARD", 4)]),   # pits after lap 2
    ]

    # Give each driver a random pace offset between -1s and +1s.
    # The seed (0) makes the "random" numbers the same every run, so the test is repeatable.
    rng = np.random.default_rng(0)
    driver_pace = {driver: rng.uniform(-1, 1) for driver, _ in strategies}

    rows = []
    for driver, stints in strategies:
        lap_number = 1
        for compound, n_laps in stints:
            for tyre_life in range(1, n_laps + 1):
                seconds = (base[compound] + deg[compound] * tyre_life
                           + fuel * lap_number + driver_pace[driver])
                rows.append({
                    "Driver": driver,
                    "LapNumber": lap_number,
                    "Compound": compound,
                    "TyreLife": float(tyre_life),
                    "LapTime": pd.Timedelta(seconds=seconds),
                    "IsAccurate": True,
                    "TrackStatus": "1",
                    "Deleted": False,
                })
                lap_number += 1
    return pd.DataFrame(rows), driver_pace


if __name__ == "__main__":
    fake_laps, true_pace = _make_fake_race()
    X, y, names, used = build_design_matrix(fake_laps)

    print("Columns:", names)
    print(X)
    print("Rank:", np.linalg.matrix_rank(X), "of", X.shape[1])

    # Fit the model and see if we get the true numbers back.
    coefficients, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    for name, value in zip(names, coefficients):
        print(f"{name:>18}: {value:.3f}")

    # Driver coefficients are gaps to the reference driver (the first alphabetically),
    # so compare them with the true gaps.
    reference = sorted(true_pace)[0]
    print(f"\nTrue gaps to {reference}:")
    for driver in sorted(true_pace)[1:]:
        print(f"{'is_' + driver.lower():>18}: {true_pace[driver] - true_pace[reference]:.3f}")