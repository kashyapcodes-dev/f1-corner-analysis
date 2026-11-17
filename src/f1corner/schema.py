"""Canonical tables. Every source backend must emit exactly these columns.

Everything downstream (lap filters, corners, clustering) only sees these, so the
pipeline is source-agnostic and FastF1 is never a hard requirement.

Conventions (identical for every backend):
- driver = 3-letter code (VER), team = team name as given by the source.
- lap_start_s, weather.t_s, race_control.t_s = UTC epoch seconds.
- car_data.t_s, location.t_s = seconds since the start of that lap.
- brake = bool; throttle in percent; speed in km/h.
"""

from __future__ import annotations

import pandas as pd

SESSION_KEYS = ["season", "event", "session"]

SESSIONS_COLS = ["season", "event", "session", "round", "date", "circuit"]

LAPS_COLS = [
    *SESSION_KEYS,
    "driver",
    "team",
    "lap_number",
    "lap_start_s",
    "lap_time_s",
    "s1_s",
    "s2_s",
    "s3_s",
    "pit_in",
    "pit_out",
    "compound",
    "tyre_age",
    "stint",
    "deleted",  # bool | NA: stewards deleted lap
    "green",  # bool | NA: green flag whole lap (no yellow/SC/VSC)
    "source_accurate",  # bool | NA: FastF1 IsAccurate if available; cross-check only
]

CAR_DATA_COLS = [
    *SESSION_KEYS,
    "driver",
    "lap_number",
    "t_s",
    "speed_kph",
    "throttle",
    "brake",
    "gear",
    "rpm",
]

LOCATION_COLS = [*SESSION_KEYS, "driver", "lap_number", "t_s", "x", "y", "z"]

WEATHER_COLS = [*SESSION_KEYS, "t_s", "air_temp_c", "track_temp_c", "rain"]

RACE_CONTROL_COLS = [*SESSION_KEYS, "t_s", "category", "message", "driver"]

TABLES = {
    "sessions": SESSIONS_COLS,
    "laps": LAPS_COLS,
    "car_data": CAR_DATA_COLS,
    "location": LOCATION_COLS,
    "weather": WEATHER_COLS,
    "race_control": RACE_CONTROL_COLS,
}


def conform(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns as NA, drop extras, fix order."""
    cols = TABLES[table]
    out = df.copy()
    for c in cols:
        if c not in out:
            out[c] = pd.NA
    return out[cols]
