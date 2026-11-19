"""Per-corner team performance, pace/profile split and track mix (spec 12-13).

Long-table conventions:
  times   : event, team, driver, corner, time_s   (time through the corner segment)
  classes : event, corner, cls                    (corner class names from `cluster`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .corners import distance_axis


def segment_times(car: pd.DataFrame, corners: pd.DataFrame, length: float | None = None):
    """Time (s) through each corner segment of one lap (index aligned to `corners`)."""
    c, d = distance_axis(car, length)
    t = c["t_s"].to_numpy(float)
    a = np.interp(corners["start_m"].to_numpy(float), d, t)
    b = np.interp(corners["end_m"].to_numpy(float), d, t)
    return pd.Series(b - a, index=corners.index, name="time_s")


def delta_pct(times: pd.DataFrame) -> pd.DataFrame:
    """Spec 12: best teammate time per (team, corner, event), % over the fastest car there.

    Returns event, team, corner, time_s, delta_pct.
    """
    keys = ["event", "corner"]
    best = times.groupby([*keys, "team"], as_index=False)["time_s"].min()
    best["delta_pct"] = 100 * (best["time_s"] / best.groupby(keys)["time_s"].transform("min") - 1)
    return best


def pace_profile(delta: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Spec 13: per (event, team): pace = median delta over corners; profile[cls] = mean(delta-pace).

    Negative profile = stronger than own average in that corner class.
    Returns event, team, pace, cls, profile.
    """
    d = delta.merge(classes, on=["event", "corner"], how="inner")
    d["pace"] = d.groupby(["event", "team"])["delta_pct"].transform("median")
    d["rel"] = d["delta_pct"] - d["pace"]
    out = d.groupby(["event", "team", "cls"], as_index=False).agg(
        pace=("pace", "first"), profile=("rel", "mean")
    )
    return out[["event", "team", "pace", "cls", "profile"]]


def track_mix(pole_times: pd.DataFrame, classes: pd.DataFrame, lap_s: pd.Series) -> pd.DataFrame:
    """Share of the pole lap's time spent in each class's corners, per event.

    `pole_times`: event, corner, time_s (pole lap only); `lap_s`: event -> pole lap time (s).
    Returns event, cls, share (shares sum to < 1; the rest is straights).
    """
    d = pole_times.merge(classes, on=["event", "corner"], how="inner")
    d = d.groupby(["event", "cls"], as_index=False)["time_s"].sum()
    d["share"] = d["time_s"] / d["event"].map(lap_s)
    return d[["event", "cls", "share"]]
