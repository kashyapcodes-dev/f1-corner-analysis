"""Per-corner features from the pole lap (raw telemetry, not the smoothed profile).

Features (one row per corner):
  min_speed_kph    apex speed, parabola-interpolated around the lowest raw sample
  speed_loss_kph   entry speed (at segment start) minus min_speed_kph
  time_off_wot_s   time from lift/brake start to full throttle again (segment duration)
  min_gear         lowest gear selected inside the segment
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .corners import distance_axis

FEATURES = ["min_speed_kph", "speed_loss_kph", "time_off_wot_s", "min_gear"]


def _parabola_min(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Vertex value of the parabola through samples i-1, i, i+1 (falls back to y[i])."""
    if i <= 0 or i >= len(y) - 1:
        return float(y[i])
    x0, x1, x2 = x[i - 1 : i + 2]
    y0, y1, y2 = y[i - 1 : i + 2]
    if not (x0 < x1 < x2):
        return float(y[i])
    h0, h1 = x1 - x0, x2 - x1
    a = ((y2 - y1) / h1 - (y1 - y0) / h0) / (h0 + h1)
    if a <= 0:  # not a convex minimum
        return float(y[i])
    # slope at x1 of the Newton parabola is s0 + a*h0; vertex where slope = 0
    xv = x1 - ((y1 - y0) / h0 + a * h0) / (2 * a)
    xv = min(max(xv, x0), x2)
    # Newton form: y = y0 + s0*(x-x0) + a*(x-x0)*(x-x1)
    v = y0 + (y1 - y0) / h0 * (xv - x0) + a * (xv - x0) * (xv - x1)
    return float(min(v, y[i]))


def corner_features(
    car: pd.DataFrame, corners: pd.DataFrame, length: float | None = None
) -> pd.DataFrame:
    """Features for each corner (index aligned to `corners`) from the pole lap's car_data.

    `corners` needs start_m/end_m (from corners.find_apexes). Pass the same `length` used to build
    the reference profile so distances line up. Corners without samples get NaN rows.
    """
    c, d = distance_axis(car, length)
    t = c["t_s"].to_numpy(float)
    v = c["speed_kph"].to_numpy(float)
    gear = c["gear"].to_numpy(float)
    rows = []
    for r in corners.itertuples():
        m = np.flatnonzero((d >= r.start_m) & (d <= r.end_m))
        if len(m) == 0:
            rows.append((np.nan,) * 4)
            continue
        i = m[np.argmin(v[m])]
        vmin = _parabola_min(d, v, int(i))
        entry = float(np.interp(r.start_m, d, v))
        dt = float(np.interp(r.end_m, d, t) - np.interp(r.start_m, d, t))
        rows.append((vmin, max(entry - vmin, 0.0), dt, float(np.nanmin(gear[m]))))
    return pd.DataFrame(rows, columns=FEATURES, index=corners.index)
