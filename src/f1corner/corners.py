"""Find and segment corners on the session's reference (pole) lap.

Pipeline: telemetry -> distance profile (speed integrated over time) -> apexes on the reference
-> fixed distance segments [brake/lift start, back on full throttle] -> applied to every lap.

Tuning constants are set once and frozen; do not tune per session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_M = 2.0  # distance grid step
SMOOTH_M = 30.0  # moving-average window on speed
MIN_DROP_KPH = 15.0  # FROZEN: min speed prominence for a speed-minimum corner
FLAT_MIN_KPH = 3.0  # minima below this are noise; between this and MIN_DROP: lift or flat-out
FULL_THROTTLE = 98.0  # percent counted as "full"
LIFT_MIN_M = 10.0  # a lift shorter than this is not "measurable"


@dataclass
class Profile:
    """One lap on a common distance grid (metres from the start line)."""

    dist: np.ndarray
    speed: np.ndarray  # smoothed km/h
    throttle: np.ndarray  # %
    brake: np.ndarray  # bool
    length: float  # integrated lap length before any rescale (m)


def _smooth(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    pad = np.pad(y, win // 2, mode="edge")
    return np.convolve(pad, np.ones(win) / win, mode="valid")[: len(y)]


def lap_profile(car: pd.DataFrame, length: float | None = None) -> Profile:
    """Distance profile for one lap's car_data (t_s, speed_kph, throttle, brake).

    Distance = integral of speed over time. If `length` is given the distance axis is rescaled so
    the lap ends at exactly `length` (cancels per-lap integration drift, aligning laps by position).
    """
    c = car.sort_values("t_s").drop_duplicates("t_s")
    t = c["t_s"].to_numpy(float)
    v = c["speed_kph"].to_numpy(float) / 3.6
    d = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])
    raw_len = float(d[-1])
    if length is not None and raw_len > 0:
        d = d * (length / raw_len)
    end = float(d[-1])
    grid = np.arange(0.0, end, GRID_M)
    speed = np.interp(grid, d, c["speed_kph"].to_numpy(float))
    speed = _smooth(speed, int(SMOOTH_M / GRID_M))
    thr = np.interp(grid, d, c["throttle"].to_numpy(float))
    brk = np.interp(grid, d, c["brake"].to_numpy(float)) > 0.5
    return Profile(grid, speed, thr, brk, raw_len)


def prominence_minima(y: np.ndarray) -> list[tuple[int, float]]:
    """Local minima with topographic prominence (how far speed rises on both sides)."""
    n = len(y)
    out = []
    # plateau-safe local minima: strictly lower than left neighbour run, <= right
    i = 1
    while i < n - 1:
        j = i
        while j + 1 < n and y[j + 1] == y[i]:
            j += 1
        if y[i - 1] > y[i] and j + 1 < n and y[j + 1] > y[i]:
            m = (i + j) // 2
            left = y[i]
            for k in range(i - 1, -1, -1):
                if y[k] < y[i]:
                    break
                left = max(left, y[k])
            right = y[i]
            for k in range(j + 1, n):
                if y[k] < y[i]:
                    break
                right = max(right, y[k])
            # edges count only if the walk reached the lap boundary at a higher level
            out.append((m, float(min(left, right) - y[i])))
        i = j + 1
    return out


def _lifts(p: Profile) -> np.ndarray:
    """Boolean mask: not at full throttle (lift, partial throttle or braking)."""
    return (p.throttle < FULL_THROTTLE) | p.brake


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where mask is True."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(idx[::2], idx[1::2])]


def find_apexes(ref: Profile) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corners, flat_out).

    corners: apex_m, apex_speed_kph, kind ("speed"|"lift"), prominence_kph, start_m, end_m.
    flat_out: apexes with a speed dip (>= FLAT_MIN_KPH) never lifted/braked: excluded, to be listed.
    """
    lift = _lifts(ref)
    lift_runs = [r for r in _runs(lift) if (r[1] - r[0] + 1) * GRID_M >= LIFT_MIN_M]
    in_lift = np.zeros(len(lift), bool)
    for a, b in lift_runs:
        in_lift[a : b + 1] = True

    cands: dict[int, tuple[str, float]] = {}
    flat: list[tuple[int, float]] = []
    for i, prom in prominence_minima(ref.speed):
        if prom < FLAT_MIN_KPH:
            continue
        if prom >= MIN_DROP_KPH:
            cands[i] = ("speed", prom)
        elif in_lift[i]:
            cands[i] = ("lift", prom)
        else:
            flat.append((i, prom))
    # lift runs with no apex inside: apex = slowest point of the lift
    for a, b in lift_runs:
        if not any(a <= i <= b for i in cands):
            i = a + int(np.argmin(ref.speed[a : b + 1]))
            cands[i] = ("lift", 0.0)

    rows = []
    for i in sorted(cands):
        kind, prom = cands[i]
        s, e = segment(ref, i)
        rows.append((ref.dist[i], ref.speed[i], kind, prom, ref.dist[s], ref.dist[e]))
    cols = ["apex_m", "apex_speed_kph", "kind", "prominence_kph", "start_m", "end_m"]
    corners = pd.DataFrame(rows, columns=cols)
    corners = corners.sort_values("apex_m").reset_index(drop=True)
    corners = add_sequences(corners)
    fo = pd.DataFrame(
        [(ref.dist[i], ref.speed[i], p) for i, p in sorted(flat)],
        columns=["apex_m", "apex_speed_kph", "prominence_kph"],
    )
    return corners, fo


def segment(ref: Profile, apex: int) -> tuple[int, int]:
    """Index range: back from the apex to the last full-throttle/no-brake sample (+1), forward to
    the first full-throttle sample. Clipped to the lap."""
    lift = _lifts(ref)
    s = apex
    while s > 0 and lift[s - 1]:
        s -= 1
    e = apex
    while e < len(lift) - 1 and (ref.throttle[e] < FULL_THROTTLE):
        e += 1
    return s, e


def add_sequences(corners: pd.DataFrame) -> pd.DataFrame:
    """Flag apexes whose segments overlap or touch. `sequence_id` groups them (NA when alone)."""
    out = corners.sort_values("apex_m").reset_index(drop=True).copy()
    groups: list[list[int]] = []
    group_end = -np.inf
    for i, r in out.iterrows():
        if groups and r.start_m <= group_end:
            groups[-1].append(i)
            group_end = max(group_end, r.end_m)
        else:
            groups.append([i])
            group_end = r.end_m
    seq = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for sid, g in enumerate(x for x in groups if len(x) > 1):
        seq[g] = sid
    out["sequence_id"] = seq
    out["in_sequence"] = seq.notna()
    return out


def segment_lap(profile: Profile, corners: pd.DataFrame) -> pd.DataFrame:
    """Apply the reference's fixed distance markers to any (rescaled) lap profile.

    Returns per corner: entry/min/exit speed, min speed location, time-in-segment is left to callers.
    """
    rows = []
    for k, c in corners.iterrows():
        m = (profile.dist >= c.start_m) & (profile.dist <= c.end_m)
        if not m.any():
            rows.append((k, np.nan, np.nan, np.nan, np.nan))
            continue
        v = profile.speed[m]
        rows.append((k, v[0], v.min(), v[-1], profile.dist[m][np.argmin(v)]))
    return pd.DataFrame(
        rows, columns=["corner", "entry_kph", "min_kph", "exit_kph", "min_at_m"]
    ).set_index("corner")


def xy_at(ref_car: pd.DataFrame, ref_loc: pd.DataFrame, dist_m: np.ndarray, length: float | None):
    """Map distances on the reference lap to track x,y via the shared time axis."""
    c = ref_car.sort_values("t_s").drop_duplicates("t_s")
    t = c["t_s"].to_numpy(float)
    v = c["speed_kph"].to_numpy(float) / 3.6
    d = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])
    if length is not None and d[-1] > 0:
        d = d * (length / d[-1])
    tt = np.interp(dist_m, d, t)
    loc = ref_loc.sort_values("t_s").drop_duplicates("t_s")
    return (
        np.interp(tt, loc["t_s"].to_numpy(float), loc["x"].to_numpy(float)),
        np.interp(tt, loc["t_s"].to_numpy(float), loc["y"].to_numpy(float)),
    )
