"""Lap filtering. A lap is kept only if it passes every rule; each drop records its reason.

Rules (numbering follows the spec; 1-2 live in ingestion):
  3 not_pit        no pit entry/exit during the lap
  4 not_green      green flag throughout (yellow/SC/VSC/red dropped; unknown = dropped)
  5 deleted        not deleted by the stewards
  6 sectors        s1+s2+s3 == lap time (within SECTOR_TOL_S)
  7 slow           lap_time <= 1.02 * driver's best lap (best among laps that passed 3-6)
  8 wet session    any INTERMEDIATE/WET tyre use -> whole session excluded
  9 car issue      driver >1% slower than teammate -> flagged for manual FIA check (not dropped)
Qualifying: additionally keep only each driver's single best valid lap.

A lap with no time is dropped as "no_time". The first failing rule is the recorded reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SLOW_FACTOR = 1.02
TEAMMATE_GAP = 0.01
SECTOR_TOL_S = 0.005
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

REASONS = ["no_time", "pit", "not_green", "deleted", "sectors", "slow"]


@dataclass
class FilterResult:
    kept: pd.DataFrame  # surviving laps
    laps: pd.DataFrame  # all input laps + `drop_reason` (NA when kept)
    reasons: pd.Series  # drop counts per reason (all REASONS present; "wet_session" if excluded)
    wet_session: bool
    car_flags: pd.DataFrame  # drivers >1% slower than teammate (for manual check)


def is_wet(laps: pd.DataFrame) -> bool:
    """Rule 8: any intermediate or wet tyre use anywhere in the session."""
    return bool(laps["compound"].astype("string").str.upper().isin(WET_COMPOUNDS).any())


def _b(s: pd.Series, default: bool) -> pd.Series:
    return s.astype("boolean").fillna(default).astype(bool)


def drop_reasons(laps: pd.DataFrame) -> pd.Series:
    """First failing rule per lap (NA = kept). Rules 3-7; session-level rule 8 handled by caller."""
    t = laps["lap_time_s"]
    reason = pd.Series(pd.NA, index=laps.index, dtype="object")

    def mark(mask, name):
        nonlocal reason
        reason = reason.where(reason.notna() | ~mask, name)

    mark(t.isna(), "no_time")
    mark(_b(laps["pit_in"], False) | _b(laps["pit_out"], False), "pit")
    mark(~_b(laps["green"], False), "not_green")
    mark(_b(laps["deleted"], False), "deleted")
    sector_sum = laps[["s1_s", "s2_s", "s3_s"]].sum(axis=1, min_count=3)
    mark(~((sector_sum - t).abs() <= SECTOR_TOL_S), "sectors")  # NaN sectors -> dropped
    # rule 7 against each driver's best among laps that passed 3-6
    best = t.where(reason.isna()).groupby(laps["driver"]).transform("min")
    mark(reason.isna() & (t > SLOW_FACTOR * best), "slow")
    return reason


def car_issue_flags(kept: pd.DataFrame) -> pd.DataFrame:
    """Rule 9: drivers whose best kept lap is >1% slower than their teammate's."""
    if kept.empty:
        return pd.DataFrame(columns=["team", "driver", "best_s", "teammate", "gap_pct"])
    best = kept.groupby(["team", "driver"])["lap_time_s"].min().reset_index(name="best_s")
    rows = []
    for team, g in best.groupby("team"):
        for r in g.itertuples():
            for o in g.itertuples():
                if o.driver != r.driver and r.best_s > o.best_s * (1 + TEAMMATE_GAP):
                    rows.append(
                        (team, r.driver, r.best_s, o.driver, 100 * (r.best_s / o.best_s - 1))
                    )
    return pd.DataFrame(rows, columns=["team", "driver", "best_s", "teammate", "gap_pct"])


def filter_laps(laps: pd.DataFrame, qualifying: bool = False) -> FilterResult:
    laps = laps.copy()
    wet = is_wet(laps)
    if wet:
        laps["drop_reason"] = "wet_session"
        counts = pd.Series(0, index=[*REASONS, "wet_session"])
        counts["wet_session"] = len(laps)
        return FilterResult(laps.iloc[0:0], laps, counts, True, car_issue_flags(laps.iloc[0:0]))

    laps["drop_reason"] = drop_reasons(laps)
    if qualifying:
        ok = laps[laps["drop_reason"].isna()]
        best_idx = ok.groupby("driver")["lap_time_s"].idxmin()
        extra = laps["drop_reason"].isna() & ~laps.index.isin(best_idx)
        laps.loc[extra, "drop_reason"] = "not_best"
    kept = laps[laps["drop_reason"].isna()].drop(columns="drop_reason")
    cols = [*REASONS, *(["not_best"] if qualifying else [])]
    counts = laps["drop_reason"].value_counts().reindex(cols, fill_value=0)
    return FilterResult(kept, laps, counts, False, car_issue_flags(kept))


def pole_lap(kept: pd.DataFrame) -> pd.Series:
    """Reference lap = fastest kept lap in the session."""
    return kept.loc[kept["lap_time_s"].idxmin()]
