"""OpenF1 backend: free, no sign-in, the reproducible default. 3 req/s, 30 req/min.

HTTP lives in OpenF1Source; payload -> canonical mapping lives in pure functions
(map_*) so it can be unit-tested without network.
"""
from __future__ import annotations

import re
import time

import httpx
import numpy as np
import pandas as pd

from ..schema import conform
from ..timeutil import epoch_s

BASE = "https://api.openf1.org/v1"

_DELETED = re.compile(r"CAR (\d+) \((\w+)\).*\bDELETED\b.*\bLAP (\d+)\b")
_YELLOWS = {"YELLOW", "DOUBLE YELLOW"}


class Throttle:
    """Sliding window: <=3/s and <=30/min."""

    def __init__(self, per_s=3, per_min=30):
        self.per_s, self.per_min, self.t = per_s, per_min, []

    def wait(self):
        while True:
            now = time.monotonic()
            self.t = [x for x in self.t if now - x < 60]
            s = sum(now - x < 1 for x in self.t)
            if s < self.per_s and len(self.t) < self.per_min:
                self.t.append(now)
                return
            time.sleep(0.25)


# ---------------------------------------------------------------- pure mapping


def _keys(season, event, session) -> dict:
    return dict(season=season, event=event, session=session)


def non_green_intervals(rc: pd.DataFrame) -> list[tuple[float, float]]:
    """Epoch-second intervals when track was not green (yellow in any sector, red, SC/VSC).

    Matches FastF1's TrackStatus != '1' notion: a yellow anywhere on track counts.
    """
    if rc.empty:
        return []
    rc = rc.assign(t=epoch_s(rc["date"])).sort_values("t")
    open_: dict[tuple, float] = {}
    out: list[tuple[float, float]] = []

    def close(key, t):
        if key in open_:
            out.append((open_.pop(key), t))

    for r in rc.itertuples():
        flag, scope, sector, msg = r.flag, r.scope, r.sector, str(r.message or "")
        if r.category == "SafetyCar":
            key = ("sc",)
            if "DEPLOYED" in msg:
                open_.setdefault(key, r.t)
            elif "IN THIS LAP" in msg or "ENDING" in msg:
                close(key, r.t)
        elif flag in _YELLOWS and scope in ("Sector", "Track"):
            open_.setdefault(("yellow", scope, sector), r.t)
        elif flag == "RED":
            open_.setdefault(("red",), r.t)
        elif flag == "CLEAR" and scope == "Sector":
            close(("yellow", "Sector", sector), r.t)
        elif flag in ("CLEAR", "GREEN") and scope == "Track":
            for k in [k for k in open_ if k[0] in ("yellow", "red")]:
                close(k, r.t)
    out += [(s, float("inf")) for s in open_.values()]
    return out


def deleted_laps(rc: pd.DataFrame) -> set[tuple[str, int]]:
    """(driver_acronym, lap_number) pairs the stewards deleted."""
    found = set()
    for m in rc.get("message", pd.Series(dtype=str)).dropna():
        g = _DELETED.search(m)
        if g:
            found.add((g.group(2), int(g.group(3))))
    return found


def map_laps(laps, drivers, stints, rc, season, event, session) -> pd.DataFrame:
    if laps.empty:
        return conform("laps", pd.DataFrame())
    acr = drivers.set_index("driver_number")["name_acronym"]
    team = drivers.set_index("driver_number")["team_name"]
    df = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True).copy()
    df["driver"] = df["driver_number"].map(acr)
    df["team"] = df["driver_number"].map(team)
    start = epoch_s(df["date_start"]) if df["date_start"].notna().any() else pd.Series(np.nan, index=df.index)
    start = start.where(df["date_start"].notna())

    # lap end = start + duration, else next lap's start for the same driver
    nxt = start.groupby(df["driver_number"]).shift(-1)
    end = (start + df["lap_duration"]).fillna(nxt)

    pit_out = df["is_pit_out_lap"].fillna(False).astype(bool)
    pit_in = pit_out.groupby(df["driver_number"]).shift(-1, fill_value=False).astype(bool)

    deleted = deleted_laps(rc)
    is_deleted = [(d, n) in deleted for d, n in zip(df["driver"], df["lap_number"])]

    ivals = non_green_intervals(rc)
    green = []
    for s, e in zip(start, end):
        if np.isnan(s) or np.isnan(e):
            green.append(pd.NA)
        else:
            green.append(not any(a < e and b > s for a, b in ivals))

    comp = pd.Series(pd.NA, index=df.index, dtype="object")
    age = pd.Series(np.nan, index=df.index)
    stint_no = pd.Series(np.nan, index=df.index)
    for st in stints.itertuples():
        m = (df["driver_number"] == st.driver_number) & df["lap_number"].between(st.lap_start, st.lap_end)
        comp[m] = st.compound
        age[m] = st.tyre_age_at_start + (df.loc[m, "lap_number"] - st.lap_start) + 1  # FastF1 TyreLife is 1-based
        stint_no[m] = st.stint_number

    out = pd.DataFrame(dict(
        **_keys(season, event, session), driver=df["driver"], team=df["team"],
        lap_number=df["lap_number"], lap_start_s=start, lap_time_s=df["lap_duration"],
        s1_s=df["duration_sector_1"], s2_s=df["duration_sector_2"], s3_s=df["duration_sector_3"],
        pit_in=pit_in, pit_out=pit_out, compound=comp, tyre_age=age, stint=stint_no,
        deleted=is_deleted, green=pd.array(green, dtype="boolean"),
    ))
    return conform("laps", out)


def _slice_by_lap(samples: pd.DataFrame, laps: pd.DataFrame, driver_number: int, value_cols: dict) -> pd.DataFrame:
    """Cut a driver's whole-session samples into laps; t_s = seconds since lap start."""
    s = samples.assign(_t=epoch_s(samples["date"])).sort_values("_t")
    t = s["_t"].to_numpy()
    parts = []
    for r in laps[(laps["driver_number"] == driver_number) & laps["date_start"].notna()].itertuples():
        a = epoch_s(pd.Series([r.date_start])).iloc[0]
        b = a + r.lap_duration if pd.notna(r.lap_duration) else np.inf
        i, j = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "left")
        if j <= i:
            continue
        chunk = s.iloc[i:j]
        d = pd.DataFrame({"lap_number": r.lap_number, "t_s": chunk["_t"].to_numpy() - a})
        for src, dst in value_cols.items():
            d[dst] = chunk[src].to_numpy()
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def map_car_data(samples_by_driver: dict[int, pd.DataFrame], laps, drivers, season, event, session) -> pd.DataFrame:
    acr = drivers.set_index("driver_number")["name_acronym"]
    parts = []
    for num, s in samples_by_driver.items():
        if s.empty:
            continue
        d = _slice_by_lap(s, laps, num, {"speed": "speed_kph", "throttle": "throttle",
                                          "brake": "brake", "n_gear": "gear", "rpm": "rpm"})
        if d.empty:
            continue
        d["brake"] = d["brake"] > 0
        d["driver"] = acr[num]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return conform("car_data", df.assign(**_keys(season, event, session)) if len(df) else df)


def map_location(samples_by_driver: dict[int, pd.DataFrame], laps, drivers, season, event, session) -> pd.DataFrame:
    acr = drivers.set_index("driver_number")["name_acronym"]
    parts = []
    for num, s in samples_by_driver.items():
        if s.empty:
            continue
        d = _slice_by_lap(s, laps, num, {"x": "x", "y": "y", "z": "z"})
        if d.empty:
            continue
        d["driver"] = acr[num]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return conform("location", df.assign(**_keys(season, event, session)) if len(df) else df)


def map_weather(w: pd.DataFrame, season, event, session) -> pd.DataFrame:
    if w.empty:
        return conform("weather", pd.DataFrame())
    df = pd.DataFrame(dict(**_keys(season, event, session), t_s=epoch_s(w["date"]),
                           air_temp_c=w["air_temperature"], track_temp_c=w["track_temperature"],
                           rain=w["rainfall"] > 0))
    return conform("weather", df)


def map_race_control(rc: pd.DataFrame, drivers, season, event, session) -> pd.DataFrame:
    if rc.empty:
        return conform("race_control", pd.DataFrame())
    acr = drivers.set_index("driver_number")["name_acronym"]
    df = pd.DataFrame(dict(**_keys(season, event, session), t_s=epoch_s(rc["date"]),
                           category=rc["category"], message=rc["message"],
                           driver=rc["driver_number"].map(acr)))
    return conform("race_control", df)


# ---------------------------------------------------------------------- source


class OpenF1Source:
    name = "openf1"

    def __init__(self, client: httpx.Client | None = None):
        self.http = client or httpx.Client(timeout=120)
        self.throttle = Throttle()
        self._memo: dict = {}

    def _get(self, endpoint: str, **params) -> pd.DataFrame:
        for attempt in range(5):
            self.throttle.wait()
            r = self.http.get(f"{BASE}/{endpoint}", params=params)
            if r.status_code == 404:  # OpenF1 answers "no results" with 404
                return pd.DataFrame()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return pd.DataFrame(r.json())
        r.raise_for_status()
        return pd.DataFrame()

    def _memoized(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def _session_key(self, season, event, session) -> int:
        def find():
            meets = self._memoized(("meetings", season), lambda: self._get("meetings", year=season))
            hit = meets[meets["meeting_name"] == event] if not meets.empty else meets
            if hit.empty:
                raise LookupError(f"No OpenF1 meeting {season} {event!r}")
            ses = self._get("sessions", meeting_key=int(hit.iloc[0]["meeting_key"]), session_name=session)
            if ses.empty:
                raise LookupError(f"No OpenF1 session {season} {event!r} {session!r}")
            return int(ses.iloc[0]["session_key"])
        return self._memoized(("sk", season, event, session), find)

    def _raw(self, endpoint, season, event, session, **extra) -> pd.DataFrame:
        sk = self._session_key(season, event, session)
        return self._memoized((endpoint, sk, tuple(sorted(extra.items()))),
                              lambda: self._get(endpoint, session_key=sk, **extra))

    def sessions(self, season):
        meets = self._get("meetings", year=season)
        ses = self._get("sessions", year=season)
        if ses.empty:
            return conform("sessions", pd.DataFrame())
        name = meets.set_index("meeting_key")["meeting_name"]
        df = pd.DataFrame(dict(season=season, event=ses["meeting_key"].map(name), session=ses["session_name"],
                               round=pd.NA, date=ses["date_start"], circuit=ses["circuit_short_name"]))
        return conform("sessions", df)

    def laps(self, season, event, session):
        return map_laps(self._raw("laps", season, event, session),
                        self._raw("drivers", season, event, session),
                        self._raw("stints", season, event, session),
                        self._raw("race_control", season, event, session),
                        season, event, session)

    def _per_driver(self, endpoint, season, event, session):
        drivers = self._raw("drivers", season, event, session)
        sk = self._session_key(season, event, session)
        return {int(n): self._get(endpoint, session_key=sk, driver_number=int(n))
                for n in drivers["driver_number"]}

    def car_data(self, season, event, session):
        return map_car_data(self._per_driver("car_data", season, event, session),
                            self._raw("laps", season, event, session),
                            self._raw("drivers", season, event, session), season, event, session)

    def location(self, season, event, session):
        return map_location(self._per_driver("location", season, event, session),
                            self._raw("laps", season, event, session),
                            self._raw("drivers", season, event, session), season, event, session)

    def weather(self, season, event, session):
        return map_weather(self._raw("weather", season, event, session), season, event, session)

    def race_control(self, season, event, session):
        return map_race_control(self._raw("race_control", season, event, session),
                                self._raw("drivers", season, event, session), season, event, session)
