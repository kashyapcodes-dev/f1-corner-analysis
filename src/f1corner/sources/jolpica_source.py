"""Jolpica (Ergast-compatible) backend: schedule + official classification. Free, no sign-in.

Limits: 4 req/s burst, 500 req/h sustained. HTTP lives in JolpicaSource; payload -> canonical
mapping lives in pure functions (map_*) so it can be unit-tested without network.
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

from ..schema import conform
from .openf1_source import Throttle

BASE = "https://api.jolpi.ca/ergast/f1"

# Jolpica race-object key -> canonical session name
SCHEDULE_SESSIONS = {
    "FirstPractice": "Practice 1",
    "SecondPractice": "Practice 2",
    "ThirdPractice": "Practice 3",
    "SprintQualifying": "Sprint Qualifying",
    "SprintShootout": "Sprint Qualifying",
    "Sprint": "Sprint",
    "Qualifying": "Qualifying",
}
# Canonical session name -> (endpoint, result-list key)
CLASSIFIED = {
    "Race": ("results", "Results"),
    "Qualifying": ("qualifying", "QualifyingResults"),
    "Sprint": ("sprint", "SprintResults"),
}


def _lap_s(t) -> float:
    """'1:29.822' / '29.822' -> seconds; ''/None -> NaN."""
    if not t:
        return float("nan")
    m, _, s = str(t).rpartition(":")
    return (int(m) * 60 if m else 0) + float(s)


def _int(x):
    try:
        return int(x)
    except TypeError, ValueError:
        return pd.NA


def _ts(d: dict) -> str | None:
    if not d or "date" not in d:
        return None
    return f"{d['date']}T{d['time']}" if d.get("time") else d["date"]


# ---------------------------------------------------------------- pure mapping


def map_schedule(races: list[dict], season: int) -> pd.DataFrame:
    rows = []
    for r in races:
        c = r.get("Circuit", {})
        loc = c.get("Location", {})
        base = dict(
            season=season,
            event=r["raceName"],
            round=int(r["round"]),
            circuit=c.get("circuitName"),
            circuit_id=c.get("circuitId"),
            locality=loc.get("locality"),
            country=loc.get("country"),
        )
        rows.append(dict(base, session="Race", date=_ts(r)))
        for key, name in SCHEDULE_SESSIONS.items():
            if key in r:
                rows.append(dict(base, session=name, date=_ts(r[key])))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
        df = df.sort_values(["round", "date"], kind="stable").reset_index(drop=True)
    return conform("schedule", df)


def map_classification(race: dict, season: int, event: str, session: str) -> pd.DataFrame:
    """One Jolpica race object (Results/QualifyingResults/SprintResults) -> classification rows."""
    key = CLASSIFIED[session][1]
    rows = []
    for r in race.get(key, []):
        fl = r.get("FastestLap", {})
        rows.append(
            dict(
                season=season,
                event=event,
                session=session,
                position=_int(r.get("position")),
                position_text=r.get("positionText", r.get("position")),
                driver=r["Driver"].get("code") or r["Driver"]["driverId"],
                driver_number=_int(r.get("number")),
                team=r["Constructor"]["name"],
                grid=_int(r.get("grid")),
                laps=_int(r.get("laps")),
                status=r.get("status"),
                time_s=float(r["Time"]["millis"]) / 1000 if "millis" in r.get("Time", {}) else None,
                points=float(r["points"]) if "points" in r else None,
                fastest_lap_rank=_int(fl.get("rank")),
                fastest_lap_s=_lap_s(fl.get("Time", {}).get("time")),
                q1_s=_lap_s(r.get("Q1")),
                q2_s=_lap_s(r.get("Q2")),
                q3_s=_lap_s(r.get("Q3")),
            )
        )
    return conform("classification", pd.DataFrame(rows))


# ------------------------------------------------------------------------ HTTP


class JolpicaSource:
    name = "jolpica"

    def __init__(self, client: httpx.Client | None = None):
        self.http = client or httpx.Client(timeout=60)
        self.throttle = Throttle(per_s=3, per_min=60)  # well inside 500/h for a season pull
        self._memo: dict = {}

    def _get(self, path: str) -> dict:
        for attempt in range(5):
            self.throttle.wait()
            r = self.http.get(f"{BASE}/{path}", params={"limit": 100})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()["MRData"]
        r.raise_for_status()
        return {}

    def _races(self, season: int) -> list[dict]:
        if season not in self._memo:
            self._memo[season] = self._get(f"{season}/races/")["RaceTable"]["Races"]
        return self._memo[season]

    def schedule(self, season: int, event: str = "all", session: str = "all") -> pd.DataFrame:
        """Whole-season calendar. event/session ignored; cache stores it at event=all/session=all."""
        return map_schedule(self._races(season), season)

    def classification(self, season: int, event: str, session: str) -> pd.DataFrame:
        if session not in CLASSIFIED:
            raise LookupError(f"No Jolpica classification for session {session!r}")
        rnd = next((r["round"] for r in self._races(season) if r["raceName"] == event), None)
        if rnd is None:
            raise LookupError(f"No Jolpica event {season} {event!r}")
        endpoint, key = CLASSIFIED[session]
        races = self._get(f"{season}/{rnd}/{endpoint}/")["RaceTable"]["Races"]
        return map_classification(races[0] if races else {}, season, event, session)
