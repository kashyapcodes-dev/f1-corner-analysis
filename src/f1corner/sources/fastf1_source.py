"""Optional FastF1 backend. Requires `uv sync --extra fastf1` and an F1TV Access subscription.

Credentials: set F1_USERNAME and F1_PASSWORD (environment or a gitignored root .env).
Imported lazily so the rest of the project works without it.
"""

from __future__ import annotations

import os

import pandas as pd

from ..env import load_dotenv
from ..schema import conform
from ..timeutil import epoch_s
from .base import SourceUnavailable


def has_credentials() -> bool:
    load_dotenv()
    return bool(os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD"))


def _secs(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds()


# ---------------------------------------------------------------- pure mapping
# Take FastF1-shaped frames, return canonical tables. Tested with fakes, no network.


def map_laps(laps: pd.DataFrame, season, event, session) -> pd.DataFrame:
    df = pd.DataFrame(
        dict(
            season=season,
            event=event,
            session=session,
            driver=laps["Driver"],
            team=laps["Team"],
            lap_number=laps["LapNumber"],
            lap_start_s=epoch_s(laps["LapStartDate"]).where(laps["LapStartDate"].notna()),
            lap_time_s=_secs(laps["LapTime"]),
            s1_s=_secs(laps["Sector1Time"]),
            s2_s=_secs(laps["Sector2Time"]),
            s3_s=_secs(laps["Sector3Time"]),
            pit_in=laps["PitInTime"].notna(),
            pit_out=laps["PitOutTime"].notna(),
            compound=laps["Compound"],
            tyre_age=laps["TyreLife"],
            stint=laps["Stint"],
            deleted=laps["Deleted"].astype("boolean"),
            green=laps["TrackStatus"].astype(str).eq("1").astype("boolean"),
            source_accurate=laps["IsAccurate"],
        )
    )
    return conform("laps", df)


def map_lap_telemetry(
    t: pd.DataFrame,
    lap_start_session_time: pd.Timedelta,
    driver,
    lap_number,
    season,
    event,
    session,
) -> pd.DataFrame:
    """One lap of FastF1 car data (has SessionTime) -> canonical car_data rows."""
    return pd.DataFrame(
        dict(
            season=season,
            event=event,
            session=session,
            driver=driver,
            lap_number=lap_number,
            t_s=_secs(t["SessionTime"] - lap_start_session_time),
            speed_kph=t["Speed"].to_numpy(),
            throttle=t["Throttle"].to_numpy(),
            brake=t["Brake"].astype(bool).to_numpy(),
            gear=t["nGear"].to_numpy(),
            rpm=t["RPM"].to_numpy(),
        )
    )


def map_weather(w: pd.DataFrame, t0_date: pd.Timestamp, season, event, session) -> pd.DataFrame:
    df = pd.DataFrame(
        dict(
            season=season,
            event=event,
            session=session,
            t_s=epoch_s(t0_date + w["Time"]),
            air_temp_c=w["AirTemp"],
            track_temp_c=w["TrackTemp"],
            rain=w["Rainfall"].astype(bool),
        )
    )
    return conform("weather", df)


def map_race_control(m: pd.DataFrame, number_to_code: dict, season, event, session) -> pd.DataFrame:
    df = pd.DataFrame(
        dict(
            season=season,
            event=event,
            session=session,
            t_s=epoch_s(m["Time"]),
            category=m["Category"],
            message=m["Message"],
            driver=m["RacingNumber"].map(number_to_code),
        )
    )
    return conform("race_control", df)


# ---------------------------------------------------------------------- source


class FastF1Source:
    name = "fastf1"

    def __init__(self, cache_dir: str = ".fastf1-cache"):
        try:
            import fastf1
        except ImportError as e:
            raise SourceUnavailable("fastf1 not installed: uv sync --extra fastf1") from e
        load_dotenv()
        self._ff1 = fastf1
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.configure(cache_dir=cache_dir)
        self._sessions: dict = {}

    def _load(self, season, event, session):
        key = (season, event, session)
        if key not in self._sessions:
            if not has_credentials():
                raise SourceUnavailable(
                    "FastF1 needs F1_USERNAME and F1_PASSWORD (F1TV Access) in the environment or root .env (copy local/.env.example to .env)"
                )
            s = self._ff1.get_session(season, event, session)
            s.load(laps=True, telemetry=True, weather=True, messages=True)
            try:
                s.laps  # noqa: B018 - raises if the load failed (e.g. auth rejected)
            except self._ff1.exceptions.DataNotLoadedError as e:
                raise SourceUnavailable(
                    f"FastF1 could not load {key}; check F1TV credentials/subscription"
                ) from e
            self._sessions[key] = s
        return self._sessions[key]

    def sessions(self, season):
        sched = self._ff1.get_event_schedule(season, include_testing=False)
        rows = []
        for _, ev in sched.iterrows():
            for n in range(1, 6):
                name = ev.get(f"Session{n}")
                if isinstance(name, str) and name:
                    rows.append(
                        dict(
                            season=season,
                            event=ev["EventName"],
                            session=name,
                            round=ev["RoundNumber"],
                            date=ev.get(f"Session{n}DateUtc"),
                            circuit=ev.get("Location"),
                        )
                    )
        return conform("sessions", pd.DataFrame(rows))

    def laps(self, season, event, session):
        return map_laps(
            pd.DataFrame(self._load(season, event, session).laps), season, event, session
        )

    def _telemetry(self, season, event, session, getter, cols):
        s = self._load(season, event, session)
        parts = []
        for _, lap in s.laps.iterlaps():
            if pd.isna(lap["LapStartTime"]):
                continue
            t = getattr(lap, getter)()
            if t.empty:
                continue
            parts.append((lap, t))
        return parts

    def car_data(self, season, event, session):
        parts = [
            map_lap_telemetry(
                t, lap["LapStartTime"], lap["Driver"], lap["LapNumber"], season, event, session
            )
            for lap, t in self._telemetry(season, event, session, "get_car_data", None)
        ]
        return conform("car_data", pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())

    def location(self, season, event, session):
        parts = []
        for lap, p in self._telemetry(season, event, session, "get_pos_data", None):
            parts.append(
                pd.DataFrame(
                    dict(
                        season=season,
                        event=event,
                        session=session,
                        driver=lap["Driver"],
                        lap_number=lap["LapNumber"],
                        t_s=_secs(p["SessionTime"] - lap["LapStartTime"]),
                        x=p["X"].to_numpy(),
                        y=p["Y"].to_numpy(),
                        z=p["Z"].to_numpy(),
                    )
                )
            )
        return conform("location", pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())

    def weather(self, season, event, session):
        s = self._load(season, event, session)
        return map_weather(s.weather_data, s.t0_date, season, event, session)

    def race_control(self, season, event, session):
        s = self._load(season, event, session)
        codes = dict(zip(s.laps["DriverNumber"], s.laps["Driver"]))
        return map_race_control(s.race_control_messages, codes, season, event, session)
