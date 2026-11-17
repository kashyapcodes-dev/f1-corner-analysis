"""Offline tests for FastF1 mapping using FastF1-shaped fake frames (no network, no F1TV)."""

import pandas as pd
import pytest

from f1corner.schema import CAR_DATA_COLS, LAPS_COLS, RACE_CONTROL_COLS, WEATHER_COLS
from f1corner.sources import fastf1_source as f
from f1corner.sources.base import SourceUnavailable

td = pd.to_timedelta


def fake_laps():
    return pd.DataFrame(
        dict(
            Driver=["NOR", "NOR"],
            Team=["McLaren"] * 2,
            LapNumber=[1.0, 2.0],
            LapStartDate=pd.to_datetime(["2025-05-24 14:00:00", None]),
            LapTime=td([69.954, None], unit="s"),
            Sector1Time=td([20.0, None], unit="s"),
            Sector2Time=td([30.0, None], unit="s"),
            Sector3Time=td([19.954, None], unit="s"),
            PitInTime=td([None, 5], unit="s"),
            PitOutTime=td([1, None], unit="s"),
            Compound=["SOFT", "SOFT"],
            TyreLife=[1.0, 2.0],
            Stint=[1.0, 1.0],
            Deleted=[False, True],
            TrackStatus=["1", "12"],
            IsAccurate=[True, False],
        )
    )


def test_map_laps():
    out = f.map_laps(fake_laps(), 2025, "Monaco Grand Prix", "Qualifying")
    assert list(out.columns) == LAPS_COLS
    assert out["lap_time_s"].iloc[0] == pytest.approx(69.954)
    assert out["lap_start_s"].iloc[0] == pytest.approx(
        pd.Timestamp("2025-05-24 14:00:00", tz="UTC").timestamp()
    )
    assert pd.isna(out["lap_start_s"].iloc[1])
    assert list(out["pit_out"]) == [True, False] and list(out["pit_in"]) == [False, True]
    assert list(out["green"]) == [True, False]  # '12' = yellow during lap
    assert list(out["deleted"]) == [False, True]
    assert list(out["source_accurate"]) == [True, False]


def test_map_lap_telemetry_uses_lap_relative_time():
    t = pd.DataFrame(
        dict(
            SessionTime=td([100.1, 100.4], unit="s"),
            Speed=[250, 251],
            Throttle=[100, 100],
            Brake=[False, True],
            nGear=[7, 7],
            RPM=[11000, 11100],
        )
    )
    out = f.map_lap_telemetry(t, td(100, unit="s"), "NOR", 26.0, 2025, "e", "s")
    assert list(out.columns) == CAR_DATA_COLS
    assert out["t_s"].tolist() == pytest.approx([0.1, 0.4])
    assert out["brake"].tolist() == [False, True]


def test_map_weather_and_race_control_are_utc_epoch():
    t0 = pd.Timestamp("2025-05-24 13:45:00")
    w = pd.DataFrame(
        dict(
            Time=td([0, 60], unit="s"),
            AirTemp=[21.0, 21.2],
            TrackTemp=[41.9, 41.8],
            Rainfall=[False, True],
        )
    )
    out = f.map_weather(w, t0, 2025, "e", "s")
    assert list(out.columns) == WEATHER_COLS
    assert out["t_s"].iloc[0] == pytest.approx(
        pd.Timestamp("2025-05-24 13:45:00", tz="UTC").timestamp()
    )
    m = pd.DataFrame(
        dict(
            Time=pd.to_datetime(["2025-05-24 14:18:00"]),
            Category=["Flag"],
            Message=["CHEQUERED FLAG"],
            RacingNumber=[None],
        )
    )
    rc = f.map_race_control(m, {"4": "NOR"}, 2025, "e", "s")
    assert list(rc.columns) == RACE_CONTROL_COLS
    assert rc["t_s"].iloc[0] == pytest.approx(
        pd.Timestamp("2025-05-24 14:18:00", tz="UTC").timestamp()
    )


def test_missing_credentials_raise_clear_error(monkeypatch):
    pytest.importorskip("fastf1")
    monkeypatch.delenv("F1_USERNAME", raising=False)
    monkeypatch.delenv("F1_PASSWORD", raising=False)
    monkeypatch.setattr(f, "load_dotenv", lambda *a, **k: None)
    src = f.FastF1Source(cache_dir="data/cache/fastf1-test")
    with pytest.raises(SourceUnavailable, match="F1_USERNAME"):
        src.laps(2025, "Monaco Grand Prix", "Qualifying")
