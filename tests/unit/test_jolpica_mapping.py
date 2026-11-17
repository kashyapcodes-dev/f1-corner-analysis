"""Offline unit tests for Jolpica payload -> canonical mapping + ingest (synthetic data, no network)."""

import pandas as pd

from f1corner import cache, ingest
from f1corner.schema import CLASSIFICATION_COLS, SCHEDULE_COLS
from f1corner.sources import jolpica_source as j

RACE = {
    "round": "6",
    "raceName": "Miami Grand Prix",
    "Circuit": {
        "circuitId": "miami",
        "circuitName": "Miami International Autodrome",
        "Location": {"locality": "Miami", "country": "USA"},
    },
    "date": "2025-05-04",
    "time": "20:00:00Z",
    "FirstPractice": {"date": "2025-05-02", "time": "17:30:00Z"},
    "Qualifying": {"date": "2025-05-03", "time": "20:00:00Z"},
    "Sprint": {"date": "2025-05-03", "time": "16:00:00Z"},
}


def test_map_schedule_one_row_per_session_sorted():
    df = j.map_schedule([RACE], 2025)
    assert list(df.columns) == SCHEDULE_COLS
    assert list(df["session"]) == ["Practice 1", "Sprint", "Qualifying", "Race"]
    assert df["round"].eq(6).all() and df["circuit_id"].eq("miami").all()
    assert df["date"].iloc[-1] == pd.Timestamp("2025-05-04T20:00:00Z")


def driver(code, num="81"):
    return {"code": code, "driverId": code.lower(), "permanentNumber": num}


def test_map_classification_race():
    race = {
        "Results": [
            {
                "number": "81",
                "position": "1",
                "positionText": "1",
                "points": "25",
                "Driver": driver("PIA"),
                "Constructor": {"name": "McLaren"},
                "grid": "4",
                "laps": "57",
                "status": "Finished",
                "Time": {"millis": "5331587", "time": "1:28:51.587"},
                "FastestLap": {"rank": "2", "lap": "35", "Time": {"time": "1:29.822"}},
            },
            {
                "number": "1",
                "position": "20",
                "positionText": "R",
                "points": "0",
                "Driver": driver("NOR"),
                "Constructor": {"name": "McLaren"},
                "grid": "2",
                "laps": "3",
                "status": "Retired",
            },
        ]
    }
    df = j.map_classification(race, 2025, "Miami Grand Prix", "Race")
    assert list(df.columns) == CLASSIFICATION_COLS
    a, b = df.iloc[0], df.iloc[1]
    assert a["driver"] == "PIA" and a["time_s"] == 5331.587 and a["fastest_lap_s"] == 89.822
    assert b["position_text"] == "R" and pd.isna(b["time_s"]) and pd.isna(b["fastest_lap_s"])


def test_map_classification_qualifying_blank_laps_are_nan():
    race = {
        "QualifyingResults": [
            {
                "number": "1",
                "position": "1",
                "Driver": driver("VER", "3"),
                "Constructor": {"name": "Red Bull"},
                "Q1": "1:27.1",
                "Q2": "1:26.5",
                "Q3": "",
            }
        ]
    }
    r = j.map_classification(race, 2025, "Miami Grand Prix", "Qualifying").iloc[0]
    assert r["q1_s"] == 87.1 and r["q2_s"] == 86.5 and pd.isna(r["q3_s"])


def test_ingest_season_skips_future_and_unclassified_sessions(tmp_path):
    class Fake:
        name = "fake"

        def __init__(self):
            self.calls = []

        def schedule(self, season, event, session):
            self.calls.append("schedule")
            far = pd.Timestamp("2999-01-01", tz="UTC")
            past = pd.Timestamp("2025-01-01", tz="UTC")
            rows = [
                ("Race", past),
                ("Practice 1", past),
                ("Qualifying", past),
                ("Sprint", far),
            ]
            return pd.DataFrame(
                [dict(season=season, event="X GP", session=s, round=1, date=d) for s, d in rows]
            )

        def classification(self, season, event, session):
            self.calls.append(session)
            return pd.DataFrame({"driver": ["VER"]})

    src = Fake()
    paths = ingest.ingest_season(src, 2025, root=tmp_path)
    assert src.calls == ["schedule", "Race", "Qualifying"]
    assert paths[0] == tmp_path / "fake/schedule/season=2025/event=all/session=all.parquet"
    assert len(cache.load("fake", "classification", 2025, "X GP", "Race", tmp_path)) == 1
