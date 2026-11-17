"""Live integration tests for ONE race: 2025 Monaco GP Qualifying. Hit real APIs; not part of the default run.

Run: uv run pytest tests/live    (OpenF1 needs no sign-in; first run takes ~2 min because of rate limits)
FastF1 tests also need F1_USERNAME / F1_PASSWORD (F1TV Access) and are skipped otherwise.
Fetched data is cached under data/cache/test (gitignored), so re-runs are fast.

Expected values below were checked against the data itself: pole = Norris 1:09.954 on lap 26.
"""
from pathlib import Path

import numpy as np
import pytest

from f1corner import cache
from f1corner.schema import TABLES
from f1corner.sources import get_source
from f1corner.sources.fastf1_source import has_credentials

SEASON, EVENT, SESSION = 2025, "Monaco Grand Prix", "Qualifying"
LIVE_CACHE = Path("data/cache/test")  # gitignored; keeps repeated live runs cheap

pytestmark = pytest.mark.live
ALL_TABLES = ["laps", "car_data", "location", "weather", "race_control"]


def _load(source_name, table):
    src = get_source(source_name)
    cache.fetch(src, table, SEASON, EVENT, SESSION, root=LIVE_CACHE)
    return cache.load(source_name, table, SEASON, EVENT, SESSION, root=LIVE_CACHE)


@pytest.fixture(scope="module", params=["openf1", pytest.param("fastf1", marks=pytest.mark.skipif(
    not has_credentials(), reason="needs F1_USERNAME/F1_PASSWORD (F1TV)"))])
def tables(request):
    return request.param, {t: _load(request.param, t) for t in ALL_TABLES}


# ---- invariants every backend must satisfy ----------------------------------------------------


def test_schema_and_nonempty(tables):
    _, t = tables
    for name, df in t.items():
        assert list(df.columns) == TABLES[name], name
        assert len(df) > 0, name


def test_twenty_drivers_two_per_team(tables):
    _, t = tables
    laps = t["laps"]
    assert laps["driver"].nunique() == 20
    assert laps.groupby("team")["driver"].nunique().eq(2).all()
    assert {"NOR", "PIA", "VER", "LEC", "HAM"} <= set(laps["driver"])


def test_pole_lap_is_norris_1_09_954(tables):
    _, t = tables
    laps = t["laps"].dropna(subset=["lap_time_s"])
    valid = laps[~laps["deleted"].fillna(False).astype(bool)]
    best = valid.loc[valid["lap_time_s"].idxmin()]
    assert best["driver"] == "NOR" and best["lap_number"] == 26
    assert best["lap_time_s"] == pytest.approx(69.954, abs=0.002)


def test_sectors_sum_to_lap_time_on_pole_lap(tables):
    _, t = tables
    l = t["laps"].query("driver == 'NOR' and lap_number == 26").iloc[0]
    assert l["s1_s"] + l["s2_s"] + l["s3_s"] == pytest.approx(l["lap_time_s"], abs=0.002)


def test_known_deleted_laps_flagged(tables):
    """Race control: GAS lap 4 (1:38.777) and RUS lap 6 (1:33.247) deleted for track limits."""
    _, t = tables
    d = t["laps"].set_index(["driver", "lap_number"])["deleted"]
    assert d[("GAS", 4)] and d[("RUS", 6)]
    assert not d[("NOR", 26)]
    assert t["laps"]["deleted"].fillna(False).astype(bool).sum() >= 10


def test_pit_flags_consistent(tables):
    _, t = tables
    laps = t["laps"]
    assert laps["pit_out"].astype(bool).sum() >= 20          # every driver leaves the pits at least once
    pole = laps.query("driver == 'NOR' and lap_number == 26").iloc[0]
    assert not pole["pit_out"] and not pole["pit_in"]        # pole lap is a flying lap


def test_pole_lap_telemetry(tables):
    _, t = tables
    c = t["car_data"].query("driver == 'NOR' and lap_number == 26").sort_values("t_s")
    assert 200 < len(c) < 340                                # ~70 s at ~3.7 Hz
    assert c["t_s"].iloc[0] < 1.0 and 69 < c["t_s"].iloc[-1] < 71
    dt = np.diff(c["t_s"])
    assert (dt > 0).all()
    assert 0.2 < np.median(dt) < 0.35                        # ~3.7 Hz
    assert 40 < c["speed_kph"].min() < 90                    # Monaco hairpin
    assert 250 < c["speed_kph"].max() < 340
    assert c["gear"].between(1, 8).all() and c["throttle"].between(0, 100).all()
    assert c["brake"].dtype == bool and c["brake"].any() and (~c["brake"]).any()


def test_pole_lap_location_moves_around_track(tables):
    _, t = tables
    p = t["location"].query("driver == 'NOR' and lap_number == 26").sort_values("t_s")
    assert len(p) > 200
    path = np.hypot(np.diff(p["x"]), np.diff(p["y"])).sum()
    assert path > 2000                                       # a lap is ~3.3 km; units differ per source so keep loose


def test_weather_dry(tables):
    _, t = tables
    w = t["weather"]
    assert w["track_temp_c"].between(30, 50).all()
    assert w["air_temp_c"].between(15, 30).all()
    assert not w["rain"].astype(bool).any()                  # dry session: passes lap-filter rule 6


def test_race_control_has_chequered_flag_and_utc_times(tables):
    _, t = tables
    rc = t["race_control"]
    assert rc["message"].str.contains("CHEQUERED", na=False).any()
    assert rc["t_s"].between(1.7e9, 1.9e9).all()             # epoch seconds, 2023-2030
    assert t["laps"]["lap_start_s"].dropna().between(1.7e9, 1.9e9).all()


# ---- cross-source agreement (only when both ran) -----------------------------------------------


@pytest.mark.skipif(not has_credentials(), reason="needs F1TV creds")
def test_openf1_and_fastf1_agree():
    a = _load("openf1", "laps").set_index(["driver", "lap_number"])
    b = _load("fastf1", "laps").set_index(["driver", "lap_number"])
    j = a[["lap_time_s", "deleted"]].join(b[["lap_time_s", "deleted"]], lsuffix="_o", rsuffix="_f", how="inner").dropna(subset=["lap_time_s_o", "lap_time_s_f"])
    assert len(j) > 300
    assert (j["lap_time_s_o"] - j["lap_time_s_f"]).abs().max() < 0.01
    assert (j["deleted_o"].astype(bool) == j["deleted_f"].astype(bool)).mean() > 0.97
    ca = _load("openf1", "car_data").query("driver == 'NOR' and lap_number == 26")
    cb = _load("fastf1", "car_data").query("driver == 'NOR' and lap_number == 26")
    assert abs(ca["speed_kph"].max() - cb["speed_kph"].max()) < 5
    assert abs(len(ca) - len(cb)) < 15
