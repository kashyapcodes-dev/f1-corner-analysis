"""Offline unit tests for OpenF1 payload -> canonical mapping (tiny synthetic data, no network)."""
import pandas as pd
import pytest

from f1corner.schema import CAR_DATA_COLS, LAPS_COLS, RACE_CONTROL_COLS, WEATHER_COLS
from f1corner.sources import openf1_source as o
from f1corner.timeutil import epoch_s

T0 = "2025-05-24T14:00:00+00:00"


def ts(sec):
    return (pd.Timestamp(T0) + pd.to_timedelta(float(sec), unit="s")).isoformat()


def rc_row(sec, category="Flag", flag=None, scope=None, sector=None, message=""):
    return dict(date=ts(sec), category=category, flag=flag, scope=scope, sector=sector, message=message)


DRIVERS = pd.DataFrame([dict(driver_number=1, name_acronym="VER", team_name="Red Bull Racing"),
                        dict(driver_number=4, name_acronym="NOR", team_name="McLaren")])


def test_deleted_laps_parses_both_message_forms():
    rc = pd.DataFrame([
        rc_row(1, "Other", message="CAR 10 (GAS) TIME 1:38.777 DELETED - TRACK LIMITS AT TURN 1 LAP 4 16:03:24"),
        rc_row(2, "Other", message="CAR 31 (OCO) LAP DELETED - TRACK LIMITS AT TURN 10 LAP 14 16:18:11 (PIT)"),
        rc_row(3, "Other", message="RISK OF RAIN FOR F1 QUALIFYING SESSION IS 20%"),
    ])
    assert o.deleted_laps(rc) == {("GAS", 4), ("OCO", 14)}


def test_non_green_sector_yellow_then_clear():
    rc = pd.DataFrame([rc_row(10, flag="DOUBLE YELLOW", scope="Sector", sector=13),
                       rc_row(20, flag="CLEAR", scope="Sector", sector=13)])
    (a, b), = o.non_green_intervals(rc)
    assert b - a == pytest.approx(10)


def test_non_green_red_closed_by_track_clear_and_open_ended_yellow():
    rc = pd.DataFrame([rc_row(5, flag="RED", scope="Track"),
                       rc_row(30, flag="CLEAR", scope="Track"),
                       rc_row(40, flag="YELLOW", scope="Sector", sector=2)])
    iv = sorted(o.non_green_intervals(rc))
    assert iv[0][1] - iv[0][0] == pytest.approx(25)
    assert iv[1][1] == float("inf")


def test_non_green_safety_car():
    rc = pd.DataFrame([rc_row(5, "SafetyCar", message="SAFETY CAR DEPLOYED"),
                       rc_row(50, "SafetyCar", message="SAFETY CAR IN THIS LAP")])
    (a, b), = o.non_green_intervals(rc)
    assert b - a == pytest.approx(45)


def _laps():
    # VER: lap1 out-lap, lap2 flying (deleted, yellow overlaps), lap3 in-lap-ish; NOR lap1 clean
    return pd.DataFrame([
        dict(driver_number=1, lap_number=1, date_start=ts(0), lap_duration=90.0, duration_sector_1=30.0,
             duration_sector_2=30.0, duration_sector_3=30.0, is_pit_out_lap=True),
        dict(driver_number=1, lap_number=2, date_start=ts(90), lap_duration=70.0, duration_sector_1=20.0,
             duration_sector_2=30.0, duration_sector_3=20.0, is_pit_out_lap=False),
        dict(driver_number=1, lap_number=3, date_start=ts(160), lap_duration=None, duration_sector_1=None,
             duration_sector_2=None, duration_sector_3=None, is_pit_out_lap=False),
        dict(driver_number=1, lap_number=4, date_start=None, lap_duration=None, duration_sector_1=None,
             duration_sector_2=None, duration_sector_3=None, is_pit_out_lap=True),
        dict(driver_number=4, lap_number=1, date_start=ts(0), lap_duration=69.9, duration_sector_1=20.0,
             duration_sector_2=30.0, duration_sector_3=19.9, is_pit_out_lap=False),
    ])


STINTS = pd.DataFrame([dict(driver_number=1, stint_number=1, lap_start=1, lap_end=2, compound="SOFT", tyre_age_at_start=0),
                       dict(driver_number=1, stint_number=2, lap_start=3, lap_end=4, compound="MEDIUM", tyre_age_at_start=2),
                       dict(driver_number=4, stint_number=1, lap_start=1, lap_end=1, compound="SOFT", tyre_age_at_start=0)])

RC = pd.DataFrame([
    rc_row(100, flag="YELLOW", scope="Sector", sector=3), rc_row(110, flag="CLEAR", scope="Sector", sector=3),
    rc_row(120, "Other", message="CAR 1 (VER) TIME 1:10.000 DELETED - TRACK LIMITS AT TURN 1 LAP 2 14:00:00"),
])


@pytest.fixture
def laps():
    return o.map_laps(_laps(), DRIVERS, STINTS, RC, 2025, "Monaco Grand Prix", "Qualifying")


def test_map_laps_columns_and_identity(laps):
    assert list(laps.columns) == LAPS_COLS
    assert set(laps["driver"]) == {"VER", "NOR"}
    assert laps.loc[laps.driver == "NOR", "team"].iloc[0] == "McLaren"


def test_map_laps_pit_flags(laps):
    ver = laps[laps.driver == "VER"].set_index("lap_number")
    assert ver.loc[1, "pit_out"] and not ver.loc[2, "pit_out"]
    assert ver.loc[3, "pit_in"]  # next lap (4) is a pit-out lap => lap 3 is an in-lap
    assert not ver.loc[1, "pit_in"]


def test_map_laps_deleted_and_green(laps):
    ver = laps[laps.driver == "VER"].set_index("lap_number")
    assert ver.loc[2, "deleted"] and not ver.loc[1, "deleted"]
    assert ver.loc[2, "green"] == False  # noqa: E712  yellow 100-110 inside lap 2 (90-160)
    assert ver.loc[1, "green"]
    assert pd.isna(ver.loc[4, "green"])  # no timestamps -> unknown, not guessed


def test_map_laps_tyres(laps):
    ver = laps[laps.driver == "VER"].set_index("lap_number")
    assert ver.loc[1, "compound"] == "SOFT" and ver.loc[1, "tyre_age"] == 1 and ver.loc[2, "tyre_age"] == 2
    assert ver.loc[3, "compound"] == "MEDIUM" and ver.loc[3, "tyre_age"] == 3  # 2 laps old at start -> 3rd lap
    assert ver.loc[4, "stint"] == 2


def test_map_laps_empty():
    out = o.map_laps(pd.DataFrame(), DRIVERS, STINTS, RC, 2025, "x", "y")
    assert list(out.columns) == LAPS_COLS and out.empty


def _samples(start_sec, n, dt=0.27, speed0=100):
    return pd.DataFrame(dict(date=[ts(start_sec + i * dt) for i in range(n)], speed=[speed0 + i for i in range(n)],
                             throttle=[100] * n, brake=[0] * (n - 1) + [100], n_gear=[5] * n, rpm=[10000] * n))


def test_car_data_sliced_per_lap_with_lap_relative_time():
    # samples span lap 1 (0-90) and lap 2 (90-160); lap 3 has no samples
    s = pd.concat([_samples(0.1, 5), _samples(90.1, 4, speed0=200)], ignore_index=True)
    out = o.map_car_data({1: s}, _laps(), DRIVERS, 2025, "Monaco Grand Prix", "Qualifying")
    assert list(out.columns) == CAR_DATA_COLS
    l1, l2 = out[out.lap_number == 1], out[out.lap_number == 2]
    assert len(l1) == 5 and len(l2) == 4 and set(out.lap_number) == {1, 2}
    assert l1["t_s"].iloc[0] == pytest.approx(0.1) and l2["t_s"].iloc[0] == pytest.approx(0.1)
    assert l2["speed_kph"].iloc[0] == 200
    assert out["brake"].dtype == bool and out["brake"].sum() == 2
    assert (out["driver"] == "VER").all()


def test_car_data_empty_driver_ignored():
    out = o.map_car_data({1: pd.DataFrame()}, _laps(), DRIVERS, 2025, "e", "s")
    assert out.empty and list(out.columns) == CAR_DATA_COLS


def test_weather_and_race_control_use_utc_epoch():
    w = pd.DataFrame([dict(date=ts(0), air_temperature=21.0, track_temperature=40.0, rainfall=0),
                      dict(date=ts(60), air_temperature=21.5, track_temperature=39.0, rainfall=1)])
    out = o.map_weather(w, 2025, "e", "s")
    assert list(out.columns) == WEATHER_COLS
    assert out["t_s"].iloc[0] == pytest.approx(epoch_s(pd.Series([T0])).iloc[0])
    assert list(out["rain"]) == [False, True]
    rc = RC.assign(driver_number=[None, None, 1])
    r = o.map_race_control(rc, DRIVERS, 2025, "e", "s")
    assert list(r.columns) == RACE_CONTROL_COLS and r["driver"].iloc[2] == "VER"


def test_throttle_enforces_per_second_limit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(o.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(o.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    th = o.Throttle(per_s=3, per_min=30)
    stamps = []
    for _ in range(7):
        th.wait()
        stamps.append(now[0])
    assert all(sum(1 for t in stamps if x <= t < x + 1) <= 3 for x in stamps)


def test_epoch_s_handles_mixed_fractional_seconds():
    """Regression: OpenF1 mixes '14:00:00+00:00' and '14:00:00.074000+00:00' in one column."""
    s = pd.Series(["2025-05-24T14:00:00+00:00", "2025-05-24T14:00:00.074000+00:00"])
    a, b = epoch_s(s)
    assert b - a == pytest.approx(0.074, abs=1e-5)  # float epoch seconds ~0.2 us resolution
