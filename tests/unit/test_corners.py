import numpy as np
import pandas as pd

from f1corner import corners as C
from f1corner.multiviewer import corners_frame, label_corners


def synth_car(hz=10.0, lap_s=60.0):
    """Straight at 300, hard corner at t=20 (min 80), lift-only corner at t=40 (min 290, throttle
    dips), flat-out kink at t=50 (speed -6 but throttle 100)."""
    t = np.arange(0, lap_s, 1 / hz)
    v = np.full_like(t, 300.0)
    thr = np.full_like(t, 100.0)
    brk = np.zeros(len(t), bool)

    def dip(c, w, depth):
        return depth * np.exp(-(((t - c) / w) ** 2))

    v -= dip(20, 2.5, 220) + dip(40, 1.5, 10) + dip(50, 1.0, 6)
    thr -= np.where(abs(t - 20) < 3.5, 100, 0)  # off throttle
    thr -= np.where(abs(t - 40) < 1.0, 40, 0)
    brk |= abs(t - 18) < 1.5
    return pd.DataFrame({"t_s": t, "speed_kph": v, "throttle": thr, "brake": brk})


def test_profile_length_and_rescale():
    car = synth_car()
    p = C.lap_profile(car)
    assert 4000 < p.length < 5500
    q = C.lap_profile(car, length=5000.0)
    assert abs(q.dist[-1] - 5000.0) <= C.GRID_M


def test_prominence_minima():
    y = np.array([10, 8, 5, 8, 9, 7, 9, 10], float)
    got = dict(C.prominence_minima(y))
    assert got[2] == 5.0  # never reaches lower ground: bounded by the lower of the two ends (10)
    assert got[5] == 2.0


def test_finds_speed_and_lift_corners_excludes_flat_out():
    p = C.lap_profile(synth_car())
    corners, flat = C.find_apexes(p)
    assert list(corners["kind"]) == ["speed", "lift"]
    assert corners["apex_speed_kph"].iloc[0] < 120
    # flat-out kink listed, not a corner
    assert len(flat) == 1 and flat["prominence_kph"].iloc[0] < C.MIN_DROP_KPH


def test_segment_spans_braking_to_full_throttle():
    car = synth_car()
    p = C.lap_profile(car)
    corners, _ = C.find_apexes(p)
    c = corners.iloc[0]
    assert c.start_m < c.apex_m < c.end_m
    i0 = np.searchsorted(p.dist, c.start_m)
    i1 = np.searchsorted(p.dist, c.end_m)
    assert p.throttle[i0 - 1] >= C.FULL_THROTTLE and p.throttle[i1] >= C.FULL_THROTTLE


def test_sequences_flag_overlap_and_touch():
    df = pd.DataFrame(
        {
            "apex_m": [100, 150, 300, 400],
            "start_m": [50, 50, 280, 320],
            "end_m": [200, 200, 320, 500],
        }
    )
    out = C.add_sequences(df)
    assert list(out["in_sequence"]) == [True, True, True, True]  # 300's end 320 touches 400's start
    assert out["sequence_id"].iloc[0] == out["sequence_id"].iloc[1]
    assert out["sequence_id"].iloc[2] == out["sequence_id"].iloc[3]
    assert out["sequence_id"].iloc[0] != out["sequence_id"].iloc[2]
    alone = C.add_sequences(df.iloc[[0, 3]].reset_index(drop=True))
    assert not alone["in_sequence"].any()


def test_segment_lap_applies_fixed_markers_to_other_lap():
    ref = C.lap_profile(synth_car())
    corners, _ = C.find_apexes(ref)
    slower = synth_car().assign(speed_kph=lambda d: d.speed_kph * 0.97)
    other = C.lap_profile(slower, length=ref.length)
    m = C.segment_lap(other, corners)
    assert m["min_kph"].iloc[0] < ref.speed.min() + 1
    assert (m["entry_kph"] > m["min_kph"]).all()
    assert abs(m["min_at_m"].iloc[0] - corners["apex_m"].iloc[0]) < 80


def test_label_nearest_multiviewer():
    mv = corners_frame(
        {
            "corners": [
                {"number": 1, "trackPosition": {"x": 0, "y": 0}},
                {"number": 11, "letter": "a", "trackPosition": {"x": 1000, "y": 0}},
            ]
        }
    )
    got = label_corners(np.array([50, 990, 5000]), np.array([0, 10, 0]), mv, max_dist=600)
    assert got == ["1", "11a", None]
