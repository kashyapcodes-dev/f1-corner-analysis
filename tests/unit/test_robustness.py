import numpy as np
import pandas as pd

from f1corner import perf
from f1corner import robustness as R
from f1corner.features import FEATURES

TEAMS = [f"T{i}" for i in range(10)]
CENTRES = {"slow": (70, 120, 4.0, 2), "medium": (150, 90, 2.5, 4), "fast": (250, 40, 1.2, 7)}


def _inputs(n_ev=(4, 6), seed=0):
    """Two seasons, 3 well-separated corner types x 3 corners, stable team strengths per type."""
    rng = np.random.default_rng(seed)
    strength = {t: {c: rng.normal(0, 1) for c in CENTRES} for t in TEAMS}
    base = {t: i * 0.3 for i, t in enumerate(TEAMS)}
    times, corners, season, order = [], [], {}, []
    for s, n in zip((2025, 2026), n_ev):
        for e in range(n):
            ev = f"{s}-{e}"
            season[ev] = s
            order.append(ev)
            k = 0
            for cls, (v, loss, dt, g) in CENTRES.items():
                for _ in range(3):
                    f = rng.normal([v, loss, dt, g], [3, 3, 0.05, 0.01])
                    corners.append((ev, k, *f))
                    for t in TEAMS:
                        pct = base[t] + 0.5 * strength[t][cls] + rng.normal(0, 0.05)
                        for d, part, slow in ((1, 3, 0.0), (2, 1, 0.5)):
                            ts = dt * (1 + (pct + slow) / 100)
                            times.append((ev, t, f"{t}{d}", k, ts, v * (1 - pct / 100), part))
                    k += 1
    times = pd.DataFrame(
        times,
        columns=["event", "team", "driver", "corner", "time_s", "min_speed_kph", "q_part"],
    )
    corners = pd.DataFrame(corners, columns=["event", "corner", *FEATURES])
    imp = pd.DataFrame({"event": order, "f12": 0.996, "f23": 0.998})
    return R.Inputs(times, corners, season, order, times_q1=times, improvement=imp)


def test_team_delta_matches_perf_and_alternatives():
    t = _inputs().times
    a = R.team_delta(t).sort_values(["event", "corner", "team"]).reset_index(drop=True)
    b = perf.delta_pct(t).sort_values(["event", "corner", "team"]).reset_index(drop=True)
    assert np.allclose(a.delta_pct, b.delta_pct)
    m = R.team_delta(t, teammate="mean")
    assert (m.delta_pct >= -1e-12).all()
    v = R.team_delta(t, metric="min_speed")
    assert (v.delta_pct >= -1e-12).all() and (
        v.groupby(["event", "corner"]).delta_pct.min() == 0
    ).all()


def test_weighted_pace_uses_segment_time():
    d = pd.DataFrame({"event": "e", "corner": [0, 1], "team": "X", "delta_pct": [1.0, 3.0]})
    cls = pd.DataFrame({"event": "e", "corner": [0, 1], "cls": ["a", "b"]})
    w = pd.DataFrame({"event": "e", "corner": [0, 1], "w": [3.0, 1.0]})
    p = R.pace_profile(d, cls, "wmean", w)
    assert np.isclose(p.pace.iloc[0], 1.5)
    assert np.isclose(R.pace_profile(d, cls).pace.iloc[0], 2.0)


def test_improvement_qpart_and_correction():
    c = pd.DataFrame(
        {
            "event": "e",
            "driver": ["A", "B", "C"],
            "q1_s": [80.0, 80.5, 81.0],
            "q2_s": [79.6, 80.1, np.nan],
            "q3_s": [79.4, np.nan, np.nan],
        }
    )
    imp = R.improvement(c)
    r = imp.iloc[0]
    assert r.n12 == 2 and r.n23 == 1 and r.pct12 < 0 and np.isclose(r.f23, 79.4 / 79.6)
    laps = pd.DataFrame({"driver": ["A", "A", "B", "C"], "lap_time_s": [79.4, 80.0, 80.1, 85.0]})
    assert R.q_part(laps, c).tolist() == [3, 1, 2, pd.NA]
    t = pd.DataFrame({"event": "e", "time_s": [1.0, 1.0, 1.0], "q_part": [1, 3, pd.NA]})
    out = R.correct_for_improvement(t, imp)
    assert out.time_s[0] == 1.0 and out.time_s[2] == 1.0
    assert np.isclose(out.time_s[1], 1 / (r.f12 * r.f23))


def test_run_flags_and_stable_defaults():
    inp = _inputs()
    base, flags = R.run(inp, n_boot=20, n_perm=100)
    assert base["k"] == 3 and base["test_a"] is not None and base["test_b"] is not None
    assert set(flags.variant) == {v for _, v, _ in R.VARIANTS}
    assert flags.result.isin(["pace", "profile", "test_a", "test_b", "predict", "-"]).all()
    # stable synthetic strengths: k-means at the same k should reproduce the clustering headline
    km = flags[(flags.variant == "k-means (same k)") & flags.result.isin(["pace", "profile"])]
    assert not km.changed.any()
    # different k always changes the class set -> flagged
    assert flags[flags.variant == "GMM k+1"].changed.any()
    # no prediction inputs -> reported as not computable, never as a number
    pr = flags[flags.result == "predict"]
    assert (pr["default"] == "not computable").all() and not pr.changed.any()


def test_compare_detects_flip():
    inp = _inputs()
    base = R.headline(inp, R.DEFAULT, n_boot=10, n_perm=50)
    alt = dict(base, season_profile=-base["season_profile"])
    alt["pace"] = -base["pace"]
    c = R.compare(base, alt).set_index("result")
    assert c.loc["pace", "changed"] and c.loc["profile", "changed"]


def test_skip_when_inputs_missing():
    inp = _inputs()
    inp.times_q1 = None
    assert "skip" in R.headline(inp, R.Config(improvement="q1_only"))


def test_apex_error_on_synthetic_lap():
    from f1corner import corners as C
    from tests.unit.test_corners import synth_car

    car = synth_car(hz=3.7).assign(gear=4)
    ref = C.lap_profile(car)
    corners, _ = C.find_apexes(ref)
    e = R.apex_error([(car, corners, ref.length)])
    assert len(e) == len(corners)
    assert (e.parabola_minus_raw.dropna() <= 1e-9).all() and (e.half_rate_err.dropna() >= 0).all()
