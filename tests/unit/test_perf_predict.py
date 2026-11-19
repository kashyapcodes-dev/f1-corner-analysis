import numpy as np
import pandas as pd
import pytest

from f1corner import perf, predict
from f1corner import profile_tests as PT


def _times():
    rows = []
    for ev in ("a", "b"):
        for corner in (0, 1):
            for team, t in (("X", 2.0), ("Y", 2.2), ("Z", 2.0)):
                rows.append((ev, team, f"{team}1", corner, t * (1.1 if corner else 1)))
                rows.append((ev, team, f"{team}2", corner, t * 1.5))
    return pd.DataFrame(rows, columns=["event", "team", "driver", "corner", "time_s"])


def test_delta_uses_best_teammate_and_pct():
    d = perf.delta_pct(_times())
    y = d[(d.event == "a") & (d.corner == 0) & (d.team == "Y")].iloc[0]
    assert abs(y.delta_pct - 10.0) < 1e-9
    assert (d[d.team == "X"].delta_pct == 0).all()


def test_pace_profile_sign_and_mix():
    t = _times()
    t.loc[(t.team == "Y") & (t.corner == 1), "time_s"] *= 1.2  # Y extra slow in corner 1 only
    d = perf.delta_pct(t)
    cls = pd.DataFrame(
        [(e, c, "slow" if c == 0 else "fast") for e in "ab" for c in (0, 1)],
        columns=["event", "corner", "cls"],
    )
    p = perf.pace_profile(d, cls)
    z = p[(p.team == "Y") & (p.event == "a")].set_index("cls").profile
    assert z["slow"] < 0 < z["fast"]  # relatively stronger in slow, weaker in fast
    pole = pd.DataFrame({"event": ["a", "a"], "corner": [0, 1], "time_s": [3.0, 1.0]})
    m = perf.track_mix(pole, cls, pd.Series({"a": 10.0}))
    assert m.set_index("cls").share.to_dict() == {"slow": 0.3, "fast": 0.1}


def _profiles(corr: bool, n_ev=6, teams=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(teams, 2))
    rows = []
    for i in range(n_ev):
        for t in range(teams):
            for c in range(2):
                v = base[t, c] + rng.normal(0, 0.1) if corr else rng.normal()
                rows.append((f"e{i}", f"T{t}", f"c{c}", v))
    return pd.DataFrame(rows, columns=["event", "team", "cls", "profile"])


def test_test_a_detects_stability_and_normalises_teams():
    p25 = _profiles(True)
    p26 = _profiles(True, seed=1).assign(event=lambda d: d.event + "_26")
    p26["team"] = p26["team"].replace({"T0": "Audi"})
    p25["team"] = p25["team"].replace({"T0": "Kick Sauber"})
    p26 = pd.concat([p26, p26.iloc[:1].assign(team="Cadillac")])
    r = PT.test_a(p25, p26, n_boot=50, n_perm=200)
    assert r.n_teams.max() == 10  # Audi matched to Sauber, Cadillac dropped
    assert r.rho.notna().all() and (r.p_perm.fillna(1) <= 1).all()
    assert r.ci_lo.notna().all()


def test_test_b_and_rolling():
    p = _profiles(True, n_ev=8)
    order = [f"e{i}" for i in range(8)]
    b = PT.test_b(p, order, n_boot=20, n_perm=100)
    assert (b.rho > 0.5).all()
    r = PT.rolling(p, order, window=3)
    assert len(r) == 3 and (r.mean_rho > 0.5).all()


def _team_df(n=12):
    rng = np.random.default_rng(0)
    rows, prof, mix = [], [], []
    for r in range(1, n + 1):
        for k in range(10):
            rows.append((r, f"gp{r}", f"T{k}", k * 0.4 + rng.normal(0, 0.1), 20 + r, 30 + r, 100.0))
            for c in ("slow", "fast"):
                prof.append((r, f"T{k}", c, rng.normal()))
        for c, s in (("slow", 0.3), ("fast", 0.2)):
            mix.append((r, c, s))
    df = pd.DataFrame(rows, columns=["round", "event", "team", "gap_pct", *predict.COND])
    return (
        df,
        pd.DataFrame(prof, columns=["round", "team", "cls", "profile"]),
        pd.DataFrame(mix, columns=["round", "cls", "share"]),
    )


def test_features_no_leak_and_ridge_kept():
    df, prof, mix = _team_df()
    f = predict.build_features(df, prof, mix)
    assert f["round"].min() == 2
    r2 = f[(f["round"] == 2) & (f.team == "T3")].iloc[0]
    assert abs(r2.recent_pace - df[(df["round"] == 1) & (df.team == "T3")].gap_pct.iloc[0]) < 1e-9
    try:
        import lightgbm  # noqa: F401
    except OSError:
        pytest.skip("lightgbm needs libomp (brew install libomp)")
    m = predict.select_model(f)
    assert m["name"] == "ridge" and set(m["rmse"]) == {"ridge", "lightgbm"}
    assert m["model"].predict(f[m["cols"]]).shape == (len(f),)


def test_driver_offset_shrinks():
    d = pd.DataFrame(
        {
            "round": [1, 1] * 1,
            "team": ["X", "X"],
            "driver": ["A", "B"],
            "gap_pct": [0.0, 1.0],
        }
    )
    o = predict.driver_offsets(d, k=3)
    assert abs(o[("X", "A")] + 0.5 * 1 / 4) < 1e-9
    pred = predict.predict_drivers(
        pd.Series({"X": 1.0}), o, pd.DataFrame({"team": ["X", "X"], "driver": ["A", "Z"]})
    )
    assert pred.iloc[1] == 1.0  # unseen driver: no offset


def test_race_pace_recovers_fuel_and_tyre_and_drops_traffic():
    rng = np.random.default_rng(0)
    rows = []
    for drv, base in (("A", 90.0), ("B", 90.5)):
        for stint, start_lap, age0 in ((1, 1, 0), (2, 20, 3), (3, 40, 8)):
            for i in range(15):
                lap, age = start_lap + i, age0 + i
                t = base - 0.03 * lap + 0.05 * age + rng.normal(0, 0.02)
                rows.append(("gp", drv, stint, lap, age, t, 5.0))
    laps = pd.DataFrame(
        rows,
        columns=["event", "driver", "stint", "lap_number", "tyre_age", "lap_time_s", "gap_ahead_s"],
    )
    laps.loc[laps.index[:3], "gap_ahead_s"] = 0.5
    laps.loc[laps.index[:3], "lap_time_s"] += 3
    p, fac = predict.race_pace(laps)
    assert abs(fac["fuel_s_per_lap"] + 0.03) < 0.01 and abs(fac["tyre_s_per_age"] - 0.05) < 0.01
    assert p.set_index("driver").gap_pct["B"] > 0.3
