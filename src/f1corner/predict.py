"""Prediction (spec 15): qualifying gap to pole, driver offsets, race pace.

Team table `df` (one row per team x race, calendar order given by `round`):
  round, event, team, gap_pct (target), air_temp_c, track_temp_c, altitude_m
`prof`: round, team, cls, profile    `mix`: round, cls, share
Features for race r use only races < r (no leakage), except track mix and conditions, which are
known beforehand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HALF_LIFE = 3.0  # races, recent-pace weighting
SHRINK_K = 3.0  # driver offset pseudo-races pulling toward zero
WIN_MARGIN = 0.05  # LightGBM must beat ridge test RMSE by >5% to replace it
COND = ["air_temp_c", "track_temp_c", "altitude_m"]


def _wavg(values: np.ndarray, rounds: np.ndarray, now: float) -> float:
    w = 0.5 ** ((now - rounds) / HALF_LIFE)
    return float(np.average(values, weights=w))


def build_features(df: pd.DataFrame, prof: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    """Rows with >=1 prior race. Columns: recent_pace, prof_x_mix_<cls>..., conditions, target."""
    classes = sorted(prof["cls"].unique())
    mixw = (
        mix.pivot(index="round", columns="cls", values="share").reindex(columns=classes).fillna(0)
    )
    rows = []
    for r in df.itertuples():
        past = df[(df.team == r.team) & (df["round"] < r.round)]
        if past.empty:
            continue
        f = {
            "round": r.round,
            "event": r.event,
            "team": r.team,
            "gap_pct": r.gap_pct,
            "recent_pace": _wavg(
                past.gap_pct.to_numpy(float), past["round"].to_numpy(float), r.round
            ),
            **{c: getattr(r, c) for c in COND},
        }
        pp = prof[(prof.team == r.team) & (prof["round"] < r.round)]
        for c in classes:
            q = pp[pp.cls == c]
            p = (
                _wavg(q.profile.to_numpy(float), q["round"].to_numpy(float), r.round)
                if len(q)
                else 0.0
            )
            f[f"px_{c}"] = p * mixw.loc[r.round, c] if r.round in mixw.index else 0.0
        rows.append(f)
    return pd.DataFrame(rows)


def feature_cols(feat: pd.DataFrame) -> list[str]:
    return ["recent_pace", *[c for c in feat if c.startswith("px_")], *COND]


def select_model(feat: pd.DataFrame, test_rounds: int = 4) -> dict:
    """Fit ridge and LightGBM on early rounds, score on the last `test_rounds` (time split).

    Ridge stays unless LightGBM's test RMSE is lower by more than WIN_MARGIN.
    Returns {"model", "name", "rmse", "cols"}; model is refit on all rows.
    """
    from lightgbm import LGBMRegressor  # lazy: needs libomp on macOS (brew install libomp)

    cols = feature_cols(feat)
    cut = sorted(feat["round"].unique())[-test_rounds]
    tr, te = feat[feat["round"] < cut], feat[feat["round"] >= cut]
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))  # noqa: E731
    gbm = lambda: LGBMRegressor(  # noqa: E731
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=5,
        verbose=-1,
        random_state=0,
    )
    rmse = {}
    for name, mk in (("ridge", ridge), ("lightgbm", gbm)):
        m = mk().fit(tr[cols], tr.gap_pct)
        rmse[name] = float(np.sqrt(np.mean((m.predict(te[cols]) - te.gap_pct) ** 2)))
    name = "lightgbm" if rmse["lightgbm"] < rmse["ridge"] * (1 - WIN_MARGIN) else "ridge"
    model = (gbm if name == "lightgbm" else ridge)().fit(feat[cols], feat.gap_pct)
    return {"model": model, "name": name, "rmse": rmse, "cols": cols}


def driver_offsets(drv: pd.DataFrame, k: float = SHRINK_K) -> pd.Series:
    """Driver offset from the teammate gap so far, shrunk to zero with few races.

    `drv`: team, driver, gap_pct (qualifying gap to pole, past races only). Offset = driver gap
    minus team mean per race, averaged, times n/(n+k).
    """
    d = drv.copy()
    d["rel"] = d["gap_pct"] - d.groupby(["team", "round"])["gap_pct"].transform("mean")
    g = d.groupby(["team", "driver"])["rel"].agg(["mean", "count"])
    return (g["mean"] * g["count"] / (g["count"] + k)).rename("offset")


def predict_drivers(team_pred: pd.Series, offsets: pd.Series, drivers: pd.DataFrame) -> pd.Series:
    """driver gap = team prediction + shrunk offset. `drivers`: team, driver."""
    idx = pd.MultiIndex.from_frame(drivers[["team", "driver"]])
    off = offsets.reindex(idx).fillna(0.0).to_numpy()
    return pd.Series(team_pred.reindex(drivers["team"]).to_numpy() + off, index=idx, name="gap_pct")


# ---- race pace (G7) ----

TRAFFIC_S = 2.0  # gap to car ahead below this = in traffic
MIN_CLEAN = 5


def correct_laps(laps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean-air laps corrected for fuel and tyre age, with factors estimated from the data.

    `laps`: event, driver, stint, lap_number, tyre_age, lap_time_s, gap_ahead_s (pre-filtered for
    green/pit/etc. by `lapfilter`). Drops laps with gap_ahead_s < TRAFFIC_S (NaN gap = leader =
    clean). Fits lap_time ~ driver-race fixed effect (needs >=2 stints) + a*lap_number + b*tyre_age (a: fuel,
    s per lap of race, usually negative; b: tyre deg, s per lap of age) and returns
    `corrected_s` = lap_time - a*(lap_number - mean) - b*tyre_age... i.e. at mean fuel, fresh tyres.
    """
    d = laps[~(laps["gap_ahead_s"] < TRAFFIC_S)].copy()
    # Fixed effect per driver-race, not per stint: inside a stint lap number and tyre age are
    # collinear; stints starting at different laps/ages are what separate fuel from tyre.
    key = d["event"].astype(str) + "|" + d["driver"].astype(str)
    ok = d.groupby(key)["stint"].transform("nunique") >= 2
    d = d[ok & (key.map(key.value_counts()) >= MIN_CLEAN)].copy()
    key = key.loc[d.index]
    X = d[["lap_number", "tyre_age"]].astype(float)
    Xd = X - X.groupby(key).transform("mean")
    y = d["lap_time_s"] - d.groupby(key)["lap_time_s"].transform("mean")
    coef, *_ = np.linalg.lstsq(Xd.to_numpy(), y.to_numpy(), rcond=None)
    fuel, tyre = float(coef[0]), float(coef[1])
    d["corrected_s"] = (
        d["lap_time_s"] - fuel * (d["lap_number"] - d["lap_number"].mean()) - tyre * d["tyre_age"]
    )
    return d, {"fuel_s_per_lap": fuel, "tyre_s_per_age": tyre}


def race_pace(laps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per (event, driver): median corrected clean-air lap, and % over the event's fastest."""
    d, factors = correct_laps(laps)
    p = d.groupby(["event", "driver"], as_index=False)["corrected_s"].median()
    p["gap_pct"] = 100 * (p["corrected_s"] / p.groupby("event")["corrected_s"].transform("min") - 1)
    return p, factors
