"""Robustness checks: re-run the headline results under each alternative and flag changes.

Checks (default -> alternatives):
  improvement  each driver's best lap -> corrected for measured Q1->Q2->Q3 improvement | Q1 laps only
  teammate     better teammate per corner -> mean of the two teammates
  method       Gaussian mixture (k by BIC) -> k-means (same k) | GMM k-1 | GMM k+1
  metric       time through the segment -> minimum speed (parabola apex estimate)
  pace         median delta -> mean weighted by segment time (fastest car's time there)

Headline results (each computed under default and every alternative):
  pace      team pace order per event            changed if Spearman < PACE_RHO_MIN or P1 differs
  profile   season profile sign per (team, cls)   changed if > PROFILE_FLIP_MAX of signs flip
  test_a/b  rho per class                         changed if sign/significance flips or |d rho| > RHO_TOL
  predict   model choice + ridge test RMSE        changed if choice flips or RMSE moves > RMSE_TOL
Results that cannot be computed from the inputs (e.g. no 2026 events) are reported as gaps.

Inputs use the `perf` long-table conventions; `event` keys must be unique across seasons.

python -m f1corner.robustness   -> runs on whatever is cached locally, writes docs/robustness.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from . import cluster as K
from . import perf, predict
from . import profile_tests as PT
from .features import FEATURES

PACE_RHO_MIN = 0.9
PROFILE_FLIP_MAX = 0.2
RHO_TOL = 0.2
ALPHA = 0.05
RMSE_TOL = 0.10
Q_TOL_S = 0.0015  # lap time vs classification Q time match
MIN_EVENTS_A = 2  # per season
MIN_EVENTS_B = 4
MIN_ROUNDS_PRED = 6


@dataclass(frozen=True)
class Config:
    improvement: str = "none"  # none | correct | q1_only
    teammate: str = "best"  # best | mean
    method: str = "gmm"  # gmm | kmeans
    k_offset: int = 0  # relative to the BIC-chosen k
    metric: str = "time"  # time | min_speed
    pace: str = "median"  # median | wmean


DEFAULT = Config()
VARIANTS = [
    ("track improvement", "corrected Q1->Q3", Config(improvement="correct")),
    ("track improvement", "Q1 laps only", Config(improvement="q1_only")),
    ("teammate rule", "mean of teammates", Config(teammate="mean")),
    ("clustering", "k-means (same k)", Config(method="kmeans")),
    ("clustering", "GMM k-1", Config(k_offset=-1)),
    ("clustering", "GMM k+1", Config(k_offset=1)),
    ("corner metric", "minimum speed", Config(metric="min_speed")),
    ("pace number", "time-weighted mean", Config(pace="wmean")),
]


@dataclass
class Inputs:
    """times: event, team, driver, corner, time_s [, min_speed_kph, q_part] (best lap per driver).
    corners: event, corner, FEATURES (reference-lap features, for clustering).
    season: event -> season; order: all events in calendar order.
    Optional: times_q1 (same schema, Q1 laps), improvement (from `improvement`),
    quali (round, event, team, gap_pct, predict.COND), pole_times (event, corner, time_s), lap_s.
    """

    times: pd.DataFrame
    corners: pd.DataFrame
    season: dict
    order: list
    times_q1: pd.DataFrame | None = None
    improvement: pd.DataFrame | None = None
    quali: pd.DataFrame | None = None
    pole_times: pd.DataFrame | None = None
    lap_s: pd.Series | None = None
    notes: list = field(default_factory=list)


# ---- track improvement -------------------------------------------------------------------------


def improvement(classif: pd.DataFrame) -> pd.DataFrame:
    """Measured improvement per event from official Q times, paired by driver (removes the
    selection effect of slower cars dropping out). f12 = median(q2/q1), f23 = median(q3/q2).

    Confounded with drivers/teams turning engines up and taking more risk in later parts.
    Returns event, n12, f12, pct12, n23, f23, pct23 (pct < 0 = faster later).
    """
    rows = []
    for ev, g in classif.groupby("event", sort=False):
        q = g[["q1_s", "q2_s", "q3_s"]].astype(float)
        r12 = (q.q2_s / q.q1_s).dropna()
        r23 = (q.q3_s / q.q2_s).dropna()
        f12 = float(r12.median()) if len(r12) else np.nan
        f23 = float(r23.median()) if len(r23) else np.nan
        rows.append((ev, len(r12), f12, 100 * (f12 - 1), len(r23), f23, 100 * (f23 - 1)))
    return pd.DataFrame(rows, columns=["event", "n12", "f12", "pct12", "n23", "f23", "pct23"])


def q_part(laps: pd.DataFrame, classif: pd.DataFrame, tol: float = Q_TOL_S) -> pd.Series:
    """Q part (1/2/3) of each lap whose time matches the driver's official Q1/Q2/Q3 time, else NA."""
    c = classif.set_index("driver")[["q1_s", "q2_s", "q3_s"]].astype(float)
    out = pd.Series(pd.NA, index=laps.index, dtype="Int64")
    for part, col in ((1, "q1_s"), (2, "q2_s"), (3, "q3_s")):
        ref = laps["driver"].map(c[col])
        hit = (laps["lap_time_s"].astype(float) - ref).abs() <= tol
        out = out.mask(hit & out.isna(), part)
    return out


def correct_for_improvement(times: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    """Scale each lap's corner times to its Q1 equivalent: Q2 lap / f12, Q3 lap / (f12*f23).
    Laps with unknown q_part (or missing factors) are left as is."""
    t = times.merge(imp[["event", "f12", "f23"]], on="event", how="left")
    part = t["q_part"].astype("Float64")
    fac = pd.Series(1.0, index=t.index)
    fac = fac.mask(part.eq(2).fillna(False), 1 / t["f12"])
    fac = fac.mask(part.eq(3).fillna(False), 1 / (t["f12"] * t["f23"]))
    t["time_s"] = t["time_s"] * fac.fillna(1.0)
    return t.drop(columns=["f12", "f23"])


# ---- deltas, pace, profile ---------------------------------------------------------------------


def team_delta(times: pd.DataFrame, teammate: str = "best", metric: str = "time") -> pd.DataFrame:
    """% deficit per (event, team, corner) to the best team there (>= 0, higher = worse).

    time: % slower through the segment; min_speed: % lower apex speed.
    best/time reproduces `perf.delta_pct`. Returns event, corner, team, delta_pct.
    """
    keys = ["event", "corner"]
    if metric == "time":
        col, agg, top = "time_s", ("min" if teammate == "best" else "mean"), "min"
    else:
        col, agg, top = "min_speed_kph", ("max" if teammate == "best" else "mean"), "max"
    t = times.dropna(subset=[col]).groupby([*keys, "team"], as_index=False)[col].agg(agg)
    ref = t.groupby(keys)[col].transform(top)
    t["delta_pct"] = 100 * (t[col] / ref - 1) if metric == "time" else 100 * (1 - t[col] / ref)
    return t[[*keys, "team", "delta_pct"]]


def segment_weights(times: pd.DataFrame) -> pd.DataFrame:
    """Weight per (event, corner) = fastest car's time through it."""
    return (
        times.groupby(["event", "corner"], as_index=False)["time_s"]
        .min()
        .rename(columns={"time_s": "w"})
    )


def pace_profile(
    delta: pd.DataFrame, classes: pd.DataFrame, pace: str = "median", weights=None
) -> pd.DataFrame:
    """`perf.pace_profile` with the pace number as median (default) or time-weighted mean."""
    if pace == "median":
        return perf.pace_profile(delta, classes)
    d = delta.merge(classes, on=["event", "corner"]).merge(weights, on=["event", "corner"])
    d["wd"] = d["delta_pct"] * d["w"]
    g = d.groupby(["event", "team"])
    d["pace"] = g["wd"].transform("sum") / g["w"].transform("sum")
    d["rel"] = d["delta_pct"] - d["pace"]
    out = d.groupby(["event", "team", "cls"], as_index=False).agg(
        pace=("pace", "first"), profile=("rel", "mean")
    )
    return out[["event", "team", "pace", "cls", "profile"]]


# ---- clustering --------------------------------------------------------------------------------


def classes_for(
    corners: pd.DataFrame, ref_events, method="gmm", k=None
) -> tuple[pd.DataFrame, int]:
    """Fit on reference-season corners (frozen), label all corners. k=None: BIC (GMM).

    Returns (event, corner, cls) and the k used.
    """
    x = corners.dropna(subset=FEATURES).reset_index(drop=True)
    ref = x[x["event"].isin(list(ref_events))].reset_index(drop=True)
    if method == "gmm":
        model, _, _ = K.fit(ref, k)
        lab, k = K.assign(model, x), model.k
    else:
        z, mean, std = K._z(ref[FEATURES].to_numpy(float))
        km = KMeans(k, n_init=K.N_INIT, random_state=K.SEED).fit(z)
        names = K.name_clusters(ref, km.labels_)
        ids = km.predict(K._z(x[FEATURES].to_numpy(float), mean, std)[0])
        lab = np.array([names[i] for i in ids])
    return x[["event", "corner"]].assign(cls=lab), int(k)


# ---- headline results --------------------------------------------------------------------------


def _ref_events(inp: Inputs):
    s = pd.Series(inp.season)
    return list(s[s == s.min()].index)


def headline(inp: Inputs, cfg: Config, k0: int | None = None, n_boot=PT.N_BOOT, n_perm=PT.N_PERM):
    """All headline results under `cfg`. k0 = default (BIC) k, needed for k-means / k +/- 1.

    Returns dict: k, pace, profile, season_profile, test_a, test_b, predict (None = not computable)
    and `skip` (reason string) if the variant itself cannot run.
    """
    times = inp.times
    if cfg.improvement == "correct":
        if inp.improvement is None or "q_part" not in times:
            return {"skip": "needs measured improvement + q_part of each lap"}
        times = correct_for_improvement(times, inp.improvement)
    elif cfg.improvement == "q1_only":
        if inp.times_q1 is None:
            return {"skip": "needs Q1-lap corner times"}
        times = inp.times_q1
    if cfg.metric == "min_speed" and "min_speed_kph" not in times:
        return {"skip": "needs min_speed_kph per lap"}

    k = None if (cfg.method == "gmm" and cfg.k_offset == 0) else max(2, (k0 or 2) + cfg.k_offset)
    classes, k = classes_for(inp.corners, _ref_events(inp), cfg.method, k)
    delta = team_delta(times, cfg.teammate, cfg.metric)
    prof = pace_profile(delta, classes, cfg.pace, segment_weights(times))
    out = {
        "k": k,
        "pace": prof.groupby(["event", "team"])["pace"].first(),
        "profile": prof,
        "season_profile": prof.groupby(["team", "cls"])["profile"].mean(),
        "test_a": None,
        "test_b": None,
        "predict": None,
    }

    season = prof["event"].map(inp.season)
    seasons = sorted(season.dropna().unique())
    if len(seasons) >= 2:
        a, b = (prof[season == s] for s in seasons[-2:])
        if a["event"].nunique() >= MIN_EVENTS_A and b["event"].nunique() >= MIN_EVENTS_A:
            out["test_a"] = PT.test_a(a, b, n_boot=n_boot, n_perm=n_perm)
    last = [e for e in inp.order if inp.season.get(e) == seasons[-1]] if seasons else []
    if len(last) >= MIN_EVENTS_B:
        out["test_b"] = PT.test_b(
            prof[prof["event"].isin(last)], last, n_boot=n_boot, n_perm=n_perm
        )

    if inp.quali is not None and inp.pole_times is not None and inp.lap_s is not None:
        rnd = {e: i + 1 for i, e in enumerate(inp.order)}
        mix = perf.track_mix(inp.pole_times, classes, inp.lap_s).assign(
            round=lambda d: d["event"].map(rnd)
        )
        pr = prof.assign(round=prof["event"].map(rnd))
        feat = predict.build_features(inp.quali, pr, mix)
        if feat["round"].nunique() >= MIN_ROUNDS_PRED:
            try:
                m = predict.select_model(feat)
                out["predict"] = {
                    "name": m["name"],
                    **{f"rmse_{n}": v for n, v in m["rmse"].items()},
                }
            except OSError as e:  # lightgbm without libomp
                out["predict"] = {"error": str(e)}
    return out


# ---- comparison --------------------------------------------------------------------------------


def _rho(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 3 or j.iloc[:, 0].nunique() < 2 or j.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0])


def _cmp_tests(name: str, a: pd.DataFrame | None, b: pd.DataFrame | None) -> list:
    if a is None or b is None:
        return [(name, "-", "not computable", "", False)] if a is None else []
    rows = []
    ca, cb = set(a["cls"]), set(b["cls"])
    if ca != cb:
        rows.append((name, "classes", ", ".join(sorted(ca)), ", ".join(sorted(cb)), True))
    ia, ib = a.set_index("cls"), b.set_index("cls")
    for c in sorted(ca & cb):
        ra, rb, pa, pb = ia.rho[c], ib.rho[c], ia.p_perm[c], ib.p_perm[c]
        changed = bool(
            np.sign(ra) != np.sign(rb) or (pa < ALPHA) != (pb < ALPHA) or abs(ra - rb) > RHO_TOL
        )
        rows.append(
            (name, f"rho[{c}]", f"{ra:+.2f} (p={pa:.3f})", f"{rb:+.2f} (p={pb:.3f})", changed)
        )
    ma, mb = a.rho.mean(), b.rho.mean()
    na, nb = int((a.p_perm < ALPHA).sum()), int((b.p_perm < ALPHA).sum())
    rows.append(
        (
            name,
            "mean rho / n sig",
            f"{ma:+.2f} / {na}",
            f"{mb:+.2f} / {nb}",
            bool(abs(ma - mb) > RHO_TOL or na != nb),
        )
    )
    return rows


def compare(base: dict, alt: dict) -> pd.DataFrame:
    """Rows: result, metric, default, alternative, changed."""
    rows = []
    # pace order per event
    pa, pb = base["pace"], alt["pace"]
    rhos, top = [], 0
    events = sorted(set(pa.index.get_level_values(0)) & set(pb.index.get_level_values(0)))
    for e in events:
        rhos.append(_rho(pa[e], pb[e]))
        top += int(pa[e].idxmin() != pb[e].idxmin())
    rmin = np.nanmin(rhos) if rhos else np.nan
    rows.append(
        (
            "pace",
            "min Spearman of team order / events with new P1",
            f"1.00 / 0 of {len(events)}",
            f"{rmin:.2f} / {top} of {len(events)}",
            bool(rmin < PACE_RHO_MIN or top > 0),
        )
    )
    # season profile signs (common team x class only)
    sa, sb = base["season_profile"], alt["season_profile"]
    j = pd.concat([sa, sb], axis=1, join="inner").dropna()
    flips = float((np.sign(j.iloc[:, 0]) != np.sign(j.iloc[:, 1])).mean()) if len(j) else np.nan
    if base["k"] != alt["k"] or len(j) < len(sa):
        note = f"k {base['k']}->{alt['k']}; {len(j)}/{len(sa)} team-class cells comparable"
    else:
        note = f"{len(j)} team-class cells"
    rows.append(
        (
            "profile",
            "share of profile signs flipped",
            note,
            f"{flips:.0%}" if len(j) else "no common classes",
            bool(not len(j) or flips > PROFILE_FLIP_MAX or len(j) < len(sa)),
        )
    )
    rows += _cmp_tests("test_a", base["test_a"], alt["test_a"])
    rows += _cmp_tests("test_b", base["test_b"], alt["test_b"])
    da, db = base["predict"], alt["predict"]
    if da is None or db is None or "error" in da or "error" in db:
        rows.append(("predict", "-", "not computable", "", False))
    else:
        rel = db["rmse_ridge"] / da["rmse_ridge"] - 1
        rows.append(
            (
                "predict",
                "model / ridge RMSE",
                f"{da['name']} / {da['rmse_ridge']:.3f}",
                f"{db['name']} / {db['rmse_ridge']:.3f}",
                bool(da["name"] != db["name"] or abs(rel) > RMSE_TOL),
            )
        )
    return pd.DataFrame(rows, columns=["result", "metric", "default", "alternative", "changed"])


def run(inp: Inputs, n_boot=PT.N_BOOT, n_perm=PT.N_PERM) -> tuple[dict, pd.DataFrame]:
    """Default headline + one comparison per variant. Returns (default results, flags table with
    columns check, variant, result, metric, default, alternative, changed)."""
    base = headline(inp, DEFAULT, n_boot=n_boot, n_perm=n_perm)
    frames = []
    for check, name, cfg in VARIANTS:
        alt = headline(inp, cfg, k0=base["k"], n_boot=n_boot, n_perm=n_perm)
        if "skip" in alt:
            c = pd.DataFrame(
                [("-", "-", "skipped", alt["skip"], False)],
                columns=["result", "metric", "default", "alternative", "changed"],
            )
        else:
            c = compare(base, alt)
        frames.append(c.assign(check=check, variant=name))
    t = pd.concat(frames, ignore_index=True)
    cols = ["check", "variant", "result", "metric", "default", "alternative", "changed"]
    return base, t[cols]


# ---- apex sampling error -----------------------------------------------------------------------


def apex_error(laps: list[tuple[pd.DataFrame, pd.DataFrame, float]]) -> pd.DataFrame:
    """Min-speed estimator check on (car, corners, length) laps at the native ~3.7 Hz.

    parabola_minus_raw: curve-fit apex minus lowest raw sample (how much the fit moves it).
    half_rate_err: |estimate from every 2nd sample - full-rate estimate| (both phases): the
    estimator's error at half rate, a proxy (lower bound) for its error at full rate.
    """
    from .features import corner_features

    rows = []
    for car, corners, length in laps:
        c = car.sort_values("t_s").drop_duplicates("t_s").reset_index(drop=True)
        full = corner_features(c, corners, length)["min_speed_kph"]
        _, d = perf.distance_axis(c, length)
        v = c["speed_kph"].to_numpy(float)
        raw = pd.Series(
            [
                float(v[(d >= r.start_m) & (d <= r.end_m)].min())
                if ((d >= r.start_m) & (d <= r.end_m)).any()
                else np.nan
                for r in corners.itertuples()
            ],
            index=corners.index,
        )
        halves = [corner_features(c.iloc[p::2], corners, length)["min_speed_kph"] for p in (0, 1)]
        err = pd.concat([(h - full).abs() for h in halves], axis=1).max(axis=1)
        rows.append(pd.DataFrame({"parabola_minus_raw": full - raw, "half_rate_err": err}))
    return pd.concat(rows, ignore_index=True)


# ---- local run + report ------------------------------------------------------------------------


def _load_local(root: Path, source="openf1", season=2025, event="Monaco Grand Prix"):
    """Inputs for one cached qualifying session (+ Jolpica classifications if cached)."""
    from . import cache
    from .corners import find_apexes, lap_profile
    from .features import corner_features
    from .lapfilter import filter_laps, pole_lap

    ld = lambda t, r=root: cache.load(source, t, season, event, "Qualifying", r)  # noqa: E731
    laps, car = ld("laps"), ld("car_data")
    fr = filter_laps(laps, qualifying=True)
    if fr.wet_session:
        raise RuntimeError("wet session: excluded")
    pole = pole_lap(fr.kept)

    def lap_car(drv, n):
        return car[(car.driver == drv) & (car.lap_number == n)]

    ref_car = lap_car(pole.driver, pole.lap_number)
    ref = lap_profile(ref_car)
    corners, _ = find_apexes(ref)
    ev = f"{season} {event}"
    feats = corner_features(ref_car, corners, ref.length).assign(event=ev, corner=corners.index)

    def corner_rows(sel: pd.DataFrame):
        rows, apex = [], []
        for r in sel.itertuples():
            c = lap_car(r.driver, r.lap_number)
            if len(c) < 10:
                continue
            ts = perf.segment_times(c, corners, ref.length)
            ms = corner_features(c, corners, ref.length)["min_speed_kph"]
            apex.append((c, corners, ref.length))
            for k in corners.index:
                rows.append((ev, r.team, r.driver, k, ts[k], ms[k], getattr(r, "q_part", pd.NA)))
        cols = ["event", "team", "driver", "corner", "time_s", "min_speed_kph", "q_part"]
        return pd.DataFrame(rows, columns=cols), apex

    classif = imp = times_q1 = None
    cpath = Path("data/raw")
    try:
        classif = cache.load("jolpica", "classification", season, event, "Qualifying", cpath)
    except FileNotFoundError:
        pass
    kept = fr.kept.copy()
    if classif is not None:
        kept["q_part"] = q_part(kept, classif).to_numpy()
        allq = laps.dropna(subset=["lap_time_s"]).copy()
        allq["q_part"] = q_part(allq, classif)
        q1 = allq[allq["q_part"] == 1].drop_duplicates("driver")
        times_q1, _ = corner_rows(q1)
        files = sorted(
            (cpath / "jolpica/classification" / f"season={season}").glob(
                "event=*/session=qualifying.parquet"
            )
        )
        imp = improvement(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
        imp_ev = imp[imp["event"] == event].assign(event=ev)
    times, apex = corner_rows(kept)
    inp = Inputs(
        times=times,
        corners=feats,
        season={ev: season},
        order=[ev],
        times_q1=times_q1,
        improvement=None if imp is None else imp_ev,
    )
    ctx = {
        "source": source,
        "event": ev,
        "n_drivers": int(times["driver"].nunique()),
        "n_teams": int(times["team"].nunique()),
        "n_corners": len(corners),
        "pole": f"{pole.driver} lap {int(pole.lap_number)} {pole.lap_time_s:.3f}s",
        "car_flags": fr.car_flags,
        "improvement_all": imp,
        "q_parts": kept[["driver", "team", "q_part"]] if classif is not None else None,
        "apex": apex_error(apex),
        "lap_gap": team_lap_gap(fr.kept),
        "dup_segments": int(corners.duplicated(["start_m", "end_m"]).sum()),
    }
    return inp, ctx


def team_lap_gap(kept: pd.DataFrame) -> pd.Series:
    """Team best lap, % over the fastest team (sanity reference for corner-based pace)."""
    b = kept.groupby("team")["lap_time_s"].min()
    return (100 * (b / b.min() - 1)).rename("lap_gap_pct")


def _md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def report(base: dict, flags: pd.DataFrame, ctx: dict) -> str:
    nc = flags["default"].eq("not computable")
    shown = flags[~nc]
    changed = shown[shown["changed"]]
    imp = ctx["improvement_all"]
    apex = ctx["apex"]
    pace = base["pace"].rename("pace_pct").reset_index().sort_values("pace_pct")
    pace = pace.merge(ctx["lap_gap"].reset_index(), on="team", how="left")
    rho_lap = _rho(pace.set_index("team").pace_pct, pace.set_index("team").lap_gap_pct)
    L = [
        "# Robustness checks",
        "",
        "Generated by `uv run python -m f1corner.robustness` (`src/f1corner/robustness.py`). "
        "Do not edit by hand; rerun after new data is cached.",
        "",
        "## Scope: what this run could use",
        "",
        f"- Telemetry: **one session only**, {ctx['event']} qualifying ({ctx['source']}, local "
        f"test cache). {ctx['n_drivers']} drivers, {ctx['n_teams']} teams, {ctx['n_corners']} "
        f"corners, pole {ctx['pole']}.",
        "- Official Q1/Q2/Q3 times: Jolpica 2025 classifications "
        f"({0 if imp is None else len(imp)} events cached).",
        "- No 2026 data and no second event with telemetry, so **Test A, Test B and the "
        "prediction model cannot be computed** (see Gaps). Everything below is one event and "
        "is illustrative; it is not the post's headline result.",
        "",
        "## Flags",
        "",
        f"{len(changed)} of {len(shown)} computable comparisons flagged as changed. Thresholds: pace order "
        f"Spearman < {PACE_RHO_MIN} or new fastest team; > {PROFILE_FLIP_MAX:.0%} of profile "
        f"signs flip; rho sign/significance (p<{ALPHA}) flips or |d rho| > {RHO_TOL}; model "
        f"choice flips or ridge RMSE moves > {RMSE_TOL:.0%}.",
        "",
        _md(shown),
        "",
        f"Not computable in this run (all variants): {', '.join(sorted(flags[nc].result.unique()))}.",
        "",
        f"Default clustering: GMM, k = {base['k']} chosen by BIC on {ctx['n_corners']} corners "
        "of one track (far too few for a stable fit; k-1/k+1/k-means rows are therefore "
        "expected to move).",
        "",
        "Default pace (median % deficit to the best team per corner, better teammate) next to "
        "each team's best lap gap:",
        "",
        _md(pace, "{:.2f}"),
        "",
        "## Sanity check: corner pace vs lap time",
        "",
        f"Spearman(corner-based pace, best-lap gap) across teams = {rho_lap:.2f}. "
        + (
            "The pole team is not the fastest by corner pace, so per-corner segment times on "
            "this event carry large noise (3.7 Hz samples, distance re-integrated per lap and "
            "rescaled). Flags for this event are therefore dominated by measurement noise and "
            "must not be read as a property of the method. "
            if pace.iloc[0].lap_gap_pct > 0
            else ""
        )
        + f"{ctx['dup_segments']} of {ctx['n_corners']} corners share an identical segment with "
        "another corner (sequences), so those segments count more than once in the median.",
        "",
    ]
    if imp is not None:
        s = imp[["pct12", "pct23"]].describe().loc[["count", "mean", "50%", "min", "max"]]
        L += [
            "## Track improvement (measured, 2025)",
            "",
            "Paired by driver: median over drivers of q2/q1 and q3/q2, per event. Negative = "
            "faster in the later part. This mixes track evolution with teams turning engines up "
            "and drivers taking more risk, so it is an upper bound on pure track gain. Events "
            "with no paired times (n12 = 0) have no Q times in the cached classification.",
            "",
            _md(s.reset_index().rename(columns={"index": "stat"}), "{:.3f}"),
            "",
            "<details><summary>Per event</summary>",
            "",
            _md(imp),
            "",
            "</details>",
            "",
        ]
        if ctx["q_parts"] is not None:
            qp = ctx["q_parts"]["q_part"].value_counts(dropna=False).sort_index()
            L += [
                "Best lap by Q part in this session: "
                + ", ".join(
                    f"Q{int(k)}: {v}" if pd.notna(k) else f"unmatched: {v}" for k, v in qp.items()
                )
                + " "
                "(matched to official times within "
                f"{Q_TOL_S * 1000:.1f} ms). The correction scales each best lap's corner times "
                "to its Q1 equivalent; the Q1-only variant uses each driver's official Q1 lap.",
                "",
            ]
    a = apex.dropna()
    L += [
        "## Min-speed sampling error (3.7 Hz)",
        "",
        f"Over {len(a)} corner passes (every driver's best lap):",
        "",
        f"- Curve-fit apex minus lowest raw sample: median {a.parabola_minus_raw.median():+.2f} "
        f"km/h, min {a.parabola_minus_raw.min():+.2f} km/h (fit can only lower it).",
        f"- Half-rate test (every 2nd sample vs all): median {a.half_rate_err.median():.2f} km/h, "
        f"p90 {a.half_rate_err.quantile(0.9):.2f} km/h, max {a.half_rate_err.max():.2f} km/h. "
        "This is the estimator's error at ~1.85 Hz; the error at 3.7 Hz should be smaller, but "
        "there is no higher-rate ground truth in open data to measure it directly.",
        "- Time through the segment stays the main metric; minimum speed is a robustness check.",
        "",
    ]
    cf = ctx["car_flags"]
    L += [
        "## Car issues (1% teammate check)",
        "",
        (
            "Flagged for manual FIA check: "
            + ", ".join(
                f"{r.driver} ({r.team}, +{r.gap_pct:.2f}% vs {r.teammate})" for r in cf.itertuples()
            )
            if len(cf)
            else "None flagged."
        ),
        "",
        "The mean-of-teammates variant includes these laps and is the main reason that row can move.",
        "",
        GAPS,
    ]
    return "\n".join(L) + "\n"


GAPS = """## Gaps (not computed; no numbers fabricated)

| Item | Status | Needed |
|---|---|---|
| Test A (2025 vs 2026 profiles) | not computable | qualifying telemetry for >= 2 events in each of 2025 and 2026 |
| Test B (2026 first vs second half) | not computable | >= 4 2026 events with telemetry |
| Rolling-window drift (upgrades) | not computable | 2026 season telemetry (`profile_tests.rolling`) |
| Prediction (ridge / LightGBM) | not computable | per-race team gap table + conditions + pole corner times for >= 6 rounds |
| Season-level clustering checks | one track only | clustering is fit on one event's corners; needs the full 2025 calendar |
| Wet/mixed sessions | handled upstream | `lapfilter` drops the whole session; nothing to vary here |
| Corner matching across years / moved corners | not implemented | position-based matching with a distance cut for Test A (`cluster.compare_labels` matches by label only) |
| Sequence corners | double counted | corners in one sequence share a segment (`corners.segment`); dedupe or split before pace |
| Bootstrap/permutation for pace and profile | not implemented | only Tests A/B carry CIs; pace/profile flags are point comparisons |
| Track-improvement correction purity | caveat | measured gain includes engine modes and driver risk, not only rubber |
| OpenF1 licence (personal use) | open question | confirm terms before publishing derived numbers |

`run()` computes Tests A/B and the prediction comparison automatically once these inputs exist
(tested on synthetic data in `tests/unit/test_robustness.py`).
"""


def main(argv=None):
    root = Path("data/cache/test")
    if not (root / "openf1/laps").exists():
        root = Path("data/raw")
    inp, ctx = _load_local(root)
    base, flags = run(inp)
    out = Path("docs/robustness.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(base, flags, ctx))
    print(flags.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
