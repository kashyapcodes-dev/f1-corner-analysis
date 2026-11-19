"""End-to-end steps behind the CLI: pick events, ingest, analyze one qualifying session, load inputs.

Derived tables (one folder per event) live in data/derived (gitignored):
  data/derived/<source>/season=2025/event=<slug>/{times,times_q1,corners,pole_times}.parquet + meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import cache, perf
from .corners import find_apexes, lap_profile
from .features import corner_features
from .lapfilter import filter_laps, pole_lap
from .robustness import Inputs, improvement, q_part
from .sources.jolpica_source import CLASSIFIED, JolpicaSource

DERIVED = Path("data/derived")
SESSION = "Qualifying"
TELEMETRY_TABLES = ["laps", "car_data", "race_control", "weather"]
TABLES = ["times", "times_q1", "corners", "pole_times"]


class WetSession(RuntimeError):
    """Wet/mixed session: excluded from the analysis by design."""


def event_key(season: int, event: str) -> str:
    return f"{season} {event}"


# ---- event selection ---------------------------------------------------------------------------


def calendar(season: int, root: Path = cache.RAW, past_only: bool = True) -> list[str]:
    """Qualifying events of a season in calendar order (from the cached Jolpica schedule)."""
    s = cache.load("jolpica", "schedule", season, "all", "all", root)
    s = s[s["session"] == SESSION].sort_values("date")
    if past_only:
        s = s[s["date"] <= pd.Timestamp.now("UTC")]
    return list(s["event"])


def ensure_schedule(season: int, root: Path = cache.RAW) -> None:
    cache.fetch(JolpicaSource(), "schedule", season, "all", "all", root)


def resolve_events(seasons: list[int], names: list[str], root: Path = cache.RAW) -> list[tuple]:
    """(season, event) pairs. No names -> every past qualifying session. Names match as a
    case-insensitive substring of the schedule's event name and must resolve to exactly one."""
    out = []
    for s in seasons:
        ensure_schedule(s, root)
        cal = calendar(s, root, past_only=not names)
        if not names:
            out += [(s, e) for e in cal]
            continue
        for n in names:
            hit = [e for e in cal if n.lower() in e.lower()]
            if len(hit) != 1:
                raise ValueError(
                    f"{s}: {n!r} matches {len(hit)} events ({hit or cal}); be more specific"
                )
            out.append((s, hit[0]))
    return out


# ---- ingest ------------------------------------------------------------------------------------


def ingest_event(src, season: int, event: str, root: Path = cache.RAW, force=False) -> None:
    """Telemetry tables (via `src`) + official Jolpica classification for one qualifying session."""
    for t in TELEMETRY_TABLES:
        cache.fetch(src, t, season, event, SESSION, root, force)
    if SESSION in CLASSIFIED:
        cache.fetch(JolpicaSource(), "classification", season, event, SESSION, root, force)


# ---- analyze -----------------------------------------------------------------------------------


def derived_dir(source: str, season: int, event: str, root: Path = DERIVED) -> Path:
    return root / source / f"season={season}" / f"event={cache._slug(event)}"


def has_derived(source: str, season: int, event: str, root: Path = DERIVED) -> bool:
    return (derived_dir(source, season, event, root) / "meta.json").exists()


def analyze_event(source: str, season: int, event: str, root: Path = cache.RAW) -> dict:
    """Corner times/features for every kept lap of one qualifying session.

    Returns dict(times, times_q1, corners, pole_times, meta). Raises WetSession.
    times_q1 / q_part need the cached Jolpica classification (empty/NA without it).
    """
    ev = event_key(season, event)
    laps = cache.load(source, "laps", season, event, SESSION, root)
    car = cache.load(source, "car_data", season, event, SESSION, root)
    fr = filter_laps(laps, qualifying=True)
    if fr.wet_session:
        raise WetSession(ev)
    pole = pole_lap(fr.kept)

    def lap_car(drv, n):
        return car[(car.driver == drv) & (car.lap_number == n)]

    ref_car = lap_car(pole.driver, pole.lap_number)
    ref = lap_profile(ref_car)
    corners, _ = find_apexes(ref)
    feats = corner_features(ref_car, corners, ref.length).assign(event=ev, corner=corners.index)

    def rows(sel: pd.DataFrame) -> pd.DataFrame:
        out = []
        for r in sel.itertuples():
            c = lap_car(r.driver, r.lap_number)
            if len(c) < 10:
                continue
            ts = perf.segment_times(c, corners, ref.length)
            ms = corner_features(c, corners, ref.length)["min_speed_kph"]
            qp = getattr(r, "q_part", pd.NA)
            out += [(ev, r.team, r.driver, k, ts[k], ms[k], qp) for k in corners.index]
        cols = ["event", "team", "driver", "corner", "time_s", "min_speed_kph", "q_part"]
        return pd.DataFrame(out, columns=cols)

    kept, times_q1 = fr.kept.copy(), None
    try:
        classif = cache.load("jolpica", "classification", season, event, SESSION, root)
    except FileNotFoundError:
        classif = None
    if classif is not None:
        kept["q_part"] = q_part(kept, classif).to_numpy()
        allq = laps.dropna(subset=["lap_time_s"]).copy()
        allq["q_part"] = q_part(allq, classif)
        times_q1 = rows(allq[allq["q_part"] == 1].drop_duplicates("driver"))
    times = rows(kept)
    pole_times = times[times["driver"] == pole.driver][["event", "corner", "time_s"]]
    meta = dict(
        event=ev,
        season=season,
        source=source,
        pole_driver=pole.driver,
        pole_lap=int(pole.lap_number),
        lap_s=float(pole.lap_time_s),
        n_drivers=int(times["driver"].nunique()),
        n_teams=int(times["team"].nunique()),
        n_corners=len(corners),
        has_classification=classif is not None,
    )
    return dict(
        times=times,
        times_q1=times_q1 if times_q1 is not None else times.iloc[0:0],
        corners=feats,
        pole_times=pole_times,
        meta=meta,
    )


def write_derived(res: dict, source: str, season: int, event: str, root: Path = DERIVED) -> Path:
    d = derived_dir(source, season, event, root)
    d.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        res[t].to_parquet(d / f"{t}.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(res["meta"], indent=2))
    return d


# ---- load for reporting ------------------------------------------------------------------------


def load_inputs(
    source: str,
    pairs: list[tuple[int, str]],
    derived: Path = DERIVED,
    raw: Path = cache.RAW,
) -> tuple[Inputs, list[dict], list[str]]:
    """Inputs for `robustness.headline` from derived events (calendar order within each season).

    Returns (inputs, metas, missing) where `missing` lists events with no derived tables.
    """
    pairs = sorted(pairs)  # season, then calendar position below
    order_in = {}
    for s in {s for s, _ in pairs}:
        order_in[s] = {e: i for i, e in enumerate(calendar(s, raw, past_only=False))}
    pairs.sort(key=lambda p: (p[0], order_in[p[0]].get(p[1], 99)))

    have = [p for p in pairs if has_derived(source, *p, root=derived)]
    missing = [event_key(*p) for p in pairs if p not in have]
    if not have:
        raise FileNotFoundError("no derived tables; run `f1 analyze` first")

    def cat(t):
        fs = [pd.read_parquet(derived_dir(source, *p, root=derived) / f"{t}.parquet") for p in have]
        return pd.concat(fs, ignore_index=True)

    metas = [
        json.loads((derived_dir(source, *p, root=derived) / "meta.json").read_text()) for p in have
    ]
    times_q1 = cat("times_q1")
    imp = None
    classifs = []
    for s, e in have:
        try:
            classifs.append(
                cache.load("jolpica", "classification", s, e, SESSION, raw).assign(
                    event=event_key(s, e)
                )
            )
        except FileNotFoundError:
            pass
    if classifs:
        imp = improvement(pd.concat(classifs, ignore_index=True))
    inp = Inputs(
        times=cat("times"),
        corners=cat("corners"),
        season={m["event"]: m["season"] for m in metas},
        order=[m["event"] for m in metas],
        times_q1=times_q1 if len(times_q1) else None,
        improvement=imp,
        pole_times=cat("pole_times"),
        lap_s=pd.Series({m["event"]: m["lap_s"] for m in metas}),
    )
    return inp, metas, missing
