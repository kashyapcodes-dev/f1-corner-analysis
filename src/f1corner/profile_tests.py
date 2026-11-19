"""Tests A and B (spec 14): are team corner-class profiles stable?

Input `prof`: event, team, cls, profile (from `perf.pace_profile`). Rank correlation is Spearman
across teams, per corner class. With ~10 teams the intervals are wide; say so up front.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_BOOT = 1000
N_PERM = 5000
SEED = 0
TEAM_ALIAS = {"Audi": "Sauber", "Kick Sauber": "Sauber", "Alfa Romeo": "Sauber"}
EXCLUDE = {"Cadillac"}
MIN_TEAMS = 4

WARNING = (
    "Only ~10 teams per correlation: confidence intervals are wide and a single team can swing "
    "rho. Treat point estimates as indicative."
)


def normalise_teams(prof: pd.DataFrame) -> pd.DataFrame:
    """Audi = Sauber continuation; Cadillac dropped (no 2025 history)."""
    p = prof.copy()
    p["team"] = p["team"].replace(TEAM_ALIAS)
    return p[~p["team"].isin(EXCLUDE)]


def season_profile(prof: pd.DataFrame, events) -> pd.Series:
    """Mean profile per (team, cls) over the given races."""
    p = prof[prof["event"].isin(list(events))]
    return p.groupby(["team", "cls"])["profile"].mean()


def _rho(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < MIN_TEAMS or j.iloc[:, 0].nunique() < 2 or j.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0])


def _by_class(pa: pd.Series, pb: pd.Series, classes) -> dict:
    return {
        c: _rho(
            pa.xs(c, level="cls")
            if c in pa.index.get_level_values("cls")
            else pd.Series(dtype=float),
            pb.xs(c, level="cls")
            if c in pb.index.get_level_values("cls")
            else pd.Series(dtype=float),
        )
        for c in classes
    }


def correlate(
    prof: pd.DataFrame,
    events_a,
    events_b,
    classes=None,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per class: rho between season profiles over `events_a` and `events_b`.

    Bootstrap resamples races within each period (95% percentile range); permutation p-value
    shuffles team labels of period B (null: no link between teams across periods, two-sided).
    Returns cls, n_teams, rho, ci_lo, ci_hi, p_perm.
    """
    prof = normalise_teams(prof)
    events_a, events_b = list(events_a), list(events_b)
    classes = sorted(prof["cls"].unique()) if classes is None else list(classes)
    rng = np.random.default_rng(seed)
    pa, pb = season_profile(prof, events_a), season_profile(prof, events_b)
    base = _by_class(pa, pb, classes)

    def resampled(events):
        pick = rng.choice(events, len(events), replace=True)
        w = pd.Series(pick).value_counts()
        p = prof[prof["event"].isin(w.index)].copy()
        p["w"] = p["event"].map(w)
        g = p.assign(wp=p["profile"] * p["w"]).groupby(["team", "cls"])
        return g["wp"].sum() / g["w"].sum()

    boots = {c: [] for c in classes}
    for _ in range(n_boot):
        r = _by_class(resampled(events_a), resampled(events_b), classes)
        for c in classes:
            boots[c].append(r[c])

    rows = []
    for c in classes:
        j = pd.concat(
            [
                pa.xs(c, level="cls")
                if c in pa.index.get_level_values("cls")
                else pd.Series(dtype=float),
                pb.xs(c, level="cls")
                if c in pb.index.get_level_values("cls")
                else pd.Series(dtype=float),
            ],
            axis=1,
            join="inner",
        ).dropna()
        rho = base[c]
        p_perm = np.nan
        if not np.isnan(rho):
            x, y = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
            null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
            p_perm = float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n_perm + 1))
        b = np.array(boots[c], float)
        b = b[~np.isnan(b)]
        lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
        rows.append((c, len(j), rho, lo, hi, p_perm))
    return pd.DataFrame(rows, columns=["cls", "n_teams", "rho", "ci_lo", "ci_hi", "p_perm"])


def test_a(prof25: pd.DataFrame, prof26: pd.DataFrame, **kw) -> pd.DataFrame:
    """2025 season profile vs 2026 season profile; both must use the 2025 (frozen) labels."""
    both = pd.concat([prof25, prof26], ignore_index=True)
    return correlate(both, prof25["event"].unique(), prof26["event"].unique(), **kw)


def test_b(prof26: pd.DataFrame, order: list[str], **kw) -> pd.DataFrame:
    """First vs second half of the 2026 season. `order` = events in calendar order."""
    h = len(order) // 2
    return correlate(prof26, order[:h], order[len(order) - h :], **kw)


def rolling(prof26: pd.DataFrame, order: list[str], window: int = 4) -> pd.DataFrame:
    """Drift: for each start i, rho between races [i, i+w) and [i+w, i+2w), mean over classes.

    Point estimates only (no bootstrap). Falling rho as upgrades land = profiles drift.
    """
    prof = normalise_teams(prof26)
    classes = sorted(prof["cls"].unique())
    rows = []
    for i in range(len(order) - 2 * window + 1):
        a = season_profile(prof, order[i : i + window])
        b = season_profile(prof, order[i + window : i + 2 * window])
        r = _by_class(a, b, classes)
        rows.append(
            (order[i], order[i + 2 * window - 1], float(np.nanmean(list(r.values()))), *r.values())
        )
    return pd.DataFrame(rows, columns=["start", "end", "mean_rho", *classes])
