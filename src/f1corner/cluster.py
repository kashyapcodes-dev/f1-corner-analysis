"""Corner-type clustering: Gaussian mixture (k by BIC), k-means as a check, bootstrap stability.

Input: one row per corner with FEATURES + `event` (race) + `label` (corner number, e.g. "11a").
2025 is fit once and frozen as the reference (`Model`, used by Test A via `assign`); 2026 is
re-clustered on its own and compared with `compare_labels`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .features import FEATURES

K_RANGE = range(2, 9)
SEED = 0
N_INIT = 5
N_BOOT = 200
ARI_STABLE = 0.7  # mean bootstrap ARI at/above this: labels count as stable
MIN_SPEED = "min_speed_kph"

# Upper bounds on a cluster's median minimum speed (km/h) -> name.
SPEED_NAMES = [
    (80, "slow hairpin"),
    (120, "slow corner"),
    (170, "medium corner"),
    (230, "fast corner"),
    (np.inf, "fast sweep"),
]


@dataclass
class Model:
    """Frozen fit: z-score stats, GMM and cluster names. Apply to new corners with `assign`."""

    mean: np.ndarray
    std: np.ndarray
    gmm: GaussianMixture
    names: list[str]  # names[cluster id]
    k: int


def _z(x: np.ndarray, mean=None, std=None):
    mean = x.mean(0) if mean is None else mean
    std = x.std(0) if std is None else std
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std, mean, std


def _gmm(z: np.ndarray, k: int) -> GaussianMixture:
    return GaussianMixture(k, covariance_type="full", n_init=N_INIT, random_state=SEED).fit(z)


def name_clusters(x: pd.DataFrame, labels: np.ndarray) -> list[str]:
    """Name each cluster by its median min speed; duplicates get a ' #n' suffix, slowest first."""
    med = pd.Series(x[MIN_SPEED].to_numpy(float)).groupby(labels).median().sort_index()
    names: dict[int, str] = {}
    for cid in med.sort_values().index:
        base = next(n for ub, n in SPEED_NAMES if med[cid] < ub)
        dup = sum(v.split(" #")[0] == base for v in names.values())
        names[int(cid)] = base if dup == 0 else f"{base} #{dup + 1}"
    return [names[i] for i in range(len(names))]


def select_k(x: pd.DataFrame) -> pd.DataFrame:
    """Per k in 2..8: GMM BIC, GMM silhouette, k-means silhouette, GMM-vs-kmeans ARI."""
    z, _, _ = _z(x[FEATURES].to_numpy(float))
    rows = []
    for k in K_RANGE:
        if k >= len(z):
            break
        g = _gmm(z, k)
        gl = g.predict(z)
        km = KMeans(k, n_init=N_INIT, random_state=SEED).fit(z).labels_
        ok = len(set(gl)) > 1
        rows.append(
            (
                k,
                g.bic(z),
                silhouette_score(z, gl) if ok else np.nan,
                silhouette_score(z, km),
                adjusted_rand_score(gl, km),
            )
        )
    return pd.DataFrame(rows, columns=["k", "bic", "sil_gmm", "sil_kmeans", "ari_gmm_kmeans"])


def fit(x: pd.DataFrame, k: int | None = None) -> tuple[Model, np.ndarray, pd.DataFrame]:
    """Fit on corners `x`. k=None picks the lowest-BIC k in 2..8.

    Returns (model, labels per row as names, k-selection table).
    """
    sel = select_k(x)
    k = int(sel.loc[sel["bic"].idxmin(), "k"]) if k is None else k
    z, mean, std = _z(x[FEATURES].to_numpy(float))
    g = _gmm(z, k)
    ids = g.predict(z)
    names = name_clusters(x, ids)
    model = Model(mean, std, g, names, k)
    return model, np.array([names[i] for i in ids]), sel


def assign(model: Model, x: pd.DataFrame) -> np.ndarray:
    """Label corners with a frozen model (2025 reference -> Test A)."""
    z, _, _ = _z(x[FEATURES].to_numpy(float), model.mean, model.std)
    return np.array([model.names[i] for i in model.gmm.predict(z)])


def stability(x: pd.DataFrame, k: int, n_boot: int = N_BOOT) -> dict:
    """Resample races (events) with replacement, refit k-component GMM, label ALL corners with the
    refit and compare with the full-data labelling by adjusted Rand index.

    A labelling counts as stable if mean ARI >= ARI_STABLE.
    """
    xf = x[FEATURES].to_numpy(float)
    zf, _, _ = _z(xf)
    base = _gmm(zf, k).predict(zf)
    events = x["event"].to_numpy()
    uniq = np.unique(events)
    rng = np.random.default_rng(SEED)
    idx_by = {e: np.flatnonzero(events == e) for e in uniq}
    aris = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_by[e] for e in pick])
        zb, mean, std = _z(xf[rows])
        if len(zb) <= k:
            continue
        g = _gmm(zb, k)
        lab = g.predict(_z(xf, mean, std)[0])
        aris.append(adjusted_rand_score(base, lab))
    a = np.array(aris)
    return {
        "k": k,
        "n_boot": len(a),
        "ari_mean": float(a.mean()),
        "ari_p05": float(np.percentile(a, 5)),
        "ari_min": float(a.min()),
        "stable": bool(a.mean() >= ARI_STABLE),
    }


def compare_labels(
    ref: pd.DataFrame, new: pd.DataFrame, ref_labels: np.ndarray, new_labels: np.ndarray
) -> pd.DataFrame:
    """Match corners across seasons by (event, label) and report class changes.

    `ref`/`new` need `event` and `label`. Returns matched corners with both class names and
    `changed`. Names (not cluster ids) are compared, so the two fits need no id alignment.
    """
    a = ref[["event", "label"]].assign(ref_class=ref_labels)
    b = new[["event", "label"]].assign(new_class=new_labels)
    m = a.dropna().merge(b.dropna(), on=["event", "label"], how="inner")
    m["changed"] = m["ref_class"] != m["new_class"]
    return m
