import numpy as np
import pandas as pd

from f1corner import cluster as K
from f1corner import corners as C
from f1corner.features import FEATURES, corner_features
from tests.unit.test_corners import synth_car


def test_corner_features_on_synthetic_lap():
    car = synth_car()
    car["gear"] = np.where(car["speed_kph"] < 150, 2, 8)
    p = C.lap_profile(car)
    corners, _ = C.find_apexes(p)
    f = corner_features(car, corners, p.length)
    assert list(f.columns) == FEATURES
    hard = f.iloc[0]
    assert 60 < hard.min_speed_kph < 110
    assert hard.speed_loss_kph > 150
    assert hard.time_off_wot_s > 2
    assert hard.min_gear == 2
    assert f.iloc[1].min_gear == 8


def _blobs(n_events=12, per=6, seed=1):
    rng = np.random.default_rng(seed)
    centres = {"slow": (65, 200, 7, 2), "med": (140, 100, 4, 4), "fast": (250, 20, 1.5, 7)}
    rows = []
    for e in range(n_events):
        for j in range(per):
            c = list(centres.values())[j % 3]
            rows.append((f"gp{e}", str(j), *(np.array(c) + rng.normal(0, [4, 8, 0.3, 0.2]))))
    return pd.DataFrame(rows, columns=["event", "label", *FEATURES])


def test_fit_recovers_three_clusters_and_names():
    x = _blobs()
    model, labels, sel = K.fit(x)
    assert model.k == 3
    assert set(labels) == {"slow hairpin", "medium corner", "fast sweep"}
    assert (sel["ari_gmm_kmeans"] > 0.9).all() or sel.loc[sel.k == 3, "ari_gmm_kmeans"].iloc[
        0
    ] > 0.9
    assert (K.assign(model, x) == labels).all()


def test_stability_high_on_separated_clusters():
    s = K.stability(_blobs(), 3, n_boot=20)
    assert s["stable"] and s["ari_mean"] > 0.9


def test_compare_labels_counts_changes():
    a = _blobs()
    ra = np.array(["x"] * len(a))
    rb = ra.copy()
    rb[:5] = "y"
    m = K.compare_labels(a, a, ra, rb)
    assert len(m) == len(a) and m["changed"].sum() == 5
