"""MultiViewer corner numbers (https://api.multiviewer.app) for readable labels.

Its trackPosition x/y share the coordinate system of OpenF1/F1 live-timing location data.
circuit_key = F1 live-timing / OpenF1 circuit_key (Monaco = 22).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

URL = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"
CACHE = Path("data/raw/multiviewer")


def fetch_corners(circuit_key: int, year: int, root: Path = CACHE) -> pd.DataFrame:
    """Corner table: number, letter, x, y. Cached as JSON."""
    p = root / f"circuit={circuit_key}_year={year}.json"
    if p.exists():
        data = json.loads(p.read_text())
    else:
        r = httpx.get(URL.format(key=circuit_key, year=year), timeout=30, follow_redirects=True)
        r.raise_for_status()
        data = r.json()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
    return corners_frame(data)


def corners_frame(data: dict) -> pd.DataFrame:
    rows = [
        (c["number"], c.get("letter", ""), c["trackPosition"]["x"], c["trackPosition"]["y"])
        for c in data["corners"]
    ]
    return pd.DataFrame(rows, columns=["number", "letter", "x", "y"])


def label_corners(
    apex_x: np.ndarray, apex_y: np.ndarray, mv: pd.DataFrame, max_dist: float = 600.0
) -> list[str | None]:
    """Nearest MultiViewer corner per apex, e.g. "5" or "11a". None if farther than max_dist
    (track units, 1/10 m)."""
    out: list[str | None] = []
    mx, my = mv["x"].to_numpy(float), mv["y"].to_numpy(float)
    for x, y in zip(apex_x, apex_y):
        d = np.hypot(mx - x, my - y)
        k = int(np.argmin(d))
        out.append(
            f"{mv['number'].iloc[k]}{mv['letter'].iloc[k] or ''}" if d[k] <= max_dist else None
        )
    return out
