"""Pull Jolpica schedule + classification into data/raw (the only code that calls the API).

python -m f1corner.ingest 2025 [2024 ...] [--force]
"""

from __future__ import annotations

import sys

import pandas as pd

from . import cache
from .sources.jolpica_source import CLASSIFIED, JolpicaSource


def ingest_season(src, season: int, root=cache.RAW, force=False) -> list:
    """Cache the calendar, then classification for every past session that has one."""
    paths = [cache.fetch(src, "schedule", season, "all", "all", root, force)]
    sched = cache.load(src.name, "schedule", season, "all", "all", root)
    now = pd.Timestamp.now("UTC")
    for r in sched.itertuples():
        if r.session in CLASSIFIED and r.date <= now:
            paths.append(
                cache.fetch(src, "classification", season, r.event, r.session, root, force)
            )
    return paths


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args
    seasons = [int(a) for a in args if a.isdigit()]
    if not seasons:
        sys.exit("usage: python -m f1corner.ingest SEASON [SEASON ...] [--force]")
    src = JolpicaSource()
    for s in seasons:
        print(f"{s}: {len(ingest_season(src, s, force=force))} files")


if __name__ == "__main__":
    main()
