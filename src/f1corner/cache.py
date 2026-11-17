"""Parquet cache partitioned by source/table/season/event/session. Pipeline code reads only from here."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schema import TABLES, conform

RAW = Path("data/raw")  # gitignored


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def path(source: str, table: str, season: int, event: str, session: str, root: Path = RAW) -> Path:
    return root / source / table / f"season={season}" / f"event={_slug(event)}" / f"session={_slug(session)}.parquet"


def fetch(src, table: str, season: int, event: str, session: str, root: Path = RAW, force=False) -> Path:
    p = path(src.name, table, season, event, session, root)
    if p.exists() and not force:
        return p
    df = getattr(src, table)(season, event, session)
    p.parent.mkdir(parents=True, exist_ok=True)
    conform(table, df).to_parquet(p, index=False)
    _manifest(root, src.name, table, season, event, session)
    return p


def load(source: str, table: str, season: int, event: str, session: str, root: Path = RAW) -> pd.DataFrame:
    return pd.read_parquet(path(source, table, season, event, session, root))


def _manifest(root: Path, source: str, table: str, season, event, session):
    mp = root / "manifest.jsonl"
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("a") as f:
        f.write(json.dumps(dict(pulled_at=datetime.now(timezone.utc).isoformat(), source=source,
                                table=table, season=season, event=event, session=session)) + "\n")
