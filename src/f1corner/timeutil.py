from __future__ import annotations

import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")


def epoch_s(ts: pd.Series) -> pd.Series:
    """UTC timestamps (tz-aware or naive-UTC) -> float epoch seconds. Unit-safe (ns/us)."""
    ts = pd.to_datetime(ts, utc=True, format="ISO8601").dt.tz_convert(None)
    return (ts - _EPOCH).dt.total_seconds()
