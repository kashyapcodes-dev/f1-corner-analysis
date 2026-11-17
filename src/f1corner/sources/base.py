from __future__ import annotations

from typing import Protocol

import pandas as pd


class Source(Protocol):
    """A data backend. Each method returns a canonical table (see f1corner.schema)."""

    name: str

    def sessions(self, season: int) -> pd.DataFrame: ...
    def laps(self, season: int, event: str, session: str) -> pd.DataFrame: ...
    def car_data(self, season: int, event: str, session: str) -> pd.DataFrame: ...
    def location(self, season: int, event: str, session: str) -> pd.DataFrame: ...
    def weather(self, season: int, event: str, session: str) -> pd.DataFrame: ...
    def race_control(self, season: int, event: str, session: str) -> pd.DataFrame: ...


class SourceUnavailable(RuntimeError):
    """Optional backend not installed / not authenticated."""
