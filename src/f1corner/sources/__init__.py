from __future__ import annotations

from .base import Source, SourceUnavailable


def get_source(name: str) -> Source:
    if name == "openf1":
        from .openf1_source import OpenF1Source
        return OpenF1Source()
    if name == "fastf1":
        from .fastf1_source import FastF1Source
        return FastF1Source()
    raise ValueError(f"unknown source {name!r}")


__all__ = ["Source", "SourceUnavailable", "get_source"]
