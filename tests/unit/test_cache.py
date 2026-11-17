import pandas as pd

from f1corner import cache
from f1corner.schema import LAPS_COLS


class FakeSource:
    name = "fake"
    calls = 0

    def laps(self, season, event, session):
        self.calls += 1
        return pd.DataFrame({"driver": ["NOR"], "lap_time_s": [69.954]})


def test_fetch_partitions_caches_and_writes_manifest(tmp_path):
    src = FakeSource()
    p = cache.fetch(src, "laps", 2025, "Monaco Grand Prix", "Qualifying", root=tmp_path)
    assert (
        p == tmp_path / "fake/laps/season=2025/event=monaco-grand-prix/session=qualifying.parquet"
    )
    cache.fetch(src, "laps", 2025, "Monaco Grand Prix", "Qualifying", root=tmp_path)
    assert src.calls == 1  # second call served from cache
    df = cache.load("fake", "laps", 2025, "Monaco Grand Prix", "Qualifying", root=tmp_path)
    assert list(df.columns) == LAPS_COLS and df["lap_time_s"].iloc[0] == 69.954
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 1
    cache.fetch(src, "laps", 2025, "Monaco Grand Prix", "Qualifying", root=tmp_path, force=True)
    assert src.calls == 2
