import pandas as pd

from f1corner.schema import LAPS_COLS, conform


def test_conform_adds_missing_and_orders():
    out = conform("laps", pd.DataFrame({"driver": ["VER"], "extra": [1]}))
    assert list(out.columns) == LAPS_COLS
