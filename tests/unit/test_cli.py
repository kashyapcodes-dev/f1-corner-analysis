import pandas as pd
import pytest
from typer.testing import CliRunner

from f1corner import cache, pipeline
from f1corner.cli import app


@pytest.fixture
def raw(tmp_path):
    p = cache.path("jolpica", "schedule", 2025, "all", "all", tmp_path)
    p.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event": ["Monaco Grand Prix"] * 2 + ["Spanish Grand Prix"] * 2 + ["Future GP"] * 2,
            "session": ["Qualifying", "Race"] * 3,
            "date": pd.to_datetime(
                [
                    "2025-05-24",
                    "2025-05-25",
                    "2025-05-31",
                    "2025-06-01",
                    "2099-01-01",
                    "2099-01-02",
                ],
                utc=True,
            ),
        }
    ).to_parquet(p)
    return tmp_path


def test_calendar_past_only_sorted(raw):
    assert pipeline.calendar(2025, raw) == ["Monaco Grand Prix", "Spanish Grand Prix"]
    assert pipeline.calendar(2025, raw, past_only=False)[-1] == "Future GP"


def test_resolve_events_substring_and_default(raw):
    assert pipeline.resolve_events([2025], ["monaco"], raw) == [(2025, "Monaco Grand Prix")]
    assert len(pipeline.resolve_events([2025], [], raw)) == 2
    with pytest.raises(ValueError):
        pipeline.resolve_events([2025], ["grand prix"], raw)  # ambiguous


def test_cli_events_and_bad_event(raw):
    r = CliRunner()
    out = r.invoke(app, ["events", "-s", "2025", "--raw", str(raw)])
    assert out.exit_code == 0 and "Monaco Grand Prix" in out.output
    bad = r.invoke(app, ["analyze", "-s", "2025", "-e", "nowhere", "--raw", str(raw)])
    assert bad.exit_code != 0


def test_analyze_not_ingested_is_skipped(raw, tmp_path):
    res = CliRunner().invoke(
        app,
        [
            "analyze",
            "-s",
            "2025",
            "-e",
            "monaco",
            "--raw",
            str(raw),
            "--derived",
            str(tmp_path / "d"),
        ],
    )
    assert res.exit_code == 0 and "not ingested" in res.output
