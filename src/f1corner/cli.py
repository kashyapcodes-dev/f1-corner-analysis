"""`f1` command line: ingest / analyze / report / run.

f1 events -s 2025
f1 ingest -s 2025 -e monaco -e spain
f1 analyze -s 2025            # every past qualifying session
f1 report -s 2025 -s 2026 --robustness
f1 run -s 2025 -e monaco      # ingest + analyze + report
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import cache, pipeline

app = typer.Typer(no_args_is_help=True, add_completion=False, help="F1 Corner DNA pipeline.")

Season = Annotated[list[int], typer.Option("--season", "-s", help="Season (repeatable).")]
Event = Annotated[
    list[str] | None,
    typer.Option("--event", "-e", help="Event name or substring (repeatable). Default: all past."),
]
Source = Annotated[str, typer.Option("--source", help="Telemetry backend: openf1 | fastf1.")]
Force = Annotated[bool, typer.Option("--force", help="Redo even if cached.")]
Raw = Annotated[Path, typer.Option(help="Raw cache root.")]
Derived = Annotated[Path, typer.Option(help="Derived tables root.")]
Out = Annotated[Path, typer.Option("--out", "-o", help="Markdown output.")]


def _pairs(seasons, events, raw):
    try:
        pairs = pipeline.resolve_events(seasons, events or [], raw)
    except (ValueError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from e
    if not pairs:
        raise typer.BadParameter("no events selected")
    return pairs


@app.command()
def events(season: Season, source: Source = "openf1", raw: Raw = cache.RAW):
    """List qualifying sessions per season and whether each is ingested/analyzed."""
    for s in season:
        pipeline.ensure_schedule(s, raw)
        for e in pipeline.calendar(s, raw, past_only=False):
            tel = cache.path(source, "car_data", s, e, pipeline.SESSION, raw).exists()
            der = pipeline.has_derived(source, s, e)
            typer.echo(
                f"{s}  {e:<32} ingested={'y' if tel else '-'} analyzed={'y' if der else '-'}"
            )


@app.command()
def ingest(
    season: Season,
    event: Event = None,
    source: Source = "openf1",
    force: Force = False,
    raw: Raw = cache.RAW,
):
    """Fetch qualifying telemetry + official classification into the Parquet cache."""
    from .sources import get_source

    src = get_source(source)
    pairs = _pairs(season, event, raw)
    for i, (s, e) in enumerate(pairs, 1):
        typer.echo(f"[{i}/{len(pairs)}] ingest {s} {e}")
        pipeline.ingest_event(src, s, e, raw, force)


@app.command()
def analyze(
    season: Season,
    event: Event = None,
    source: Source = "openf1",
    force: Force = False,
    raw: Raw = cache.RAW,
    derived: Derived = pipeline.DERIVED,
):
    """Corner segmentation + per-lap corner times for cached sessions -> data/derived."""
    pairs = _pairs(season, event, raw)
    for i, (s, e) in enumerate(pairs, 1):
        tag = f"[{i}/{len(pairs)}] analyze {s} {e}"
        if pipeline.has_derived(source, s, e, derived) and not force:
            typer.echo(f"{tag}: cached")
            continue
        try:
            res = pipeline.analyze_event(source, s, e, raw)
        except pipeline.WetSession:
            typer.secho(f"{tag}: wet session, excluded", fg="yellow")
            continue
        except FileNotFoundError:
            typer.secho(f"{tag}: not ingested (run `f1 ingest`)", fg="yellow")
            continue
        pipeline.write_derived(res, source, s, e, derived)
        m = res["meta"]
        typer.echo(f"{tag}: pole {m['pole_driver']} {m['lap_s']:.3f}s, {m['n_corners']} corners")


def _report(season, event, source, out, robustness, raw, derived):
    from . import report as R
    from .robustness import DEFAULT, headline, run

    pairs = _pairs(season, event, raw)
    try:
        inp, metas, missing = pipeline.load_inputs(source, pairs, derived, raw)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from e
    if robustness:
        base, flags = run(inp)
    else:
        base, flags = headline(inp, DEFAULT), None
    text = R.build(base, metas, missing, flags)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    typer.echo(R.pivot_pace(base).round(2).to_string(index=False))
    typer.echo(f"\nk={base['k']}  events={len(metas)}  wrote {out}")


@app.command()
def report(
    season: Season,
    event: Event = None,
    source: Source = "openf1",
    out: Out = Path("reports/report.md"),
    robustness: Annotated[bool, typer.Option(help="Also re-run under every alternative.")] = False,
    raw: Raw = cache.RAW,
    derived: Derived = pipeline.DERIVED,
):
    """Team pace + corner-class profile (+ Tests A/B, robustness) from analyzed sessions."""
    _report(season, event, source, out, robustness, raw, derived)


@app.command()
def run(
    season: Season,
    event: Event = None,
    source: Source = "openf1",
    force: Force = False,
    out: Out = Path("reports/report.md"),
    robustness: bool = False,
    raw: Raw = cache.RAW,
    derived: Derived = pipeline.DERIVED,
):
    """ingest -> analyze -> report."""
    ingest(season, event, source, force, raw)
    analyze(season, event, source, force, raw, derived)
    _report(season, event, source, out, robustness, raw, derived)


if __name__ == "__main__":
    app()
