# F1 Corner DNA

Classify every F1 corner from open telemetry, measure each team's strength by corner type, and test whether that strength predicts qualifying and race results. Classical ML only (clustering, ridge regression), no LLMs.

**Status:** early. Only the data ingestion layer exists so far. Design spec is in `notes.txt` (local, gitignored).

## Setup

### 1. Install uv

```sh
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

Check with `uv --version`. Python 3.14+ is required; uv will download it if missing.

### 2. Install from the lockfile

`uv.lock` pins exact versions. Commit it.

```sh
uv sync --locked            # core + dev tools (OpenF1 backend), fails if lock is out of date
uv sync --locked --extra fastf1   # + optional FastF1 backend
```

Dev tools (pytest, ruff, pre-commit, poethepoet) live in the `dev` dependency group, which `uv sync` installs by default; skip with `--no-dev`. Use `--extra` multiple times to combine extras. After editing `pyproject.toml`, refresh the lock with `uv lock`, then `uv sync`.

Run things in the env with `uv run`, e.g. `uv run pytest`.

## Linting, formatting, hooks

Ruff handles both linting and formatting (its formatter is black-compatible, so black is not used). Tasks run via [poethepoet](https://poethepoe.natn.io/):

```sh
uv run poe fix     # ruff check --fix + ruff format (auto-fix everything)
uv run poe check   # same checks, no changes (what CI would run)
```

Pre-commit runs ruff check, ruff format and the unit tests on every commit. Enable it once per clone:

```sh
uv sync --all-extras
uv run pre-commit install
```

A failed hook aborts the commit; auto-fixes are left unstaged, so review, `git add`, and commit again. Run all hooks manually with `uv run pre-commit run --all-files`.

## Ingestion layer

Code lives in `src/f1corner/`.

```
schema.py              canonical tables every backend must produce
cache.py               Parquet cache + manifest
sources/base.py        Source protocol
sources/openf1_source.py    open backend (no sign-in)
sources/fastf1_source.py    optional backend (F1TV subscription)
```

### Design

- **Source-agnostic.** Each backend returns the same canonical tables (`sessions`, `laps`, `car_data`, `location`, `weather`, `race_control`). Everything downstream reads only those, so FastF1 is never required.
- **FastF1 is optional.** It adds history from 2018, no rate limits and built-in lap flags. It needs an F1TV Access subscription and `--extra fastf1`. The backend is imported lazily.
- **OpenF1 is the open default.** No sign-in, limited to 3 requests/s and 30/min (throttled in the client). Both sources come from the same live timing feed, so telemetry is about 3.7 Hz either way.
- **Cache first.** Analysis code never calls an API. Data is fetched once into Parquet, partitioned by source, table, season, event and session:

  ```
  data/raw/<source>/<table>/season=2025/event=<slug>/session=<slug>.parquet
  data/raw/manifest.jsonl    # pull time, source, table, session
  ```

### Usage

```python
from f1corner.sources import get_source
from f1corner import cache

src = get_source("fastf1")  # or "openf1"
cache.fetch(src, "laps", 2025, "Monaco Grand Prix", "Qualifying")
laps = cache.load("fastf1", "laps", 2025, "Monaco Grand Prix", "Qualifying")
```

### FastF1 credentials (optional)

FastF1 needs your F1TV login. The template is `local/.env.example`. Copy it to a root `.env` (gitignored; never commit it):

```sh
cp local/.env.example .env   # then fill in the values
```

| Variable | Used for |
|---|---|
| `F1_USERNAME` | F1TV account email (FastF1 login) |
| `F1_PASSWORD` | F1TV account password |

Only the root `.env` is read. Real environment variables win over it.

Without it, `FastF1Source` raises `SourceUnavailable` with this hint and everything else still works.

### What works today

| Piece | State |
|---|---|
| Canonical schema (`schema.py`) | done |
| Parquet cache + manifest (`cache.py`) | done, tested |
| OpenF1 backend | all six tables mapped; **verified live** on 2025 Monaco Qualifying |
| FastF1 backend | mapping unit-tested on fake frames; **not yet verified live** (needs F1TV credentials) |

Conventions: `lap_start_s`, weather and race-control `t_s` are UTC epoch seconds; car data and location `t_s` are seconds since lap start.

## Tests

```sh
uv run pytest             # tests/unit: offline, fast, no network (default)
uv run pytest tests/live  # tests/live: real APIs, one race (2025 Monaco GP Qualifying)
```

- Live OpenF1 tests need no sign-in. The first run takes ~2 min because of rate limits, then data is cached in `data/cache/test` (gitignored).
- Live FastF1 tests and the OpenF1-vs-FastF1 cross-check are skipped unless `F1_USERNAME`/`F1_PASSWORD` are set.

## Data licensing

- `data/raw/`, `data/cache/`, `.fastf1-cache/` and all `*.parquet` files are gitignored. The only exception is `data/published/`, for derived outputs such as corner labels.
- FastF1 data comes from an F1TV subscription (personal use). **Never commit or publish raw or cached FastF1 data.**
- OpenF1 is licensed for personal use too.
