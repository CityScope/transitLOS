# transitLOS

`transitLOS` computes a **transit level of service (LOS)** accessibility score by tying
together three previously separate pieces of infrastructure in the AccessibilityMIT /
City Scope project family:

- **[geohierarchy](https://github.com/CityScope/geohierarchy)** — nested geographic unit
  hierarchies (e.g. block/tract/place, or H3 cells) used to aggregate and join spatial
  data consistently.
- **[pyCensus](https://github.com/CityScope/pyCensus)** — common-schema loaders for
  national census/population sources, keyed to a `geohierarchy.GeoHierarchy`.
- **[pyGTFSHandler](https://github.com/CityScope/pyGTFSHandler)** — GTFS transit feed
  parsing, giving stop locations, frequencies, and schedule/reliability data.
- **[UrbanAccessAnalyzer](https://github.com/CityScope/UrbanAccessAnalyzer)** —
  street-network routing and walk-distance/walk-time accessibility analysis.

Combining these, `transitLOS` produces a single **0-1 transit accessibility score** per
location/population unit, based on: (1) the quality of nearby transit stops (mode/speed,
frequency, reliability) and (2) the walking accessibility of those stops from the street
network. It also renders the results as an interactive MapLibre map (`transitlos.map`)
with hexagon/circle/census layers, a stats panel (distribution, regression, ANOVA,
place-rank), and a what-if scenario editor.

The literature review and methodology backing the score's parameters (kept locally
under `documentation/`, not part of this repo) documents the reasoning behind every
constant used in `src/transitlos/scoring/parameters.py`.

## Install

```bash
uv sync
```

During development this package depends on `geohierarchy`, `pyCensus`, `pyGTFSHandler`,
and `UrbanAccessAnalyzer` as local editable path dependencies (see
`[tool.uv.sources]` in `pyproject.toml`) — check those repos out as sibling directories.
Once all four are published, `transitLOS` can instead depend on their git URLs directly.

## Usage

See `examples/transit_level_of_service_walkthrough.ipynb` for computing a transit LOS
score for a real area, and `examples/equity_analysis_walkthrough.ipynb` for the
equity/demographic breakdown of accessibility results.

## Tests

```bash
uv run pytest
```

## Structure

- `src/transitlos/` — scoring pipeline (`scoring/`) and map rendering (`map/`).
- `examples/` — runnable notebooks.
- `tests/` — unit tests.
