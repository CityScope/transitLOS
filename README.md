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

The scoring formulas and their default constants are in `src/transitlos/scoring/` (`parameters.py`).

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

## Use as a library

transitlos never writes into the current working directory. `prepare_street_network(aoi, cache_dir=...)` /
`level_of_service(..., cache_dir=...)` persist the simplified street network only when `cache_dir` is given
(`None` = no caching). `transitlos.cache.get_cache_dir(namespace, cache_dir=None)` resolves a cache folder:
explicit argument, else `$TRANSITLOS_CACHE_DIR/<namespace>`, else `<tempdir>/transitlos-cache/<namespace>`.
Census/GTFS caches are configured in pycensus (`PYCENSUS_CACHE_DIR`) and pyGTFSHandler (`PYGTFSHANDLER_CACHE_DIR`).

```python
from transitlos.network import prepare_street_network
net = prepare_street_network(aoi, cache_dir="/data/my_project/cache/streets", pbf_path="region.osm.pbf")
```

OSM street extracts (`.osm.pbf`) use the `osm` namespace (`transitlos.cache.get_osm_dir()`); when `pbf_path` is omitted the extract is downloaded there.


## Flexible scoring

`transitlos.flex` recomputes the scores after processing, with other parameters, curve shapes or aggregation, on the server or in the browser: see [docs/FLEX_SCORING.md](docs/FLEX_SCORING.md).

## Large files

A pre-commit hook (`tools/check_large_files.py`, installed with `python3 tools/check_large_files.py --install`) refuses a commit with a file over 10 MB once: run the same commit again to keep it. Files over 50 MB are never committed.

## Local files

This repository is public. API keys (`api_keys.json`, `.env`), notes about a particular machine or its processes (hosting, uploads,
local paths) and upload scripts are never committed: keep notes as `*.local.md` or under `local/`, which `.gitignore` excludes.
