# transitLOS

A **transit level-of-service score** from 0 to 1 for every place: how good the public transport within walking distance
is.

The score of a place comes from two things:

1. **The stops it can walk to**, each rated by its mode (bus, tram, rail), how often vehicles come (headway) and how
   fast they go. The speed is the straight-line distance a vehicle gets in the 15 minutes before and after the stop,
   so winding lines count as slower.
2. **How far the walk is**: a stop keeps its full value up to a short walk and loses value beyond it.

The best stop decides the score. Results come per H3 cell and per census unit. `transitlos.map` turns them into
interactive maps: hexagons and census areas, a stats panel and a what-if editor.

transitLOS builds on four sibling packages:

| Package | Used for |
|---|---|
| geohierarchy | moving data between H3 cells and census levels |
| pyCensus | census and population data |
| pyGTFSHandler | reading the GTFS feeds: stops, headways, speeds |
| UrbanAccessAnalyzer | the street network and walking distances |

## Install

```bash
uv sync      # the four sibling packages are expected as sibling folders (see pyproject.toml)
```

## Example

The notebooks in `examples/` compute the score for a real area
(`transit_level_of_service_walkthrough.ipynb`) and break it down by population group
(`equity_analysis_walkthrough.ipynb`).

```python
from transitlos.stops import download_and_prepare_stops

stops = download_and_prepare_stops("path/to/gtfs", aoi=area)   # one row per station, with its stop score
```

## Stop speed

`speed_method="straight"` (default) or `"shape"` (distance along the route), with `window_min=15` minutes on each side
of the stop and `window_direction="both"`. These options exist on every function that scores stops; the computation
itself is pyGTFSHandler's `Feed.get_window_distances`.

## Caching

Nothing is written to the current folder. Pass `cache_dir=` to keep street networks between runs, or set
`TRANSITLOS_CACHE_DIR`. Census and GTFS caches belong to pyCensus (`PYCENSUS_CACHE_DIR`) and pyGTFSHandler
(`PYGTFSHANDLER_CACHE_DIR`).

## Recomputing with other parameters

[Flexible scoring](FLEX_SCORING.md): stations and cells are stored once, so the score can be recomputed with other
curves or weights in milliseconds, also in the browser.
