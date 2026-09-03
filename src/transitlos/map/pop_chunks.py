"""On-demand static population chunks for the map editor's client-side access recompute.

The scenario editor (`build.py`'s editor section, Phase 4) recomputes
`accessibility_score` **entirely in the browser**, from a plain static file
server -- no server-side logic at all. That recompute needs, for every H3
res-11 cell near a newly drawn/edited stop:

    (h3_cell, lat, lng, population, baseline level_of_service)

Boston's metro grid is 1.6M res-11 cells, so inlining all of it into
`map.html` is not an option (~100 MB of JSON). Instead this module
pre-partitions the grid into many small static JSON files -- the same
general idea as the vector tiles the rest of the map already serves, but far
simpler, because the recompute is a *straight-line* buffer operation
(explicit user instruction: "just do simple buffer operations, not through
the street network, but just straight lines, using the h3 cells with res
11") and therefore needs only a **point + two numbers** per cell, never a
hexagon polygon.

Partitioning scheme: a plain lat/lon degree grid, NOT H3 parents.
------------------------------------------------------------------
A coarse H3 parent (res 6/7) would be the "natural" key, but the client
would then need an H3 implementation in JavaScript just to work out *which*
chunk covers a given point -- a new third-party dependency on a page that
otherwise has none of its own. A fixed lat/lon degree grid is computable in
the browser with a single `Math.floor`, so the client stays dependency-free:

    key = f"{floor(lat / CHUNK_DEG)}_{floor(lon / CHUNK_DEG)}"

`CHUNK_DEG = 0.04` deg is ~4.4 km north-south and ~3.3 km east-west at
Boston's latitude (~14 km^2). A res-11 cell is ~2150 m^2, so a fully
populated chunk holds at most a few thousand rows -- a couple hundred KB of
JSON -- and the 2 km buffer around a single stop (the real pipeline's own
`walk_distance_steps[-1]` outer cutoff, which the JS mirrors) touches only
about 2x2 chunks. A route drawn anywhere in the metro therefore downloads a
small local slice, never the whole metro.

Only chunks that actually contain populated cells are written, and an
`index.json` lists every key that exists -- so the client never fires 404s
at empty ocean/forest.

Row format is a compact array-of-arrays rather than array-of-objects,
which roughly halves the byte count for identical information:

    {"key": "1063_-1772", "deg": 0.04, "res": 11,
     "cells": [["8b2a...ffff", 42.3512, -71.0634, 118.4, 0.5213], ...]}
            #    h3_cell,      lat,     lng,      pop,   level_of_service
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

# Directory name (relative to the saved `map.html`) the client fetches from,
# and the degree size of one chunk. Both are baked into the editor JS by
# `build.py` (`window.__popChunkCfg`) so the two sides can never disagree.
POP_CHUNK_DIRNAME = "pop_chunks"
POP_CHUNK_DEG = 0.04
POP_CHUNK_INDEX = "index.json"

# Coordinate rounding. 6 decimal places is ~11 cm -- far finer than a res-11
# cell (~25 m across) and far finer than a straight-line walk buffer cares
# about, while cutting ~40% of the bytes float repr would otherwise emit.
_COORD_DP = 6
_VALUE_DP = 4


def chunk_key(lat: float, lon: float, deg: float = POP_CHUNK_DEG) -> str:
    """The chunk key covering `(lat, lon)` -- mirrored exactly in the editor JS."""
    return f"{math.floor(lat / deg)}_{math.floor(lon / deg)}"


def build_population_chunks(
    h3_table: Any,
    out_dir: str,
    deg: float = POP_CHUNK_DEG,
    h3_column: str = "h3_cell",
    population_column: str = "population",
    access_column: str = "level_of_service",
    min_population: float = 0.0,
    resolution: int = 11,
) -> Dict[str, Any]:
    """Write the per-chunk population/access files plus their `index.json`.

    Args:
        h3_table: Any object exposing the needed columns as sequences --
            a Polars DataFrame (what `read_h3_grid_table` returns, and the
            cheapest option since no geometry is ever needed here), a pandas
            DataFrame or a GeoDataFrame all work.
        out_dir: Directory to write into (created if missing). Existing
            `*.json` files in it are removed first, so a rebuild never
            leaves stale chunks from a previous, differently-partitioned run
            behind for the client to fetch.
        deg: Chunk size in degrees. Must match the client's
            `window.__popChunkCfg.deg`.
        h3_column, population_column, access_column: Column names.
        min_population: Cells at/below this population are dropped. They
            carry zero weight in every consumer (population-weighted access,
            `impact = sum(gain * population)`), so keeping them would only
            inflate the download.
        resolution: H3 resolution recorded in each chunk's metadata (the
            grid's own native resolution; not re-derived per cell).

    Returns:
        A summary dict: `{"n_chunks", "n_cells", "bytes", "deg",
        "resolution", "dir"}`.
    """
    import h3 as _h3

    cells = list(h3_table[h3_column])
    pops = [float(v) if v is not None and v == v else 0.0 for v in h3_table[population_column]]
    if access_column in getattr(h3_table, "columns", []):
        accs = [float(v) if v is not None and v == v else 0.0 for v in h3_table[access_column]]
    else:
        accs = [0.0] * len(cells)

    buckets: Dict[str, list] = {}
    n_cells = 0
    for i, cell in enumerate(cells):
        pop = pops[i]
        if pop <= min_population:
            continue
        lat, lng = _h3.cell_to_latlng(cell)
        key = chunk_key(lat, lng, deg)
        buckets.setdefault(key, []).append(
            [cell, round(lat, _COORD_DP), round(lng, _COORD_DP),
             round(pop, _VALUE_DP), round(accs[i], _VALUE_DP)]
        )
        n_cells += 1

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.endswith(".json"):
            os.remove(os.path.join(out_dir, name))

    total_bytes = 0
    for key, rows in buckets.items():
        payload = {"key": key, "deg": deg, "res": resolution, "cells": rows}
        path = os.path.join(out_dir, key + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        total_bytes += os.path.getsize(path)

    index = {
        "deg": deg,
        "res": resolution,
        "n_chunks": len(buckets),
        "n_cells": n_cells,
        # A plain list of the keys that exist, so the client can skip
        # fetching (and 404ing on) chunks that hold no populated cells.
        "keys": sorted(buckets.keys()),
    }
    index_path = os.path.join(out_dir, POP_CHUNK_INDEX)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    total_bytes += os.path.getsize(index_path)

    return {
        "n_chunks": len(buckets),
        "n_cells": n_cells,
        "bytes": total_bytes,
        "deg": deg,
        "resolution": resolution,
        "dir": out_dir,
    }


def build_population_chunks_from_parquet(
    parquet_path: str,
    out_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper: read a cached `h3_grid.parquet` and chunk it.

    Reads only the three columns needed (`h3_cell`, `population`,
    `level_of_service`) -- never the WKB geometry column, which on Boston's grid
    is ~4 GB once decoded and is entirely unnecessary here.
    """
    import polars as pl

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(parquet_path)), POP_CHUNK_DIRNAME)
    df = pl.read_parquet(parquet_path, columns=["h3_cell", "population", "level_of_service"])
    return build_population_chunks(df, out_dir, **kwargs)


# ----------------------------------------------------------------------------
# GEOID <-> H3-cell chunks (2026-08-16): the census-polygon counterpart of the
# population chunks above.
#
# The scenario editor's post-Compute recolor already covers hexagons/circles
# by keying `window.__h3AccessOverrides` straight off each feature's own
# `h3_cell` id. Census polygons are keyed by GEOID instead and have no H3
# relationship on the tile itself, so that lookup is a documented no-op for
# them (see `build.py`'s `_h3_override_lookup_js`).
#
# Rather than bake a full GEOID -> [member h3 cells] table (which for a
# metro-scale study would mean one entry per h3 cell duplicated across a
# whole separate payload, on top of the population chunks that already list
# every one of those cells), this reuses the SAME lat/lon chunk grid the
# population chunks already partition the metro into. A chunk file here is
# just `{h3_cell: GEOID}` for every populated cell in that square. Because
# `computeAccess()` only ever fetches the population chunks covering the
# scenario's own stop buffers, fetching the matching GEOID chunks for
# those exact same keys is bounded by the same "just the drawn area" locality
# -- never the whole metro -- and adds only a short GEOID string per cell
# already being downloaded.
#
# SIMPLIFICATION (documented, not a bug): only the finest ("blockgroup")
# census level gets this treatment. A cell's GEOID is assigned by a plain
# centroid-in-polygon test with NO nearest-nonmatch fallback (unlike
# `code.pipeline._census_geometries_with_score`'s full aggregation, which
# does fall back for the ~8% of cells whose centroid lands just outside
# their true polygon) -- a cell that misses simply contributes nothing to
# any GEOID's live recolor and that polygon's fill falls back to its
# original baked score for that cell's share, which is a minor precision
# loss at polygon edges, not a correctness bug. Tract/county levels are left
# as followup: they would need the same join one level up (or a rollup of
# this blockgroup mapping via `code.pipeline`'s blockgroup->tract/county
# GEOID prefix relationship), out of scope for this pass.
# ----------------------------------------------------------------------------

GEOID_CHUNK_DIRNAME = "geoid_chunks"


def build_geoid_chunks(
    h3_cells: Any,
    lats: Any,
    lngs: Any,
    geoids: Any,
    out_dir: str,
    level: str = "blockgroup",
    deg: float = POP_CHUNK_DEG,
) -> Dict[str, Any]:
    """Write `{h3_cell: GEOID}` chunk files (same grid as `build_population_chunks`).

    Args:
        h3_cells, lats, lngs: Parallel sequences, one per h3 cell (lat/lng
            are that cell's own centroid, in degrees).
        geoids: Parallel sequence of GEOID strings (or `None` for a cell with
            no matched polygon -- dropped, see module docstring above).
        out_dir: Directory to write into (existing `*.json` removed first).
        level: Census level name recorded in each chunk's metadata.
        deg: Chunk size in degrees; must match `build_population_chunks`'s.

    Returns:
        Summary dict: `{"n_chunks", "n_cells", "bytes", "deg", "level", "dir"}`.
    """
    buckets: Dict[str, Dict[str, str]] = {}
    n_cells = 0
    for cell, lat, lng, geoid in zip(h3_cells, lats, lngs, geoids):
        if geoid is None:
            continue
        key = chunk_key(float(lat), float(lng), deg)
        buckets.setdefault(key, {})[cell] = geoid
        n_cells += 1

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.endswith(".json"):
            os.remove(os.path.join(out_dir, name))

    total_bytes = 0
    for key, mapping in buckets.items():
        payload = {"key": key, "deg": deg, "level": level, "map": mapping}
        path = os.path.join(out_dir, key + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        total_bytes += os.path.getsize(path)

    index = {
        "deg": deg,
        "level": level,
        "n_chunks": len(buckets),
        "n_cells": n_cells,
        "keys": sorted(buckets.keys()),
    }
    index_path = os.path.join(out_dir, POP_CHUNK_INDEX)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    total_bytes += os.path.getsize(index_path)

    return {
        "n_chunks": len(buckets),
        "n_cells": n_cells,
        "bytes": total_bytes,
        "deg": deg,
        "level": level,
        "dir": out_dir,
    }


def build_geoid_chunks_from_h3_and_census(
    h3_table: Any,
    census_gdf: Any,
    out_dir: str,
    id_col: str = "GEOID",
    level: str = "blockgroup",
    deg: float = POP_CHUNK_DEG,
    h3_column: str = "h3_cell",
) -> Dict[str, Any]:
    """Convenience wrapper: join an h3 grid table onto a census polygon layer, then chunk it.

    Args:
        h3_table: Object exposing `h3_column` as a sequence (Polars/pandas
            DataFrame, or a GeoDataFrame -- geometry, if present, is ignored;
            centroids are computed H3-natively, matching
            `code.pipeline._census_geometries_with_score`'s own approach).
        census_gdf: Census polygon GeoDataFrame (e.g. `census_by_level["blockgroup"]`).
        out_dir: Directory to write chunk files into.
        id_col: GEOID column name on `census_gdf`.
        level, deg: See `build_geoid_chunks`.
        h3_column: H3 cell id column name on `h3_table`.
    """
    import h3 as _h3
    import geopandas as gpd
    import pandas as pd

    cells = list(h3_table[h3_column])
    latlngs = [_h3.cell_to_latlng(c) for c in cells]
    lats = [ll[0] for ll in latlngs]
    lngs = [ll[1] for ll in latlngs]

    pts = gpd.GeoDataFrame(
        {"_row": range(len(cells))},
        geometry=gpd.points_from_xy(lngs, lats),
        crs=4326,
    ).to_crs(census_gdf.crs)
    joined = gpd.sjoin(pts, census_gdf[[id_col, "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="_row").sort_values("_row")
    geoid_by_row = pd.Series(joined[id_col].to_numpy(), index=joined["_row"].to_numpy())
    geoids = [geoid_by_row.get(i) for i in range(len(cells))]
    geoids = [None if g is None or g != g else str(g) for g in geoids]

    return build_geoid_chunks(cells, lats, lngs, geoids, out_dir, level=level, deg=deg)
