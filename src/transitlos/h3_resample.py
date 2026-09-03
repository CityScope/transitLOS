"""Resample street-edges `level_of_service` onto an H3 grid.

This is the transit-specific counterpart to `population.cross_with_population`
(which targets a population polygon layer via `geohierarchy.edges_to_level`,
an overlap-based Mean by default). H3 access-score resampling has different
semantics, spelled out by the product spec this module implements directly:

    Each H3 cell gets the *maximum* level_of_service of any street that
    intersects/nears it (not an average -- a cell touched by both a great
    street and a mediocre one should read as "well served", not blended
    down). Every cell must end up with a value: a cell with no street
    within the normal search margin falls back to its single nearest
    street, as long as that street is within `fallback_radius_multiplier *
    cell_edge_length` -- cells with literally nothing that close keep an
    explicit "no access" sentinel instead of a silent null.

The actual geometry work (line-to-polygon distance matching, the
nearest-neighbor fallback, and the `Max` aggregation itself) all happens in
`geohierarchy.edges.edges_to_h3_by_distance` -- a generic, transit-agnostic
primitive. This module only supplies the transit-specific policy on top of
it: which column, which aggregation (`Max`), what fallback radius, and what
sentinel value an uncovered cell gets.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import polars as pl
from geohierarchy.aggregation import Max
from geohierarchy.edges import edges_to_h3_by_distance
from geohierarchy.utils import h3_cells


def level_of_service_to_h3(
    access_gdf: gpd.GeoDataFrame,
    resolution: int,
    aoi: Optional[gpd.GeoDataFrame] = None,
    score_col: str = "level_of_service",
    margin_m: float = 10.0,
    fallback_radius_multiplier: Optional[float] = 5.0,
    no_access_value: float = 0.0,
) -> gpd.GeoDataFrame:
    """Resample a street-edges `level_of_service` column onto an H3 grid.

    Each returned H3 cell holds the maximum `score_col` among streets that
    intersect it (or, absent that, its single nearest street within
    `fallback_radius_multiplier` cell-edge-lengths) -- see module docstring
    for the full policy. Cells with no qualifying street at all get
    `no_access_value` rather than a null, matching the `fillna(0.0)`
    convention used elsewhere in the downstream pipeline for "no coverage".

    Args:
        access_gdf: Street-edges GeoDataFrame with `score_col` (e.g. the
            output of `transitlos.level_of_service.compute_level_of_service`).
        resolution: H3 resolution the grid is built at (typically the
            maximum/finest resolution configured for the AOI).
        aoi: Geometry whose extent the H3 grid should cover. Defaults to
            `access_gdf`'s own extent (buffered implicitly by
            `edges_to_h3_by_distance`'s search margin) if not given.
        score_col: Column in `access_gdf` to resample.
        margin_m: Extra distance (in meters) beyond one cell's own
            footprint still counted as "touching" a street -- forwarded to
            `edges_to_h3_by_distance`.
        fallback_radius_multiplier: How many cell-edge-lengths a cell may
            look for its nearest street once the normal margin turns up
            nothing. `None` disables the fallback (matches
            `edges_to_h3_by_distance`'s own default meaning).
        no_access_value: Value assigned to a cell that found no street
            even within the fallback radius.

    Returns:
        GeoDataFrame with `"h3"`, `"geometry"`, and `score_col` -- one row
        per H3 cell covering `aoi` (or `access_gdf`'s extent), every row
        populated.
    """
    if access_gdf.empty:
        raise ValueError("access_gdf is empty -- nothing to resample.")

    grid_source = aoi if aoi is not None else access_gdf
    grid = h3_cells(grid_source, resolution=resolution)
    all_cells = grid["h3"].tolist()

    scored = edges_to_h3_by_distance(
        access_gdf,
        all_cells,
        columns=score_col,
        resolution=resolution,
        margin_m=margin_m,
        agg=Max(),
        fallback_radius_multiplier=fallback_radius_multiplier,
    )

    covered = set(scored["h3_cell"].to_list())
    missing = [c for c in all_cells if c not in covered]

    if missing:
        filler = pl.DataFrame({"h3_cell": missing, score_col: [no_access_value] * len(missing)})
        scored = pl.concat([scored, filler], how="vertical")

    scored_pd = scored.rename({"h3_cell": "h3"}).to_pandas()
    result = grid.merge(scored_pd, on="h3", how="left")
    # Belt-and-suspenders: any row the merge still left null (shouldn't
    # happen given the filler above, but a duplicate/degenerate h3 id
    # would otherwise silently slip through as NaN) gets the sentinel too.
    result[score_col] = result[score_col].fillna(no_access_value)
    return result
