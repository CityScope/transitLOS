"""Who does each stop serve? Counts (population, jobs, any additive census column) of the H3 cells each stop WINS.

A stop wins a cell when it is the station that gives the cell its level of service (the station of the winning slot, `core.recompute_cells*`).
Which stop wins depends on the scoring, so the catchment is recomputed together with the scores.

    counts = block_counts_to_cells(blocks_gdf, cells, ["population", "lodes_wac_jobsTotal"])    # [nc, ncol] counts per served cell
    los, slot, prof = core.recompute_cells_fast(res.station, res.dist, prof, params)
    winner = winner_station(res.station, slot)                                                      # station index per cell, -1 none
    totals = stop_catchment(winner, counts.to_numpy(), n_stations)                                  # [S, ncol]

`block_counts_to_cells` spreads a polygon layer's counts over the cells (uniform density inside each polygon): a polygon larger than ~2 cells gives each
served cell whose centre lies inside it `cell_area / polygon_area` of its count; a smaller polygon gives its whole count to the cell of its representative point.
Counts of people in cells that no stop reaches are simply not attributed to any stop.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def winner_station(cell_station: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """Station index that wins each cell (-1 when the cell has no station): `cell_station[c, slot[c]]`.

    Args:
        cell_station: ``[cells, K]`` station index per slot.
        slot: Winning slot per cell (-1: none).

    Returns:
        Station index per cell.
    """
    cs = np.asarray(cell_station)
    sl = np.asarray(slot).astype(np.int64)
    ok = sl >= 0
    out = np.full(len(cs), -1, dtype=np.int64)
    out[ok] = cs[np.flatnonzero(ok), sl[ok]]
    return out


def stop_catchment(winner: np.ndarray, counts: np.ndarray, n_stations: int) -> np.ndarray:
    """[S, ncol] sum of `counts` [nc, ncol] over the cells each station wins; also returns nothing for cells without a winner.

    Args:
        winner: Winning station per cell (-1: none).
        counts: ``[cells, columns]`` counts (population, jobs...).
        n_stations: Number of stations.

    Returns:
        ``[stations, columns]`` sums.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts[:, None]
    ok = winner >= 0
    out = np.zeros((n_stations, counts.shape[1]))
    for j in range(counts.shape[1]):
        out[:, j] = np.bincount(winner[ok], weights=counts[ok, j], minlength=n_stations)[:n_stations]
    return out


def cells_won(winner: np.ndarray, n_stations: int) -> np.ndarray:
    """Number of cells each station wins.

    Args:
        winner: Winning station per cell (-1: none).
        n_stations: Number of stations.

    Returns:
        Cells per station.
    """
    return np.bincount(winner[winner >= 0], minlength=n_stations)[:n_stations]


def cell_centres(cells: np.ndarray):
    """(lat, lon) arrays of H3 cell centres (vectorised).

    Args:
        cells: uint64 H3 cell ids.

    Returns:
        ``(lat, lon)`` arrays.
    """
    from h3ronpy.vector import cells_to_coordinates
    r = cells_to_coordinates(np.asarray(cells, dtype=np.uint64))
    return np.asarray(r.column("lat")), np.asarray(r.column("lng"))


def block_counts_to_cells(blocks, cells: np.ndarray, columns: Sequence[str], resolution: int = 10) -> pd.DataFrame:
    """Counts per served cell from a polygon layer (GeoDataFrame, EPSG:4326) with additive `columns`. Returns a DataFrame indexed like `cells`.

    Args:
        blocks: Polygons (EPSG:4326) with additive columns.
        cells: Sorted unique uint64 H3 ids.
        columns: Additive columns to move onto the cells.
        resolution: H3 resolution of the cells.

    Raises:
        ValueError: If ``cells`` is not sorted and unique.
    """
    import geopandas as gpd
    import h3

    cells = np.asarray(cells, dtype=np.uint64)
    if len(cells) > 1 and not (np.diff(cells.astype(np.float64)) > 0).all():
        raise ValueError("cells must be sorted and unique")
    out = np.zeros((len(cells), len(columns)))
    if len(cells) == 0 or len(blocks) == 0:
        return pd.DataFrame(out, columns=list(columns))
    cell_area = h3.average_hexagon_area(resolution, unit="m^2")
    lat, lon = cell_centres(cells)
    pts = gpd.GeoDataFrame({"ci": np.arange(len(cells))}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
    b = blocks[list(columns) + ["geometry"]].reset_index(drop=True).copy()
    b["area"] = b.to_crs(5070).geometry.area.to_numpy()                      # equal-area metres (CONUS albers; fine for the shares, other regions: pass a projected layer)
    large = b["area"] >= 2.0 * cell_area
    if large.any():
        j = gpd.sjoin(pts, b[large].reset_index().rename(columns={"index": "bi"}), how="inner", predicate="within")
        w = (cell_area / j["area"].to_numpy())
        for k, col in enumerate(columns):
            np.add.at(out[:, k], j["ci"].to_numpy(), w * j[col].to_numpy(dtype=np.float64))
    small = b[~large]
    if len(small):
        from h3ronpy.vector import coordinates_to_cells
        rp = small.geometry.representative_point()
        scell = np.asarray(coordinates_to_cells(np.asarray(rp.y), np.asarray(rp.x), resolution)).astype(np.uint64)
        pos = np.searchsorted(cells, scell)                                  # cells are sorted (unique)
        hit = (pos < len(cells)) & (cells[np.minimum(pos, len(cells) - 1)] == scell)
        for k, col in enumerate(columns):
            np.add.at(out[:, k], pos[hit], small[col].to_numpy(dtype=np.float64)[hit])
    return pd.DataFrame(out, columns=list(columns))
