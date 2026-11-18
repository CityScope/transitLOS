"""Neighbourhood (buffer) statistics of H3 grids.

`density_within(cells, values, radius_m=1000)`: for every H3 cell, the sum of `values` over the grid's cells whose centres lie within
`radius_m` of its centre, divided by those cells' total area (per km2). The grid is rasterised (cell centres -> pixels of `pixel_m`) and
convolved with a disk, tile by tile with a halo, so a whole US state at resolution 11 fits in memory.
"""
from __future__ import annotations

import math

import numpy as np

# Residents per km2 within each radius of the cell (user 2026-11-14: 500 m, 1 km, 2 km and 3 km on every map and census layer;
# 2026-11-16: also 4 km and 5 km, to find the radius with the highest R2 against the score)
DENSITY_RADII = (("pop_density_500m", 500.0), ("pop_density_1km", 1000.0), ("pop_density_2km", 2000.0), ("pop_density_3km", 3000.0),
                 ("pop_density_4km", 4000.0), ("pop_density_5km", 5000.0))
DENSITY_COLUMNS = tuple(c for c, _ in DENSITY_RADII)


def _local_xy(lat: np.ndarray, lng: np.ndarray):
    lat0, lng0 = float(np.nanmean(lat)), float(np.nanmean(lng))
    x = (lng - lng0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_574.0
    return x, y


def density_within(cells, values, radius_m: float = 1000.0, pixel_m: float = 50.0, tile_px: int = 4096) -> np.ndarray:
    """Per cell: sum(values) / sum(cell area km2) over the grid cells whose centres are within `radius_m` (the cell itself included).

    Args:
        cells: H3 cell ids.
        values: Value per cell (residents).
        radius_m: Radius (m).
        pixel_m: Raster pixel size (m) of the computation.
        tile_px: Tile size in pixels.

    Returns:
        Density per cell (per km2).
    """
    return density_within_multi(cells, values, (radius_m,), pixel_m, tile_px)[0]


def density_within_multi(cells, values, radii, pixel_m: float = 50.0, tile_px: int = 4096) -> list:
    """`density_within` for several radii, the cell centres and areas computed once: a list of arrays in the order of `radii`.

    Args:
        cells: H3 cell ids.
        values: Value per cell.
        radii: Radii (m).
        pixel_m: Raster pixel size (m).
        tile_px: Tile size in pixels.
    """
    import h3
    from scipy.signal import fftconvolve

    cells = [str(c) for c in cells]
    n = len(cells)
    if n == 0:
        return [np.full(0, np.nan) for _ in radii]
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    area = np.array([h3.cell_area(c, unit="km^2") for c in cells])
    v = np.nan_to_num(np.asarray(values, dtype=float))
    x, y = _local_xy(ll[:, 0], ll[:, 1])
    res = h3.get_resolution(cells[0])
    pixel_m = max(pixel_m, 0.5 * math.sqrt(h3.average_hexagon_area(res, unit="m^2")))   # coarse grids: about two pixels per cell
    ix = np.floor((x - x.min()) / pixel_m).astype(np.int64)
    iy = np.floor((y - y.min()) / pixel_m).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    outs = []
    for radius_m in radii:
        out = np.full(n, np.nan)
        r = int(math.ceil(radius_m / pixel_m))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        kernel = ((xx * pixel_m) ** 2 + (yy * pixel_m) ** 2 <= radius_m ** 2).astype(np.float64)
        core = max(tile_px - 2 * r, 256)
        for tx in range(0, nx, core):
            for ty in range(0, ny, core):
                inside = (ix >= tx) & (ix < tx + core) & (iy >= ty) & (iy < ty + core)
                if not inside.any():
                    continue
                x0, y0 = tx - r, ty - r
                w, h = min(core, nx - tx) + 2 * r, min(core, ny - ty) + 2 * r
                halo = (ix >= x0) & (ix < x0 + w) & (iy >= y0) & (iy < y0 + h)
                P = np.zeros((h, w)); A = np.zeros((h, w))
                np.add.at(P, (iy[halo] - y0, ix[halo] - x0), v[halo])
                np.add.at(A, (iy[halo] - y0, ix[halo] - x0), area[halo])
                cp = fftconvolve(P, kernel, mode="same"); ca = fftconvolve(A, kernel, mode="same")
                py, px = iy[inside] - y0, ix[inside] - x0
                a = ca[py, px]
                out[inside] = np.where(a > 1e-9, np.maximum(cp[py, px], 0.0) / np.where(a > 1e-9, a, 1.0), np.nan)
        outs.append(out)
    return outs


def add_densities(df, cell_col: str = "h3", pop_col: str = "population"):
    """`df` with every DENSITY_RADII column (residents per km2 within 500 m / 1 / 2 / 3 / 4 / 5 km); unchanged without the two columns.

    Args:
        df: Table of cells.
        cell_col: H3 id column.
        pop_col: Residents column.
    """
    if df is None or len(df) == 0 or cell_col not in df.columns or pop_col not in df.columns:
        return df
    df = df.copy()
    vals = density_within_multi(df[cell_col].astype(str).to_numpy(), df[pop_col].to_numpy(dtype=float), [r for _, r in DENSITY_RADII])
    for (col, _), v in zip(DENSITY_RADII, vals):
        df[col] = v
    return df


def add_density_1km(df, cell_col: str = "h3", pop_col: str = "population", out_col: str = "pop_density_1km"):
    """`df` with `out_col` = residents per km2 within 1 km of each cell (see `density_within`); unchanged without the two columns.

    Args:
        df: Table of cells.
        cell_col: H3 id column.
        pop_col: Residents column.
        out_col: Output column.
    """
    if df is None or len(df) == 0 or cell_col not in df.columns or pop_col not in df.columns:
        return df
    df = df.copy()
    df[out_col] = density_within(df[cell_col].astype(str).to_numpy(), df[pop_col].to_numpy(dtype=float))
    return df


def densities_to_polygons(polys, cells, cols=DENSITY_COLUMNS, cell_col: str = "h3", pop_col: str = "population"):
    """`polys` with each of `cols` = residents-weighted mean of that cell column over the H3 cells whose centre lies in each polygon (the
    densities its residents live in); NaN where no populated cell centre falls inside. One spatial join for all columns.

    Args:
        polys: Polygons.
        cells: Cells with the density columns.
        cols: Density columns.
        cell_col: H3 id column.
        pop_col: Residents column.
    """
    import geopandas as gpd
    import h3
    import pandas as pd

    cols = [c for c in cols if c in cells.columns]
    c = cells[[cell_col, pop_col] + cols].copy()
    c = c[pd.to_numeric(c[pop_col], errors="coerce").fillna(0) > 0]
    out = polys.copy()
    if not len(c) or not len(out) or not cols:
        for k in cols:
            out[k] = np.nan
        return out
    ll = np.array([h3.cell_to_latlng(str(x)) for x in c[cell_col]])
    pts = gpd.GeoDataFrame({"_w": c[pop_col].to_numpy(dtype=float), **{f"_d{i}": c[k].to_numpy(dtype=float) for i, k in enumerate(cols)}},
                           geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326).to_crs(out.crs or 4326)
    left = gpd.GeoDataFrame({"_pi": np.arange(len(out))}, geometry=out.geometry.to_numpy(), crs=out.crs or 4326)
    j = gpd.sjoin(pts, left, predicate="within", how="inner")
    for i, k in enumerate(cols):
        ok = np.isfinite(j[f"_d{i}"].to_numpy())
        jj = j[ok]
        g = jj.assign(_wd=jj["_w"] * jj[f"_d{i}"]).groupby("_pi")[["_wd", "_w"]].sum()
        v = np.full(len(out), np.nan)
        v[g.index.to_numpy()] = (g["_wd"] / g["_w"]).to_numpy()
        out[k] = v
    return out


def density_1km_to_polygons(polys, cells, cell_col: str = "h3", pop_col: str = "population", dens_col: str = "pop_density_1km",
                            out_col: str = "pop_density_1km"):
    """`polys` (GeoDataFrame) with `out_col` = residents-weighted mean of `dens_col` over the H3 cells whose centre lies in each polygon
    (the 1 km density the polygon's residents live in); NaN where no populated cell centre falls inside.

    Args:
        polys: Polygons.
        cells: Cells with the density column.
        cell_col: H3 id column.
        pop_col: Residents column.
        dens_col: Density column of the cells.
        out_col: Output column.
    """
    import geopandas as gpd
    import h3
    import pandas as pd

    c = cells[[cell_col, pop_col, dens_col]].copy()
    c = c[pd.to_numeric(c[pop_col], errors="coerce").fillna(0) > 0]
    out = polys.copy()
    if not len(c) or not len(out):
        out[out_col] = np.nan
        return out
    ll = np.array([h3.cell_to_latlng(str(x)) for x in c[cell_col]])
    pts = gpd.GeoDataFrame({"_w": c[pop_col].to_numpy(dtype=float), "_d": c[dens_col].to_numpy(dtype=float)},
                           geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326).to_crs(out.crs or 4326)
    pts = pts[np.isfinite(pts["_d"])]
    left = gpd.GeoDataFrame({"_pi": np.arange(len(out))}, geometry=out.geometry.to_numpy(), crs=out.crs or 4326)
    j = gpd.sjoin(pts, left, predicate="within", how="inner")
    g = j.assign(_wd=j["_w"] * j["_d"]).groupby("_pi")[["_wd", "_w"]].sum()
    v = np.full(len(out), np.nan)
    v[g.index.to_numpy()] = (g["_wd"] / g["_w"]).to_numpy()
    out[out_col] = v
    return out


def nearest_cell_density(polys, cells, cell_col: str = "h3", dens_col="pop_density_1km"):
    """For each polygon: `dens_col` of the populated cell whose centre is nearest to the polygon's representative point (a list of
    columns: a 2-D array, one column each).

    Args:
        polys: Polygons.
        cells: Cells with the density column(s).
        cell_col: H3 id column.
        dens_col: Density column or list of columns.
    """
    import geopandas as gpd
    import h3

    many = not isinstance(dens_col, str)
    cols = list(dens_col) if many else [dens_col]
    c = cells[[cell_col] + cols].dropna()
    ll = np.array([h3.cell_to_latlng(str(x)) for x in c[cell_col]])
    pts = gpd.GeoDataFrame({f"_d{i}": c[k].to_numpy(dtype=float) for i, k in enumerate(cols)}, geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326).to_crs(3857)
    rp = gpd.GeoDataFrame({"_pi": np.arange(len(polys))}, geometry=polys.geometry.representative_point().to_numpy(), crs=polys.crs or 4326).to_crs(3857)
    j = gpd.sjoin_nearest(rp, pts, how="left").drop_duplicates("_pi").sort_values("_pi")
    a = j[[f"_d{i}" for i in range(len(cols))]].to_numpy(dtype=float)
    return a if many else a[:, 0]
