"""Join an access-score GeoDataFrame with population data.

Composition decision (read before changing this file): `access_gdf` is
street-*edge* (line) geometry, while population sources are polygon (US
Census ACS/DHC tract/blockgroup) or raster (WorldPop) geometry. Rather than
rasterizing the edges to H3 first, this module treats the *population
source* as the target polygon level and aggregates `access_gdf`'s
`level_of_service` (and any other requested edge columns) onto it via
`geohierarchy.edges_to_level` -- this is the same composition
`UrbanAccessAnalyzer.api.AccessibilityAnalyzer.to_level`/`.to_h3` already use
internally, just with the population source's own polygons as `level_gdf`
instead of an H3 grid or admin boundary layer supplied by the caller. This
keeps population counts exactly as reported by the source (no raster
resampling error from forcing them onto an edge-derived grid) and produces
one row per population polygon with both a `population` and an aggregated
`level_of_service` column, ready for `equity.equity_statistics`.

For WorldPop (`population_source="worldpop"`), which ships a raster rather
than a polygon layer, the raster is first vectorized into a per-pixel
polygon grid (`_worldpop_raster_to_gdf`) so the same `edges_to_level` call
can be reused. This is O(n_pixels) in memory, so pass a coarse `resolution`
(`"1km"` is the WorldPop default probed here) for large AOIs.
"""

from __future__ import annotations

from typing import Optional, Union

import geopandas as gpd
import numpy as np
from geohierarchy import edges_to_level


def _worldpop_raster_to_gdf(raster_path: str, aoi: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """Vectorize a WorldPop population GeoTIFF into a per-pixel polygon GeoDataFrame.

    WorldPop has no per-AOI download endpoint -- `download_worldpop_population`
    always returns a *whole-country* raster (e.g. the continental US at 1km
    is ~270 million pixels), so vectorizing the full band unconditionally
    was never tractable for any real AOI smaller than a whole country. If
    `aoi` is given, only the raster window covering its bounds (padded by a
    couple of pixels) is read and vectorized -- everything downstream
    (`cross_with_population`/`edges_to_level`) only needs pixels near the
    AOI anyway.

    Args:
        raster_path: Path to a GeoTIFF downloaded by
            `pycensus.countries.worldwide.worldpop.loader.download_worldpop_population`.
        aoi: Optional Area of Interest GeoDataFrame/GeoSeries used to crop
            the raster (via a `rasterio` windowed read) before
            vectorization. Its CRS is reprojected to the raster's CRS as
            needed. If omitted, the entire raster is vectorized -- only
            safe for small/already-cropped rasters.

    Returns:
        GeoDataFrame with one row per non-nodata pixel: a square `geometry`
        polygon and a `population` column holding that pixel's value.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.geometry import box

    with rasterio.open(raster_path) as src:
        window = None
        if aoi is not None:
            aoi_bounds = aoi.to_crs(src.crs).total_bounds
            window = from_bounds(*aoi_bounds, transform=src.transform).round_offsets().round_lengths()
            # Pad by a couple of pixels so edge pixels right at the AOI
            # boundary aren't clipped off, then clamp back to raster bounds.
            window = window.round_lengths().round_offsets()
            window = rasterio.windows.Window(
                col_off=max(window.col_off - 2, 0),
                row_off=max(window.row_off - 2, 0),
                width=min(window.width + 4, src.width - max(window.col_off - 2, 0)),
                height=min(window.height + 4, src.height - max(window.row_off - 2, 0)),
            )
        band = src.read(1, window=window)
        nodata = src.nodata
        transform = src.window_transform(window) if window is not None else src.transform
        crs = src.crs

    rows, cols = np.where(band != nodata if nodata is not None else np.isfinite(band))
    values = band[rows, cols]

    geoms = []
    for r, c in zip(rows, cols):
        x0, y0 = transform * (c, r)
        x1, y1 = transform * (c + 1, r + 1)
        geoms.append(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    return gpd.GeoDataFrame({"population": values}, geometry=geoms, crs=crs)


def cross_with_population(
    access_gdf: gpd.GeoDataFrame,
    population_source: str = "worldpop",
    aoi: Optional[gpd.GeoDataFrame] = None,
    region_for_census: Optional[Union[str, list]] = None,
    id_col: str = "edge_id",
    access_columns: Optional[list] = None,
    census_level: str = "tract",
    census_dataset: str = "acs",
    worldpop_date=None,
    worldpop_resolution: str = "1km",
    worldpop_folder: Optional[str] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Combine `access_gdf` (street edges with an `level_of_service` column) with population data.

    Args:
        access_gdf: Street-edges GeoDataFrame with an `level_of_service` column
            (e.g. from `transitlos.level_of_service.compute_level_of_service`).
        population_source: `"worldpop"` (global gridded population, any
            country) or `"us_census"` (`pycensus.countries.usa.acs5`/`.dhc`, US only).
        aoi: Area of interest, required for `"worldpop"` (used to download
            the raster) and used as a fallback to derive `us_census`'s
            `states` if `region_for_census` is not given.
        region_for_census: For `"us_census"`: state name(s)/abbreviation(s)/
            FIPS code(s) forwarded to `pycensus.countries.usa.acs5.load`/`.dhc.load` as
            `states`. If `None`, derived from `aoi`.
        id_col: Name to give the population polygon's unique-id column in
            the returned GeoDataFrame (`GEOID` for `us_census` is renamed to
            this; a synthetic pixel index is used for `worldpop`).
        access_columns: Edge columns from `access_gdf` to aggregate onto the
            population polygons. Defaults to `["level_of_service"]`.
        census_level: Geometry level forwarded to `pycensus.countries.usa.acs5.load`/
            `.dhc.load` (only used when `population_source="us_census"`).
        census_dataset: `"acs"` or `"dhc"` -- which `pycensus.usa` submodule
            to use for `"us_census"`.
        worldpop_date: Year (or date/datetime) forwarded to
            `download_worldpop_population`; required for `"worldpop"`.
        worldpop_resolution: `"100m"` or `"1km"`, forwarded to
            `download_worldpop_population`. `"1km"` is the safer default to
            keep the pixel-grid vectorization tractable.
        worldpop_folder: Download destination forwarded to
            `download_worldpop_population`.
        **kwargs: Extra keyword arguments forwarded to the underlying
            `pycensus` loader.

    Returns:
        GeoDataFrame with one row per population polygon: `id_col`,
        geometry, `population`, and the aggregated `access_columns`.

    Raises:
        ValueError: If `population_source` is not `"worldpop"`/`"us_census"`,
            or a required argument for the chosen source is missing.
    """
    access_columns = access_columns if access_columns is not None else ["level_of_service"]

    if population_source == "worldpop":
        from pycensus.countries.worldwide.worldpop.loader import download_worldpop_population

        if aoi is None or worldpop_date is None:
            raise ValueError("aoi and worldpop_date are required for population_source='worldpop'.")
        raster_path = download_worldpop_population(
            aoi, worldpop_date, folder=worldpop_folder, resolution=worldpop_resolution, **kwargs
        )
        pop_gdf = _worldpop_raster_to_gdf(raster_path, aoi=aoi)
        pop_gdf[id_col] = np.arange(len(pop_gdf))

    elif population_source == "us_census":
        from pycensus.countries.usa import acs, dhc

        loader = {"acs": acs, "dhc": dhc}.get(census_dataset)
        if loader is None:
            raise ValueError(f"Unknown census_dataset {census_dataset!r}; expected 'acs' or 'dhc'.")
        pop_gdf = loader.load(aoi=aoi, states=region_for_census, level=census_level, **kwargs)
        pop_gdf["population"] = pop_gdf.acs["population"] if census_dataset == "acs" else pop_gdf.dhc["population"]
        pop_gdf = pop_gdf.rename(columns={"GEOID": id_col})

    else:
        raise ValueError(f"Unknown population_source {population_source!r}; expected 'worldpop' or 'us_census'.")

    return edges_to_level(access_gdf, pop_gdf, id_col=id_col, columns=access_columns).merge(
        pop_gdf[[id_col, "population"]], on=id_col, how="left"
    )
