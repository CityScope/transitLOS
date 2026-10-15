"""One call from a place name to the transit level-of-service map and statistics."""

from __future__ import annotations

import os
from datetime import date
from typing import Optional, Sequence, Union

import geopandas as gpd

from .level_of_service import compute_level_of_service


def level_of_service_map(
    place: Union[str, gpd.GeoDataFrame],
    folder: Optional[str] = None,
    region: str = "global",
    service_day: Optional[date] = None,
    max_walk_distance: float = 2000,
    h3_resolution: int = 10,
    population: str = "worldpop",
    empty_cells: str = "nearest_value",
    empty_max_distance: Optional[float] = None,
    cache_dir: str = "cache",
    feed_source: Optional[Sequence] = None,
    refresh_token: str = "",
    map_file: str = "transit_map.html",
):
    """Transit level of service (0-100) of every street, on a map, with who is well served.

    Downloads the GTFS feeds of the place from the Mobility Database (a free token is needed: pass
    ``refresh_token``, set ``MOBILITY_DATABASE_REFRESH_TOKEN``, or it is asked for), scores every stop
    for a service day, spreads the score along the walking network, averages it into H3 hexagons
    and adds the population.

    Args:
        place: Place name, or a GeoDataFrame with the study area (then ``feed_source`` is required).
        folder: Output folder (default ``output/<place>``).
        region: ``"europe"``, ``"north_america"``, ``"global_south"`` or ``"global"``.
        service_day: Day to analyze, e.g. ``date(2026, 8, 21)``; ``None`` detects a typical day (slower).
        max_walk_distance: Streets farther than this (m) from any stop score 0.
        h3_resolution: H3 resolution (10 is ~0.015 km2).
        population: ``"worldpop"`` or ``"census"``.
        empty_cells: What to do with hexagons that have people but no street: ``"nearest_value"`` (take the
            level of service of the nearest hexagon with streets), ``"relocate_weight"`` (move their people
            there, within 300 m by default) or ``"ignore"`` (leave them without score).
            See :func:`geohierarchy.handle_empty_cells`.
        empty_max_distance: Meters, center to center, for ``empty_cells``.
        cache_dir: Folder where downloads are kept and reused (street networks, regional extracts, population).
        feed_source: Your own GTFS ``.zip`` files or folders, instead of downloading them.
        refresh_token: Mobility Database refresh token.
        map_file: File name of the map in ``folder/maps/``.

    Returns:
        A :class:`UrbanAccessAnalyzer.workflows.Result` (``edges``, ``hexes``, ``stats``, ``map``...).
    """
    from UrbanAccessAnalyzer import workflows

    aoi, folder = workflows._setup(place, folder)
    if feed_source is None:
        if not isinstance(place, str):
            raise ValueError("feed_source is required when the area is a GeoDataFrame")
        from pyGTFSHandler.downloaders.mobility_database import download_gtfs_for_place

        from UrbanAccessAnalyzer.utils import sanitize_filename

        feed_source = download_gtfs_for_place(place, os.path.join(cache_dir, "gtfs", sanitize_filename(place)), refresh_token)

    edges = compute_level_of_service(
        aoi, feed_source=feed_source, region=region, cache_dir=os.path.join(cache_dir, "networks"), cache_by_aoi=True,
        pbf_path=os.path.join(cache_dir, "pbf", "region.osm.pbf"), max_walk_distance_m=max_walk_distance,
        date_range=(service_day, service_day) if service_day else None,
    )
    edges.to_file(os.path.join(folder, "transit_streets.gpkg"))

    hexes, stats, m = level_of_service_results(
        edges, aoi, folder, h3_resolution, population, map_file, empty_cells, empty_max_distance, cache_dir
    )
    return workflows.Result(folder, aoi, edges, hexes, stats, m)


def top_stops(stops, n: int = 10):
    """The ``n`` best stops: name, mode, headway (min between vehicles), speed and ``stop_score`` (0-1)."""
    cols = ["stop_name", "mode_category", "headway_minutes", "avg_speed_kmh", "stop_score"]
    return stops[cols].sort_values("stop_score", ascending=False).head(n)


def level_of_service_results(
    edges: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    folder: str,
    h3_resolution: int = 10,
    population: str = "worldpop",
    map_file: str = "transit_map.html",
    empty_cells: str = "nearest_value",
    empty_max_distance: Optional[float] = None,
    cache_dir: str = "cache",
):
    """Hexagons + population, map and statistics of the level of service (see :func:`compute_level_of_service`).

    ``empty_cells`` / ``empty_max_distance``: what to do with hexagons that have people but no street
    (``"nearest_value"``, ``"relocate_weight"`` or ``"ignore"``; see :func:`geohierarchy.handle_empty_cells`).

    Returns:
        ``(hexes, stats, map)``: ``stats`` counts people by level of service.
    """
    from UrbanAccessAnalyzer import workflows

    labels = {"population": "👥 Population", "level_of_service": "🚌 Transit level of service (0-100)"}
    return workflows.report(
        edges, aoi, folder, ["level_of_service"], labels, "level_of_service", map_file, h3_resolution, population,
        bands=(0, 10, 50, 100), band_labels=["almost none (< 10)", "some (10-50)", "well served (50+)"],
        empty_cells=empty_cells, empty_max_distance=empty_max_distance, cache_dir=cache_dir,
    )
