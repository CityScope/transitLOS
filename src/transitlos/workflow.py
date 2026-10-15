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

        feed_source = download_gtfs_for_place(place, os.path.join(folder, "gtfs"), refresh_token)

    edges = compute_level_of_service(
        aoi, feed_source=feed_source, region=region, cache_dir=os.path.join(folder, "network_cache"),
        pbf_path=os.path.join(folder, "region.osm.pbf"), max_walk_distance_m=max_walk_distance,
        date_range=(service_day, service_day) if service_day else None,
    )
    edges.to_file(os.path.join(folder, "access_streets.gpkg"))

    labels = {"population": "👥 Population", "level_of_service": "🚌 Transit level of service (0-100)"}
    hexes, stats, m = workflows.report(
        edges, aoi, folder, ["level_of_service"], labels, "level_of_service", map_file, h3_resolution, population,
        bands=(0, 10, 50, 100), band_labels=["almost none (< 10)", "some (10-50)", "well served (50+)"],
    )
    return workflows.Result(folder, aoi, edges, hexes, stats, m)
