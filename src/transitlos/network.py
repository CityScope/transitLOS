"""Build (or load a cached) simplified, AOI-cropped walking street network.

Thin wrapper around `UrbanAccessAnalyzer.api.StreetNetwork`: downloads (or
reads a local PBF), simplifies, and crops to the AOI -- unless a previously
saved+simplified network already exists at `cache_dir`, in which case it is
loaded instead of rebuilt.
"""

from __future__ import annotations

import os
from typing import Optional

from UrbanAccessAnalyzer.api import AreaOfInterest, StreetNetwork


def prepare_street_network(
    aoi: AreaOfInterest,
    cache_dir: Optional[str] = None,
    cluster_distance: float = 10.0,
    pbf_path: Optional[str] = None,
    network_type: str = "all",
    ignore_oneway: bool = True,
) -> StreetNetwork:
    """Return a simplified, AOI-cropped `StreetNetwork`, using a cache if available.

    Args:
        aoi: Area of interest the network is built/cropped for.
        cache_dir: Directory to load a previously `.save()`d network from
            (via `StreetNetwork.load`), and to `.save()` a freshly built one
            to. If `None`, no caching is attempted -- the network is always
            rebuilt and never persisted.
        cluster_distance: Cluster distance (meters) passed to
            `StreetNetwork.simplify`.
        pbf_path: Path to a local `.osm.pbf` file, forwarded to
            `StreetNetwork.from_pbf`. If it doesn't exist, it is downloaded
            for `aoi` via Geofabrik (see `StreetNetwork.from_pbf`'s
            docstring). Required when `cache_dir` is `None` or has no cached
            network yet.
        network_type: Defaults to `"all"` (item 3 fix), not the old
            `"walk"` profile -- `"walk"` only kept a highway-tag whitelist
            (`osm_io.WALK_HIGHWAYS`: footway/pedestrian/path/living_street/
            steps/residential/service/unclassified/track) and silently
            dropped every other OSM street/way type (primary/secondary/
            trunk arterials, etc), even though pedestrians walk along those
            too (sidewalks/shoulders). `"all"` applies no highway-tag
            filter at all.
        ignore_oneway: Defaults to `True` (item 3 fix) -- forwarded to
            `StreetNetwork.from_pbf`/`osm_io.load_pbf` so every included
            edge is traversable in both directions for this walking-access
            graph, regardless of any OSM `oneway` tag (a vehicle-traffic
            restriction pedestrians aren't bound by). Redundant with
            `network_type="walk"` (which already ignores oneway
            unconditionally, independent of this flag), but load-bearing
            now that the default `network_type` is `"all"`.

    Returns:
        A `StreetNetwork` already simplified and cropped to `aoi`.

    Raises:
        ValueError: If no cached network is found and `pbf_path` is `None`.
    """
    if cache_dir is not None and os.path.isfile(os.path.join(cache_dir, "meta.txt")):
        return StreetNetwork.load(cache_dir)

    if pbf_path is None:
        raise ValueError(
            "pbf_path is required to build a street network when no cached "
            "network is found at cache_dir."
        )

    network = StreetNetwork.from_pbf(pbf_path, aoi=aoi, network_type=network_type, ignore_oneway=ignore_oneway)
    network = network.simplify(cluster_distance)
    network = network.crop(aoi)

    if cache_dir is not None:
        network.save(cache_dir)

    return network
