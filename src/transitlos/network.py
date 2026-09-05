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
        network_type: Defaults to `"all"` -- every public road included
            regardless of walk-permission, including highway/trunk. This
            was briefly `"walk"` (2026-09-04, an explicit request to
            exclude highway/trunk roads except where needed for
            connectivity), reverted the same day (explicit follow-up:
            "delete this idea of excluding highway and only include edges
            really needed for the graph. Include all public roads
            regardless if they are walk or not including highway"). No
            highway-tag filter at all -- see `osm_io.NETWORK_PROFILES["all"]`.
        ignore_oneway: Defaults to `True` -- forwarded to
            `StreetNetwork.from_pbf`/`osm_io.load_pbf` so every included
            edge is traversable in both directions for this walking-access
            graph, regardless of any OSM `oneway` tag (a vehicle-traffic
            restriction pedestrians aren't bound by).

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
