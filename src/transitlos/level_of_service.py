"""Orchestrates network + stops + scoring into a street-edges level_of_service GeoDataFrame.

Pipeline: `network.prepare_street_network` (if `network` not given) +
`stops.download_and_prepare_stops` (if `stops` not given) +
`stop_scores.compute_stop_scores` (if `stops` lacks a `stop_score` column
yet) feed a `UrbanAccessAnalyzer.api.PointsOfInterest` (transit stops, scored
by `stop_score`) into an `AccessibilityAnalyzer`, whose `.run()` + `.to_gdf()`
produce the final street-edges `level_of_service` GeoDataFrame.

**2026-08-12 rewrite, at the user's explicit request**: this module used to
rely on `AccessibilityAnalyzer.default_distance_matrix`'s own generic
rank-based falloff, deliberately *not* calling `transitlos.scoring`'s own
`walk_decay`/`accessibility_score` to avoid double-counting walk decay (once
via that generic falloff, once via `walk_decay`). That tradeoff is now
resolved the other way: `transitlos.scoring.distance_matrix
.build_intelligent_distance_matrix` builds the `distance_matrix` argument
`AccessibilityAnalyzer.run` expects *from* `transitlos.scoring`'s own
offset-power `D(t; stop_score)` decay (see `walk_access.py`), so the
isochrone engine now runs transitLOS's literature-grounded decay directly --
there is no longer a second, independent falloff to double-count against.

This also implements "intelligent discretization" (see
`distance_matrix.py`'s module docstring for the full rationale):

1. **Distance steps are no longer a flat convention** (the old
   `walk_distance_steps=(400, 800, 1200)` default) -- they are derived by
   inverting the decay curve at each output-grid tier's midpoint, so every
   isochrone distance computed maps directly onto an output-visible
   `{0.0, 0.1, ..., 1.0}` boundary.
2. **Stops are grouped ("bucketed") by output-equivalent stop_score**, not
   left as hundreds of near-duplicate continuous values -- two stops whose
   `stop_score` would produce identical ceiling-discretized output at every
   chosen distance collapse into one distance-matrix row, directly
   minimizing the number of `(poi_score, distance)` isochrone/Dijkstra runs
   `UrbanAccessAnalyzer.isochrones` has to perform (that module already
   batches every POI sharing a tier into one multi-source search -- see its
   docstring -- so fewer distinct tiers is exactly fewer runs).
3. **The final `level_of_service` is ceiling-discretized** onto
   `{0.0, 0.1, ..., 1.0}` both inside the matrix and, as a safety net, on
   the final per-edge output column (`UrbanAccessAnalyzer`'s
   `exact_edge_access` can produce edge-interpolated values between matrix
   grid points that a matrix-only discretization wouldn't catch).

`walk_distance_steps` remains an accepted parameter for backward
compatibility and manual overrides, but is no longer used by default --
`compute_level_of_service` only falls back to it if
`build_intelligent_distance_matrix` is explicitly bypassed (`region=None`
is not a valid escape hatch here; pass `distance_matrix`/`bucket_map`
yourself if you need the old flat-convention behavior).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely

from UrbanAccessAnalyzer.api import AccessibilityAnalyzer, AreaOfInterest, PointsOfInterest, StreetNetwork

from .network import prepare_street_network
from .scoring.accessibility_score import accessibility_score
from .scoring.distance_matrix import build_intelligent_distance_matrix, ceil_to_grid, ceil_to_grid_array
from .scoring.mrc import mode_category
from .scoring.parameters import REGIONS
from .stop_scores import compute_stop_scores
from .stops import download_and_prepare_stops


def _stops_to_points_of_interest(stops_df: gpd.GeoDataFrame, crs=None) -> PointsOfInterest:
    """Convert a stops GeoDataFrame (lon/lat points + `stop_score`) into a `PointsOfInterest`.

    Args:
        stops_df: GeoDataFrame with a point `geometry` column (lon/lat) and
            a `stop_score` column.
        crs: CRS to reproject `stops_df` into before converting to WKB --
            must match the `StreetNetwork`'s CRS (a projected, metric CRS),
            since `StreetNetwork.snap_points` does no reprojection of its
            own and does a purely coordinate-based nearest-node search. If
            `None`, `stops_df` is used as-is (only correct if it is already
            in the network's CRS).

    Returns:
        `PointsOfInterest` wrapping a Polars DataFrame with a WKB
        `geometry` column and the numeric columns of `stops_df`.

    Note:
        Rows with a missing/empty/NaN-coordinate geometry (real GTFS feeds
        occasionally have stops with blank `stop_lat`/`stop_lon`) are
        dropped here -- `StreetNetwork.snap_points`'s STRtree `.nearest()`
        query raises `ValueError: Cannot determine nearest geometry for
        empty geometry or missing value` on the very first such point
        otherwise, which would abort the whole city's run over a handful of
        bad GTFS rows rather than just excluding those stops.
    """
    if crs is not None:
        stops_df = stops_df.to_crs(crs)
    # `stops_df.geometry.is_empty`/`.isna()` do *not* catch this: a
    # `Point(nan, nan)` reports `is_empty=False` pre-WKB-roundtrip, but
    # `shapely.to_wkb`/`from_wkb` (used just below, and again inside
    # `snap_points`) silently turns NaN-coordinate points into a true empty
    # geometry -- which is exactly what trips the STRtree `.nearest()`
    # error this filter exists to prevent. Checking the raw x/y for
    # finiteness catches it before that round trip.
    xy = shapely.get_coordinates(stops_df.geometry.to_numpy())
    valid = np.isfinite(xy).all(axis=1) if len(xy) == len(stops_df) else np.ones(len(stops_df), dtype=bool)
    n_dropped = int((~valid).sum())
    if n_dropped:
        import warnings

        warnings.warn(f"Dropping {n_dropped} stop(s) with missing/invalid coordinates before snapping to the network")
    stops_df = stops_df[valid]
    pdf = stops_df.drop(columns="geometry")
    wkb = shapely.to_wkb(stops_df.geometry.to_numpy())
    df = pl.from_pandas(pdf).with_columns(pl.Series("geometry", wkb, dtype=pl.Binary))
    return PointsOfInterest(df)


def compute_level_of_service(
    aoi: AreaOfInterest,
    feed_source=None,
    network: Optional[StreetNetwork] = None,
    stops: Optional[gpd.GeoDataFrame] = None,
    region: str = "global",
    cache_dir: Optional[str] = None,
    walk_distance_steps: Sequence[float] = (400, 800, 1200),
    weights: Optional[tuple[float, float, float]] = None,
    pbf_path: Optional[str] = None,
    date_range: Optional[tuple[Union[date, datetime], Union[date, datetime]]] = None,
    score_bins: Optional[int] = 50,
    max_walk_distance_m: Optional[float] = None,
    chunk_h3_resolution: Optional[int] = None,
    chunk_buffer_m: float = 1000.0,
    stops_crop_buffer_m: float = 500.0,
    **stops_kwargs,
) -> gpd.GeoDataFrame:
    """Compute a street-edges GeoDataFrame with a transit-access `level_of_service` column.

    Args:
        aoi: Area of interest for the network and (if `stops` not given)
            the GTFS feed.
        feed_source: GTFS feed source forwarded to
            `stops.download_and_prepare_stops` if `stops` is not given.
        network: A pre-built `StreetNetwork`; if `None`,
            `network.prepare_street_network(aoi, cache_dir=cache_dir,
            pbf_path=pbf_path)` is called.
        stops: A pre-built stops GeoDataFrame; if `None`,
            `stops.download_and_prepare_stops(feed_source, aoi=aoi,
            date_range=date_range, **stops_kwargs)` is called. If given but
            missing a `stop_score` column, `stop_scores.compute_stop_scores`
            is applied to it.
        region: Region key forwarded to `compute_stop_scores` and to
            `build_intelligent_distance_matrix` (selects the decay-shape
            defaults `walk_decay`/`accessibility_score` use).
        cache_dir: Passed to `prepare_street_network` for network caching.
        walk_distance_steps: **Deprecated/unused by default** as of
            2026-08-12 -- distance thresholds are now derived automatically
            by `build_intelligent_distance_matrix` from transitLOS's own
            decay curve (see module docstring). Kept only so existing calls
            don't break; passing a non-default value has no effect unless
            you also bypass `build_intelligent_distance_matrix` yourself.
        weights: Optional `(w1, w2, w3, w4)` weights forwarded to
            `compute_stop_scores`.
        pbf_path: Passed to `prepare_street_network` when a network must be
            built from scratch.
        date_range: Optional `(start_date, end_date)` forwarded to
            `download_and_prepare_stops` (ignored if `stops` is given
            directly). **Strongly recommended** whenever you already know a
            valid service date for `feed_source`: it skips
            `download_and_prepare_stops`'s expensive service-day
            auto-detection fallback, which alone can cost ~7s of a ~10s
            call on a real multi-file GTFS feed (see that function's
            docstring for details). If omitted, the safe (but slower)
            auto-detection default is used.
        score_bins: **Deprecated/unused by default** as of 2026-08-12 --
            `build_intelligent_distance_matrix` already collapses stop_score
            into output-equivalent buckets (see module docstring), which
            supersedes this parameter's quantile-bucketing approach.
        max_walk_distance_m: Optional hard outer walk-distance cutoff
            (meters) forwarded to `build_intelligent_distance_matrix`. The
            decay curve (`walk_access.walk_decay`) never truly reaches
            zero, so without this the isochrone search radius can extend
            several kilometers for the lowest access tier; set this to cap
            it (e.g. 2000 m) while still searching the *entire* requested
            radius (see that function's docstring for how the cutoff is
            applied). `None` (default) leaves the search radius unbounded.
        chunk_h3_resolution: Opt-in memory-bounded isochrone computation --
            forwarded to `UrbanAccessAnalyzer.api.AccessibilityAnalyzer.run`'s
            `chunk_h3_resolution`. `None` (default) is the original
            whole-network behavior, unaffected. Set to e.g. `4` for
            oversized AOIs (Boston, Shanghai) where the unchunked path OOMs.
        chunk_buffer_m: Forwarded as `chunk_buffer_m` when
            `chunk_h3_resolution` is set. Must be >= the largest isochrone
            search radius the chosen decay curve can produce (see
            `max_walk_distance_m` above and
            `AccessibilityAnalyzer.run`/`compute_node_access_chunked`'s own
            docstrings) or results near chunk boundaries will be truncated.
        stops_crop_buffer_m: Metres to buffer `aoi` by before querying
            stops (only when `stops` is not given directly) -- see the
            note where it's used, below. `0` for the exact-AOI behavior.
        **stops_kwargs: Extra keyword arguments forwarded to
            `download_and_prepare_stops` (e.g. `route_types`).

    Returns:
        `geopandas.GeoDataFrame` of street edges (network CRS) with a
        `level_of_service` column, ceiling-discretized onto `{0.0, 0.1, ...,
        1.0}` internally (see module docstring, point 3) and then rescaled
        to `{0, 10, ..., 100}` -- 2026-09-23, explicit user request that
        every 0-1 score be displayed/stored on a 0-100 scale.
    """
    if network is None:
        network = prepare_street_network(aoi, cache_dir=cache_dir, pbf_path=pbf_path)

    if stops is None:
        # Buffered (2026-09-04, explicit user request: "for stops and
        # streets use a buffer specified by user (or 500m) to avoid
        # boundary issues. Everything else should be for the exact aoi")
        # -- a real stop just outside the exact AOI edge (e.g. a station
        # whose platform/entrance sits a few dozen metres past the
        # boundary) still legitimately serves people just inside it; an
        # unbuffered crop would silently drop it and understate access
        # right at the border. Population/census attribution stays on the
        # real, unbuffered AOI elsewhere in the pipeline -- this buffer is
        # scoped to the stops query alone.
        stops_aoi_gdf = aoi.gdf.copy()
        if stops_crop_buffer_m:
            utm_crs = stops_aoi_gdf.estimate_utm_crs()
            stops_aoi_gdf = stops_aoi_gdf.to_crs(utm_crs)
            stops_aoi_gdf["geometry"] = stops_aoi_gdf.geometry.buffer(stops_crop_buffer_m)
            stops_aoi_gdf = stops_aoi_gdf.to_crs(aoi.gdf.crs)
        stops = download_and_prepare_stops(feed_source, aoi=stops_aoi_gdf, date_range=date_range, **stops_kwargs)

    if "stop_score" not in stops.columns:
        stops = compute_stop_scores(stops, region=region, weights=weights)

    # `D(t; stop_score)`'s scale (`t0`) is now mode-dependent (see
    # `walk_access.py`), so the distance_matrix/bucketing must be built, and
    # `AccessibilityAnalyzer` run, once per mode -- a single shared
    # distance-column set would silently use the wrong mode's t0_base for
    # every stop but one. Results are combined by taking the per-edge
    # maximum access_score across mode-runs, matching
    # `UrbanAccessAnalyzer.isochrones`'s own "best tier any stop/distance
    # combination can justify" reconciliation rule (see that module's
    # docstring) extended across mode-runs instead of just within one.
    #
    # **2026-09-29 bug fix**: this used to run `network.snap_points` and
    # `AccessibilityAnalyzer` once per mode, then combine the per-mode
    # `access_score` columns positionally via `np.maximum`. That is invalid:
    # `network.snap_points` (and `AccessibilityAnalyzer.run`'s own internal
    # re-snap) *splits* street edges at each POI's projected snap point, so
    # different modes' point sets produce structurally different edge tables
    # (different length, different `u`/`v` node ids from the newly-inserted
    # snap nodes) -- there is no valid positional (or even a stable `u,v`-id)
    # alignment between two separate per-mode runs' edge outputs. This
    # crashed in production on every multi-mode city with
    # `ValueError: operands could not be broadcast together with shapes
    # (...) (...)` as soon as two modes produced different edge counts.
    #
    # The fix: snap every mode's stops against the SAME network in a single
    # `AccessibilityAnalyzer` run, so edges are only ever split once (by the
    # union of all points), by building one combined distance_matrix whose
    # `poi_score` values are made unique per mode (via a disjoint numeric
    # offset) and whose distance columns are the *union* of every mode's own
    # natural distance steps -- each mode's bucket score is (cheaply)
    # re-evaluated at every column in the union using that mode's own
    # `D(t; stop_score)` curve (`accessibility_score`), not just the subset
    # of columns its own `build_intelligent_distance_matrix` call chose, so
    # every mode's stops remain scored by their own genuine mode-specific
    # decay curve at every distance actually searched.
    stops = stops.copy()
    if "mode_category" not in stops.columns or stops["mode_category"].isna().any():
        rtype = stops["route_type"] if "route_type" in stops.columns else pd.Series([None] * len(stops), index=stops.index)
        stops["mode_category"] = [mode_category(None if pd.isna(rt) else rt) for rt in rtype]

    per_mode_bucket_maps: dict[str, dict[float, float]] = {}
    per_mode_distance_steps_m: dict[str, list[float]] = {}
    union_distance_steps_m: set[float] = set()
    for mode, mode_stops in stops.groupby("mode_category"):
        _dm, bucket_map, distance_steps_m = build_intelligent_distance_matrix(
            mode_stops["stop_score"].to_numpy(), mode, region=region, max_distance_m=max_walk_distance_m
        )
        per_mode_bucket_maps[mode] = bucket_map
        per_mode_distance_steps_m[mode] = distance_steps_m
        union_distance_steps_m.update(distance_steps_m)

    union_steps_m = sorted(union_distance_steps_m)
    walking_speed_mps = REGIONS[region].walking_speed_mps

    # Disjoint per-mode offset (a full order of magnitude past any real
    # bucket score, which live in [0, ~1.05]) so one shared `poi_score`
    # column never collides across modes.
    mode_offset = {mode: 10.0 * i for i, mode in enumerate(sorted(per_mode_bucket_maps))}

    combined_stop_parts = []
    combined_matrix_rows: list[dict] = []
    for mode, mode_stops in stops.groupby("mode_category"):
        mode_stops = mode_stops.copy()
        mode_stops["stop_score_bucket"] = mode_stops["stop_score"].map(per_mode_bucket_maps[mode]) + mode_offset[mode]
        combined_stop_parts.append(mode_stops)

        for bucket_score in sorted(set(per_mode_bucket_maps[mode].values())):
            row = {"poi_score": bucket_score + mode_offset[mode]}
            for d_m in union_steps_m:
                t_min = d_m / walking_speed_mps / 60.0
                row[str(d_m)] = ceil_to_grid(
                    accessibility_score(bucket_score, t_min, mode, region=region)
                )
            combined_matrix_rows.append(row)

    stops = pd.concat(combined_stop_parts).sort_index()
    distance_matrix = pl.DataFrame(combined_matrix_rows) if combined_matrix_rows else pl.DataFrame({"poi_score": []})

    # Single analyzer run across every mode's stops together (see the fix
    # note above) -- edges are snapped/split exactly once, so there is no
    # cross-run combination step needed at all.
    points = _stops_to_points_of_interest(stops, crs=network.crs)
    network, points = network.snap_points(points)

    analyzer = AccessibilityAnalyzer(network, points)
    analyzer.run(
        distance_matrix, poi_score_col="stop_score_bucket",
        chunk_h3_resolution=chunk_h3_resolution, chunk_buffer_m=chunk_buffer_m,
    )
    result = analyzer.to_gdf()
    # `AccessibilityAnalyzer.to_gdf()` names its output column `access_score`
    # (UrbanAccessAnalyzer's own generic naming); this module's public
    # contract (see this function's docstring) is a `level_of_service`
    # column, so rename here rather than downstream everywhere it's
    # consumed.
    #
    # Final safety-net discretization: `exact_edge_access` can produce
    # edge-interpolated values between matrix grid points, so re-apply the
    # ceiling-to-0.1 rule to the actual output column, not just the matrix
    # cells that fed it.
    # 2026-09-23, explicit user request: every 0-1 score is now
    # displayed/stored on a 0-100 scale. `ceil_to_grid_array` itself is left
    # untouched (still operates on its native [0, 1] grid -- it's also used
    # internally elsewhere in this module's own decay-curve/bucket math,
    # which must stay in [0, 1] for the multiplicative stop_score * D(t)
    # math to remain correct), so the x100 rescale happens only here, once,
    # on this function's actual public-contract output column -- everything
    # downstream (population-weighted means/medians, `code.stats`'s
    # regression + equity-flag residual/std math, distribution bucketing)
    # is linear in `level_of_service` and therefore scale-consistent
    # automatically; `equity_flag`/`equity_flag_thresholds` in particular
    # need no code change at all, since a regression's residual-vs-std test
    # is scale-invariant (multiplying y by a constant scales slope,
    # intercept, residuals, and std together, so which cells land outside
    # +-1 std is unchanged).
    result["level_of_service"] = ceil_to_grid_array(result["access_score"].to_numpy()) * 100.0
    result = result.drop(columns=["access_score"])
    return result
