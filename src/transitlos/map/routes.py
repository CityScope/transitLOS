"""Build one map-ready line feature per GTFS `route_id`, for the map's transit-lines overlay.

Deliberately a small, dependency-light GTFS reader (plain `pandas.read_csv`
over `routes.txt`/`trips.txt`/`shapes.txt`/`stop_times.txt`/`stops.txt`)
rather than a `pyGTFSHandler.feed.Feed` load: this needs *geometry only*, no
service-calendar filtering, no headway/speed analysis, and no stop grouping --
all of which is what makes a full `Feed` load expensive (tens of seconds to
minutes per city). Nothing here is city-specific; a caller passes GTFS
directories in and gets a GeoDataFrame back.

Geometry source, in priority order (per the "shapes if available, else
straight lines between consecutive stops" requirement):

1. `shapes.txt` polylines referenced by that route's trips (the real,
   on-street alignment).
2. If a route has no usable shape, the stop sequence of its longest trip
   (`stop_times.txt` ordered by `stop_sequence`, joined to `stops.txt`
   coordinates) connected by straight segments.

One feature per `route_id` (the explicit deliverable), not one per trip or
shape: a route's many trip shapes are collapsed by a greedy "does this
candidate add geometry the already-kept ones don't cover?" pass, so a route
with genuinely separate branches (e.g. a light-rail trunk splitting into
several tails) keeps them as a MultiLineString while the dozens of
near-identical per-trip variants of an ordinary bus route collapse to one
LineString.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from ..scoring.mode import mode_category

#: Fallback line color per mode when GTFS supplies no `route_color`
#: (dark green rail / brown tram / dark red bus, per the map's spec). Also
#: used as the fallback background for route badges in stop popups and the
#: on-map "routes" annotation layer, so the two always agree.
MODE_FALLBACK_COLOR = {"rail": "#006400", "tram": "#8b4513", "bus": "#8b0000"}

#: Simplification tolerance in degrees (~5 m at Boston's latitude). Shapes go
#: straight into the page's inline JSON, so full-resolution GTFS polylines
#: (hundreds of thousands of points for a big feed) would bloat `map.html`
#: for detail no screen can resolve.
_SIMPLIFY_DEG = 0.00005

#: A candidate shape is kept as an extra branch only if this much of it (in
#: degrees of length, ~200 m) lies outside a ~30 m buffer of the shapes
#: already kept for that route.
_BRANCH_MIN_NEW_DEG = 0.002
_BRANCH_BUFFER_DEG = 0.0003
_MAX_BRANCHES_PER_ROUTE = 12

#: `shape_id`s whose underlying `shapes.txt` points are known-corrupt in the
#: source GTFS feed itself (not a bug in the geometry-building code here), so
#: rendering them straight would draw a nonsensical line. Each entry recorded
#: with the specific defect found, so a future re-check of the upstream feed
#: can confirm whether it's been fixed and this entry can be dropped.
#:
#: - "T14B_r1", "T14B_r2" (Guadalajara, feed `mdb2366_public_tra_mi_transpo`,
#:   route T14B "Troncal T14B - 50 'Reyes Heroles'"): `T14B_r1` has ~317
#:   `shape_pt_sequence` values that repeat (the sequence counts 0..381 but
#:   the shape has 699 points -- two overlapping point sets share the same
#:   sequence numbers), and `T14B_r2` has a single-hop ~16.7 km jump between
#:   consecutive sequence numbers 296 and 297 (from the route's southern
#:   terminus straight to its northern one, with no actual return path
#:   recorded). Both make `_shape_lines`' sort-by-sequence produce a
#:   self-crossing/teleporting line no matter how it's sorted -- the raw data
#:   has no single valid ordering. Excluded here rather than fixed in code
#:   since this is specific to this shape's data, not a general sequencing
#:   bug; other shapes in this and other feeds were spot-checked and don't
#:   show this pattern. Routes affected fall back to the straight
#:   stop-sequence line (see `_stop_sequence_lines`) if this was their only
#:   shape.
EXCLUDED_SHAPE_IDS: frozenset[str] = frozenset({"T14B_r1", "T14B_r2"})


def _read_csv(path: Path, usecols: Sequence[str]) -> Optional[pd.DataFrame]:
    """Read `path` keeping whichever of `usecols` actually exist; None if unusable."""
    if not path.exists():
        return None
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = [c for c in usecols if c in header.columns]
    if not cols:
        return None
    try:
        return pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
    except Exception:
        return None


def _hex_color(value, fallback: str) -> str:
    """Normalize a GTFS color field (`"DA291C"`, `"#DA291C"`, empty) to `#RRGGBB`."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        return fallback
    try:
        int(text, 16)
    except ValueError:
        return fallback
    return "#" + text.upper()


def _shape_lines(shapes: pd.DataFrame) -> dict[str, LineString]:
    """Group `shapes.txt` points into one LineString per `shape_id`."""
    df = shapes.copy()
    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    df = df.sort_values(["shape_id", "shape_pt_sequence"])
    out: dict[str, LineString] = {}
    for shape_id, grp in df.groupby("shape_id", sort=False):
        coords = list(zip(grp["shape_pt_lon"].to_numpy(), grp["shape_pt_lat"].to_numpy()))
        if len(coords) >= 2:
            out[str(shape_id)] = LineString(coords)
    return out


def _stop_sequence_lines(feed_dir: Path, route_ids: set[str], trips: pd.DataFrame) -> dict[str, LineString]:
    """Straight-line fallback geometry: the longest trip's stop sequence per route.

    Only called for routes that have no usable `shapes.txt` geometry, and only
    then is `stop_times.txt` (by far the largest GTFS file) read at all.
    """
    if not route_ids:
        return {}
    stop_times = _read_csv(feed_dir / "stop_times.txt", ["trip_id", "stop_id", "stop_sequence"])
    stops = _read_csv(feed_dir / "stops.txt", ["stop_id", "stop_lat", "stop_lon"])
    if stop_times is None or stops is None:
        return {}
    wanted_trips = trips[trips["route_id"].isin(route_ids)][["trip_id", "route_id"]]
    if wanted_trips.empty:
        return {}
    st = stop_times[stop_times["trip_id"].isin(set(wanted_trips["trip_id"]))].copy()
    if st.empty:
        return {}
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"])
    st = st.merge(wanted_trips, on="trip_id", how="inner")

    # One representative trip per route: the one calling at the most stops,
    # i.e. the variant that traverses the greatest extent of the route.
    counts = st.groupby(["route_id", "trip_id"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("route_id")
    st = st[st["trip_id"].isin(set(best["trip_id"]))]

    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    st = st.merge(stops, on="stop_id", how="inner").sort_values(["route_id", "stop_sequence"])

    out: dict[str, LineString] = {}
    for route_id, grp in st.groupby("route_id", sort=False):
        coords = list(zip(grp["stop_lon"].to_numpy(), grp["stop_lat"].to_numpy()))
        # Consecutive duplicate stop coordinates (loop routes revisiting a
        # stop, or a stop split across platforms) make a degenerate segment.
        deduped = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
        if len(deduped) >= 2:
            out[str(route_id)] = LineString(deduped)
    return out


def _collapse_branches(lines: list[LineString]) -> Optional[Union[LineString, MultiLineString]]:
    """Reduce a route's many trip shapes to one feature, keeping genuine branches.

    Greedy: take the longest shape, then keep each next-longest only if a
    meaningful stretch of it falls outside a small buffer of everything kept
    so far. That turns dozens of near-identical per-trip variants into a
    single LineString while preserving structurally distinct branches as
    additional parts of a MultiLineString.
    """
    lines = [ln for ln in lines if ln is not None and not ln.is_empty and ln.length > 0]
    if not lines:
        return None
    lines = sorted(lines, key=lambda ln: ln.length, reverse=True)
    kept = [lines[0]]
    covered = lines[0].buffer(_BRANCH_BUFFER_DEG)
    for candidate in lines[1:]:
        if len(kept) >= _MAX_BRANCHES_PER_ROUTE:
            break
        try:
            extra = candidate.difference(covered)
        except Exception:
            continue
        if extra.is_empty or extra.length < _BRANCH_MIN_NEW_DEG:
            continue
        kept.append(candidate)
        covered = unary_union([covered, candidate.buffer(_BRANCH_BUFFER_DEG)])
    simplified = [ln.simplify(_SIMPLIFY_DEG, preserve_topology=False) for ln in kept]
    simplified = [ln for ln in simplified if len(ln.coords) >= 2]
    if not simplified:
        return None
    return simplified[0] if len(simplified) == 1 else MultiLineString(simplified)


def build_route_lines(
    gtfs_dirs: Union[str, Path, Iterable[Union[str, Path]]],
    aoi: Optional[gpd.GeoDataFrame] = None,
    exclude_no_shape_routes: bool = True,
) -> gpd.GeoDataFrame:
    """Build one line feature per GTFS `route_id` across one or more feeds.

    Args:
        gtfs_dirs: A GTFS directory, or an iterable of them (one per feed).
            Zipped feeds are not handled here -- pass extracted directories.
        aoi: Optional area of interest (any CRS); routes whose geometry does
            not intersect its union are dropped, so a metro-scale map isn't
            carrying intercity coach routes running hundreds of km away.
        exclude_no_shape_routes: 2026-09-01, user-requested default: a route
            with no real `shapes.txt` geometry for ANY of its trips (either
            the whole feed lacks `shapes.txt`, or every trip on that route
            has a null/missing `shape_id`) falls back to a straight-line
            connect-the-stops geometry (see `_stop_sequence_lines`) -- a
            straight line is not a real route alignment and can look
            actively wrong on a map (cutting through blocks/water/etc.).
            When `True` (the default -- "I want all the maps generated that
            way"), such routes are dropped from the returned GeoDataFrame
            entirely rather than drawn with fabricated geometry. Every row
            (including ones that WOULD be dropped) still carries a real
            `has_real_shape` boolean column when `False`, so a caller that
            wants the full route list regardless (e.g. for a downloadable
            routes list, as opposed to the map's own line layer) can pass
            `False` and filter on that column itself.

    Returns:
        A WGS84 GeoDataFrame with one row per `(feed, route_id)`:
        `route_uid` (feed-qualified, unique across feeds), `route_id`,
        `feed`, `route_label` (`route_short_name` -> `route_long_name` ->
        `route_id`), `route_short_name`, `route_long_name`, `mode`
        (`transitlos.scoring.mode.mode_category` of `route_type`), `color`
        (GTFS `route_color` or `MODE_FALLBACK_COLOR[mode]`), `text_color`
        (GTFS `route_text_color` or white), `has_real_shape` (`True` if any
        of the route's trips had real `shapes.txt` geometry, `False` if
        every trip fell back to the straight-line stop-sequence
        connect-the-dots geometry), plus LineString/MultiLineString
        `geometry`.
    """
    if isinstance(gtfs_dirs, (str, Path, os.PathLike)):
        dirs = [Path(gtfs_dirs)]
    else:
        dirs = [Path(d) for d in gtfs_dirs]

    records: list[dict] = []
    for feed_dir in dirs:
        routes = _read_csv(
            feed_dir / "routes.txt",
            ["route_id", "route_short_name", "route_long_name", "route_type", "route_color", "route_text_color"],
        )
        trips = _read_csv(feed_dir / "trips.txt", ["route_id", "trip_id", "shape_id"])
        if routes is None or trips is None or "route_id" not in routes.columns:
            continue
        routes = routes.dropna(subset=["route_id"]).drop_duplicates("route_id")

        shape_lines: dict[str, LineString] = {}
        shapes = _read_csv(feed_dir / "shapes.txt", ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
        if shapes is not None and {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.issubset(shapes.columns):
            shape_lines = _shape_lines(shapes)
            for bad_id in EXCLUDED_SHAPE_IDS:
                shape_lines.pop(bad_id, None)

        by_route: dict[str, list[LineString]] = {}
        if "shape_id" in trips.columns and shape_lines:
            pairs = trips.dropna(subset=["route_id", "shape_id"]).drop_duplicates(["route_id", "shape_id"])
            for route_id, shape_id in zip(pairs["route_id"], pairs["shape_id"]):
                line = shape_lines.get(str(shape_id))
                if line is not None:
                    by_route.setdefault(str(route_id), []).append(line)

        # Snapshot BEFORE the stop-sequence fallback below adds its own
        # entries -- any route already present here got at least one real
        # `shapes.txt`-derived line; anything added after is fallback-only.
        real_shape_route_ids = set(by_route)

        missing = {str(r) for r in routes["route_id"]} - set(by_route)
        for route_id, line in _stop_sequence_lines(feed_dir, missing, trips).items():
            by_route.setdefault(route_id, []).append(line)

        for _, row in routes.iterrows():
            route_id = str(row["route_id"])
            geom = _collapse_branches(by_route.get(route_id, []))
            if geom is None:
                continue
            short = row.get("route_short_name")
            long = row.get("route_long_name")
            short = None if short is None or pd.isna(short) or str(short).strip() == "" else str(short).strip()
            long = None if long is None or pd.isna(long) or str(long).strip() == "" else str(long).strip()
            route_type = row.get("route_type")
            try:
                mode = mode_category(int(route_type))
            except (TypeError, ValueError):
                mode = mode_category(None)
            fallback = MODE_FALLBACK_COLOR.get(mode, MODE_FALLBACK_COLOR["bus"])
            records.append(
                {
                    "route_uid": f"{feed_dir.name}:{route_id}",
                    "route_id": route_id,
                    "feed": feed_dir.name,
                    "route_short_name": short,
                    "route_long_name": long,
                    "route_label": short or long or route_id,
                    "mode": mode,
                    "color": _hex_color(row.get("route_color"), fallback),
                    "text_color": _hex_color(row.get("route_text_color"), "#FFFFFF"),
                    "has_real_shape": route_id in real_shape_route_ids,
                    "geometry": geom,
                }
            )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "route_uid", "route_id", "feed", "route_short_name", "route_long_name",
                "route_label", "mode", "color", "text_color", "has_real_shape", "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if aoi is not None and len(aoi):
        aoi_union = aoi.to_crs(4326).union_all()
        gdf = gdf[gdf.geometry.intersects(aoi_union)].reset_index(drop=True)
    if exclude_no_shape_routes:
        gdf = gdf[gdf["has_real_shape"]].reset_index(drop=True)
    return gdf
