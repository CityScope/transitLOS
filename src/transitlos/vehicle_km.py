"""Daily vehicle-kilometres of a GTFS feed inside any number of areas.

`vehicle_km = sum over the trips that run on the day of (runs of the trip) x (length of the trip inside the area)`:
  * trips that run: services active on `day` (`calendar.txt` + `calendar_dates.txt`);
  * runs: 1 for a normal trip; a trip listed in `frequencies.txt` runs `(end_time - start_time) // headway_secs` times per window
    (a trip every 5 minutes is 12 trips an hour);
  * length: the `shapes.txt` line of the trip clipped to the area; a trip without a shape uses straight lines between its stops.
Shared by the city studies (metro / core) and the US study (every county and place).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import polars as pl
import pyproj
import shapely


def scan_csv(path: Path) -> Optional[pl.LazyFrame]:
    """Lazy GTFS table with BOM-stripped column names; quote handling disabled for files with literal quotes inside fields.

    Args:
        path: GTFS text file.

    Returns:
        LazyFrame of text columns, or None.
    """
    if not path.is_file():
        return None
    from pyGTFSHandler.utils.io import _has_balanced_quotes

    quoted = _has_balanced_quotes(str(path))
    lf = pl.scan_csv(path, infer_schema=False, raise_if_empty=False, truncate_ragged_lines=True, quote_char='"' if quoted else None)
    names = lf.collect_schema().names()
    if not names:
        return None
    if not quoted:  # fields still carry their literal quotes: drop them so ids / dates / headers parse
        lf = lf.with_columns(pl.all().str.replace_all('"', "", literal=True))
        return lf.rename({c: c.replace('"', "").strip().lstrip("﻿") for c in names})
    return lf.rename({c: c.strip().lstrip("﻿") for c in names})


def _secs(col: str):
    p = pl.col(col).str.strip_chars().str.split(":")
    return (p.list.get(0).cast(pl.Int64, strict=False) * 3600 + p.list.get(1).cast(pl.Int64, strict=False) * 60
            + p.list.get(2).cast(pl.Int64, strict=False).fill_null(0))


def active_trips(gtfs_dir: Path, day: date) -> pl.DataFrame:
    """`trip_id, runs, shape_id` of every trip running on `day` (frequency windows expanded into runs).

    Args:
        gtfs_dir: Unzipped GTFS folder.
        day: Service date.
    """
    cal = scan_csv(gtfs_dir / "calendar.txt")
    if cal is not None:   # a feed whose calendar ended before (or starts after) the analysis day: use the same weekday of its nearest week, like a stale feed is read everywhere else
        span = cal.filter(pl.col("start_date").str.len_chars() == 8, pl.col("end_date").str.len_chars() == 8).select(pl.col("start_date").min().alias("lo"), pl.col("end_date").max().alias("hi")).collect()
        lo, hi = span["lo"][0], span["hi"][0]
        if lo is not None and hi is not None:
            lo_d, hi_d = datetime.strptime(str(lo), "%Y%m%d").date(), datetime.strptime(str(hi), "%Y%m%d").date()
            if day > hi_d:
                day = hi_d - timedelta(days=(hi_d.weekday() - day.weekday()) % 7)
            elif day < lo_d:
                day = lo_d + timedelta(days=(day.weekday() - lo_d.weekday()) % 7)
            # never a holiday of the feed: a mapped date with calendar_dates exceptions (e.g. 25 December, which ran the Sunday
            # service on top of the Monday-Saturday one in Guadalajara's feed) moves a week further from the edge, up to 8 weeks
            cdx = scan_csv(gtfs_dir / "calendar_dates.txt")
            if cdx is not None:
                exc = set(cdx.select(pl.col("date").cast(pl.Utf8)).collect()["date"])
                step = -7 if day >= lo_d + (hi_d - lo_d) / 2 else 7
                for _ in range(8):
                    if day.strftime("%Y%m%d") not in exc:
                        break
                    nd = day + timedelta(days=step)
                    if nd < lo_d or nd > hi_d:
                        break
                    day = nd
    ymd = day.strftime("%Y%m%d")
    wd = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][day.weekday()]
    active: set = set()
    if cal is not None:
        active = set(cal.filter((pl.col("start_date") <= ymd) & (pl.col("end_date") >= ymd) & (pl.col(wd) == "1"))
                     .select("service_id").collect()["service_id"])
    cd = scan_csv(gtfs_dir / "calendar_dates.txt")
    if cd is not None:
        e = cd.filter(pl.col("date") == ymd).collect()
        active = (active | set(e.filter(pl.col("exception_type") == "1")["service_id"])) - set(
            e.filter(pl.col("exception_type") == "2")["service_id"])
    trips = scan_csv(gtfs_dir / "trips.txt")
    cols = trips.collect_schema().names()
    sel = ["trip_id", "service_id"] + (["shape_id"] if "shape_id" in cols else [])
    t = trips.select(sel).filter(pl.col("service_id").is_in(list(active))).collect()
    if "shape_id" not in t.columns:
        t = t.with_columns(pl.lit(None, dtype=pl.Utf8).alias("shape_id"))
    t = t.with_columns(pl.lit(1, dtype=pl.Int64).alias("runs"))
    fr = scan_csv(gtfs_dir / "frequencies.txt")
    if fr is not None:
        f = fr.collect()
        if len(f):
            f = (f.with_columns(((_secs("end_time") - _secs("start_time")) // pl.col("headway_secs").cast(pl.Int64, strict=False))
                                .clip(lower_bound=1).alias("fr")).group_by("trip_id").agg(pl.col("fr").sum()))
            t = t.join(f, on="trip_id", how="left").with_columns(pl.coalesce(["fr", "runs"]).alias("runs")).drop("fr")
    return t.select("trip_id", "runs", "shape_id")


def metric_crs_for(lon: float, lat: float) -> pyproj.CRS:
    """The UTM zone CRS of a point.

    Args:
        lon: Longitude.
        lat: Latitude.

    Returns:
        pyproj CRS.
    """
    return pyproj.CRS.from_proj4(f"+proj=utm +zone={int((lon + 180) // 6) + 1} +{'north' if lat >= 0 else 'south'} +datum=WGS84")


def _lines_and_weights(gtfs_dir: Path, trips: pl.DataFrame) -> Tuple[list, np.ndarray]:
    """Unique line geometries (lon/lat coordinate arrays) of the active trips and the number of runs each carries."""
    lines, weights = [], []
    sh = scan_csv(gtfs_dir / "shapes.txt")
    grouped = trips.group_by("shape_id").agg(pl.col("runs").sum())   # grouped ONCE: group order differs between calls
    by_shape = {sid: r for sid, r in zip(grouped["shape_id"].to_list(), grouped["runs"].to_list()) if sid is not None}
    have_shape: set = set()
    if sh is not None and by_shape:
        g = (sh.filter(pl.col("shape_id").is_in(list(by_shape)))
             .with_columns(pl.col("shape_pt_lat").cast(pl.Float64, strict=False), pl.col("shape_pt_lon").cast(pl.Float64, strict=False),
                           pl.col("shape_pt_sequence").cast(pl.Int64, strict=False))
             .drop_nulls(["shape_pt_lat", "shape_pt_lon"]).sort("shape_id", "shape_pt_sequence")
             .group_by("shape_id", maintain_order=True).agg(pl.col("shape_pt_lon"), pl.col("shape_pt_lat")).collect())
        for sid, lon, lat in zip(g["shape_id"].to_list(), g["shape_pt_lon"].to_list(), g["shape_pt_lat"].to_list()):
            if len(lon) >= 2:
                lines.append(np.column_stack([lon, lat]))
                weights.append(by_shape[sid])
                have_shape.add(sid)
    rest = trips.filter(pl.col("shape_id").is_null() | ~pl.col("shape_id").is_in(list(have_shape)))
    if len(rest):
        st = (scan_csv(gtfs_dir / "stop_times.txt").select("trip_id", "stop_id", "stop_sequence")
              .filter(pl.col("trip_id").is_in(rest["trip_id"].to_list()))
              .with_columns(pl.col("stop_sequence").cast(pl.Int64, strict=False)).sort("trip_id", "stop_sequence")
              .group_by("trip_id", maintain_order=True).agg(pl.col("stop_id")).collect())
        stops = scan_csv(gtfs_dir / "stops.txt").select("stop_id", "stop_lat", "stop_lon").collect().with_columns(
            pl.col("stop_lat").cast(pl.Float64, strict=False), pl.col("stop_lon").cast(pl.Float64, strict=False)).drop_nulls()
        xy = dict(zip(stops["stop_id"].to_list(), zip(stops["stop_lon"].to_list(), stops["stop_lat"].to_list())))
        runs = dict(zip(rest["trip_id"].to_list(), rest["runs"].to_list()))
        patterns: Dict[tuple, int] = {}
        for tid, seq in zip(st["trip_id"].to_list(), st["stop_id"].to_list()):
            pat = tuple(s for s in seq if s in xy)
            if len(pat) >= 2:
                patterns[pat] = patterns.get(pat, 0) + runs.get(tid, 1)
        for pat, w in patterns.items():
            lines.append(np.array([xy[s] for s in pat]))
            weights.append(w)
    return lines, np.asarray(weights, dtype=float)


def vehicle_km_by_area(gtfs_dir: Path, day: date, areas_metric: gpd.GeoSeries, to_metric: pyproj.Transformer) -> Tuple[np.ndarray, np.ndarray]:
    """`(vehicle_km, trip_runs)` of the feed on `day` inside each area (same order as `areas_metric`, polygons in the metric CRS).

    Args:
        gtfs_dir: Unzipped GTFS folder.
        day: Service date.
        areas_metric: Areas in a metric CRS.
        to_metric: Transformer from EPSG:4326 to that CRS.
    """
    n = len(areas_metric)
    km, runs_in = np.zeros(n), np.zeros(n)
    trips = active_trips(gtfs_dir, day)
    if not len(trips):
        return km, runs_in
    coords, weights = _lines_and_weights(gtfs_dir, trips)
    if not coords:
        return km, runs_in
    counts = np.array([len(c) for c in coords])
    flat = np.vstack(coords)
    x, y = to_metric.transform(flat[:, 0], flat[:, 1])
    idx = np.repeat(np.arange(len(coords)), counts)
    ok = np.isfinite(x) & np.isfinite(y)
    xy = np.column_stack([x[ok], y[ok]])
    kept = idx[ok]
    keep_lines = np.unique(kept)
    if len(keep_lines) == 0:
        return km, runs_in
    # lines with fewer than 2 finite points cannot be built
    per = np.bincount(kept, minlength=len(coords))
    good = per >= 2
    sel = good[kept]
    lines = shapely.linestrings(xy[sel], indices=kept[sel])
    line_ids = np.unique(kept[sel])
    w = weights[line_ids]
    tree = shapely.STRtree(np.asarray(areas_metric.values))
    li, ai = tree.query(lines, predicate="intersects")
    if len(li) == 0:
        return km, runs_in
    inter = shapely.intersection(lines[li], np.asarray(areas_metric.values)[ai])
    length_km = shapely.length(inter) / 1000.0
    np.add.at(km, ai, length_km * w[li])
    np.add.at(runs_in, ai, np.where(length_km > 0, w[li], 0.0))
    return km, runs_in
