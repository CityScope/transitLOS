"""GTFS -> station profiles (up to P (mode, headway, speed) combinations per parent station).

Same method as `transitlos.scoring.direction_subset` (feeds grouped by window, stops grouped within 150 m, the same departure table, the same candidate
search: three mode tiers x minimum trip speed in 5 km/h steps, headway = the better of the two directions, speed = sum(distance)/sum(time)). The
difference: instead of keeping only the best candidate per station, every candidate is kept, then floors + dominance + the cap (`core.select_profiles`)
choose at most P of them.

Modes: 0 bus, 1 tram, 2 rail (the tier, as `direction_subset` scores it).

    cand = state_candidates(gtfs_dirs, aoi)                # slow part (loads the feeds); cache it
    stations, prof, cutoffs = select(cand, Config(P=10))   # fast; redo with other caps/floors without reloading
"""
from __future__ import annotations

import os
import math
import time
from datetime import date, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

import logging

from . import core
from .scoring import DEFAULT, ScoreParams

logger = logging.getLogger(__name__)

ANALYSIS_DATE = date(2026, 10, 7)
WINDOW = (date(2026, 10, 1), date(2026, 10, 15))
START, END = dtime(6, 0), dtime(22, 0)
STOP_GROUP_M = 150.0
os.environ["GTFS_STOP_GROUPING"] = "complete"   # strict 150 m stations (no two stops of a station farther apart), as the US/Spain/city studies; read by pyGTFSHandler at Feed construction
SPEED_STEP = 5.0
_TIERS = ((0, "bus", 0), (1, "tram", 1), (2, "rail", 2))   # mode id, name, minimum mode rank kept


REQUIRED = ("routes.txt", "trips.txt", "stop_times.txt", "stops.txt")


def usable_feeds(gtfs_dirs: Sequence[Path]) -> List[Path]:
    """Feed folders that really hold a static GTFS (some downloads are not: empty folders, partial zips).

    Args:
        gtfs_dirs: Candidate feed folders.

    Returns:
        The usable folders.
    """
    return [Path(d) for d in gtfs_dirs if all((Path(d) / f).is_file() for f in REQUIRED)]


def drop_duplicate_feeds(gtfs_dirs: Sequence[Path]) -> List[Path]:
    """The same service published twice (agency feed + Mobility Database copy, ...) would count every trip twice and change the stop clustering;
    keep the larger copy, as the original study does.

    Args:
        gtfs_dirs: Feed folders.

    Returns:
        The folders without the duplicates.
    """
    from pyGTFSHandler.utils.feed_dedupe import drop_duplicate_feeds as _drop
    return _drop(list(gtfs_dirs), service_date=ANALYSIS_DATE)   # of two copies, the one running on the analysis date


def feed_route_modes(feed) -> pl.DataFrame:
    """route_id -> scoring_mode for the route ids the loaded `Feed` really uses, from each route's own GTFS route_type (BRT marker 702 -> tram).

    Keyed by the Feed's final ids: when several feeds are loaded together the Feed renames a route id that exists in more than one of them to
    `<id>_file_<n>`. A table built from the feeds' own routes.txt files and joined on the final id gave those routes no mode, and an id shared by
    a bus feed and a rail feed took the first feed's mode -- LA Metro Rail, LIRR, NJ Transit rail... came out as buses or were lost
    (the original study fixed the same trap in usa_study/code/best_subset_scoring.feed_route_modes).

    Args:
        feed: A loaded pyGTFSHandler Feed.

    Returns:
        ``route_id, scoring_mode`` table.
    """
    from transitlos.scoring.mrc import mode_category
    if feed.routes is None:
        return pl.DataFrame(schema={"route_id": pl.Utf8, "scoring_mode": pl.Utf8})
    r = feed.routes.lf.select(["route_id", "route_type"]).collect()
    rts = [3 if rt is None else int(rt) for rt in r["route_type"].to_list()]   # empty route_type: bus (the GTFS default for local routes)
    modes = ["tram" if rt == 702 else mode_category(rt) for rt in rts]
    return pl.DataFrame({"route_id": r["route_id"].cast(pl.Utf8), "scoring_mode": modes}).unique("route_id", keep="first")


def feed_candidates(feed, analysis_date, route_modes, start=START, end=END, step=SPEED_STEP, speed_method=None, window_min=None,
                    window_direction=None) -> pd.DataFrame:
    """Every (tier, minimum speed) candidate of every parent station: columns parent_station, mode, cutoff, headway_min, speed_kmh.

    ``speed_method`` / ``window_min`` / ``window_direction``: how the speed at the station is measured
    (`transitlos.scoring.direction_subset.departures_with_direction`); ``None`` = its defaults (straight line, 15 min, both sides).

    Args:
        feed: A loaded pyGTFSHandler Feed.
        analysis_date: Service date.
        route_modes: ``route_id -> scoring_mode`` table.
        start: Start of the scoring window.
        end: End of the scoring window.
        step: Minimum-speed step (km/h).
        speed_method: ``"straight"`` or ``"shape"``; ``None`` = the default (straight).
        window_min: Minutes on each side of the station; ``None`` = 15.
        window_direction: ``"both"``, ``"forward"`` or ``"backward"``; ``None`` = both.

    Returns:
        One row per candidate.
    """
    from transitlos.scoring import direction_subset as ds
    from transitlos.scoring.direction_subset import departures_with_direction, direction_headway_minutes, _to_seconds
    df = departures_with_direction(feed, analysis_date, start, end, route_modes, speed_method or ds.SPEED_METHOD,
                                   ds.SPEED_WINDOW_MIN if window_min is None else window_min, window_direction or ds.SPEED_WINDOW_DIRECTION)
    if df.is_empty():
        return pd.DataFrame(columns=["parent_station", "mode", "cutoff", "headway_min", "speed_kmh"])
    s_sec, e_sec = _to_seconds(start), _to_seconds(end)
    st = df["parent_station"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
    t_all, sec_all, rank_all = df["departure_time"].to_numpy(), df["sector"].to_numpy(), df["rank"].to_numpy()
    spd_all, dist_all, tm_all = df["speed_kmh"].to_numpy(), df["dist_m"].to_numpy(), df["time_s"].to_numpy()
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t, sec, rank, spd, dist, tm = t_all[a:b], sec_all[a:b], rank_all[a:b], spd_all[a:b], dist_all[a:b], tm_all[a:b]
        for mode, _, min_rank in _TIERS:
            in_tier = rank >= min_rank
            if not in_tier.any():
                continue
            known = spd[in_tier]
            known = known[np.isfinite(known)]
            top = math.floor(known.max() / step) * step if known.size else 0.0
            seen = set()
            for i in range(int(round(top / step)) + 1):
                cutoff = i * step
                keep = in_tier & (~np.isfinite(spd) | (spd >= cutoff))
                n_keep = int(keep.sum())
                if n_keep == 0 or n_keep in seen:
                    continue
                seen.add(n_keep)
                hw = min(direction_headway_minutes(t[keep & (sec == s)], s_sec, e_sec) for s in (0, 1))
                ok = keep & np.isfinite(dist) & np.isfinite(tm) & (tm > 0)
                if not math.isfinite(hw) or not ok.any():
                    continue
                rows.append((st[a], mode, cutoff, hw, float(dist[ok].sum() / 1000.0 / (tm[ok].sum() / 3600.0))))
    return pd.DataFrame(rows, columns=["parent_station", "mode", "cutoff", "headway_min", "speed_kmh"])


def split_feeds_by_window(gtfs_dirs: Sequence[Path], aoi, analysis_date=ANALYSIS_DATE, stop_group_distance: float = STOP_GROUP_M):
    """(feeds with a trip on `analysis_date`, the others). The others are scored alone on their own typical weekday (a feed with no service on the
    shared day would otherwise lose all its stops silently).

    Args:
        gtfs_dirs: Feed folders.
        aoi: Study area.
        analysis_date: The shared analysis date.
        stop_group_distance: Stop grouping distance (m).
    """
    from pyGTFSHandler.feed import Feed
    inside, outside = [], []
    for d in gtfs_dirs:
        try:
            feed = Feed(gtfs_dirs=[str(d)], aoi=aoi, start_date=analysis_date, end_date=analysis_date, stop_group_distance=stop_group_distance)
            ok = feed.trips.lf.limit(1).collect().height > 0
        except Exception as exc:   # real feeds vary widely
            logger.warning("feed %s: could not check its service window, scored alone: %s", d, exc)
            ok = False
        if ok:   # a holiday or a normal day, never a mix: reduced / holiday service on the shared day -> the feed's own typical weekday
            from ..day_check import day_volume_ratio

            try:
                r = day_volume_ratio(d, analysis_date)
            except Exception:  # noqa: BLE001
                r = None
            if r is not None and not (0.8 <= r <= 1.25):
                logger.warning("feed %s: %s runs %.0f%% of its usual trips for that weekday -- scored alone on its own typical day", d, analysis_date, 100 * r)
                ok = False
        (inside if ok else outside).append(d)
    return inside, outside


def weighted_stations(feed, parents: pd.DataFrame):
    """Real stops of the stations in `parents` (column parent_station) with the station location as the centroid of its stop_ids weighted by their
    service (stop_times rows of the loaded window, proportional to 1/headway) -- so a station sits where its service is, not at the plain mean.
    Returns (stops: stop_id, stop_name, stop_lat, stop_lon, parent_station, weight; stations: parent_station, lat, lon).

    Args:
        feed: A loaded pyGTFSHandler Feed.
        parents: Table with a ``parent_station`` column.
    """
    st = feed.stops.lf.select([c for c in ("stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station") if c in feed.stops.lf.collect_schema().names()]).collect().to_pandas()
    st["stop_id"] = st["stop_id"].astype(str)
    if "parent_station" not in st.columns:
        st["parent_station"] = st["stop_id"]
    w = feed.stop_times.lf.select(pl.col("stop_id").cast(pl.Utf8)).group_by("stop_id").agg(pl.len().alias("weight")).collect(engine="streaming").to_pandas()
    st = st.merge(w, on="stop_id", how="inner")   # only stops with service
    st = st[st.parent_station.isin(parents.parent_station)].copy()
    st["wlat"] = st.stop_lat * st.weight
    st["wlon"] = st.stop_lon * st.weight
    g = st.groupby("parent_station")
    stations = pd.DataFrame({"lat": g.wlat.sum() / g.weight.sum(), "lon": g.wlon.sum() / g.weight.sum()}).reset_index()
    return st.drop(columns=["wlat", "wlon"]), stations


def state_candidates(gtfs_dirs: Sequence[Path], state_aoi, log=print, stops_sink: list | None = None, speed_method=None, window_min=None,
                     window_direction=None) -> pd.DataFrame:
    """Candidates of all stations inside `state_aoi` (any area; a state in the US study): columns lat, lon, mode, cutoff, headway_min, speed_kmh (feeds loaded as the original does:
    in-window feeds together, every other feed alone on its own typical weekday). Speed options as in `feed_candidates`.

    Args:
        gtfs_dirs: Feed folders of the area.
        state_aoi: The area (GeoDataFrame).
        log: Progress function.
        stops_sink: Optional list that receives the real stops of each feed group.
        speed_method: ``"straight"`` or ``"shape"``; ``None`` = the default (straight).
        window_min: Minutes on each side of the station; ``None`` = 15.
        window_direction: ``"both"``, ``"forward"`` or ``"backward"``; ``None`` = both.

    Returns:
        The candidates.
    """
    from pyGTFSHandler.feed import Feed

    gtfs_dirs = usable_feeds(gtfs_dirs)
    gtfs_dirs = drop_duplicate_feeds(gtfs_dirs)
    in_window, fallback = split_feeds_by_window(list(gtfs_dirs), state_aoi)
    frames = []

    def one(dirs, fallback_feed=False):
        if fallback_feed:   # no service in the shared window: alone, on its own typical weekday
            feed = Feed(gtfs_dirs=[str(d) for d in dirs], aoi=state_aoi, stop_group_distance=STOP_GROUP_M)
            day = feed.get_representative_date(date_type="weekday")
        else:
            feed = Feed(gtfs_dirs=[str(d) for d in dirs], aoi=state_aoi, start_date=WINDOW[0], end_date=WINDOW[1], stop_group_distance=STOP_GROUP_M)
            day = ANALYSIS_DATE
        if day is None:
            return None
        c = feed_candidates(feed, day, feed_route_modes(feed), speed_method=speed_method, window_min=window_min, window_direction=window_direction)
        if c.empty:
            return None
        stops, xy = weighted_stations(feed, c[["parent_station"]].drop_duplicates())
        c = c.merge(xy, on="parent_station", how="inner")
        if stops_sink is not None:   # every real stop_id at its own location, with the station (rounded lat/lon = the station key of `select`) it belongs to
            stops = stops.merge(xy.rename(columns={"lat": "station_lat", "lon": "station_lon"}), on="parent_station", how="inner")
            stops_sink.append(stops.drop(columns="parent_station"))
        return c.drop(columns="parent_station")

    if in_window:
        try:
            f = one(in_window)
            if f is not None:
                frames.append(f)
        except Exception as exc:   # one bad feed must not discard the others
            log(f"combined load failed ({exc}); feeds one by one")
            for d in in_window:
                try:
                    f = one([d])
                    if f is not None:
                        frames.append(f)
                except Exception as exc2:
                    log(f"feed {d} skipped: {exc2}")
    for d in fallback:
        try:
            f = one([d], fallback_feed=True)
            if f is not None:
                frames.append(f)
        except Exception as exc:
            log(f"fallback feed {d} skipped: {exc}")
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "mode", "cutoff", "headway_min", "speed_kmh"])
    return pd.concat(frames, ignore_index=True)


def select(cand: pd.DataFrame, cfg: core.Config = core.Config(), params: ScoreParams = DEFAULT):
    """Group candidates by station location (same place in different feed groups = one station), apply floors/dominance/cap.
    Returns (stations DataFrame [lat, lon, n_candidates, n_profiles, dropped], core.Profiles, selected candidate rows).

    Args:
        cand: Candidates from :func:`state_candidates`.
        cfg: Caps and floors.
        params: Scoring parameters.
    """
    cand = cand.copy()
    cand["key"] = list(zip(cand.lat.round(6), cand.lon.round(6)))
    cand["sid"] = pd.factorize(cand["key"])[0]
    cand = cand.sort_values("sid", kind="stable").reset_index(drop=True)
    S = int(cand.sid.max()) + 1 if len(cand) else 0
    counts = np.bincount(cand.sid.to_numpy(), minlength=S)
    off = np.r_[0, np.cumsum(counts)].astype(np.int32)
    idx, n, dr = core.select_profiles(off, cand["mode"].to_numpy(), cand.headway_min.to_numpy(), cand.speed_kmh.to_numpy(), cfg, params)
    P = cfg.P
    mode = np.full((S, P), -1, dtype=np.int8); h = np.zeros((S, P), dtype=np.float32); v = np.zeros((S, P), dtype=np.float32)
    cut = np.zeros((S, P), dtype=np.float32)
    ok = idx >= 0
    rows = np.where(ok, idx, 0)
    mode[ok] = cand["mode"].to_numpy()[rows][ok]
    h[ok] = cand.headway_min.to_numpy()[rows][ok]
    v[ok] = cand.speed_kmh.to_numpy()[rows][ok]
    cut[ok] = cand.cutoff.to_numpy()[rows][ok]
    first = cand.drop_duplicates("sid")[["sid", "lat", "lon"]].sort_values("sid")
    stations = pd.DataFrame({"lat": first.lat.to_numpy(), "lon": first.lon.to_numpy(), "n_candidates": counts, "n_profiles": n, "dropped": dr})
    keep = n > 0   # a station with no profile left (all below the floors) is not stored at all
    prof = core.Profiles(mode[keep], h[keep], v[keep])
    stations = stations[keep].reset_index(drop=True)
    stations["cutoff_json"] = [",".join(f"{c:g}" for c in row[m >= 0]) for row, m in zip(cut[keep], mode[keep])]
    return stations, prof, cut[keep]
