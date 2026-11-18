"""Best stop score per parent station from merged departures, split into two travel directions.

For every parent station all departures of every route are pooled and split into the two opposite directions of travel (the angle of
the line through the station that best separates the shape directions, "uptown" / "downtown"); route ids play no role. The headway of a
direction is computed on the MERGED, sorted departure times -- ``(sum(gap^2) + (first - start + end - last)^2) / window`` -- so two buses
that leave at the same moment count once (0,5,0,5 scores like 5,5) and a duplicated timetable changes nothing.

Each departure carries the speed of its trip at the station and the mode tier of its route. The speed (``speed_method``, 2026-11-17: the
way speed is computed on every map) is measured over a window of ``window_min`` minutes (default 15) before and after the station (``window_direction``: both sides by
default, or only after / before):

* ``"straight"`` (default): straight-line distance from the station to where the trip is 15 min before plus 15 min after, over the
  time taken -- a winding line is slower than a direct one at the same running speed;
* ``"shape"``: the same window, distance along the shape.

(Before 2026-11-17 the speed was the whole trip's shape distance / scheduled travel time.)

Near a trip's ends the window is cut at the first / last stop (the time shrinks with it). For each
mode tier (bus = every mode, tram = tram + rail, rail = rail only) and each minimum trip speed (0, 5, 10, ... km/h up to the fastest
trip of the tier, steps of `speed_bucket_kmh`) the departures that pass are kept, the headway of each direction is recomputed without the slow trips and
the better direction is used; the mean speed of the kept trips and the tier's mode give the three scores. The combination with the
highest `stop_score` wins, so excluding slow lines sometimes raises a stop's score.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from .frequency import frequency_score
from .mrc import mrc_score
from .speed import speed_score
from .stop_score import stop_score as compute_final_stop_score

_TIERS = (("bus", 0, 3), ("tram", 1, 0), ("rail", 2, 2))   # name, minimum mode rank kept, canonical GTFS route_type of the tier
_RANK = {"bus": 0, "tram": 1, "rail": 2}
# speed defaults (2026-11-17, user); every function below takes them as parameters. Computed by pyGTFSHandler's Feed.get_window_distances.
SPEED_METHOD = "straight"        # "straight" | "shape"
SPEED_WINDOW_MIN = 15.0          # minutes on each side of the station
SPEED_WINDOW_DIRECTION = "both"  # "both" | "forward" | "backward"


def _to_seconds(t) -> float:
    if isinstance(t, datetime):
        t = t.time()
    return t.hour * 3600 + t.minute * 60 + t.second


def direction_headway_minutes(times: np.ndarray, start_sec: float, end_sec: float) -> float:
    """Headway (minutes) of one direction from its sorted departure times (seconds): the mean-wait style measure used by the library.

    Args:
        times: Sorted departure times (s).
        start_sec: Window start (s).
        end_sec: Window end (s).

    Returns:
        Headway in minutes (inf with no departure).
    """
    if len(times) == 0:
        return math.inf
    gaps = np.diff(times)
    initial = (times[0] - start_sec) + (end_sec - times[-1])
    return float((np.sum(gaps * gaps) + initial * initial) / (end_sec - start_sec) / 60.0)


def departures_with_direction(feed, analysis_date, start_time, end_time, route_modes: pl.DataFrame, speed_method: str = SPEED_METHOD,
                              window_min: float = SPEED_WINDOW_MIN, window_direction: str = SPEED_WINDOW_DIRECTION) -> pl.DataFrame:
    """One row per departure at a parent station: ``parent_station, departure_time, route_id, rank, speed_kmh, dist_m, time_s, sector``.

    Args:
        feed: A loaded ``pyGTFSHandler`` Feed.
        analysis_date: Service date.
        start_time: Start of the scoring window.
        end_time: End of the scoring window.
        route_modes: ``route_id -> scoring_mode`` table.
        speed_method: ``"straight"`` (default) or ``"shape"`` distance (module docstring).
        window_min: Minutes on each side of the station over which the speed is measured (default 15).
        window_direction: ``"both"`` (default), ``"forward"`` or ``"backward"``.

    Returns:
        One row per departure; ``dist_m`` / ``time_s`` are the window distance and time its speed comes from.
    """
    from pyGTFSHandler.utils import geo_polars

    lf = feed.filter(date=analysis_date, start_time=start_time, end_time=end_time, in_aoi=True, frequencies=True, delete_last_stop=True)
    names = lf.collect_schema().names()
    need = ["parent_station", "route_id", "direction_id", "shape_id", "departure_time", "arrival_time", "start_time", "end_time", "headway_secs",
            "shape_time_traveled", "shape_direction", "shape_direction_backwards", "stop_sequence", "service_id", "day_offset", "route_type",
            "shape_total_distance", "shape_total_travel_time", "trip_id"]
    lf = lf.filter(pl.col("parent_station").is_not_null()).select([c for c in need if c in names])
    # window speed per (trip, stop), joined before the frequency expansion (every departure of a pattern shares it)
    ws = feed.get_window_distances(analysis_date, window_min=window_min, distance=speed_method, direction=window_direction).rename(
        {"distance_weight": "dist_m", "time_weight": "time_s"})
    lf = lf.join(ws, on=["trip_id", "stop_sequence"], how="left")
    lf = feed._frequencies_to_departures_lean(lf)
    start_sec, end_sec = _to_seconds(start_time), _to_seconds(end_time)
    df = (
        lf.with_columns(pl.col("departure_time").cast(pl.Float64, strict=False))
        .filter(pl.col("departure_time").is_finite() & (pl.col("departure_time") >= start_sec) & (pl.col("departure_time") <= end_sec))
        .collect()
    )
    if df.is_empty():
        return df
    # forward / backward shape directions agree 180 degrees apart (same reconciliation as the library's shape_direction headway)
    d, b = pl.col("shape_direction"), pl.col("shape_direction_backwards")
    diff = pl.when(((d + 360 - b) % 360) > ((b + 360 - d) % 360)).then(-1 * (180 - ((b + 360 - d) % 360)) / 2).otherwise((180 - ((d + 360 - b) % 360)) / 2)
    df = df.with_columns(diff.fill_nan(0).fill_null(0).alias("_sd")).with_columns(((pl.col("shape_direction") + 360 + pl.col("_sd")) % 360).alias("shape_direction")).drop("_sd")
    per_station = df.group_by("parent_station").agg(pl.col("shape_direction"))
    split = geo_polars.max_separation_angle(per_station, "shape_direction")
    per_station = per_station.select("parent_station").with_columns(split.alias("_split"))
    df = df.join(per_station, on="parent_station", how="left").with_columns(
        (((pl.col("shape_direction") - pl.col("_split") + 360) % 360) * 2 / 360).floor().cast(pl.Int8, strict=False).fill_null(0).clip(0, 1).alias("sector")
    )
    modes = route_modes.select(pl.col("route_id").cast(pl.Utf8), pl.col("scoring_mode")).unique("route_id", keep="first")
    df = df.with_columns(pl.col("route_id").cast(pl.Utf8)).join(modes, on="route_id", how="left")
    rank = pl.col("scoring_mode").replace_strict(_RANK, default=0, return_dtype=pl.Int8)
    dist = pl.col("dist_m").cast(pl.Float64)
    tm = pl.when(pl.col("time_s") > 0).then(pl.col("time_s").cast(pl.Float64)).otherwise(None)
    return df.select(
        "parent_station", "departure_time", "route_id", "sector", rank.alias("rank"), dist.alias("dist_m"), tm.alias("time_s"),
        (dist / 1000.0 / (tm / 3600.0)).alias("speed_kmh"),
    ).sort("parent_station", "departure_time")


@lru_cache(maxsize=200_000)
def _scores(headway_min: float, speed_kmh: float, route_type: int, region: str):
    mrc = mrc_score(route_type=route_type, headway_cv=None, region=region)
    sp = speed_score(speed_kmh, region=region)
    fr = frequency_score(headway_min, region=region)
    return mrc, sp, fr, compute_final_stop_score(mrc, sp, fr, region=region)


def best_stop_scores_by_direction(
    feed,
    analysis_date: Union[date, datetime],
    route_modes: pl.DataFrame,
    start_time: Union[datetime, time] = time(6, 0),
    end_time: Union[datetime, time] = time(22, 0),
    region: str = "global",
    speed_bucket_kmh: float = 5.0,
    speed_method: str = SPEED_METHOD,
    window_min: float = SPEED_WINDOW_MIN,
    window_direction: str = SPEED_WINDOW_DIRECTION,
) -> pd.DataFrame:
    """Best (mode tier, minimum trip speed) per parent station; see the module docstring.

    Returns one row per parent station with ``parent_station, route_ids, scoring_mode, headway_minutes, avg_speed_kmh,
    min_speed_cutoff_kmh, mrc_score, speed_score, frequency_score, stop_score`` (same columns as the route-based search it replaces).

    Args:
        feed: A loaded pyGTFSHandler Feed.
        analysis_date: Service date.
        route_modes: ``route_id -> scoring_mode`` table.
        start_time: Start of the scoring window.
        end_time: End of the scoring window.
        region: Scoring region (curve set).
        speed_bucket_kmh: Minimum-speed step (km/h).
        speed_method: ``"straight"`` or ``"shape"``.
        window_min: Minutes on each side of the station.
        window_direction: ``"both"``, ``"forward"`` or ``"backward"``.
    """
    df = departures_with_direction(feed, analysis_date, start_time, end_time, route_modes, speed_method, window_min, window_direction)
    if df.is_empty():
        return pd.DataFrame()
    start_sec, end_sec = _to_seconds(start_time), _to_seconds(end_time)
    st = df["parent_station"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
    t_all = df["departure_time"].to_numpy()
    sec_all = df["sector"].to_numpy()
    rank_all = df["rank"].to_numpy()
    spd_all = df["speed_kmh"].to_numpy()
    dist_all = df["dist_m"].to_numpy()
    tm_all = df["time_s"].to_numpy()
    rid_all = df["route_id"].to_numpy()
    results = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t, sec, rank, spd, dist, tm = t_all[a:b], sec_all[a:b], rank_all[a:b], spd_all[a:b], dist_all[a:b], tm_all[a:b]
        best = None
        for name, min_rank, route_type in _TIERS:
            in_tier = rank >= min_rank
            if not in_tier.any():
                continue
            known = spd[in_tier]
            known = known[np.isfinite(known)]
            top = math.floor(known.max() / speed_bucket_kmh) * speed_bucket_kmh if known.size else 0.0
            cutoffs = [i * speed_bucket_kmh for i in range(int(round(top / speed_bucket_kmh)) + 1)]   # 0, 5, 10, ... up to the fastest trip's step
            seen_keep = set()
            for cutoff in cutoffs:
                keep = in_tier & (~np.isfinite(spd) | (spd >= cutoff))
                n_keep = int(keep.sum())
                if n_keep == 0 or n_keep in seen_keep:   # a higher cutoff that drops nobody new gives the same result
                    continue
                seen_keep.add(n_keep)
                hw = min(direction_headway_minutes(t[keep & (sec == s)], start_sec, end_sec) for s in (0, 1))
                ok = keep & np.isfinite(dist) & np.isfinite(tm) & (tm > 0)
                if not math.isfinite(hw) or not ok.any():
                    continue
                speed = float(dist[ok].sum() / 1000.0 / (tm[ok].sum() / 3600.0))
                mrc, sp, fr, score = _scores(round(hw, 3), round(speed, 2), route_type, region)
                if best is None or score > best["stop_score"]:
                    best = {"parent_station": st[a], "route_ids": ";".join(sorted({str(r) for r in rid_all[a:b][keep]})), "scoring_mode": name,
                            "headway_minutes": hw, "avg_speed_kmh": speed, "min_speed_cutoff_kmh": cutoff, "mrc_score": mrc,
                            "speed_score": sp, "frequency_score": fr, "stop_score": score}
        if best is not None:
            results.append(best)
    return pd.DataFrame(results)
