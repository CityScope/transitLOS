"""Load a GTFS feed and build a per-stop table of raw transit level-of-service inputs.

Wraps `pyGTFSHandler.feed.Feed` loading plus its `FeedAnalysisMixin` methods
(`get_headway_at_stops`, `get_speed_at_stops`, `add_stop_coords`) into a
single call that produces the raw inputs `transitlos.scoring` needs:
`headway_minutes` (from `get_headway_at_stops`, already returned in minutes)
and `avg_speed_kmh` (from `get_speed_at_stops`'s `speed` column, which is
already km/h -- see its implementation: `(distance_weight/1000) /
(time_weight/3600)`).

Honesty note on reliability: `get_headway_at_stops` returns only a harmonic
*mean* headway per stop/route, not a distribution, so there is no headway
coefficient-of-variation (`headway_cv`) available from pyGTFSHandler today.
This module does **not** fabricate one -- the returned frame's `headway_cv`
column is all-null, and `stop_scores.compute_stop_scores` documents how it
handles that honestly (it does not silently assume perfect reliability
without saying so).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional, Sequence, Union

import geopandas as gpd
import polars as pl
from pyGTFSHandler.feed import Feed

#: How far from "today" (in either direction) the auto-detected
#: representative-date search is allowed to look when no explicit
#: `date_range` is given. See the "Bug found and fixed 2026-08-24" note in
#: `download_and_prepare_stops` for why this bound exists: without it, a
#: multi-feed load's representative-date pick can drift months past a
#: dominant agency's real `calendar.txt` validity window, silently zeroing
#: out that agency's stops for the whole run.
#:
#: **Symmetric, not forward-only** (fixed same day, after a forward-only
#: version regressed San Francisco): a downloaded GTFS snapshot can go
#: stale in *either* direction relative to wall-clock "today" -- Boston's
#: MBTA feed was still valid at today's date (2026-08-24) (valid through
#: 2026-09-05), but San Francisco's BART feed had already expired by then (valid only
#: through 2026-08-07, 17 days stale) and Muni's by even more (valid only
#: through 2026-07-01, 54 days stale). A forward-only `[today, today+45d]`
#: window caught Boston fine but excluded both of SF's dominant agencies
#: entirely, so the search fell back to whichever minor regional operators
#: still had far-future calendar coverage -- SF's stop count collapsed
#: from 11,507 to 1,114 even though the remaining stops themselves scored
#: correctly. Reaching backward too catches a snapshot that's gone stale
#: by up to this many days without losing the original fix's protection
#: against a pick drifting hundreds of days into the future.
_REPRESENTATIVE_DATE_HORIZON_DAYS = 60


def download_and_prepare_stops(
    feed_source: Union[str, Path, Sequence[Union[str, Path]]],
    aoi: Optional[gpd.GeoDataFrame] = None,
    date_range: Optional[tuple[Union[date, datetime], Union[date, datetime]]] = None,
    start_time: Union[datetime, time] = time.min,
    end_time: Union[datetime, time] = time.max,
    route_types: Optional[Union[list, int, str]] = None,
    by: str = "route_id",
    at: str = "parent_station",
    how: str = "add",
    **kwargs,
) -> gpd.GeoDataFrame:
    """Load a GTFS feed and build a per-stop headway/speed table.

    `feed_source` is passed straight through to `pyGTFSHandler.feed.Feed`'s
    `gtfs_dirs` constructor argument -- this function does not implement any
    downloading/geocoding-to-feed-source resolution of its own, it only
    orchestrates a `Feed` that has already been (or can already be) loaded
    from a path/URL/list understood by `pyGTFSHandler`.

    Args:
        feed_source: One or more GTFS directory/zip paths, forwarded to
            `Feed(gtfs_dirs=feed_source, ...)`.
        aoi: Optional Area of Interest GeoDataFrame/GeoSeries used to
            geospatially filter stops while loading the feed.
        date_range: Optional `(start_date, end_date)` used as `Feed`'s
            `start_date`/`end_date` filters, and as the single analysis
            `date` (`start_date`) for `get_headway_at_stops`/
            `get_speed_at_stops`.

            **Performance:** passing `date_range` explicitly skips the
            auto-detection fallback below entirely. That fallback (calling
            `feed.get_representative_date`, which groups dates into
            calendar weeks, scores each week by its median daily service
            intensity, and returns the day within the best week with the
            most services -- see
            `pyGTFSHandler.representative_date.select_representative_date`
            for the full algorithm) is safe but expensive -- on a real
            multi-file Cambridge, MA GTFS feed it accounted for ~7.3s of
            this function's ~10.2s total runtime. If you already know a
            valid service date for the feed (e.g. from a prior run, from
            inspecting `calendar.txt`/`calendar_dates.txt`, or from
            documentation), pass `date_range=(that_date, that_date)` --
            it is strongly recommended whenever you have this information,
            since it turns a ~10s call into roughly ~3s. If omitted, the
            auto-detection fallback runs as described below (the safe
            default for callers who don't know a good date up front).
        start_time: Start of the daily analysis window, forwarded to both
            `get_headway_at_stops` and `get_speed_at_stops`. Defaults to
            midnight (full day). Pass e.g. `time(6, 0)` to restrict headway
            and speed to a specific service window (e.g. excluding
            overnight low-frequency service from the score).
        end_time: End of the daily analysis window, forwarded the same way.
            Defaults to end of day.
        route_types: Optional GTFS route type filter, forwarded to both
            `Feed` and the two analysis calls.
        by: Grouping key forwarded to `get_headway_at_stops`/
            `get_speed_at_stops` (`"route_id"` or `"shape_direction"`).
        at: Spatial stop-identity column forwarded to both analysis calls
            (`"parent_station"` or `"stop_id"`).
        how: Aggregation strategy forwarded to both analysis calls. Note
            `get_headway_at_stops` and `get_speed_at_stops` interpret
            `how` independently (see their own docstrings); for speed
            there is no `"best"` option so `"best"` is mapped to `"max"`.

            **Changed from `"best"` to `"add"` (2026-08-12), at the user's
            explicit request** -- diagnosed against a real multi-branch
            trunk stop (MBTA Green Line at Boylston, served by 4 branches):
            `how="best"` groups headway per `(stop, route, direction)` and
            keeps only the single most-frequent one, which can never
            reflect the combined frequency riders actually experience at a
            stop served by multiple routes/branches -- it reported 49.68
            min there (one branch's own headway) when the real combined
            Green Line frequency is close to 4-6 min. This was not a rare
            edge case: on a full Boston run, 89% of stops ended up with a
            null `stop_score` and the median reported headway was ~7.3
            hours, because most real-world stops are served by >=2
            routes/directions and `"best"` structurally can't combine
            them. `how="add"` (see `get_headway_at_stops`'s own docstring)
            combines routes via a harmonic sum of their rates
            (`1 / (1/headway).sum()`) within each direction, then --
            with `mix_directions=False`, the default forwarded here --
            keeps whichever direction has the better (lower) combined
            headway, rather than requiring both directions' data to be
            good. Verified against the same real Green Line stop: headway
            dropped from 49.68 min to 11.07 min, matching the
            user-confirmed real-world frequency far more closely.
        **kwargs: Extra keyword arguments forwarded to `Feed(...)`
            (e.g. `stop_group_distance`, `check_files`).

    Returns:
        geopandas.GeoDataFrame with columns: `stop_id` (or the `at` column
        used), `geometry` (Point, lon/lat, from `add_stop_coords`),
        `headway_minutes`, `headway_cv` (always null -- see module
        docstring), `avg_speed_kmh`, and `route_id`/`route_ids` if produced
        by the underlying calls.

    Raises:
        ValueError: If the feed has no usable date to analyze (no
            `date_range` given and the feed exposes no service dates).
    """
    start_date = date_range[0] if date_range else None
    end_date = date_range[1] if date_range else None

    feed = Feed(
        gtfs_dirs=feed_source,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        route_types=route_types,
        **kwargs,
    )

    analysis_date = start_date
    if analysis_date is None:
        # `feed.calendar.min_date`/`max_date` bound the union of every
        # calendar.txt row across every GTFS file loaded into this `Feed`,
        # which is not the same as a date any *service* actually runs on --
        # and, critically, not the same across different agencies/feeds
        # stacked into one multi-directory load. `feed.get_representative_date`
        # scores candidate dates by *total active service_id count across
        # every stacked feed*, which silently favors a date where some
        # minor operator (e.g. a seasonal ferry/shuttle with an unusually
        # long-running calendar.txt) has lots of service_ids active, even
        # if the metro area's dominant, highest-stop-density agency (e.g.
        # a subway operator) has NO calendar coverage that day at all.
        #
        # **Bug found and fixed 2026-08-24**, reproduced on a real Boston
        # run: across Boston's 60 stacked GTFS feeds, `get_representative_date`
        # picked 2026-11-18 -- more than two months past MBTA's own
        # `calendar.txt` validity window (2026-08-12 to 2026-09-05, from
        # that feed's `feed_info.txt`). Every MBTA stop_id (including every
        # Red Line platform, e.g. Alewife/Braintree) then had *zero* active
        # trips on the picked date, so `get_headway_at_stops` produced no
        # row for them at all -- not a null/bad score, an entirely missing
        # stop, since `download_and_prepare_stops`'s `at="parent_station"`
        # expansion below inner-joins raw stops onto the computed headway
        # table. Meanwhile the map's route lines (`transitlos.map.routes
        # .build_route_lines`) are calendar-agnostic (reads `shapes.txt`/
        # `routes.txt` directly, see that module's docstring), so the Red
        # Line still drew in full -- exactly the "routes show but stops
        # don't" symptom this was diagnosed from. This is a general risk
        # for ANY multi-feed metro load, not Boston-specific.
        #
        # Fix: bound the representative-date search to a near-term window
        # (today .. today + `_REPRESENTATIVE_DATE_HORIZON_DAYS`) whenever
        # that intersects the feed's actual calendar range, instead of
        # letting one feed's outlying-far-future calendar.txt coverage
        # drag the pick past another feed's real validity window. Falls
        # back to the unbounded search (old behavior) if the near-term
        # window doesn't overlap the feed's calendar at all (e.g. a
        # deliberate historical-replay analysis on old GTFS data), so nothing
        # regresses for that case.
        cal_lo = getattr(feed.calendar, "min_date", None)
        cal_hi = getattr(feed.calendar, "max_date", None)
        today = date.today()
        horizon_lo = today - timedelta(days=_REPRESENTATIVE_DATE_HORIZON_DAYS)
        horizon_hi = today + timedelta(days=_REPRESENTATIVE_DATE_HORIZON_DAYS)
        bounded_lo, bounded_hi = None, None
        if cal_lo is not None and cal_hi is not None:
            candidate_lo = max(cal_lo, horizon_lo)
            candidate_hi = min(cal_hi, horizon_hi)
            if candidate_lo <= candidate_hi:
                bounded_lo, bounded_hi = candidate_lo, candidate_hi

        analysis_date = feed.get_representative_date(
            start_date=bounded_lo, end_date=bounded_hi, date_type="weekday", route_types=route_types
        )
        if analysis_date is None:
            # Bounded window found nothing (feed has no near-term service --
            # e.g. historical data) -- fall back to the original unbounded
            # search across the feed's full calendar range.
            analysis_date = feed.get_representative_date(
                start_date=None, end_date=None, date_type="weekday", route_types=route_types
            )
        if analysis_date is None:
            raise ValueError(
                "No date_range given and the feed exposes no service-intensity "
                "dates to pick a representative day from; pass date_range explicitly."
            )
        analysis_date = datetime(analysis_date.year, analysis_date.month, analysis_date.day)

    headway_lf = feed.get_headway_at_stops(
        date=analysis_date, start_time=start_time, end_time=end_time,
        route_types=route_types, by=by, at=at, how=how,
        # Explicit, not relying on pyGTFSHandler's own default: combine all
        # routes serving a stop within each direction (via how="add"), then
        # keep whichever direction has the better combined headway, rather
        # than requiring good data in both directions -- see the `how`
        # docstring above for why this replaced "best".
        mix_directions=False,
    )
    headway_df = headway_lf.collect() if isinstance(headway_lf, pl.LazyFrame) else headway_lf
    headway_df = headway_df.rename({"headway": "headway_minutes"})
    headway_df = headway_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("headway_cv"))

    # `get_speed_at_stops` only recognizes how={"max","min","mean"} -- unlike
    # `get_headway_at_stops`, it has no "add"/"best"/"all" concept (summing
    # or "best-of" speeds the way headway rates combine doesn't make sense).
    # **Bug found and fixed 2026-08-12**: this used to be
    # `"max" if how == "best" else how`, which forwarded the headway `how`
    # value verbatim whenever it wasn't literally `"best"` -- since this
    # function's own default `how` changed to `"add"` (see that parameter's
    # docstring), that meant `get_speed_at_stops(how="add")` was being
    # called, which silently falls through every branch of that function's
    # internal `if/elif` chain, leaving its result **ungrouped** (many rows
    # per stop instead of one) and exploding the downstream
    # `headway_df.join(speed_df, ...)` into duplicate rows. Speed has no
    # sensible non-"max" default, so it is now always "max" regardless of
    # the headway `how`.
    speed_df = feed.get_speed_at_stops(
        date=analysis_date, start_time=start_time, end_time=end_time,
        route_types=route_types, by=by, at=at, how="max",
    )
    speed_df = speed_df.rename({"speed": "avg_speed_kmh"})

    join_cols = [c for c in (at, by) if c in headway_df.columns and c in speed_df.columns]
    stops_df = headway_df.join(
        speed_df.select(join_cols + ["avg_speed_kmh"]), on=join_cols, how="left"
    )

    if by == "route_id" and "route_id" in stops_df.columns:
        routes_lf = feed.routes.lf if hasattr(feed.routes, "lf") else feed.routes
        if routes_lf is not None:
            routes_df = routes_lf.collect() if isinstance(routes_lf, pl.LazyFrame) else routes_lf
            if "route_type" in routes_df.columns:
                route_type_map = routes_df.select("route_id", "route_type").unique(subset=["route_id"], keep="first")
                stops_df = stops_df.join(route_type_map, on="route_id", how="left")
            # Human-readable route label for display (map popups, figures): GTFS's own
            # `route_id` is often an opaque internal code, not what riders call the
            # route -- `route_short_name` (e.g. "65") is what's actually posted at the
            # stop, falling back to `route_long_name` for routes with no short name.
            name_cols = [c for c in ("route_short_name", "route_long_name") if c in routes_df.columns]
            if name_cols:
                route_name_map = routes_df.select("route_id", *name_cols).unique(subset=["route_id"], keep="first")
                stops_df = stops_df.join(route_name_map, on="route_id", how="left")
                label_expr = pl.col("route_short_name") if "route_short_name" in name_cols else pl.lit(None, dtype=pl.Utf8)
                if "route_long_name" in name_cols:
                    label_expr = label_expr.fill_null(pl.col("route_long_name")) if "route_short_name" in name_cols else pl.col("route_long_name")
                stops_df = stops_df.with_columns(
                    label_expr.fill_null(pl.col("route_id")).alias("route_label")
                ).drop([c for c in name_cols])

    own_mode_df = None
    per_stop_id_route_list = None
    _stop_key_map = None
    if at in stops_df.columns:
        # `route_label` above is only the single route this *row* represents (headway/
        # speed are computed per (stop, route) then `how="best"` collapses to one row per
        # physical stop) -- a stop actually served by several routes only shows its
        # busiest one there. `all_routes` instead lists *every* route ever calling at
        # that physical stop, built independently from raw `stop_times` -> `trips` ->
        # `routes`, grouped by the same physical-stop key (`parent_station`, falling back
        # to `stop_id`) as `at` uses, so it matches what a rider actually sees posted.
        def _collect(obj):
            lf = obj.lf if hasattr(obj, "lf") else obj
            return lf.collect() if isinstance(lf, pl.LazyFrame) else lf

        st_df = _collect(feed.stop_times)
        tr_df = _collect(feed.trips)
        stops_meta_df = _collect(feed.stops)
        routes_df = _collect(feed.routes)

        name_cols = [c for c in ("route_short_name", "route_long_name") if c in routes_df.columns]
        required = {"trip_id", "stop_id"}.issubset(st_df.columns) and {"trip_id", "route_id"}.issubset(tr_df.columns)
        if required and name_cols and "route_id" in routes_df.columns and "stop_id" in stops_meta_df.columns:
            label_expr = pl.col("route_short_name") if "route_short_name" in name_cols else pl.col("route_long_name")
            if "route_short_name" in name_cols and "route_long_name" in name_cols:
                label_expr = label_expr.fill_null(pl.col("route_long_name"))
            route_label_all = routes_df.select(
                "route_id", label_expr.fill_null(pl.col("route_id")).alias("route_label")
            ).unique(subset=["route_id"], keep="first")

            if at == "parent_station" and "parent_station" in stops_meta_df.columns:
                key_expr = pl.col("parent_station").fill_null(pl.col("stop_id"))
            else:
                key_expr = pl.col("stop_id")
            stop_key_map = stops_meta_df.select("stop_id", key_expr.alias("_stop_key"))
            _stop_key_map = stop_key_map

            per_stop_routes = (
                st_df.select("trip_id", "stop_id")
                .unique()
                .join(tr_df.select("trip_id", "route_id"), on="trip_id", how="inner")
                .join(route_label_all, on="route_id", how="left")
                .join(stop_key_map, on="stop_id", how="left")
                .drop_nulls(["_stop_key", "route_label"])
                .unique(subset=["_stop_key", "route_label"])
                .group_by("_stop_key")
                .agg(pl.col("route_label").sort().alias("_route_labels"))
                .with_columns(pl.col("_route_labels").list.join(", ").alias("all_routes"))
                .drop("_route_labels")
                .rename({"_stop_key": at})
            )
            stops_df = stops_df.join(per_stop_routes, on=at, how="left")

            # Per-raw-`stop_id` (not per-`at`-station) route list, distinct from
            # `all_routes` above whenever `at="parent_station"` (the default):
            # `all_routes` already answers "every route serving this physical
            # STATION" (grouped by `_stop_key`), which is of limited use for a
            # map popup that wants to show "routes at THIS platform/stop_id"
            # separately from "other routes reachable via the same parent
            # station" -- the classic multi-platform-station distinction (e.g.
            # a bus bay vs. the rail platform sharing one station). Same
            # stop_times -> trips -> routes expansion, just grouped by the raw
            # `stop_id` instead of the `_stop_key`/`at` station key. When
            # `at="stop_id"` this is identical to `all_routes` (each stop IS
            # its own group), so it's still computed and joined for
            # consistency, just redundant in that mode.
            per_stop_id_route_list = (
                st_df.select("trip_id", "stop_id")
                .unique()
                .join(tr_df.select("trip_id", "route_id"), on="trip_id", how="inner")
                .join(route_label_all, on="route_id", how="left")
                .drop_nulls(["stop_id", "route_label"])
                .unique(subset=["stop_id", "route_label"])
                .group_by("stop_id")
                .agg(pl.col("route_label").sort().alias("_route_labels"))
                .with_columns(pl.col("_route_labels").list.join(", ").alias("stop_routes"))
                .drop("_route_labels")
            )
            # Joined later (after the `at == "parent_station"` expansion below,
            # if any) since that expansion is keyed by raw `stop_id` too and
            # would otherwise need this column threaded through its own
            # `station_cols`/`keep_cols` bookkeeping -- simpler to attach once
            # at the very end, right before the GeoDataFrame is built.

        # Per-stop_id (not per-`at`-station) best mode, for map-icon purposes.
        # `route_type`/`route_id` on `stops_df` above are computed at the `at`
        # grouping (one value per parent_station when `at="parent_station"`),
        # which is correct for headway/speed but WRONG for the mode icon each
        # individual stop_id shows on the map: a bus-only platform and a
        # rail-only platform sharing one parent_station would otherwise both
        # inherit the station's single winning route. Build a genuine
        # per-stop_id route_type here (reusing the same
        # stop_times -> trips -> routes expansion as `all_routes` above, just
        # keyed by the raw `stop_id` instead of the `at` station key) and
        # apply rail > tram > bus priority per stop_id to pick its own
        # dominant mode.
        if required and "route_type" in routes_df.columns:
            per_stop_id_routes = (
                st_df.select("trip_id", "stop_id")
                .unique()
                .join(tr_df.select("trip_id", "route_id"), on="trip_id", how="inner")
                .join(routes_df.select("route_id", "route_type"), on="route_id", how="left")
                .drop_nulls(["stop_id", "route_type"])
                .unique(subset=["stop_id", "route_id"])
                .with_columns(
                    pl.when(pl.col("route_type").is_in([1, 2]))
                    .then(0)
                    .when(pl.col("route_type").is_in([0, 5, 12]))
                    .then(1)
                    .otherwise(2)
                    .alias("_priority")
                )
            )
            if per_stop_id_routes.height > 0:
                min_priority = per_stop_id_routes.group_by("stop_id").agg(
                    pl.col("_priority").min().alias("_min_priority")
                )
                own_mode_df = (
                    per_stop_id_routes.join(min_priority, on="stop_id")
                    .filter(pl.col("_priority") == pl.col("_min_priority"))
                    .unique(subset=["stop_id"], keep="first")
                    .select(
                        "stop_id",
                        pl.col("route_type").alias("own_route_type"),
                        pl.col("route_id").alias("own_route_id"),
                    )
                )

                # Bug found 2026-08-29 (user report, "El Arenal" station):
                # the STATION-WIDE headway/speed computed above
                # (`headway_df`/`speed_df`, grouped at `at="parent_station"`
                # with `how="add"`) harmonic-sums the headway of EVERY route
                # serving the station regardless of mode -- a rail platform
                # sharing a parent_station with several frequent bus routes
                # ends up reporting the *combined* bus+rail frequency (e.g.
                # 0.42 min) as its station-level headway, even though the
                # per-stop_id "own mode" fix above correctly labels that
                # platform "rail". A first attempt at this fix recomputed
                # headway/speed restricted to just the dominant mode's own
                # route_types via 1-2 EXTRA full `feed.get_headway_at_stops`/
                # `get_speed_at_stops` calls -- correct in principle, but
                # verified live (2026-08-29, real Concepcion GTFS/BusMaps
                # feed) to blow well past 22 GB and get OOM-killed, because
                # each extra call reprocesses the whole feed's
                # stop_times/trips joins again. Reverted: this pipeline runs
                # under a hard memory budget (see the standing 30 GB/2 GB
                # swap machine constraint), so re-deriving a genuinely
                # mode-restricted headway number is not affordable here.
                #
                # Cheaper, honest fix instead: compute (from data already
                # materialized above -- `per_stop_id_routes`, no new feed
                # queries) the SET of real mode classes actually present at
                # each station, as `station_modes` (e.g. "rail" for a
                # single-mode station, "bus + rail" for a mixed one, ordered
                # rail > tram > bus). The popup (see `transitlos.map.build`)
                # uses this to stop mislabeling a mixed-mode station's
                # combined headway/speed as belonging to a single mode --
                # instead of a false "mode: rail", a mixed station now
                # honestly reads "mode: bus + rail", so the combined number
                # is no longer attributed to a mode it doesn't solely
                # belong to.
                station_modes_df = None
                if _stop_key_map is not None:
                    _MODE_NAME_BY_PRIORITY = {0: "rail", 1: "tram", 2: "bus"}
                    station_routes = per_stop_id_routes.join(_stop_key_map, on="stop_id", how="left").drop_nulls(["_stop_key"])
                    # `map_elements` must run on a column whose values are
                    # already whole per-row lists (one full `_priorities`
                    # list per station) -- inside `.agg(...)` itself,
                    # `pl.col("_priority")` gets fed to the UDF one SCALAR
                    # at a time despite appearing list-like in the group-by
                    # context, so `ps` inside the old lambda was an `int`,
                    # not the group's Series (`TypeError: 'int' object is
                    # not iterable`, surfaced 2026-08-30 once the
                    # `parent_station`-empty-string ingestion fix below made
                    # this code path actually run on real multi-route
                    # stations for the first time -- it never had a chance
                    # to error before that fix, since every station used to
                    # collapse into one shared fake group).
                    station_modes_df = (
                        station_routes.select("_stop_key", "_priority").unique()
                        .group_by("_stop_key")
                        .agg(pl.col("_priority").unique().sort().alias("_priorities"))
                        .with_columns(
                            pl.col("_priorities")
                            .map_elements(
                                lambda ps: " + ".join(_MODE_NAME_BY_PRIORITY[p] for p in ps),
                                return_dtype=pl.Utf8,
                            )
                            .alias("station_modes")
                        )
                        .select(pl.col("_stop_key").alias(at), "station_modes")
                    )

                if station_modes_df is not None and station_modes_df.height > 0:
                    stops_df = stops_df.join(station_modes_df, on=at, how="left")

    if at == "parent_station":
        # Individual-marker requirement: `feed.add_stop_coords` (below, in the
        # `else` branch) collapses every row to ONE point per `at` group (a
        # `.mean().over("parent_station")` centroid -- see its implementation
        # in pyGTFSHandler), which would merge e.g. two opposite-platform
        # stops into a single marker. Each physical GTFS `stop_id` needs its
        # own marker/coordinates (task requirement), while *sharing* the
        # `station_key` group's headway/speed/route/score inputs computed
        # above -- so instead of collapsing, expand `stops_df` (one row per
        # `at`=`parent_station` group) back out to one row per raw `stop_id`,
        # left-joining each stop's own coordinates on. `parent_station` here
        # already reflects both real GTFS linkage *and* the `stop_group_distance`
        # proximity fallback (see `Feed`/`Stops.group_stops`, forwarded via
        # `**kwargs` above) if the caller passed `stop_group_distance > 0` --
        # so this also carries that 150m grouping through to individual points.
        raw_stops_lf = feed.stops.lf if hasattr(feed.stops, "lf") else feed.stops
        raw_stops_df = raw_stops_lf.collect() if isinstance(raw_stops_lf, pl.LazyFrame) else raw_stops_lf
        keep_cols = [c for c in ("stop_id", "stop_lat", "stop_lon", "stop_name", "parent_station") if c in raw_stops_df.columns]
        raw_stops_df = raw_stops_df.select(keep_cols).unique(subset=["stop_id"], keep="first")
        # Same "real parent_station, else fall back to the stop's own id"
        # key used to build `stops_df`'s `at` groups in the first place (see
        # the `all_routes` join above), so every raw stop matches the right group.
        join_key = pl.col("parent_station").fill_null(pl.col("stop_id")) if "parent_station" in raw_stops_df.columns else pl.col("stop_id")
        raw_stops_df = raw_stops_df.with_columns(join_key.alias(at))

        # Bug found 2026-08-29 (user report): the popup had no way to show
        # the parent station's own real name at all -- only this platform's
        # own `stop_name`. When the GTFS agency provides a genuine
        # hierarchical `parent_station` link, that id IS itself a `stop_id`
        # with its own `stops.txt` row (typically `location_type=1`,
        # e.g. "El Arenal" the station, distinct from "El Arenal
        # Andén 1"/"...Andén 2" the platforms) -- look that row up by its
        # `stop_id` and carry its `stop_name` through as `parent_station_name`.
        # Left null when there's no such row (e.g. the `stop_group_distance`
        # proximity-fallback path, whose synthetic id is a joined string of
        # member stop_ids with no `stops.txt` row of its own) -- the client
        # already degrades gracefully for that case.
        if "stop_name" in raw_stops_df.columns:
            station_name_map = raw_stops_df.select(
                pl.col("stop_id").alias(at), pl.col("stop_name").alias("parent_station_name")
            ).unique(subset=[at], keep="first")
            raw_stops_df = raw_stops_df.join(station_name_map, on=at, how="left")

        station_cols = [c for c in stops_df.columns if c not in ("stop_lat", "stop_lon", "stop_name")]
        expanded = raw_stops_df.join(stops_df.select(station_cols), on=at, how="inner")

        # Overwrite the station-wide `route_type`/`route_id` each row just
        # inherited from the `at`-level headway/speed aggregation (correct
        # for headway/speed, wrong for the mode icon -- see `own_mode_df`
        # above) with each raw stop_id's OWN dominant mode, so a bus-only
        # platform and a rail-only platform sharing one parent_station get
        # their own icons instead of both showing the station's single
        # winning route. Falls back to the station-wide value (via
        # coalesce) for any stop_id `own_mode_df` has no data for (e.g. a
        # stop with no stop_times rows at all).
        if own_mode_df is not None and "stop_id" in expanded.columns:
            expanded = expanded.join(own_mode_df, on="stop_id", how="left")
            if "route_type" in expanded.columns:
                expanded = expanded.with_columns(
                    pl.coalesce(["own_route_type", "route_type"]).alias("route_type")
                )
            if "route_id" in expanded.columns:
                expanded = expanded.with_columns(
                    pl.coalesce(["own_route_id", "route_id"]).alias("route_id")
                )
            expanded = expanded.drop([c for c in ("own_route_type", "own_route_id") if c in expanded.columns])

        stops_df = expanded

    if per_stop_id_route_list is not None and "stop_id" in stops_df.columns:
        stops_df = stops_df.join(per_stop_id_route_list, on="stop_id", how="left")

    stops_df = stops_df if "stop_lat" in stops_df.columns and "stop_lon" in stops_df.columns else feed.add_stop_coords(stops_df)
    stops_pd = stops_df.to_pandas() if isinstance(stops_df, pl.DataFrame) else stops_df

    geometry = gpd.points_from_xy(stops_pd["stop_lon"], stops_pd["stop_lat"])
    return gpd.GeoDataFrame(stops_pd, geometry=geometry, crs="EPSG:4326")


def download_and_prepare_stops_per_feed(
    gtfs_dirs: Sequence[Union[str, Path]],
    **kwargs,
) -> gpd.GeoDataFrame:
    """Score each GTFS directory in `gtfs_dirs` independently, then concatenate.

    `download_and_prepare_stops` normally stacks every directory into one
    `Feed` and picks a SINGLE representative date shared across all of them
    (see its docstring's "Bug found and fixed 2026-08-24" note) -- correct
    and cheaper for the common case where every feed's `calendar.txt`
    covers roughly the same real-world window. That assumption breaks when
    feeds are genuinely from different times: **real case found 2026-08-24,
    Taipei** -- its `tdx_bus` feed's calendar is valid only 2026-05-01 to
    2026-06-30 (a narrow real-time TDX export snapshot), while its
    `trtc_metro` feed's calendar spans 2025-08-18 to 2026-12-31. No single
    date can be representative of both feeds' real service at once: a
    near-term-bounded search barely overlaps `tdx_bus`'s narrow window
    (starves it down to a handful of stops), while an unbounded search can
    only get both feeds right by coincidence.

    This function sidesteps the problem entirely by never stacking feeds in
    the first place: each directory in `gtfs_dirs` gets its own `Feed`, its
    own representative-date pick (independently, via the normal
    `download_and_prepare_stops` call), and its own stop scoring -- then
    the resulting per-feed `GeoDataFrame`s are concatenated. Each feed's
    `parent_station` grouping (`stop_group_distance`, if passed via
    `**kwargs`) is computed WITHIN that feed only, not across feeds -- a
    bus stop and a metro entrance from two different feeds within 150m of
    each other will NOT be merged into one `parent_station` by this path
    (unlike the single-`Feed` path, where they would be). This is a
    deliberate, documented simplification, not an oversight: the point of
    this function is to fix the "no representative date works for every
    feed" problem, not to also solve cross-feed station deduplication --
    use `download_and_prepare_stops` (single stacked `Feed`) whenever every
    feed's calendar genuinely overlaps well enough for one shared date.

    Args:
        gtfs_dirs: One GTFS directory/zip path per feed -- each is scored
            completely independently (own `Feed`, own representative date).
        **kwargs: Forwarded unchanged to `download_and_prepare_stops` for
            every feed (`aoi`, `date_range`, `start_time`, `end_time`,
            `route_types`, `by`, `at`, `how`, `stop_group_distance`, ...).
            `date_range`, if a plain `(start, end)` tuple, is used as-is
            for every feed alike. If a `{feed_directory_name: (start,
            end)}` dict, only the named feed(s) get that explicit pin --
            every other feed still auto-detects its own date (matching by
            `Path(gtfs_dir).name`, e.g. `{"tdx_bus": (d, d)}` pins only
            the directory literally named `tdx_bus`). Leave `None` (the
            default, recommended) to let every feed auto-detect
            independently, searching each ONE feed's full real calendar
            range with no horizon bound -- see the "Not just unneeded,
            actively harmful" note above for why this deliberately does
            NOT reuse `download_and_prepare_stops`'s own horizon-bounded
            auto-detect. A real per-feed pin is still sometimes necessary
            even with unbounded per-feed search, though -- see
            `CityConfig`'s Taipei entry for a real feed (`tdx_bus`) whose
            median-service-based auto-detect can't reliably tell its
            genuinely dense weeks apart from sparse ones (most low-traffic
            stations have near-constant baseline service across its whole
            range, masking a real ~6-week span of much denser real data
            in the per-station MEDIAN this algorithm scores by).

    Returns:
        Concatenated `geopandas.GeoDataFrame`, same columns as
        `download_and_prepare_stops`'s single-feed return, one row set per
        feed stacked on top of the others (`ignore_index=True` -- no
        cross-feed index meaning to preserve).

    Raises:
        ValueError: If every feed in `gtfs_dirs` fails to produce stops
            (each feed's own `download_and_prepare_stops` failure is
            logged but not fatal -- one bad/empty feed among several real
            ones shouldn't abort the whole city, matching this codebase's
            "one bad workbook/level shouldn't abort everything" convention
            elsewhere -- see `pycensus.countries.germany.ba.loader`).
    """
    import warnings

    # `download_and_prepare_stops`'s own auto-detect (when `date_range` is
    # omitted) bounds its search to `_REPRESENTATIVE_DATE_HORIZON_DAYS` of
    # "today" -- a real protection against one STACKED feed's outlying
    # calendar dragging the shared pick away from another feed's real
    # window (see that function's docstring). That protection is not just
    # unneeded once a feed is scored in isolation, it actively HURTS a
    # feed whose own real calendar validity happens to sit further than
    # the horizon from "today" -- reproduced live 2026-08-24 on Taipei's
    # `tdx_bus` feed (calendar valid 2026-05-01 to 2026-06-30, i.e. its
    # OWN real window barely reaches the horizon-bounded search's lower
    # edge): scoring it in isolation with the horizon-bounded auto-detect
    # still only found 23 real stops, vs 51,864 across the whole city when
    # the old unbounded multi-feed search happened to land inside this
    # exact feed's window. So per-feed scoring here deliberately bypasses
    # `download_and_prepare_stops`'s own bounded auto-detect and instead
    # searches each isolated feed's FULL real calendar range directly --
    # safe here in a way it is not for a stacked multi-feed `Feed`, since
    # there is only one feed's calendar to consider, so nothing else can
    # get contaminated by searching its entire real range.
    explicit_date_range = kwargs.pop("date_range", None)
    aoi = kwargs.get("aoi")
    route_types = kwargs.get("route_types")

    per_feed_results = []
    for gtfs_dir in gtfs_dirs:
        try:
            # `explicit_date_range` may be a single `(start, end)` tuple
            # (applied to every feed alike, as documented above), OR a
            # `{feed_directory_name: (start, end)}` dict for pinning only
            # SOME feeds while leaving the rest on auto-detect -- real case
            # (Taipei, 2026-08-24): `tdx_bus` needed an explicit pin (its
            # own real calendar has a ~6-week span of genuinely dense
            # service, e.g. 2026-06-10, with ~80/day sparse baseline
            # coverage everywhere else in its range that the median-based
            # auto-detect can't reliably distinguish -- see this
            # function's module-level fix notes), while `trtc_metro`'s
            # service is stable day-to-day (241 real stops on every date
            # checked) and needs no pin at all.
            if isinstance(explicit_date_range, dict):
                feed_date_range = explicit_date_range.get(Path(gtfs_dir).name)
            else:
                feed_date_range = explicit_date_range
            if feed_date_range is None:
                probe_feed = Feed(gtfs_dirs=gtfs_dir, aoi=aoi, route_types=route_types)
                picked = probe_feed.get_representative_date(
                    start_date=None, end_date=None, date_type="weekday", route_types=route_types
                )
                if picked is None:
                    raise ValueError(
                        f"feed {gtfs_dir!r} exposes no service-intensity dates to pick a "
                        "representative day from"
                    )
                feed_date_range = (picked, picked)
            per_feed_results.append(
                download_and_prepare_stops(gtfs_dir, **{**kwargs, "date_range": feed_date_range})
            )
        except Exception as exc:
            warnings.warn(f"download_and_prepare_stops_per_feed: skipping feed {gtfs_dir!r}: {exc}")

    if not per_feed_results:
        raise ValueError(f"download_and_prepare_stops_per_feed: every feed in {list(gtfs_dirs)!r} failed")

    import pandas as pd

    combined = pd.concat(per_feed_results, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=per_feed_results[0].crs)


def _proximity_cluster_ids(lon, lat, distance_m: float = 150.0):
    """Union-find clustering of points within `distance_m` of each other.

    Uses a flat equirectangular projection (accurate enough at ~150m scale)
    plus `scipy.spatial.cKDTree.query_pairs` to find all point pairs within
    `distance_m`, then unions them. Returns a list of small-int cluster ids
    aligned to the input arrays (every point gets a cluster, including
    singletons with no neighbor).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    n = len(lon)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    lat_r = np.radians(np.asarray(lat, dtype=float))
    lon_r = np.radians(np.asarray(lon, dtype=float))
    earth_radius_m = 6371000.0
    x = earth_radius_m * lon_r * np.cos(lat_r)
    y = earth_radius_m * lat_r
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    for a, b in tree.query_pairs(r=distance_m):
        union(a, b)

    roots = [find(i) for i in range(n)]
    renumber = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    return [renumber[r] for r in roots]


def build_parent_station_keys(
    stops_meta_df: pl.DataFrame,
    distance_m: float = 150.0,
) -> pl.DataFrame:
    """Assign every GTFS stop a `station_key` grouping it with its physical station.

    Prefers the real GTFS `parent_station` field where present (a station's
    platforms/entrances are usually already linked there by the agency).
    Stops with no `parent_station` (null/missing, or the column absent
    entirely) are instead grouped by 150m spatial-proximity clustering
    (`_proximity_cluster_ids`) -- this is the fallback the module docstring
    and the stop-clustering requirement call for: real-world GTFS feeds
    frequently split one physical station into several unlinked `stop_id`s
    (opposite-direction platforms, separate entrances) with no
    `parent_station` set at all.

    Args:
        stops_meta_df: A polars DataFrame with at least `stop_id`, `stop_lat`,
            `stop_lon` columns (as on `feed.stops`), and optionally
            `parent_station`.
        distance_m: Proximity threshold in meters for the spatial fallback.
            Defaults to 150m per the clustering requirement.

    Returns:
        `stops_meta_df` with a `station_key` column added: `parent_station`
        where present, else `"cluster_<n>"` for stops grouped purely by
        proximity. Each individual `stop_id` keeps its own row/coordinates --
        this only adds the grouping key, it does not collapse rows.
    """
    df = stops_meta_df
    has_real_parent = "parent_station" in df.columns and df["parent_station"].is_not_null().any()

    if not has_real_parent:
        lon = df["stop_lon"].to_list()
        lat = df["stop_lat"].to_list()
        cluster_ids = _proximity_cluster_ids(lon, lat, distance_m=distance_m)
        return df.with_columns(
            pl.Series("station_key", [f"cluster_{c}" for c in cluster_ids])
        )

    # Real parent_station present for at least some stops: use it where set,
    # and spatially cluster only the remaining unlinked stops (falling back
    # to proximity within that subset, per-row spatial clusters get merged
    # with any parent_station group they land within distance_m of).
    unlinked_mask = df["parent_station"].is_null()
    unlinked = df.filter(unlinked_mask)
    linked = df.filter(~unlinked_mask)

    station_key = df["parent_station"].to_list()
    if len(unlinked) > 0:
        # Cluster unlinked stops against themselves AND against the centroid
        # of each real parent_station group, so an unlinked platform 20m from
        # a named station joins that station instead of forming its own.
        station_centroids = (
            linked.group_by("parent_station")
            .agg(pl.col("stop_lon").mean().alias("stop_lon"), pl.col("stop_lat").mean().alias("stop_lat"))
        )
        combined_lon = station_centroids["stop_lon"].to_list() + unlinked["stop_lon"].to_list()
        combined_lat = station_centroids["stop_lat"].to_list() + unlinked["stop_lat"].to_list()
        combined_ids = _proximity_cluster_ids(combined_lon, combined_lat, distance_m=distance_m)

        n_stations = len(station_centroids)
        station_ids = combined_ids[:n_stations]
        unlinked_ids = combined_ids[n_stations:]
        station_names = station_centroids["parent_station"].to_list()
        cluster_to_station = {cid: name for cid, name in zip(station_ids, station_names)}

        unlinked_keys = [
            cluster_to_station.get(cid, f"cluster_{cid}") for cid in unlinked_ids
        ]
        key_iter = iter(unlinked_keys)
        station_key = [
            (k if k is not None else next(key_iter)) for k in station_key
        ]

    return df.with_columns(pl.Series("station_key", station_key))
