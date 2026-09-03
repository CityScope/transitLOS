"""Verifies whether `download_and_prepare_stops`'s per-`stop_id` output rows
(at the default `at="parent_station"`) carry each raw stop's OWN serving
mode, or the whole parent station's aggregated mode.

This matters because `map/build.py`'s stop markers are supposed to show
"the mode icon of the BEST mode served by that SPECIFIC stop_id" per
`CS_transitLOS/code/pipeline.py`'s `stops["mode_category"] = stops["route_type"].apply(...)`,
computed directly off each row's `route_type` -- so the real question is
whether that per-row `route_type` genuinely differs between two stop_ids
in the same parent_station when they serve genuinely different modes.

Real GTFS/pyGTFSHandler isn't needed to answer this: `Feed` is monkeypatched
with a tiny fake exposing only the attributes `download_and_prepare_stops`
touches, with two synthetic stop_ids under one shared parent_station --
one served only by a bus route, the other only by a rail route.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from transitlos import stops as stops_mod


class _FakeFeed:
    """Minimal stand-in for `pyGTFSHandler.feed.Feed` covering only the
    attributes/methods `download_and_prepare_stops` actually calls.
    """

    def __init__(self, **kwargs):
        pass

    # -- analysis calls -----------------------------------------------
    def get_headway_at_stops(self, date, start_time, end_time, route_types, by, at, how, mix_directions):
        # One combined row per parent_station (this is what `how="add"`
        # collapsing to a single station-level frequency looks like in
        # practice) -- arbitrarily "wins" with the rail route_id, exactly
        # like a real combined-headway result would report *some* single
        # `route_id` for the group.
        return pl.DataFrame({"parent_station": ["PARENT1"], "route_id": ["ROUTE_RAIL"], "headway": [5.0]})

    def get_speed_at_stops(self, date, start_time, end_time, route_types, by, at, how):
        return pl.DataFrame({"parent_station": ["PARENT1"], "route_id": ["ROUTE_RAIL"], "speed": [40.0]})

    def add_stop_coords(self, df):
        return df

    def get_representative_date(self, **kwargs):
        return None

    # -- raw GTFS tables ------------------------------------------------
    @property
    def stop_times(self):
        return pl.DataFrame({
            "trip_id": ["T_BUS", "T_RAIL"],
            "stop_id": ["STOP_BUS", "STOP_RAIL"],
        })

    @property
    def trips(self):
        return pl.DataFrame({
            "trip_id": ["T_BUS", "T_RAIL"],
            "route_id": ["ROUTE_BUS", "ROUTE_RAIL"],
        })

    @property
    def stops(self):
        return pl.DataFrame({
            "stop_id": ["STOP_BUS", "STOP_RAIL"],
            "stop_lat": [42.0, 42.001],
            "stop_lon": [-71.0, -71.001],
            "stop_name": ["Bus Platform", "Rail Platform"],
            "parent_station": ["PARENT1", "PARENT1"],
        })

    @property
    def routes(self):
        return pl.DataFrame({
            "route_id": ["ROUTE_BUS", "ROUTE_RAIL"],
            "route_type": [3, 2],  # 3 = bus, 2 = rail (transitlos.scoring.mode._RAIL_ROUTE_TYPES)
            "route_short_name": ["B1", "R1"],
        })


def test_per_stop_route_type_is_genuinely_per_stop_id(monkeypatch):
    """Verifies the fixed behavior: each raw stop_id under a shared
    parent_station carries its OWN dominant route_type/route_id (derived
    from a genuine per-stop_id stop_times -> trips -> routes expansion with
    rail > tram > bus priority), not the whole station's aggregated
    headway-winning route.

    STOP_BUS is served only by a bus route (route_type 3) and STOP_RAIL only
    by a rail route (route_type 2), even though both share parent_station
    PARENT1 and PARENT1's combined headway/speed aggregation arbitrarily
    "won" with the rail route_id. The per-stop mode must differ accordingly,
    while the shared headway (aggregated at the parent_station level) must
    still be identical for both rows -- that aggregation is correct and
    must not change.
    """
    monkeypatch.setattr(stops_mod, "Feed", _FakeFeed)

    result = stops_mod.download_and_prepare_stops(
        "fake-gtfs-dir",
        date_range=(datetime(2026, 1, 5), datetime(2026, 1, 5)),
    )

    by_stop = {
        row["stop_id"]: row
        for row in result[["stop_id", "route_type", "route_id", "headway_minutes"]].to_dict("records")
    }
    assert set(by_stop) == {"STOP_BUS", "STOP_RAIL"}

    bus_row = by_stop["STOP_BUS"]
    rail_row = by_stop["STOP_RAIL"]

    # The crux of the fix: each stop_id's own route_type/route_id, not the
    # station-wide winner from headway aggregation.
    assert bus_row["route_type"] == 3, f"expected STOP_BUS route_type=3 (bus), got {bus_row['route_type']!r}"
    assert bus_row["route_id"] == "ROUTE_BUS"
    assert rail_row["route_type"] == 2, f"expected STOP_RAIL route_type=2 (rail), got {rail_row['route_type']!r}"
    assert rail_row["route_id"] == "ROUTE_RAIL"

    # Headway/speed remain correctly aggregated at the parent_station level
    # and must still be shared identically by both stop_ids -- only the
    # mode/route columns should have become per-stop_id.
    assert bus_row["headway_minutes"] == rail_row["headway_minutes"] == 5.0
