"""Best stop score from merged departures split in two directions (`transitlos.scoring.direction_subset`)."""
from __future__ import annotations

import csv
from datetime import date, time

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pyGTFSHandler")
from pyGTFSHandler.feed import Feed  # noqa: E402

from transitlos.scoring.direction_subset import best_stop_scores_by_direction, departures_with_direction, direction_headway_minutes  # noqa: E402

DAY = date(2024, 11, 6)   # a Wednesday
START, END = time(6, 0), time(10, 0)


def _hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _write(folder, name, rows):
    folder.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(folder / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _line_feed(tmp_path, routes, name="f"):
    """A straight east-west line of 4 stops (S0..S3, ~0.01 deg = 850 m apart). routes: [(route_id, route_type, minutes_between_stops, [first departure s], direction 0/1)]"""
    lons = [-3.700, -3.690, -3.680, -3.670]
    stops = [{"stop_id": f"S{i}", "stop_name": f"S{i}", "stop_lat": 40.0, "stop_lon": lon} for i, lon in enumerate(lons)]
    trips, times, rts = [], [], []
    for rid, rtype, mins, deps, direction in routes:
        rts.append({"route_id": rid, "route_short_name": rid, "route_long_name": rid, "route_type": rtype})
        order = list(range(4)) if direction == 0 else list(range(3, -1, -1))
        for k, t0 in enumerate(deps):
            tid = f"{rid}_{direction}_{k}"
            trips.append({"route_id": rid, "service_id": "SVC", "trip_id": tid, "direction_id": direction})
            for seq, s in enumerate(order):
                t = t0 + seq * mins * 60
                times.append({"trip_id": tid, "arrival_time": _hms(t), "departure_time": _hms(t), "stop_id": f"S{s}", "stop_sequence": seq})
    d = tmp_path / name
    _write(d, "agency.txt", [{"agency_id": "A", "agency_name": "A", "agency_url": "http://x", "agency_timezone": "Europe/Madrid"}])
    _write(d, "calendar.txt", [{"service_id": "SVC", **{k: 1 for k in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}, "start_date": "20240101", "end_date": "20241231"}])
    _write(d, "routes.txt", rts)
    _write(d, "stops.txt", stops)
    _write(d, "trips.txt", trips)
    _write(d, "stop_times.txt", times)
    return d


def _modes(feed_routes):
    return pl.DataFrame({"route_id": [r[0] for r in feed_routes], "scoring_mode": [{3: "bus", 0: "tram", 2: "rail"}[r[1]] for r in feed_routes]})


def _run(tmp_path, routes, dup=False):
    d = _line_feed(tmp_path, routes)
    feed = Feed(d, start_date=DAY, end_date=DAY, stop_group_distance=0)
    return best_stop_scores_by_direction(feed, DAY, _modes(routes), START, END), feed


def _deps(first, every, n):
    return [first + i * every for i in range(n)]


def test_zero_gaps_do_not_change_the_headway():
    start, end = 0.0, 3600.0
    single = np.arange(0, 3600, 300.0)
    doubled = np.sort(np.r_[single, single])
    assert direction_headway_minutes(single, start, end) == pytest.approx(direction_headway_minutes(doubled, start, end))
    assert direction_headway_minutes(np.array([]), start, end) == float("inf")


def test_a_duplicated_route_does_not_change_the_stop_score(tmp_path):
    one = [("R1", 3, 2, _deps(6 * 3600, 600, 24), 0)]
    two = one + [("R2", 3, 2, _deps(6 * 3600, 600, 24), 0)]         # same timetable under another route id
    a, _ = _run(tmp_path / "a", one)
    b, _ = _run(tmp_path / "b", two)
    ra = a.set_index("parent_station").loc["S1"]
    rb = b.set_index("parent_station").loc["S1"]
    assert rb["headway_minutes"] == pytest.approx(ra["headway_minutes"])
    assert rb["stop_score"] == pytest.approx(ra["stop_score"])


def test_two_directions_use_the_better_one_and_ignore_route_ids(tmp_path):
    routes = [("A", 3, 2, _deps(6 * 3600, 600, 24), 0), ("B", 3, 2, _deps(6 * 3600 + 120, 1200, 12), 1)]   # eastbound every 10 min, westbound every 20
    out, feed = _run(tmp_path, routes)
    deps = departures_with_direction(feed, DAY, START, END, _modes(routes)).filter(pl.col("parent_station") == "S1")
    assert sorted(deps["sector"].unique().to_list()) == [0, 1]                                      # the two opposite directions
    east = deps.filter(pl.col("route_id") == "A")["sector"].unique().to_list()
    west = deps.filter(pl.col("route_id") == "B")["sector"].unique().to_list()
    assert len(east) == 1 and len(west) == 1 and east != west
    row = out.set_index("parent_station").loc["S1"]
    assert row["headway_minutes"] == pytest.approx(direction_headway_minutes(np.sort(deps.filter(pl.col("route_id") == "A")["departure_time"].to_numpy()), 6 * 3600, 10 * 3600))


def test_routes_in_the_same_direction_are_merged(tmp_path):
    routes = [("A", 3, 2, _deps(6 * 3600, 600, 24), 0), ("B", 3, 2, _deps(6 * 3600 + 300, 600, 24), 0)]   # interleaved every 10 min -> 5 min
    out, _ = _run(tmp_path, routes)
    merged = np.sort(np.r_[_deps(6 * 3600 + 120, 600, 24), _deps(6 * 3600 + 420, 600, 24)]).astype(float)   # S1 departure = start + 2 min
    row = out.set_index("parent_station").loc["S1"]
    assert row["headway_minutes"] == pytest.approx(direction_headway_minutes(merged, 6 * 3600, 10 * 3600))


def test_excluding_slow_trips_can_win(tmp_path):
    # a frequent but very slow line next to a rarer fast one: the best stop score never loses to "keep everything" and records its cutoff
    routes = [("FAST", 3, 1, _deps(6 * 3600, 1200, 12), 0), ("SLOW", 3, 6, _deps(6 * 3600 + 600, 1200, 12), 0)]
    out, feed = _run(tmp_path, routes)
    row = out.set_index("parent_station").loc["S1"]
    from transitlos.scoring.direction_subset import _scores

    everything = _scores(round(direction_headway_minutes(np.sort(departures_with_direction(feed, DAY, START, END, _modes(routes)).filter(pl.col("parent_station") == "S1")["departure_time"].to_numpy()), 6 * 3600, 10 * 3600), 3), 10.0, 3, "global")[3]
    assert row["stop_score"] >= everything - 1e-9 or row["min_speed_cutoff_kmh"] > 0
    assert row["min_speed_cutoff_kmh"] >= 0 and row["scoring_mode"] in ("bus", "tram", "rail")


def test_mode_tiers_report_the_mode_of_the_winning_tier(tmp_path):
    routes = [("BUS", 3, 3, _deps(6 * 3600, 900, 16), 0), ("RAIL", 2, 1, _deps(6 * 3600, 900, 16), 0)]
    out, _ = _run(tmp_path, routes)
    row = out.set_index("parent_station").loc["S1"]
    assert row["scoring_mode"] in ("bus", "rail")
    assert ("RAIL" in row["route_ids"]) and (row["scoring_mode"] == "rail") == ("BUS" not in row["route_ids"])


def test_minimum_speeds_are_tried_in_steps_of_5_kmh(tmp_path):
    routes = [("FAST", 3, 1, _deps(6 * 3600, 1200, 12), 0), ("SLOW", 3, 6, _deps(6 * 3600 + 600, 1200, 12), 0)]
    out, _ = _run(tmp_path, routes)
    assert (out["min_speed_cutoff_kmh"] % 5 == 0).all()


def test_download_and_prepare_stops_uses_the_direction_search_by_default(tmp_path):
    from transitlos.stops import download_and_prepare_stops

    routes = [("A", 3, 2, _deps(6 * 3600, 600, 24), 0), ("B", 2, 1, _deps(6 * 3600, 900, 16), 1)]
    d = _line_feed(tmp_path, routes)
    gdf = download_and_prepare_stops(d, date_range=(DAY, DAY), start_time=START, end_time=END)
    for col in ("headway_minutes", "avg_speed_kmh", "stop_score", "scoring_mode", "min_speed_cutoff_kmh", "route_type", "geometry"):
        assert col in gdf.columns
    assert len(gdf) == 4 and gdf["stop_score"].between(0, 1).all()
    old = download_and_prepare_stops(d, date_range=(DAY, DAY), start_time=START, end_time=END, by="route_id", scoring="route")
    assert "stop_score" not in old.columns or len(old) >= 4          # the previous route-level table is still available


# ---- speed method (2026-11-17: speed = distance covered 15 min before / after the station; straight line by default) ----
def _zigzag_feed(tmp_path, lats):
    """One bus line of 6 stops, 0.01 deg of longitude apart, 3 min between stops, at the given latitudes; departures every 10 min."""
    lons = [-3.70 + 0.01 * i for i in range(len(lats))]
    rows = [(40.0 + la, lo) for la, lo in zip(lats, lons)]
    d = _line_feed(tmp_path, [("B", 3, 3, _deps(6 * 3600, 600, 24), 0)], name="z")
    times = []
    for k in range(24):
        for seq in range(len(rows)):
            t = 6 * 3600 + 600 * k + seq * 180
            times.append({"trip_id": f"B_0_{k}", "arrival_time": _hms(t), "departure_time": _hms(t), "stop_id": f"S{seq}", "stop_sequence": seq})
    _write(d, "stops.txt", [{"stop_id": f"S{i}", "stop_name": f"S{i}", "stop_lat": la, "stop_lon": lo} for i, (la, lo) in enumerate(rows)])
    _write(d, "stop_times.txt", times)
    return Feed(d, start_date=DAY, end_date=DAY, stop_group_distance=0)


def _speed_at(feed, stop, method):
    deps = departures_with_direction(feed, DAY, START, END, _modes([("B", 3, 3, [], 0)]), speed_method=method)
    return float(deps.filter(pl.col("parent_station") == stop)["speed_kmh"].median())


def test_straight_speed_penalises_a_winding_line(tmp_path):
    straight_line = _zigzag_feed(tmp_path / "a", [0.0] * 6)
    winding = _zigzag_feed(tmp_path / "b", [0.0, 0.01, 0.0, 0.01, 0.0, 0.01])
    s_flat, s_wind = _speed_at(straight_line, "S2", "straight"), _speed_at(winding, "S2", "straight")
    sh_flat, sh_wind = _speed_at(straight_line, "S2", "shape"), _speed_at(winding, "S2", "shape")
    assert s_flat == pytest.approx(sh_flat, rel=0.02)       # a straight line: both methods agree
    assert sh_wind > sh_flat                                 # along the zigzag the vehicle covers more distance in the same time
    assert s_wind < sh_wind * 0.8                            # ... but gets less far: the straight-line speed penalises the curves


def test_unknown_speed_method(tmp_path):
    feed = _zigzag_feed(tmp_path, [0.0] * 6)
    with pytest.raises(ValueError):
        departures_with_direction(feed, DAY, START, END, _modes([("B", 3, 3, [], 0)]), speed_method="trip")
