"""Idealised checks of the merged-departure headway and of the best-combination search against hand-computed / brute-force answers."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pyGTFSHandler")
from test_direction_subset import DAY, END, START, _deps, _modes, _run, departures_with_direction  # noqa: E402

from transitlos.scoring.direction_subset import _scores, best_stop_scores_by_direction, direction_headway_minutes  # noqa: E402

S, E = 6 * 3600.0, 22 * 3600.0     # the scoring window
T = E - S
WS, WE = 6 * 3600.0, 10 * 3600.0     # the window the Feed tests score (test_direction_subset START / END)


def _regular(h_min, start=S, end=E):
    """Departures every h minutes, the first half a headway after the window start (the symmetric, ideal placement)."""
    h = h_min * 60.0
    return np.arange(start + h / 2, end, h)


# ---------- the headway formula itself ----------
@pytest.mark.parametrize("h", [5, 7.5, 10, 12, 15, 20, 30, 60])
def test_regular_service_has_a_headway_equal_to_its_spacing(h):
    t = _regular(h)
    assert direction_headway_minutes(t, S, E) == pytest.approx(h, rel=1e-9)


def test_one_departure_in_the_window_is_a_headway_of_the_whole_window_over_four_halves():
    # one departure in the middle: the two edge gaps are added into one wrap-around gap of the whole window
    assert direction_headway_minutes(np.array([14 * 3600.0]), S, E) == pytest.approx(T / 60.0)


@pytest.mark.parametrize("h", [10, 15, 30])
def test_exact_duplicates_of_any_multiplicity_change_nothing(h):
    t = _regular(h)
    base = direction_headway_minutes(t, S, E)
    for k in (2, 3, 7):
        assert direction_headway_minutes(np.sort(np.repeat(t, k)), S, E) == pytest.approx(base, rel=1e-12)


def test_zero_five_zero_five_is_five_five():
    assert direction_headway_minutes(np.sort(np.repeat(_regular(5), 2)), S, E) == pytest.approx(5.0)


def test_interleaving_two_regular_services_halves_the_headway_and_stacking_them_does_not():
    a = _regular(10)
    shifted = a + 300.0                                                  # half a headway later
    assert direction_headway_minutes(np.sort(np.r_[a, shifted]), S, E) == pytest.approx(5.0, rel=0.02)
    assert direction_headway_minutes(np.sort(np.r_[a, a]), S, E) == pytest.approx(10.0)


def test_adding_departures_never_makes_the_headway_longer_and_order_does_not_matter():
    rng = np.random.default_rng(7)
    for _ in range(200):
        base = np.sort(rng.uniform(S, E, rng.integers(1, 40)))
        extra = np.sort(np.r_[base, rng.uniform(S, E, rng.integers(1, 10))])
        assert direction_headway_minutes(extra, S, E) <= direction_headway_minutes(base, S, E) + 1e-9
        assert direction_headway_minutes(np.sort(rng.permutation(base)), S, E) == pytest.approx(direction_headway_minutes(base, S, E))


# ---------- through a real Feed ----------
@pytest.mark.parametrize("h", [10, 15, 30])
def test_a_regular_route_in_a_feed_scores_its_spacing(tmp_path, h):
    # S1 is reached 2 minutes after the first stop; departures are placed half a headway after the window start at S1
    first = int(S + h * 30 - 120)
    out, _ = _run(tmp_path, [("R", 3, 2, _deps(first, h * 60, int((E - S) // (h * 60)) + 1), 0)])
    assert out.set_index("parent_station").loc["S1", "headway_minutes"] == pytest.approx(h, rel=0.02)


def test_the_same_timetable_under_one_two_or_five_route_ids_gives_the_same_score(tmp_path):
    dep = _deps(6 * 3600, 600, 24)
    scores = []
    for k in (1, 2, 5):
        routes = [(f"R{i}", 3, 2, dep, 0) for i in range(k)]
        out, _ = _run(tmp_path / str(k), routes)
        scores.append(out.set_index("parent_station").loc["S1", ["headway_minutes", "stop_score"]].to_list())
    assert scores[1] == pytest.approx(scores[0]) and scores[2] == pytest.approx(scores[0])


def test_more_frequent_service_never_scores_lower(tmp_path):
    prev = -1.0
    for h in (60, 30, 20, 10, 5):
        out, _ = _run(tmp_path / str(h), [("R", 3, 2, _deps(6 * 3600, h * 60, int(4 * 3600 // (h * 60)) + 1), 0)])
        sc = out.set_index("parent_station").loc["S1", "stop_score"]
        assert sc >= prev - 1e-9
        prev = sc


# ---------- the search against a brute-force oracle ----------
def _oracle(feed, routes, station="S1"):
    """Best score over every mode tier and every minimum speed in 5 km/h steps, recomputed from the raw departures."""
    d = departures_with_direction(feed, DAY, START, END, _modes(routes)).filter(pl.col("parent_station") == station).to_dict(as_series=False)
    t, sec, rank, spd = (np.array(d[k], dtype=float) for k in ("departure_time", "sector", "rank", "speed_kmh"))
    dist, tm = np.array(d["dist_m"], dtype=float), np.array(d["time_s"], dtype=float)
    best = (-1.0, None, None)
    for name, min_rank, rtype in (("bus", 0, 3), ("tram", 1, 0), ("rail", 2, 2)):
        if not (rank >= min_rank).any():
            continue
        top = np.nanmax(spd[rank >= min_rank])
        for c in np.arange(0, math.floor(top / 5) * 5 + 1, 5.0):
            keep = (rank >= min_rank) & (spd >= c)
            if not keep.any():
                continue
            hw = min(direction_headway_minutes(np.sort(t[keep & (sec == s)]), WS, WE) for s in (0, 1))
            if not math.isfinite(hw):
                continue
            speed = dist[keep].sum() / 1000 / (tm[keep].sum() / 3600)
            score = _scores(round(hw, 3), round(speed, 2), rtype, "global")[3]
            if score > best[0] + 1e-12:
                best = (score, name, c)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_the_search_matches_a_brute_force_enumeration(tmp_path, seed):
    rng = np.random.default_rng(seed)
    routes = []
    for i in range(int(rng.integers(2, 5))):
        rtype = int(rng.choice([3, 3, 0, 2]))
        mins = float(rng.choice([1, 2, 3, 5, 8]))                       # minutes between stops -> trip speed
        every = int(rng.choice([300, 600, 900, 1200, 1800]))
        first = int(6 * 3600 + rng.integers(0, 900))
        routes.append((f"R{i}", rtype, mins, _deps(first, every, int(4 * 3600 // every)), int(rng.integers(0, 2))))
    out, feed = _run(tmp_path, routes)
    row = out.set_index("parent_station").loc["S1"]
    score, name, cutoff = _oracle(feed, routes)
    assert row["stop_score"] == pytest.approx(score, abs=1e-6)


def test_dropping_a_slow_frequent_line_can_raise_the_score(tmp_path):
    # a fast line every 15 min (~50 km/h) plus a very slow one every 15 min in between (~8 km/h): the slow line halves the headway but
    # drags the mean speed down. Whatever wins, the search must never do worse than either extreme and must report its cutoff.
    routes = [("FAST", 2, 1, _deps(6 * 3600, 900, 16), 0), ("SLOW", 3, 6, _deps(6 * 3600 + 450, 900, 16), 0)]
    out, feed = _run(tmp_path, routes)
    row = out.set_index("parent_station").loc["S1"]
    score, name, cutoff = _oracle(feed, routes)
    assert row["stop_score"] == pytest.approx(score, abs=1e-6) and row["scoring_mode"] == name and row["min_speed_cutoff_kmh"] == cutoff
