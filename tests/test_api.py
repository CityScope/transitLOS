"""Tests for the transitlos orchestration layer (stops/stop_scores/network/
level_of_service/population/equity), without hitting any network or downloading
real GTFS/OSM/census data.

Coverage:
    - `compute_stop_scores` and `equity_statistics` are exercised end-to-end
      against small synthetic pandas DataFrames (pure computation, no I/O).
    - `network.py`, `level_of_service.py`, `population.py`, `stops.py` are
      checked for clean import and correct public-function signatures via
      `inspect.signature`, without invoking anything network-dependent.

No true offline end-to-end integration test (real `StreetNetwork` +
`PointsOfInterest` + `AccessibilityAnalyzer` run) is included: UrbanAccessAnalyzer's
own `tests/` directory has no small pre-built network/POI fixture -- its
`sample.osm.pbf` fixture would still require the full `from_pbf` -> `simplify`
-> `crop` -> `snap_points` -> `run` pipeline to run for real, which is out of
scope for a fast unit-test file.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from transitlos import level_of_service, network, population, stops
from transitlos.equity import equity_statistics
from transitlos.stop_scores import compute_stop_scores


# ---------------------------------------------------------------------------
# compute_stop_scores
# ---------------------------------------------------------------------------


def test_compute_stop_scores_adds_expected_columns():
    df = pd.DataFrame(
        {
            "stop_id": ["a", "b", "c"],
            "headway_minutes": [5.0, 30.0, np.nan],
            "headway_cv": [0.1, np.nan, 0.2],
            "avg_speed_kmh": [20.0, 10.0, 15.0],
        }
    )
    out = compute_stop_scores(df, region="global")

    for col in ("mode_score", "speed_score", "frequency_score", "reliability_score", "stop_score"):
        assert col in out.columns

    # Row 0: fully specified, everything should be a finite score in [0, 1].
    row0 = out.iloc[0]
    assert 0.0 <= row0["stop_score"] <= 1.0
    assert not row0["reliability_score_is_assumed"]

    # Row 1: headway_cv was null -> default_headway_cv substituted, and the
    # frame should say so honestly via reliability_score_is_assumed.
    row1 = out.iloc[1]
    assert row1["reliability_score_is_assumed"]
    assert 0.0 <= row1["stop_score"] <= 1.0

    # Row 2: missing headway_minutes -> every score is NaN, not fabricated.
    row2 = out.iloc[2]
    assert np.isnan(row2["stop_score"])

    # A more frequent, faster, more reliable stop (row 0) should score
    # higher than an infrequent, slower one (row 1).
    assert row0["stop_score"] > row1["stop_score"]


def test_compute_stop_scores_respects_custom_weights():
    df = pd.DataFrame(
        {"headway_minutes": [5.0], "headway_cv": [0.1], "avg_speed_kmh": [20.0]}
    )
    out_equal = compute_stop_scores(df, weights=(0.25, 0.25, 0.25, 0.25))
    out_speed_heavy = compute_stop_scores(df, weights=(0.05, 0.85, 0.05, 0.05))
    # Different weights on identical sub-scores should generally change the
    # combined stop_score (unless all four sub-scores are equal).
    assert out_equal["stop_score"].iloc[0] != out_speed_heavy["stop_score"].iloc[0] or (
        out_equal["mode_score"].iloc[0]
        == out_equal["speed_score"].iloc[0]
        == out_equal["frequency_score"].iloc[0]
        == out_equal["reliability_score"].iloc[0]
    )


# ---------------------------------------------------------------------------
# equity_statistics
# ---------------------------------------------------------------------------


def test_equity_statistics_overall_stats_are_sane():
    df = pd.DataFrame(
        {
            "level_of_service": [0.0, 0.25, 0.5, 0.75, 1.0],
            "population": [100, 100, 100, 100, 100],
        }
    )
    result = equity_statistics(df)
    overall = result["overall"]

    assert overall["weighted_mean"] == pytest.approx(0.5)
    assert overall["weighted_median"] == pytest.approx(0.5)
    # Equal population weights and a symmetric spread of scores.
    assert 0.0 <= overall["gini"] <= 1.0
    assert overall["pct_population_above_threshold"][0.5] == pytest.approx(0.6)
    assert overall["pct_population_below_threshold"][0.5] == pytest.approx(0.4)


def test_equity_statistics_perfect_equality_has_zero_gini():
    df = pd.DataFrame({"level_of_service": [0.6, 0.6, 0.6], "population": [10, 20, 30]})
    result = equity_statistics(df)
    assert result["overall"]["gini"] == pytest.approx(0.0, abs=1e-9)


def test_equity_statistics_by_group():
    df = pd.DataFrame(
        {
            "level_of_service": [0.2, 0.8, 0.3, 0.9],
            "population": [10, 10, 20, 20],
            "region": ["north", "north", "south", "south"],
        }
    )
    result = equity_statistics(df, groups="region")
    by_group = result["by_group"]
    assert set(by_group.index) == {"north", "south"}
    assert by_group.loc["north", "weighted_mean"] == pytest.approx(0.5)
    assert by_group.loc["south", "weighted_mean"] == pytest.approx(0.6)


def test_equity_statistics_all_nan_scores_returns_nan_not_crash():
    df = pd.DataFrame({"level_of_service": [np.nan, np.nan], "population": [10, 10]})
    overall = equity_statistics(df)["overall"]
    assert np.isnan(overall["weighted_mean"])
    assert np.isnan(overall["gini"])


# ---------------------------------------------------------------------------
# Import + signature checks for the network-dependent modules
# ---------------------------------------------------------------------------


def test_stops_download_and_prepare_stops_signature():
    sig = inspect.signature(stops.download_and_prepare_stops)
    params = sig.parameters
    assert "feed_source" in params
    assert "aoi" in params
    assert "date_range" in params
    assert "route_types" in params
    assert params["date_range"].default is None


def test_stops_download_and_prepare_stops_headway_default_combines_routes():
    # Regression guard (2026-08-12): `how="best"` groups headway per
    # (stop, route, direction) and keeps only the single most-frequent one,
    # which can never reflect the combined frequency riders experience at a
    # stop served by multiple routes/branches -- confirmed against the real
    # MBTA Green Line at Boylston (49.68 min reported vs. ~4-6 min real
    # combined frequency). `how="add"` (harmonic-sum combination across
    # routes, see pyGTFSHandler's `get_headway_at_stops` docstring) is the
    # fix; `mix_directions=False` keeps whichever direction has the better
    # combined headway rather than requiring both directions' data to be
    # good. If this default silently reverts to "best", every multi-route
    # stop in every downstream city study goes back to reporting an
    # implausibly sparse headway (it was ~89% of stops with a null
    # stop_score on a real Boston run) with no visible error -- this test
    # exists specifically so that regression fails loudly instead.
    sig = inspect.signature(stops.download_and_prepare_stops)
    params = sig.parameters
    assert params["how"].default == "add"


def test_network_prepare_street_network_signature():
    sig = inspect.signature(network.prepare_street_network)
    params = sig.parameters
    assert list(params)[0] == "aoi"
    assert "cache_dir" in params
    assert "cluster_distance" in params
    assert params["cluster_distance"].default == 10.0


def test_level_of_service_compute_level_of_service_signature():
    sig = inspect.signature(level_of_service.compute_level_of_service)
    params = sig.parameters
    for name in ("aoi", "feed_source", "network", "stops", "region", "cache_dir", "walk_distance_steps"):
        assert name in params
    assert params["walk_distance_steps"].default == (400, 800, 1200)


def test_population_cross_with_population_signature():
    sig = inspect.signature(population.cross_with_population)
    params = sig.parameters
    assert params["population_source"].default == "worldpop"
    for name in ("access_gdf", "aoi", "region_for_census", "id_col"):
        assert name in params
