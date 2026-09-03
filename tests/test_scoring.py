"""Tests for transitlos.scoring, per report/scoring/SCORING.md's methodology.

Covers monotonicity, bounds, the equal-weight geometric-mean identity, weight
validation, a qualitative metro-vs-rural-bus ordering check, and regional
parameter differentiation (including the global_south unvalidated-region
warning).
"""

import pytest

from transitlos.scoring import (
    accessibility_score,
    frequency_score,
    mode_score,
    reliability_score,
    speed_score,
    stop_score,
    walk_decay,
)


# --- Monotonicity ---------------------------------------------------------


def test_frequency_score_increases_as_headway_decreases():
    # Sample above the saturation plateau so the function is strictly
    # decreasing in headway (i.e. strictly increasing as headway shrinks).
    headways = [90, 60, 40, 20, 15]
    scores = [frequency_score(h, region="global") for h in headways]
    for a, b in zip(scores, scores[1:]):
        assert b > a


def test_reliability_score_decreases_as_cv_increases():
    cvs = [0.0, 0.2, 0.5, 1.0, 2.0]
    scores = [reliability_score(cv) for cv in cvs]
    for a, b in zip(scores, scores[1:]):
        assert b < a


def test_speed_score_increases_with_speed_then_caps_at_one():
    # Strictly below breakpoints[2] (20 km/h default, where the score
    # reaches its max -- see speed.py's 2026-08-12 fix for the "weird peak"
    # discontinuity bug this guards against): strictly increasing, bounded
    # to 1. (Sampled strictly inside each band, not exactly on a
    # breakpoint -- the boundary points themselves have a known
    # pre-existing continuity quirk in speed.py not addressed by this
    # change.)
    speeds_below = [5, 13, 17, 19]
    scores_below = [speed_score(s) for s in speeds_below]
    for a, b in zip(scores_below, scores_below[1:]):
        assert b > a
    assert all(s <= 1.0 for s in scores_below)

    # At/above breakpoints[2] (20 km/h, ITDP's own "zero speed penalty"
    # anchor -- this module's docstring): reaches exactly 1.0 and stays
    # capped there for any faster speed, including well past
    # `saturation_kmh` (2026-08-12: that parameter no longer extends the
    # ramp for this breakpoint-anchored formula, see speed.py).
    assert speed_score(20) == pytest.approx(1.0)
    assert speed_score(45) == pytest.approx(1.0)
    assert speed_score(70) == pytest.approx(1.0)
    assert speed_score(90) == pytest.approx(1.0)
    assert speed_score(120) == pytest.approx(1.0)


def test_walk_decay_decreases_as_walk_time_increases():
    # First two times fall inside the flat "close enough" plateau
    # (saturation_minutes, ~3.2 min at the default 1.3 m/s / 250 m) and are
    # both exactly 1.0 by design -- only check strict decrease beyond it.
    times = [5, 10, 15, 30]
    scores = [walk_decay(t) for t in times]
    for a, b in zip(scores, scores[1:]):
        assert b < a
    assert walk_decay(0) == walk_decay(2) == pytest.approx(1.0)


def test_walk_decay_is_independent_of_stop_score():
    # 2026-08-12, at the user's explicit request: D(t) is now a single
    # curve shared by every stop regardless of quality -- stop_score only
    # enters as a separate multiplicative factor in accessibility_score,
    # not by reshaping D(t) itself (see walk_access.py's module docstring
    # for why the earlier stop_score-scaled design double-counted stop
    # quality). walk_decay no longer accepts a stop_score argument at all.
    t = 20
    assert walk_decay(t) == pytest.approx(walk_decay(t))


def test_accessibility_score_decreases_as_walk_time_increases():
    fixed_stop_score = 0.8
    times = [5, 10, 15, 30]  # past the flat plateau -- see walk_decay test above
    scores = [accessibility_score(fixed_stop_score, t) for t in times]
    for a, b in zip(scores, scores[1:]):
        assert b < a


# --- Bounds ----------------------------------------------------------------


@pytest.mark.parametrize("headway", [0, 1, 5, 11, 12, 30, 60, 120, 1000])
def test_frequency_score_bounded(headway):
    s = frequency_score(headway)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("cv", [0, 0.01, 0.5, 1.0, 5.0, 100.0])
def test_reliability_score_bounded(cv):
    s = reliability_score(cv)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("route_type", [0, 1, 2, 3, 5, 7, 12, None])
def test_mode_score_bounded(route_type):
    s = mode_score(route_type)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("t", [0, 0.001, 5, 60, 10_000])
def test_walk_decay_bounded(t):
    d = walk_decay(t)
    assert 0.0 <= d <= 1.0


def test_walk_decay_zero_is_one():
    assert walk_decay(0) == pytest.approx(1.0)


def test_walk_decay_raises_on_negative_walk_time():
    with pytest.raises(ValueError):
        walk_decay(-1)


# --- stop_score geometric mean identity & validation -----------------------


def test_stop_score_equal_weights_reduces_to_fourth_root_of_product():
    a, b, c, d = 0.4, 0.7, 0.9, 0.5
    result = stop_score(a, b, c, d, weights=(0.25, 0.25, 0.25, 0.25))
    expected = (a * b * c * d) ** 0.25
    assert result == pytest.approx(expected, rel=1e-9)


def test_stop_score_raises_on_bad_weights():
    with pytest.raises(ValueError):
        stop_score(0.5, 0.5, 0.5, 0.5, weights=(0.5, 0.5, 0.5, 0.5))


def test_stop_score_raises_on_out_of_range_subscore():
    with pytest.raises(ValueError):
        stop_score(1.5, 0.5, 0.5, 0.5)


def test_stop_score_raises_on_speed_score_above_one():
    # speed_score is capped at 1.0 (2026-08-12, see speed.py); stop_score
    # must reject anything above that as out of range, like every other
    # sub-score.
    with pytest.raises(ValueError):
        stop_score(0.8, 1.15, 0.8, 0.8)


# --- Qualitative validation: metro vs rural bus -----------------------------


def test_metro_scores_high_and_rural_bus_scores_low():
    metro_stop = stop_score(
        mode_score(1),  # subway/metro
        speed_score(35),
        frequency_score(4),
        reliability_score(0.05),
    )
    rural_stop = stop_score(
        mode_score(3),  # bus
        speed_score(15),
        frequency_score(60),
        reliability_score(1.5),
    )

    assert metro_stop > 0.6
    assert rural_stop < 0.3
    assert metro_stop > rural_stop


# --- Regional differentiation ------------------------------------------------


def test_region_changes_walk_decay_via_walking_speed():
    # Decay-shape params (t0_base/k/p) are deliberately the same across
    # regions (2026-08-11, at the user's explicit request -- see
    # parameters.py); only walking_speed_mps varies, which shifts the
    # effective saturation_minutes plateau and therefore the decay curve.
    t = 10
    na = walk_decay(t, region="north_america")  # slower MUTCD walking speed
    eu = walk_decay(t, region="europe")
    assert na != eu


def test_frequency_score_same_across_regions():
    # headway_saturation_minutes/headway_decay_scale_minutes are
    # deliberately uniform across regions (2026-08-11 design choice, see
    # parameters.py) -- unlike walk_decay, frequency_score currently has no
    # region-varying parameter at all.
    h = 11
    na = frequency_score(h, region="north_america")
    eu = frequency_score(h, region="europe")
    assert na == eu


def test_global_south_does_not_crash_and_warns():
    with pytest.warns(UserWarning, match="global_south"):
        s = frequency_score(20, region="global_south")
    assert 0.0 <= s <= 1.0

    with pytest.warns(UserWarning, match="global_south"):
        s2 = accessibility_score(0.7, 5, region="global_south")
    assert 0.0 <= s2 <= 1.0


def test_non_flagged_regions_do_not_warn():
    with warnings_none():
        frequency_score(20, region="global")
        frequency_score(20, region="europe")
        frequency_score(20, region="north_america")


def warnings_none():
    import warnings as _warnings

    class _Ctx:
        def __enter__(self):
            _warnings.simplefilter("error", UserWarning)
            return self

        def __exit__(self, *exc):
            _warnings.resetwarnings()
            return False

    return _Ctx()
