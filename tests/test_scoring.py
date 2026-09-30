"""Tests for transitlos.scoring, per scoring.md's finalized methodology.

Covers monotonicity, bounds, the equal-weight geometric-mean identity, weight
validation, a qualitative metro-vs-rural-bus ordering check, the MRC(·)
mode/reliability/comfort sub-functions, and the now-universal (no longer
region-differentiated) parameter set.
"""

import math

import pytest

from transitlos.scoring import (
    MR_CV,
    MR_mode,
    accessibility_score,
    comfort_score,
    frequency_score,
    mode_category,
    mode_score,
    mrc_score,
    speed_score,
    stop_score,
    walk_decay,
    walk_time_for_decay,
)


# --- Monotonicity ---------------------------------------------------------


def test_frequency_score_increases_as_headway_decreases():
    # Sample above the saturation plateau (5 min) so the function is
    # strictly decreasing in headway (i.e. strictly increasing as headway
    # shrinks).
    headways = [90, 60, 40, 20, 15]
    scores = [frequency_score(h, region="global") for h in headways]
    for a, b in zip(scores, scores[1:]):
        assert b > a


def test_frequency_score_flat_at_and_below_saturation():
    assert frequency_score(5) == pytest.approx(1.0)
    assert frequency_score(2) == pytest.approx(1.0)
    assert frequency_score(0) == pytest.approx(1.0)


def test_mr_cv_decreases_as_cv_increases():
    cvs = [0.0, 0.2, 0.5, 1.0, 2.0]
    scores = [MR_CV(cv) for cv in cvs]
    for a, b in zip(scores, scores[1:]):
        assert b < a
    assert MR_CV(0.0) == pytest.approx(1.0)
    assert MR_CV(1.0) == pytest.approx(0.5)


def test_speed_score_increases_with_speed_then_slows_above_anchor():
    # 2026-09-29 THIRD SHAPE-FIX revision: a single smooth, strictly
    # increasing Hill-function curve V(v)=M*v^n/(v^n+K^n) (n=2), fit through
    # the single anchor (30, 1.0) -- re-derived from real metro
    # commercial-speed data -- with M=1.15 an explicit design-judgment
    # asymptote (an intermediate same-day revision tried pinning a second
    # exact point at (80, 1.15), matching German Regional-Express's speed,
    # and it was reverted after review found that single-country benchmark
    # didn't hold up the way the anchor's 4-system convergence did). No kink
    # at the anchor, and no low-speed defect (the n=1 plain-Michaelis-Menten
    # predecessor gave V(5)~=0.49-0.59 depending on fitting setup, found too
    # generous by review; this curve gives V(5)~=0.180/V(10)~=0.489). Still
    # strictly increasing everywhere below the anchor.
    speeds_below = [5, 10, 15, 20, 25]
    scores_below = [speed_score(s) for s in speeds_below]
    for a, b in zip(scores_below, scores_below[1:]):
        assert b > a
    assert speed_score(5) == pytest.approx(0.1797, abs=1e-3)
    assert speed_score(10) == pytest.approx(0.4894, abs=1e-3)
    assert speed_score(30) == pytest.approx(1.0)

    # Above 30 km/h: the curve keeps rising, very slowly, asymptoting to
    # M = 1.15 (never reached at any finite speed, never capped by a
    # min(...), and no longer fit to hit exactly at any named speed).
    assert speed_score(100) == pytest.approx(1.1347, abs=1e-3)
    assert speed_score(150) > speed_score(100)  # no hard cap anymore
    assert speed_score(150) < 1.15  # still below the asymptote
    assert speed_score(1_000_000) == pytest.approx(1.15, rel=1e-3)


def test_walk_decay_decreases_as_walk_time_increases():
    # First time falls inside the flat plateau (t_plateau ~= 2.56 min at the
    # default 1.3 m/s / 200 m) and is exactly 1.0 by design -- only check
    # strict decrease beyond it.
    times = [5, 10, 15, 30]
    scores = [walk_decay(t, 0.5, "bus") for t in times]
    for a, b in zip(scores, scores[1:]):
        assert b < a
    assert walk_decay(0, 0.5, "bus") == walk_decay(2, 0.5, "bus") == pytest.approx(1.0)


def test_walk_decay_depends_on_stop_score_and_mode():
    # 2026-09-28 restored-coupling revision: D(t; stop_score) is a joint
    # function again -- a higher stop_score (and a mode with a longer
    # t0_base) should tolerate a longer walk before decaying as much.
    t = 20
    low_quality = walk_decay(t, 0.0, "bus")
    high_quality = walk_decay(t, 1.0, "bus")
    assert high_quality > low_quality

    bus = walk_decay(t, 0.5, "bus")
    rail = walk_decay(t, 0.5, "rail")
    tram = walk_decay(t, 0.5, "tram")
    assert tram > rail > bus  # t0_base: tram > rail > bus


def test_walk_time_for_decay_inverts_walk_decay():
    for stop_score_val in (0.0, 0.3, 0.8):
        for mode in ("bus", "rail", "tram"):
            for target in (1.0, 0.8, 0.5, 0.1):
                t = walk_time_for_decay(target, stop_score_val, mode)
                d = walk_decay(t, stop_score_val, mode)
                assert d == pytest.approx(target, rel=1e-6, abs=1e-9)


def test_accessibility_score_decreases_as_walk_time_increases():
    fixed_stop_score = 0.8
    times = [5, 10, 15, 30]  # past the flat plateau -- see walk_decay test above
    scores = [accessibility_score(fixed_stop_score, t, "bus") for t in times]
    for a, b in zip(scores, scores[1:]):
        assert b < a


# --- Bounds ----------------------------------------------------------------


@pytest.mark.parametrize("headway", [0, 1, 5, 11, 12, 30, 60, 120, 1000])
def test_frequency_score_bounded(headway):
    s = frequency_score(headway)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("cv", [0, 0.01, 0.5, 1.0, 5.0, 100.0])
def test_mr_cv_bounded(cv):
    s = MR_CV(cv)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("route_type", [0, 1, 2, 3, 5, 7, 12, None])
def test_mode_score_bounded(route_type):
    s = mode_score(route_type)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("t", [0, 0.001, 5, 60, 10_000])
def test_walk_decay_bounded(t):
    d = walk_decay(t, 0.5, "bus")
    assert 0.0 <= d <= 1.0


def test_walk_decay_zero_is_one():
    assert walk_decay(0, 0.5, "bus") == pytest.approx(1.0)


def test_walk_decay_raises_on_negative_walk_time():
    with pytest.raises(ValueError):
        walk_decay(-1, 0.5, "bus")


def test_walk_decay_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        walk_decay(20, 0.5, "spaceship")


# --- mrc_score / MR_mode / comfort_score -------------------------------------


def test_mode_category_classification():
    assert mode_category(1) == "rail"
    assert mode_category(2) == "rail"
    assert mode_category(0) == "tram"
    assert mode_category(5) == "tram"
    assert mode_category(12) == "tram"
    assert mode_category(3) == "bus"
    assert mode_category(None) == "bus"
    assert mode_category("not-a-number") == "bus"


def test_mr_mode_values():
    assert MR_mode("rail") == pytest.approx(1.000)
    assert MR_mode("tram") == pytest.approx(0.894, abs=1e-3)
    assert MR_mode("bus") == pytest.approx(0.800)


def test_mr_mode_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        MR_mode("spaceship")


def test_comfort_score_neutral_by_default():
    assert comfort_score(None) == pytest.approx(1.0)


def test_comfort_score_known_values():
    assert comfort_score("EMPTY") == pytest.approx(1.00)
    assert comfort_score("FEW_SEATS_AVAILABLE") == pytest.approx(0.85)
    assert comfort_score("STANDING_ROOM_ONLY") == pytest.approx(0.55)
    assert comfort_score("CRUSHED_STANDING_ROOM_ONLY") == pytest.approx(0.40)
    assert comfort_score("FULL") == pytest.approx(0.0)
    assert comfort_score("NOT_ACCEPTING_PASSENGERS") == pytest.approx(0.0)


def test_mrc_score_defaults_to_mode_categorical():
    # Common case: no headway_cv, no occupancy_status -- MRC == MR_mode.
    assert mrc_score(route_type=1) == pytest.approx(MR_mode("rail"))
    assert mrc_score(route_type=3) == pytest.approx(MR_mode("bus"))


def test_mrc_score_prefers_cv_when_supplied():
    assert mrc_score(route_type=3, headway_cv=0.3) == pytest.approx(MR_CV(0.3))


def test_mrc_score_applies_comfort_on_top():
    base = mrc_score(route_type=1)
    with_crowding = mrc_score(route_type=1, occupancy_status="STANDING_ROOM_ONLY")
    assert with_crowding == pytest.approx(base * 0.55)


# --- stop_score aggregation structure & validation --------------------------


def test_stop_score_default_is_nested_sqrt_mrc_v_times_h():
    # 2026-09-29 aggregation-STRUCTURE revision (scoring.md Section 1.2): the
    # default is now the nested form sqrt(MRC * V) * H, not a flat mean of
    # all three sub-scores.
    a, b, c = 0.4, 0.7, 0.9
    result = stop_score(a, b, c)
    expected = math.sqrt(a * b) * c
    assert result == pytest.approx(expected, rel=1e-9)


def test_stop_score_nested_rejects_weights_and_p():
    with pytest.raises(ValueError):
        stop_score(0.5, 0.5, 0.5, weights=(1 / 3, 1 / 3, 1 / 3))
    with pytest.raises(ValueError):
        stop_score(0.5, 0.5, 0.5, p=-1.0)


def test_stop_score_nested_not_idempotent_at_all_equal():
    # Unlike a true mean, the nested form is NOT idempotent: MRC=V=H=x gives
    # x**2, not x -- an honest, documented consequence of gating H in by
    # multiplication rather than averaging it (scoring.md Section 1.2).
    x = 0.90
    assert stop_score(x, x, x) == pytest.approx(x**2, rel=1e-9)


def test_stop_score_nested_catastrophic_v_stress_case():
    # The "does pooling MRC/V let one rescue the other" stress case: a
    # catastrophic V alongside a good MRC. sqrt(0.90*0.10) = 0.30 is still
    # visibly harsh, and headway then still gates it further.
    result = stop_score(0.90, 0.10, 0.90)
    assert result == pytest.approx(math.sqrt(0.90 * 0.10) * 0.90, rel=1e-9)
    assert result < 0.30  # H<1 gates the sub-mean down further


def test_stop_score_nested_catastrophic_headway_gates_harder_than_legacy_means():
    # The original motivating production example (scoring.md Section 1.2):
    # mode=bus, MRC=0.800, V=0.8401, headway=285.73min -> H=0.1312. The
    # nested form should punish this at least as hard as the legacy p=-1
    # harmonic mean did.
    mrc, v, h = 0.800, 0.8401, 0.1312
    nested = stop_score(mrc, v, h)
    harmonic = stop_score(mrc, v, h, mode="harmonic")
    geometric = stop_score(mrc, v, h, mode="geometric")
    assert nested == pytest.approx(math.sqrt(mrc * v) * h, rel=1e-9)
    assert nested < harmonic < geometric


def test_stop_score_harmonic_mode_reproduces_legacy_flat_harmonic_mean():
    # mode="harmonic" reproduces the pre-2026-09-29 default exactly, for
    # back-compat/comparison purposes.
    a, b, c = 0.4, 0.7, 0.9
    result = stop_score(a, b, c, weights=(1 / 3, 1 / 3, 1 / 3), mode="harmonic")
    expected = 3.0 / (1 / a + 1 / b + 1 / c)
    assert result == pytest.approx(expected, rel=1e-9)


def test_stop_score_geometric_mode_reduces_to_cube_root_of_product():
    a, b, c = 0.4, 0.7, 0.9
    result = stop_score(a, b, c, weights=(1 / 3, 1 / 3, 1 / 3), mode="geometric")
    expected = (a * b * c) ** (1 / 3)
    assert result == pytest.approx(expected, rel=1e-9)


def test_stop_score_raises_on_bad_weights():
    with pytest.raises(ValueError):
        stop_score(0.5, 0.5, 0.5, weights=(0.5, 0.5, 0.5), mode="harmonic")


def test_stop_score_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        stop_score(0.5, 0.5, 0.5, mode="bogus")


def test_stop_score_raises_on_out_of_range_mrc_score():
    with pytest.raises(ValueError):
        stop_score(1.5, 0.5, 0.5)


def test_stop_score_allows_speed_score_above_one():
    # speed_score can reach up to 1.15 (V(speed)'s ceiling bonus); unlike
    # mrc_score/frequency_score, stop_score must NOT reject this.
    result = stop_score(0.8, 1.15, 0.8)
    expected = math.sqrt(0.8 * 1.15) * 0.8
    assert result == pytest.approx(expected)


def test_stop_score_raises_on_negative_speed_score():
    with pytest.raises(ValueError):
        stop_score(0.8, -0.1, 0.8)


# --- Qualitative validation: metro vs rural bus -----------------------------


def test_metro_scores_high_and_rural_bus_scores_low():
    metro_stop = stop_score(
        mrc_score(route_type=1),  # subway/metro
        speed_score(35),
        frequency_score(4),
    )
    rural_stop = stop_score(
        mrc_score(route_type=3),  # bus
        speed_score(15),
        frequency_score(60),
    )

    assert metro_stop > 0.6
    assert rural_stop < 0.6
    assert metro_stop > rural_stop


# --- Universal parameters (no more real regional differentiation) ----------


def test_frequency_score_same_across_regions():
    h = 11
    na = frequency_score(h, region="north_america")
    eu = frequency_score(h, region="europe")
    g = frequency_score(h, region="global")
    assert na == eu == g


def test_walk_decay_same_across_regions():
    # 2026-09-28: no region-specific override remains for any of the six
    # functions (scoring.md Section 3.3) -- unlike the prior model, walking
    # speed itself no longer varies by region either.
    t = 10
    na = walk_decay(t, 0.5, "bus", region="north_america")
    eu = walk_decay(t, 0.5, "bus", region="europe")
    assert na == pytest.approx(eu)


def test_global_south_does_not_crash_or_warn():
    # UNVALIDATED_REGIONS is now empty (every region shares identical,
    # universal parameters) -- see parameters.py's module docstring.
    with warnings_none():
        s = frequency_score(20, region="global_south")
    assert 0.0 <= s <= 1.0

    with warnings_none():
        s2 = accessibility_score(0.7, 5, "bus", region="global_south")
    assert s2 >= 0.0


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
