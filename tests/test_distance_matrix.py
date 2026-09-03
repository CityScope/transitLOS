"""Tests for transitlos.scoring.distance_matrix (2026-08-12 UrbanAccessAnalyzer integration).

Covers `ceil_to_grid`'s ceiling-not-rounding semantics, and
`build_intelligent_distance_matrix`'s core invariants: bucket outputs are
truly indistinguishable at every chosen distance, every matrix value is on
the {0, 0.1, ..., 1.0} grid, and a stop_score of exactly 1.0 reaches a
perfect 1.0 at/near the stop.
"""

import numpy as np
import pytest

from transitlos.scoring import accessibility_score
from transitlos.scoring.parameters import REGIONS
from transitlos.scoring.distance_matrix import (
    OUTPUT_GRID_STEP,
    build_intelligent_distance_matrix,
    ceil_to_grid,
    ceil_to_grid_array,
)


# --- ceil_to_grid ------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (0.91, 1.0),
        (0.90, 0.9),  # exact multiple should not jump to the next tier
        (0.899999999, 0.9),  # float noise just under an exact multiple
        (0.30, 0.3),
        (0.05, 0.1),
        (1.0, 1.0),
        (1.2, 1.0),  # stop_score can exceed 1.0 (uncapped speed_score) -- clip
    ],
)
def test_ceil_to_grid(value, expected):
    assert ceil_to_grid(value) == pytest.approx(expected)


def test_ceil_to_grid_array_matches_scalar():
    values = np.array([0.0, 0.91, 0.90, 0.30, 1.2])
    expected = np.array([ceil_to_grid(v) for v in values])
    np.testing.assert_allclose(ceil_to_grid_array(values), expected)


# --- build_intelligent_distance_matrix ---------------------------------------


def test_matrix_values_are_all_on_grid():
    scores = np.random.default_rng(0).beta(2, 3, 200)
    matrix, bucket_map, distances = build_intelligent_distance_matrix(scores, region="global")
    value_cols = [c for c in matrix.columns if c != "poi_score"]
    for col in value_cols:
        for v in matrix[col].to_list():
            remainder = v / OUTPUT_GRID_STEP
            assert abs(remainder - round(remainder)) < 1e-6


def test_buckets_are_truly_output_equivalent():
    # Whatever bucket each raw score maps to, every score sharing a bucket
    # must have an identical output signature -- the whole point of
    # bucketing. Use a dense sweep so at least some values are forced to
    # collapse together (there are far more scores here than possible
    # distinct output signatures).
    scores = np.linspace(0.01, 1.0, 300)
    matrix, bucket_map, distances = build_intelligent_distance_matrix(scores, region="global")
    assert matrix.height < len(scores)  # bucketing actually collapsed something

    from collections import defaultdict

    members_by_rep = defaultdict(list)
    for raw, rep in bucket_map.items():
        members_by_rep[rep].append(raw)
    collapsed_reps = [rep for rep, members in members_by_rep.items() if len(members) > 1]
    assert collapsed_reps, "expected at least one bucket with >1 member"

    for rep in collapsed_reps:
        row = matrix.filter(matrix["poi_score"] == rep).to_dicts()[0]
        for member in members_by_rep[rep]:
            expected = tuple(
                ceil_to_grid(accessibility_score(member, d / REGIONS["global"].walking_speed_mps / 60.0, region="global"))
                for d in distances
            )
            actual = tuple(row[str(d)] for d in distances)
            assert actual == expected


def test_perfect_stop_score_reaches_one_at_plateau():
    matrix, bucket_map, distances = build_intelligent_distance_matrix([1.0], region="global")
    rep = bucket_map[1.0]
    row = matrix.filter(matrix["poi_score"] == rep).to_dicts()[0]
    assert max(v for k, v in row.items() if k != "poi_score") == pytest.approx(1.0)


def test_distances_are_monotonically_increasing():
    scores = np.random.default_rng(1).beta(2, 3, 100)
    _matrix, _bucket_map, distances = build_intelligent_distance_matrix(scores, region="global")
    assert distances == sorted(distances)
    assert len(distances) == len(set(distances))


def test_empty_scores_returns_empty_matrix():
    matrix, bucket_map, distances = build_intelligent_distance_matrix([], region="global")
    assert matrix.height == 0
    assert bucket_map == {}
    assert distances == []


def test_bucket_representative_gives_same_accessibility_score_signature():
    # The matrix cell values should equal accessibility_score(rep, t) at the
    # chosen distances, ceiling-discretized -- not some other formula.
    scores = [0.7]
    matrix, bucket_map, distances_m = build_intelligent_distance_matrix(scores, region="global")
    rep = bucket_map[0.7]
    row = matrix.filter(matrix["poi_score"] == rep).to_dicts()[0]
    from transitlos.scoring.parameters import REGIONS

    speed = REGIONS["global"].walking_speed_mps
    for d_m in distances_m:
        t_min = d_m / speed / 60.0
        expected = ceil_to_grid(accessibility_score(rep, t_min, region="global"))
        assert row[str(d_m)] == pytest.approx(expected)


def test_max_distance_m_caps_the_search_radius():
    # Without a cap, the lowest tier's natural threshold extends several
    # kilometers out (the decay curve never truly reaches zero) -- with a
    # cap, no distance step should exceed it, and the cap itself should be
    # the final step with a real (not truncated-early) discretized value.
    scores = [0.3, 0.5, 0.7, 0.9, 1.0]
    _matrix, _bucket_map, uncapped = build_intelligent_distance_matrix(scores, region="global")
    assert max(uncapped) > 2000

    matrix, bucket_map, capped = build_intelligent_distance_matrix(
        scores, region="global", max_distance_m=2000
    )
    assert max(capped) == pytest.approx(2000.0)
    assert all(d <= 2000.0 for d in capped)

    # The 2000m column's value must be the real, ceiling-discretized D(t)
    # at exactly 2000m -- not a leftover/truncated value from whatever
    # tier happened to fall just under the cap.
    rep = bucket_map[1.0]
    row = matrix.filter(matrix["poi_score"] == rep).to_dicts()[0]
    from transitlos.scoring.parameters import REGIONS

    speed = REGIONS["global"].walking_speed_mps
    t_min = 2000.0 / speed / 60.0
    expected = ceil_to_grid(accessibility_score(rep, t_min, region="global"))
    assert row["2000.0"] == pytest.approx(expected)
