"""Unit test for `equity_flag`'s `more_transit`/`more_housing` split, against a
synthetic dataset with hand-computed expected results.

`code/stats.py` lives in `TransitLOSStudies/CS_transitLOS/code/`, a sibling
project, not a transitLOS dependency -- import it by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load `stats.py` directly by file path rather than via the `code` package
# name: `code` shadows Python's own stdlib `code` module (already imported
# by pytest/the interpreter by the time this test collects), so a normal
# `sys.path` + `from code.stats import ...` picks up the wrong module.
_STATS_PATH = (
    Path(__file__).resolve().parents[2]
    / "TransitLOSStudies"
    / "CS_transitLOS"
    / "code"
    / "stats.py"
)
_spec = importlib.util.spec_from_file_location("cs_transitlos_stats", _STATS_PATH)
_stats = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _stats  # dataclass field resolution needs this in sys.modules first
_spec.loader.exec_module(_stats)
development_priority_flag = _stats.development_priority_flag
housing_priority_flag = _stats.housing_priority_flag
equity_flag = _stats.equity_flag


def _synthetic_dataset():
    # 8 cells, equal population weight. Density is 1..8 (already sorted).
    # Level of service is chosen so the low-density/high-access cell (density=1,
    # access=0.9) should be flagged "more_housing", and the high-density/
    # low-access cell (density=8, access=0.1) should be flagged "more_transit".
    density = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    access = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4, 0.1], dtype=float)
    population = np.ones(8)
    return density, access, population


def test_development_priority_flag_more_transit():
    density, access, population = _synthetic_dataset()
    # Hand-computed: median_all(density)=4 -> group_a(density>4)=[5,6,7,8],
    # median_a(density in group_a)=6 -> density_flag = density>6 -> {7,8}.
    # group_b(density<=4)=[1,2,3,4], median_b(access in group_b)=0.3 ->
    # access_flag = access<0.3 over ALL rows -> density in {4,8} (access .2,.1).
    # AND -> only density=8.
    expected = np.array([False, False, False, False, False, False, False, True])
    result = development_priority_flag(density, access, population)
    np.testing.assert_array_equal(result, expected)


def test_housing_priority_flag_more_housing():
    density, access, population = _synthetic_dataset()
    # Hand-computed: median_all(density)=4 -> group_b(density<=4)=[1,2,3,4],
    # median_b(density in group_b)=2 -> density_flag_low = density<2 -> {1}.
    # group_a(density>4)=[5,6,7,8], median_a(access in group_a)=0.4 ->
    # access_flag_high = access>0.4 over ALL rows -> density in {1,2,5,6}
    #   (access .9, .8, .7, .6).
    # AND -> only density=1.
    expected = np.array([True, False, False, False, False, False, False, False])
    result = housing_priority_flag(density, access, population)
    np.testing.assert_array_equal(result, expected)


def test_equity_flag_labels_match_mirrored_splits():
    density, access, population = _synthetic_dataset()
    flags = equity_flag(density, access, population)
    expected = np.array(
        ["more_housing", None, None, None, None, None, None, "more_transit"],
        dtype=object,
    )
    np.testing.assert_array_equal(flags, expected)
