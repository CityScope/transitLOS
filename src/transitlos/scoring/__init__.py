"""Public API for the transitLOS scoring module.

Implements the Transit Level-of-Service scoring methodology documented in
`report/scoring/SCORING.md` (and its authoritative parameter files under
`report/scoring/parameters/`). This package is pure math on floats -- it has
no dependency on the sibling geohierarchy/pycensus/pygtfshandler/
urbanaccessanalyzer packages and needs only the Python standard library.

Re-exports:
    mode_score, speed_score, frequency_score, reliability_score: the four
        stop_score sub-score functions.
    mode_category: classifies a GTFS route_type into "rail"/"tram"/"bus"
        (see `mode.py`).
    stop_score: weighted geometric mean of the four sub-scores
        (SCORING.md §1.1, extended with a separate mode factor).
    walk_decay: the walk-time decay function D(t) (SCORING.md §1.2/§2).
    accessibility_score: stop_score * walk_decay(...) (SCORING.md §1.2).
    REGIONS, RegionParameters, UNVALIDATED_REGIONS: the parameter table from
        `parameters.py`.
"""

from .accessibility_score import accessibility_score
from .distance_matrix import build_intelligent_distance_matrix, ceil_to_grid, ceil_to_grid_array
from .frequency import frequency_score
from .mode import mode_category, mode_score
from .parameters import REGIONS, RegionParameters, UNVALIDATED_REGIONS
from .reliability import reliability_score
from .speed import speed_score
from .stop_score import stop_score
from .walk_access import walk_decay, walk_time_for_decay

__all__ = [
    "mode_score",
    "mode_category",
    "speed_score",
    "frequency_score",
    "reliability_score",
    "stop_score",
    "walk_decay",
    "walk_time_for_decay",
    "accessibility_score",
    "build_intelligent_distance_matrix",
    "ceil_to_grid",
    "ceil_to_grid_array",
    "REGIONS",
    "RegionParameters",
    "UNVALIDATED_REGIONS",
]
