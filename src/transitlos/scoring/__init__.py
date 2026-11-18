"""Public API of the transitLOS scoring module: pure math on floats (standard library only).

Re-exports:
    mrc_score, MR_mode, MR_CV, comfort_score: the mode / reliability / comfort sub-score and its components (`mrc.py`).
    mode_category: classifies a GTFS route_type as "rail", "tram" or "bus".
    mode_score: `MR_mode(mode_category(route_type))`.
    speed_score: V(speed) (`speed.py`).
    frequency_score: H(headway) (`frequency.py`).
    stop_score: the combination of MRC, speed and frequency (`stop_score.py`).
    walk_decay: the walk-access decay D(t; stop_score) (`walk_access.py`).
    walk_time_for_decay: closed-form inverse of walk_decay.
    accessibility_score: stop_score * walk_decay(...) = PTLOS (`accessibility_score.py`).
"""

from .accessibility_score import accessibility_score
from .distance_matrix import build_intelligent_distance_matrix, ceil_to_grid, ceil_to_grid_array
from .frequency import frequency_score
from .mrc import MR_CV, MR_mode, comfort_score, mode_category, mode_score, mrc_score
from .parameters import REGIONS, RegionParameters, UNVALIDATED_REGIONS
from .speed import speed_score
from .stop_score import stop_score
from .walk_access import walk_decay, walk_time_for_decay

__all__ = [
    "mrc_score",
    "MR_mode",
    "MR_CV",
    "comfort_score",
    "mode_category",
    "mode_score",
    "speed_score",
    "frequency_score",
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
