"""Public API for the transitLOS scoring module.

Implements the finalized Transit Quality-of-Service scoring methodology
documented in `transitLOS/documentation/scoring/scoring.md`. This package is
pure math on floats -- it has no dependency on the sibling geohierarchy/
pycensus/pygtfshandler/urbanaccessanalyzer packages and needs only the
Python standard library.

**2026-09-28 model rewrite**: the prior four-factor
(mode/speed/frequency/reliability) model is replaced by scoring.md's
finalized three-factor model (MRC/speed/frequency), with mode and
reliability/comfort merged into one `MRC(·)` axis (`mrc.py`, replacing the
old `mode.py`/`reliability.py`), and walk-access decay `D(t; stop_score)`
restored to being a joint function of walk time AND stop quality/mode
(`walk_access.py`), not a stop_score-independent curve.

Re-exports:
    mrc_score, MR_mode, MR_CV, comfort_score: the combined mode/reliability/
        comfort sub-score and its (a)/(b)/(c) components (`mrc.py`).
    mode_category: classifies a GTFS route_type into "rail"/"tram"/"bus".
    mode_score: convenience wrapper, `MR_mode(mode_category(route_type))`.
    speed_score: V(speed) (`speed.py`).
    frequency_score: H(headway) (`frequency.py`).
    stop_score: weighted geometric mean of MRC/speed/frequency
        (scoring.md §1.2, `stop_score.py`).
    walk_decay: the walk-access decay function D(t; stop_score) (`walk_access.py`).
    walk_time_for_decay: closed-form inverse of walk_decay.
    accessibility_score: stop_score * walk_decay(...) = PTLOS (scoring.md §1.1).
    REGIONS, RegionParameters, UNVALIDATED_REGIONS: the parameter table from
        `parameters.py` (now a backward-compatible universal-parameters
        shim -- see that module's docstring).
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
