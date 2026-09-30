"""MRC(mode/reliability/comfort): the combined mode/reliability/comfort axis.

Implements scoring.md §2.3: mode, reliability, and comfort are modeled as
**one combined axis**, not three separately-weighted terms -- there is no
clean literature-backed "mode premium" independent of realized reliability
(protected right-of-way reduces bunching/signal delay/headway variance,
which *is* the mode-correlated quality gap), and comfort/crowding is the
same kind of refinement on the mode label, available only with richer
real-time data. Replaces the previous version's separate `mode.py`
(`mode_score`, a `MR`-flavored but unrelated categorical scheme) and
`reliability.py` (`reliability_score`, an `exp(-CV/scale)` design-choice
falloff with no direct literature source) with this single module.

Three layered components (scoring.md §2.3 table):

    (a) MR_mode(mode)           -- categorical default, GTFS-static `route_type` only.
                                   PRIMARY, load-bearing path for this project's
                                   actual GTFS-static-only pipeline.
    (b) MR_CV(headway_cv)       -- optional, preferred over (a) when scheduled-headway
                                   CV data is available (still GTFS-static-only).
    (c) C(occupancy_status)     -- optional comfort refinement on top of (a)/(b),
                                   needs GTFS-realtime `OccupancyStatus`; NOT wired
                                   into the default pipeline (real-time occupancy
                                   is rarely available in practice).

Combination: `MRC = MR_component * C_component`, where `MR_component` is (b)
if headway_cv is supplied, else (a); `C_component` defaults to 1.0
(comfort-neutral) with no occupancy_status supplied -- the common case, so
`MRC = MR_mode(mode)` exactly in practice, matching scoring.md §2.3's stated
"common case" behavior. This internal product mirrors the document's
non-compensatory multiplicative philosophy (§1.2) at the sub-function level.

Cross-checked against `transitLOS/documentation/scoring/scripts/
compare_methods.py`'s own `MR_MODE` table (the CV/comfort refinements are
not implemented in that comparison script, which only ever has mode data).
"""

from __future__ import annotations

from typing import Optional

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region

# GTFS `route_type` (standard 0-7 codes) -> mode category.
#   - "rail": heavy rail / subway / commuter train (1=subway/metro, 2=rail).
#   - "tram": tram/streetcar/light rail and other rail-adjacent fixed-guideway
#     modes not already counted as "rail" (0=tram/light rail, 5=cable tram, 12=monorail).
#   - "bus": everything else, including any route with a missing/unrecognized
#     route_type -- bus is the fallback/default category, not a stricter filter.
_RAIL_ROUTE_TYPES = {1, 2}
_TRAM_ROUTE_TYPES = {0, 5, 12}


def mode_category(route_type: Optional[int]) -> str:
    """Classify a GTFS `route_type` into `"rail"`, `"tram"`, or `"bus"`.

    Args:
        route_type: Standard GTFS `route_type` code (0-7), or `None`/NaN for
            a route with no usable route_type -- falls back to `"bus"`.

    Returns:
        `"rail"`, `"tram"`, or `"bus"`.
    """
    if route_type is None:
        return "bus"
    try:
        rt = int(route_type)
    except (TypeError, ValueError):
        return "bus"
    if rt in _RAIL_ROUTE_TYPES:
        return "rail"
    if rt in _TRAM_ROUTE_TYPES:
        return "tram"
    return "bus"


def MR_mode(mode: str, region: str = "global") -> float:
    """Compute the mode-based categorical default `MR_mode(mode)`.

    scoring.md §2.3(a): rail/subway/metro/commuter rail = 1.000 (anchor);
    tram/light rail/dedicated-ROW BRT = 0.894 (geometric-mean interpolation
    between rail and bus); local bus (mixed traffic) = 0.800 (Wardman,
    Chintakayala & de Jong 2016, Table 7, "metro valued/bus valued" ratio).

    Args:
        mode: One of `"rail"`, `"tram"`, `"bus"` (see `mode_category`).
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal values.

    Returns:
        A float in (0, 1].

    Raises:
        ValueError: If mode is not one of "rail", "tram", "bus".
    """
    warn_if_unvalidated_region(region)
    table = REGIONS[region].mr_mode
    if mode not in table:
        raise ValueError(f"mode must be one of {sorted(table)}, got {mode!r}")
    return table[mode]


def mode_score(route_type: Optional[int], region: str = "global") -> float:
    """Compute `MR_mode(mode_category(route_type))` directly from a GTFS route_type.

    Convenience wrapper combining `mode_category` + `MR_mode`.
    """
    return MR_mode(mode_category(route_type), region=region)


def MR_CV(headway_cv: float) -> float:
    """Compute the optional CV-of-scheduled-headway reliability refinement.

    scoring.md §2.3(b): `MR_CV(CV) = 1 / (1 + CV^2)`, the algebraic inverse
    of the average-wait inflation factor `(1+CV^2)` from Bowman & Turnquist's
    (1981) passenger-choice wait model -- NOT the prior version's
    `exp(-CV/scale)` design-choice falloff, which had no direct literature
    source. `MR_CV(0)=1.0`, `MR_CV(1)=0.5`, `MR_CV(2)~=0.2`. Mode-agnostic by
    construction; preferred over `MR_mode` whenever scheduled-headway CV data
    exists, since it reflects the stop's own actual pattern.

    Args:
        headway_cv: Coefficient of variation of scheduled headways
            (stddev / mean), a unitless non-negative number.

    Returns:
        A float in (0, 1].

    Raises:
        ValueError: If headway_cv is negative.
    """
    if headway_cv < 0:
        raise ValueError(f"headway_cv must be >= 0, got {headway_cv}")
    return 1.0 / (1.0 + headway_cv**2)


def comfort_score(occupancy_status: Optional[str], region: str = "global") -> float:
    """Compute the optional comfort/crowding refinement `C(occupancy_status)`.

    scoring.md §2.3(c): a categorical lookup keyed to the standardized
    GTFS-realtime `VehiclePosition.OccupancyStatus` enum. Defaults to 1.0
    (comfort-neutral) when occupancy data is unavailable -- the common case,
    since this refinement needs GTFS-realtime data this project's pipeline
    typically does not have.

    Args:
        occupancy_status: One of the GTFS-realtime `OccupancyStatus` enum
            string values (`"EMPTY"`, `"MANY_SEATS_AVAILABLE"`,
            `"FEW_SEATS_AVAILABLE"`, `"STANDING_ROOM_ONLY"`,
            `"CRUSHED_STANDING_ROOM_ONLY"`, `"FULL"`,
            `"NOT_ACCEPTING_PASSENGERS"`), or `None` if unavailable.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal values.

    Returns:
        A float in [0, 1]. Exactly 1.0 if `occupancy_status` is None.

    Raises:
        ValueError: If occupancy_status is a string not in the known enum.
    """
    warn_if_unvalidated_region(region)
    if occupancy_status is None:
        return 1.0
    table = REGIONS[region].comfort_scores
    if occupancy_status not in table:
        raise ValueError(f"occupancy_status must be one of {sorted(table)}, got {occupancy_status!r}")
    return table[occupancy_status]


def mrc_score(
    route_type: Optional[int] = None,
    headway_cv: Optional[float] = None,
    occupancy_status: Optional[str] = None,
    region: str = "global",
) -> float:
    """Compute the combined `MRC(mode/reliability/comfort)` sub-score.

    scoring.md §2.3: `MRC = MR_component * C_component`, where
    `MR_component` is `MR_CV(headway_cv)` if `headway_cv` is supplied
    (preferred, richer data), else `MR_mode(mode_category(route_type))`
    (the primary, load-bearing GTFS-static-only default); `C_component` is
    `comfort_score(occupancy_status)`, defaulting to 1.0 (comfort-neutral)
    when `occupancy_status` is None -- the common case, so
    `mrc_score(route_type) == MR_mode(mode_category(route_type))` exactly
    when no other data is supplied.

    Args:
        route_type: Standard GTFS `route_type` code, or `None`. Used only
            if `headway_cv` is not supplied.
        headway_cv: Coefficient of variation of scheduled headways, or
            `None` if unavailable. When supplied, takes precedence over
            `route_type` for the mode/reliability component.
        occupancy_status: GTFS-realtime `OccupancyStatus` enum value, or
            `None` if unavailable (the common case; defaults to
            comfort-neutral).
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal values.

    Returns:
        A float in [0, 1] (or up to (0, 1] * (0, 1] -- always <= 1.0 since
        both components are individually <= 1.0).
    """
    if headway_cv is not None:
        mr_component = MR_CV(headway_cv)
    else:
        mr_component = MR_mode(mode_category(route_type), region=region)
    c_component = comfort_score(occupancy_status, region=region)
    return mr_component * c_component
