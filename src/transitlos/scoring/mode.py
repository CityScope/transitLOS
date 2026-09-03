"""Mode scoring: converts a GTFS `route_type` into a discrete-category [0, 1] score.

This module exists at the user's explicit direction, overriding this
project's own prior, documented decision to score transit *only* by
observed speed/quality (see `mode_speed.py`/`speed.py`'s module docstring
and `report/scoring/parameters/06_mode_in_vehicle_time_valuation.md`) rather
than a fixed mode-label premium -- that file's own conclusion was that the
literature-estimated "rail factor" is real on average but geographically
concentrated (UK/Europe/North America) and collapses toward zero once
service quality is actually equalized, which is why a mode premium was
originally rejected in favor of the speed-based `speed_score`. The user
asked for a mode score anyway, still grounded in the same literature file's
findings rather than an arbitrary constant.

Parameter source: `06_mode_in_vehicle_time_valuation.md` §2 (Abrantes &
Wardman 2011, underlying UK DfT/WebTAG in-vehicle-time appraisal
parameters) reports rail in-vehicle time valued at a reference weight of
~1.0 and local bus in-vehicle time valued ~1.0-1.3x higher (i.e. a bus
minute is perceived as more onerous than an equivalent rail minute for
otherwise-comparable service). That file is explicit this range is a
"synthesized order-of-magnitude range from the Wardman research programme,
not a verbatim quoted table value" and is UK-specific, not a global
constant -- carried forward here with the same caveat. Using the midpoint
of that range (1.15x) as bus's relative in-vehicle-time disutility gives a
quality-equivalent score (higher = better, inverse of the disutility
weight) of `1 / 1.15 ≈ 0.87` for bus relative to rail's reference `1.0`.

Tram/light rail has no separate numeric anchor in that file -- Scherer
(2010)/Scherer & Dziekan (2012)'s "rail factor" literature (also reviewed
there) finds light rail attracts a similar attractiveness premium to heavy
rail rather than tracking bus, so tram is placed close to rail: the
geometric mean of bus and rail (~0.93), a clearly-labeled interpolation,
not a separately-sourced figure.
"""

from __future__ import annotations

from typing import Optional

# GTFS `route_type` (standard 0-7 codes, already normalized by
# `pyGTFSHandler`) -> mode category, per the user's explicit grouping:
#   - "rail": only heavy rail / subway / commuter train (1=subway/metro, 2=rail).
#   - "tram": tram/streetcar/light rail and other rail-adjacent fixed-guideway
#     modes not already counted as "rail" (0=tram/light rail, 5=cable tram, 12=monorail).
#   - "bus": everything else, including any route with a missing/unrecognized
#     route_type -- bus is the fallback/default category, not a stricter filter.
_RAIL_ROUTE_TYPES = {1, 2}
_TRAM_ROUTE_TYPES = {0, 5, 12}

# Mode-category scores (see module docstring for the literature derivation).
# Not literature-mandated in the way `speed_score`'s outer breakpoints are --
# this project's own prior research concluded a mode premium's magnitude is
# too context-dependent to fix as a universal constant; these values are the
# best available literature anchor (Abrantes & Wardman 2011 WebTAG-style
# bus:rail IVT ratio), applied per the user's explicit request for a mode
# score, not an internally validated global default.
MODE_SCORES = {
    "rail": 1.0,
    "tram": (1.0 * (1.0 / 1.15)) ** 0.5,  # geometric mean of rail (1.0) and bus (1/1.15)
    "bus": 1.0 / 1.15,
}


def mode_category(route_type: Optional[int]) -> str:
    """Classify a GTFS `route_type` into `"rail"`, `"tram"`, or `"bus"`.

    Args:
        route_type: Standard GTFS `route_type` code (0-7), or `None`/NaN for
            a route with no usable route_type -- falls back to `"bus"`,
            per the user's explicit instruction ("bus: all routes including
            ... anything even if no route type given").

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


def mode_score(route_type: Optional[int]) -> float:
    """Compute a [0, 1] mode score from a GTFS `route_type` (see module docstring).

    Args:
        route_type: Standard GTFS `route_type` code, or `None`.

    Returns:
        A float in [0, 1]: `MODE_SCORES[mode_category(route_type)]`.
    """
    return MODE_SCORES[mode_category(route_type)]
