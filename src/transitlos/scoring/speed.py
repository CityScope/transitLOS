"""Speed scoring: converts an observed average operating speed into a [0, 1] score.

Formerly `mode_speed_score` -- this project's original, still literature-
preferred position (SCORING.md §2 row "Mode/speed classification"; Ben-Akiva
& Morikawa 2002, Scherer 2010/2012, corroborated by Ingvardson & Nielsen
2018) is that speed alone should carry this classification, not a fixed
rail/bus mode premium. At the user's explicit request, mode is now scored
*separately* as its own factor (`mode.py`) alongside this one, rather than
folded in here -- this function's own formula/breakpoints are unchanged, it
is just no longer the sole "how good is this vehicle's service" term in
`stop_score`.

The outer two km/h breakpoints (11.0, 20.0) are anchored to the ITDP BRT
Standard 2024's verbatim "Low Commercial Speeds" deduction table, read in
full 2026-08-10 (parameters/07_speed_based_service_classification.md §1):
ITDP applies zero speed penalty above 20 km/h and its steepest penalty band
below 11 km/h. The middle breakpoint (15.5, roughly the midpoint) still has
no direct ITDP/ARE/ÖROK numeric anchor and remains the report author's own
interpolated design choice. This module implements a continuous,
piecewise-linear interpolation between those banded breakpoints rather than
a hard step function, consistent with SCORING.md's general preference for
continuous over hard-cutoff scoring.

**2026-08-11 update:** above `breakpoints[2]` the score used to approach but
never reach 1.0 (an unbounded asymptote). At the user's explicit request,
`speed_saturation_kmh` (default 70 km/h) now marks the point where the score
reaches exactly 1.0 -- rising linearly from `breakpoints[2]` (0.85) up to 1.0
there, on the same slope as the `breakpoints[2]`-to-`saturation_kmh` segment.
70 km/h is a **researcher's design choice**: a realistic ceiling for ordinary
urban transit line commercial speeds (which rarely exceed 40-60 km/h even
for express service), not an artificially distant asymptote-avoidance value
like the earlier 150 km/h default.

**2026-08-12 update, at the user's explicit request** ("stop score cannot be
above 1"): `speed_score` is now capped at exactly 1.0 above `saturation_kmh`,
like every other sub-score in this package. An earlier design (2026-08-11)
left it deliberately uncapped so exceptionally fast service (e.g. intercity
rail well above 70 km/h) could keep scoring higher than an ordinary rapid
line -- but that let `stop_score` (a weighted geometric mean of four
sub-scores, `stop_score.py`) exceed 1.0 whenever `speed_score` did, breaking
the invariant that `stop_score`/`accessibility_score` are genuinely bounded
to [0, 1]. See SCORING.md §2/§4.
"""

from __future__ import annotations

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def speed_score(
    avg_speed_kmh: float,
    region: str = "global",
    breakpoints_kmh: tuple[float, float, float] | None = None,
    saturation_kmh: float | None = None,
) -> float:
    """Compute a [0, 1] mode/speed score from average operating speed.

    Speed bands (see parameters/07_speed_based_service_classification.md §1
    and §4; outer breakpoints anchored to the ITDP BRT Standard 2024's
    verbatim speed-deduction table as of the 2026-08-10 verification pass):
        < breakpoints[0] km/h (default 11.0, ITDP's steepest-penalty
            threshold): "slow/congested local" -> score near 0
        breakpoints[0] to breakpoints[1]: "ordinary local" -> linear 0-0.5
        breakpoints[1] to breakpoints[2]: "semi-rapid/priority" -> linear 0.5-0.85
        breakpoints[2] to saturation_kmh (default 20.0 to 70.0): "rapid/
            segregated-quality" -> linear 0.85-1.0
        > saturation_kmh: capped at exactly 1.0, like every other sub-score

    Args:
        avg_speed_kmh: Average operating (commercial) speed in km/h. Must be
            >= 0.
        region: Region key into `parameters.REGIONS` used to select default
            breakpoints if not explicitly overridden.
        breakpoints_kmh: Ascending 3-tuple of km/h breakpoints
            (slow/ordinary boundary, ordinary/semi-rapid boundary,
            semi-rapid/rapid boundary). Defaults to the region's
            `mode_speed_breakpoints_kmh`. The outer two are anchored to the
            ITDP BRT Standard's speed-deduction table; the middle one
            remains a researcher's interpolated design choice
            (SCORING.md §4), not directly literature-sourced.
        saturation_kmh: Speed at which the score reaches exactly 1.0.
            Defaults to the region's `speed_saturation_kmh` (70 km/h
            globally) -- a researcher's design choice anchored to a
            realistic urban-transit ceiling. The score is capped at 1.0
            above this value, see module docstring.

    Returns:
        A float in [0, 1], monotonically non-decreasing in avg_speed_kmh,
        equal to 1.0 exactly at and above `saturation_kmh`.

    Raises:
        ValueError: If avg_speed_kmh is negative.
    """
    if avg_speed_kmh < 0:
        raise ValueError(f"avg_speed_kmh must be >= 0, got {avg_speed_kmh}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    b0, b1, b2 = breakpoints_kmh if breakpoints_kmh is not None else params.mode_speed_breakpoints_kmh
    sat = saturation_kmh if saturation_kmh is not None else params.speed_saturation_kmh

    if avg_speed_kmh <= 0:
        return 0.0
    if avg_speed_kmh < b0:
        return 0.5 * (avg_speed_kmh / b0)
    if avg_speed_kmh < b1:
        return 0.5 + 0.35 * (avg_speed_kmh - b0) / (b1 - b0)
    if avg_speed_kmh < b2:
        return 0.85 + 0.15 * (avg_speed_kmh - b1) / (b2 - b1)
    # Bug found 2026-08-12 (visible as a "weird peak" in the map's
    # Discretization tab plot -- a real, non-monotonic *drop* right at
    # `b2`, e.g. 0.933 at 18 km/h -> 0.853 at 21 km/h): the segment above
    # already reaches exactly 1.0 at `b2` (0.85 + 0.15*(b2-b1)/(b2-b1) ==
    # 1.0), which is also ITDP's own real anchor -- this module's opening
    # docstring states "ITDP applies zero speed penalty above 20 km/h"
    # (the default `b2`). A later, separate change (2026-08-11) added
    # `saturation_kmh` to avoid a *different*, now-removed formula's
    # unbounded asymptote, but grafting a fourth segment onto *this*
    # breakpoint-anchored formula made it restart from a hardcoded 0.85
    # instead of continuing from the 1.0 the previous segment already
    # reached -- there's no headroom left to "rise to 1.0" a second time.
    # Once at/above `b2`, the score is already at its maximum; `sat` no
    # longer does anything for this breakpoint-based formula (kept as an
    # accepted parameter for API stability, silently unused below `b2`'s
    # value of 1.0).
    return 1.0
