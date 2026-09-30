"""Headway/frequency scoring: converts a service headway into a [0, 1] score.

Implements `H(headway)`, scoring.md §2.2: a single, continuous closed-form
curve whose local elasticity magnitude grows log-linearly with headway, fit
directly to a real, tiered empirical elasticity table (TCRP Report 95 Ch.9
Table 9-2, source Lago, Mayworm & McEnroe 1981):

    H(headway) = 1.0                                            for headway <= 5 min
    H(headway) = (headway/5)^(-eps5) * exp(-(q/2)*ln^2(headway/5))   for headway > 5 min
        where eps(headway) = p + q*ln(headway), eps5 = eps(5)

This is the exact closed-form solution of `d(ln H)/d(ln headway) =
-eps(headway)` when the local elasticity magnitude is log-linear in headway.
`p`/`q` are a least-squares fit to three geometric-mean-of-band-edges
representative points (7.07, 22.36, 70.71 min) derived from TCRP 95's three
elasticity tiers (<10 min: -0.22; 10-50 min: -0.46; >50 min: -0.58).

This is a genuinely different design from the prior version of this module
(a flat plateau + plain negative-exponential tail, a shape-only design
choice with no direct empirical fit) -- see scoring.md §2.2 for the full
derivation, why log-linear elasticity growth beats a fixed exponent or a
quadratic-in-headway growth rate, and the honesty caveats on the underlying
1960s-80s North American bus data (small per-tier n, wide SDs).

Cross-checked against `transitLOS/documentation/scoring/scripts/
compare_methods.py`'s own `H_headway` reference implementation.
"""

from __future__ import annotations

import math

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def frequency_score(
    headway_minutes: float,
    region: str = "global",
    saturation_minutes: float | None = None,
    elasticity_p: float | None = None,
    elasticity_q: float | None = None,
) -> float:
    """Compute `H(headway)`, a [0, 1] frequency score from a service headway.

    Below `saturation_minutes` the score is exactly 1.0 (scoring.md §2.2
    Anchor: 5 min). Above it, the score follows the log-linear-elasticity
    closed form described in the module docstring.

    Args:
        headway_minutes: Scheduled or average headway in minutes. Must be
            >= 0.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region now shares identical, universal
            parameters (see `parameters.py`'s module docstring).
        saturation_minutes: Headway (min) at/below which score is 1.0.
            Defaults to the region's `h_saturation_minutes` (5.0).
        elasticity_p: Intercept of `eps(headway) = p + q*ln(headway)`.
            Defaults to the region's `h_elasticity_p`.
        elasticity_q: Slope of the same fit. Defaults to the region's
            `h_elasticity_q`.

    Returns:
        A float in (0, 1]: 1.0 at/below the saturation headway, decaying
        smoothly (but never reaching exactly 0) as headway grows.

    Raises:
        ValueError: If headway_minutes is negative.
    """
    if headway_minutes < 0:
        raise ValueError(f"headway_minutes must be >= 0, got {headway_minutes}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    sat = saturation_minutes if saturation_minutes is not None else params.h_saturation_minutes
    p = elasticity_p if elasticity_p is not None else params.h_elasticity_p
    q = elasticity_q if elasticity_q is not None else params.h_elasticity_q

    if headway_minutes <= sat:
        return 1.0

    eps_sat = p + q * math.log(sat)
    log_ratio = math.log(headway_minutes / sat)
    ln_h = -eps_sat * log_ratio - (q / 2.0) * log_ratio * log_ratio
    return math.exp(ln_h)
