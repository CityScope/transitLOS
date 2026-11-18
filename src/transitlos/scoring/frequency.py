"""Headway score `H(headway)`: converts a service headway into a [0, 1] score.

    H(headway) = 1                                                     for headway <= 5 min
    H(headway) = (headway / 5)^(-eps5) * exp(-(q / 2) * ln^2(headway / 5))   for headway > 5 min
        with eps(headway) = p + q * ln(headway), eps5 = eps(5)

the closed-form solution of `d(ln H) / d(ln headway) = -eps(headway)`: the elasticity of the score to the headway grows with the
headway. Defaults: `parameters.RegionParameters` (`h_saturation_minutes`, `h_elasticity_p`, `h_elasticity_q`).
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

    Below `saturation_minutes` (5 min) the score is exactly 1.0. Above it, the score follows the log-linear-elasticity
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
