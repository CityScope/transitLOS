"""Reliability scoring: converts headway coefficient-of-variation into a [0, 1] score.

SCORING.md §2 row "Reliability metric" states there is no single standardized
reliability formula in the literature, but that regularity indices
(Henderson et al. 1991; Trompet et al. 2011) and headway
coefficient-of-variation (CV = stddev(headway) / mean(headway)) are the
literature-consistent building blocks (parameters/04_reliability_headway_regularity.md).
The exact CV -> [0, 1] mapping used here is a design choice (not itself
sourced), implemented as a simple monotonically decreasing exponential-style
falloff: CV = 0 (perfectly even headways) maps to 1.0, and score decays
smoothly as CV grows.
"""

from __future__ import annotations

import math

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def reliability_score(
    headway_cv: float,
    region: str = "global",
    cv_scale: float | None = None,
) -> float:
    """Compute a [0, 1] reliability score from headway coefficient-of-variation.

    Args:
        headway_cv: Coefficient of variation of observed headways
            (stddev / mean), a unitless non-negative number. 0 means
            perfectly regular service; values around 0.5-1.0+ indicate
            substantial irregularity. See SCORING.md §2 "Reliability
            metric"; parameters/04_reliability_headway_regularity.md.
        region: Region key into `parameters.REGIONS` used to select the
            default `cv_scale` if not explicitly overridden.
        cv_scale: Scale parameter controlling how quickly score falls off
            with increasing CV. This mapping is a design choice
            (SCORING.md §4), not itself literature-derived -- only the
            underlying CV metric is literature-grounded.

    Returns:
        A float in [0, 1], monotonically decreasing in `headway_cv`.

    Raises:
        ValueError: If headway_cv is negative.
    """
    if headway_cv < 0:
        raise ValueError(f"headway_cv must be >= 0, got {headway_cv}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    scale = cv_scale if cv_scale is not None else params.reliability_cv_scale

    return math.exp(-headway_cv / scale)
