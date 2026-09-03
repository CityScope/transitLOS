"""Combines mode, speed, frequency, and reliability sub-scores into one stop-level score.

Implements SCORING.md §1.1's weighted geometric mean, extended to four
factors at the user's explicit request (mode and speed as two separate
scores rather than the single combined `speed_score` this project's own
prior research had settled on -- see `mode.py`'s module docstring for the
full rationale/caveats):

    stop_score = (mode_score)^w1 * (speed_score)^w2 * (frequency_score)^w3 * (reliability_score)^w4,
                 w1 + w2 + w3 + w4 = 1

The multiplicative (geometric-mean) combination -- rather than an additive
weighted sum -- is literature-grounded per SCORING.md §4: OECD/JRC composite
indicator methodology (Nardo et al. 2008; Munda & Nardo 2009) and every
institutional precedent reviewed (TfL PTAL, ÖV-Güteklassen, TCRP Transit
Capacity & Quality of Service Manual) treat service-quality dimensions
non-compensatorily, so a very poor sub-score cannot be masked by good ones.
The exact weights (default splits `mode`/`speed` evenly across the former
`mode_speed` factor's 1/3 share, `frequency`/`reliability` unchanged at 1/3
each) are a researcher's design choice, not empirically calibrated
(SCORING.md §2 rows w1/w2/w3; §4).
"""

from __future__ import annotations

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region

_WEIGHT_SUM_TOLERANCE = 1e-6


def stop_score(
    mode_score: float,
    speed_score: float,
    frequency_score: float,
    reliability_score: float,
    weights: tuple[float, float, float, float] | None = None,
    region: str = "global",
) -> float:
    """Compute the weighted geometric mean stop_score from four sub-scores.

    Args:
        mode_score: Sub-score in [0, 1] from `mode.mode_score`.
        speed_score: Sub-score in [0, 1] from `speed.speed_score` -- capped
            at 1.0 above its saturation point (2026-08-12, at the user's
            explicit request), like every other sub-score here, see
            `speed.py`'s module docstring.
        frequency_score: Sub-score in [0, 1] from `frequency.frequency_score`.
        reliability_score: Sub-score in [0, 1] from `reliability.reliability_score`.
        weights: (w1, w2, w3, w4) weights for (mode, speed, frequency,
            reliability) respectively, must sum to 1 within floating-point
            tolerance. Defaults to the region's default weights. See
            SCORING.md §2 rows w1/w2/w3.
        region: Region key into `parameters.REGIONS` used to select default
            weights if `weights` is not explicitly passed.

    Returns:
        A float in [0, 1]: the weighted geometric mean of the four
        sub-scores.

    Raises:
        ValueError: If any sub-score is outside [0, 1], or if the weights
            do not sum to 1 within tolerance.
    """
    for name, value in (
        ("mode_score", mode_score),
        ("speed_score", speed_score),
        ("frequency_score", frequency_score),
        ("reliability_score", reliability_score),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    warn_if_unvalidated_region(region)
    w1, w2, w3, w4 = weights if weights is not None else REGIONS[region].weights

    if abs((w1 + w2 + w3 + w4) - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got w1+w2+w3+w4={w1 + w2 + w3 + w4}")

    # Handle exact-zero sub-scores explicitly: 0**w for w>0 is 0, which
    # Python's ** already handles correctly, so a plain product-of-powers
    # is safe here without special-casing.
    return (mode_score**w1) * (speed_score**w2) * (frequency_score**w3) * (reliability_score**w4)
