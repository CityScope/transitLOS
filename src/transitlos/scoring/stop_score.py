"""Combines the MRC, speed and frequency sub-scores into one stop-level score.

Default structure (`"nested"`):

    stop_score = sqrt(MRC * V) * H

`MRC` (mode / reliability / comfort) and `V` (speed) are pooled by an equal geometric mean (a faster vehicle can make up for a lower
mode category and vice versa); `H` (headway) multiplies the result, so a service that hardly ever comes is not rescued by being fast
or comfortable. The flat weighted geometric mean of the three is kept as an option (`mode=`).
"""

from __future__ import annotations

import math

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region

_WEIGHT_SUM_TOLERANCE = 1e-6

#: Default aggregation structure.
#: "nested" = sqrt(MRC * V) * H (the new default). "harmonic" / "geometric" reproduce the
#: prior flat generalized-power-mean of all three sub-scores at p=-1 / p=0 respectively,
#: kept for comparison/back-compatibility -- see module docstring.
DEFAULT_AGGREGATION_MODE = "nested"

#: Generalized (power) mean exponents backing the legacy flat-mean modes.
_LEGACY_MODE_EXPONENTS = {"harmonic": -1.0, "geometric": 0.0}


def _flat_generalized_mean(mrc_score, speed_score, frequency_score, weights, p):
    """The pre-2026-09-29 flat weighted generalized (power) mean of all three
    sub-scores, kept for `mode="harmonic"`/`mode="geometric"` back-compatibility.
    """
    w_mrc, w_v, w_h = weights

    if abs((w_mrc + w_v + w_h) - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got w_mrc+w_v+w_h={w_mrc + w_v + w_h}")

    # A sub-score of exactly 0 forces stop_score to 0 regardless of p -- the
    # non-compensatory boundary behaviour. Handled
    # explicitly because x**p for negative p and x=0 raises ZeroDivisionError in
    # Python rather than the mathematically-consistent limiting value of 0.
    if mrc_score == 0.0 or speed_score == 0.0 or frequency_score == 0.0:
        return 0.0

    if p == 0.0:
        return (mrc_score**w_mrc) * (speed_score**w_v) * (frequency_score**w_h)

    weighted_sum = w_mrc * mrc_score**p + w_v * speed_score**p + w_h * frequency_score**p
    return weighted_sum ** (1.0 / p)


def stop_score(
    mrc_score: float,
    speed_score: float,
    frequency_score: float,
    weights: tuple[float, float, float] | None = None,
    region: str = "global",
    mode: str = DEFAULT_AGGREGATION_MODE,
    p: float | None = None,
) -> float:
    """Compute `stop_score` from the three MRC/V/H sub-scores.

    Args:
        mrc_score: Sub-score from `mrc.mrc_score`. Must be in [0, 1].
        speed_score: Sub-score from `speed.speed_score`. Must be >= 0 (may
            exceed 1.0 up to 1.15, see module docstring).
        frequency_score: Sub-score from `frequency.frequency_score`. Must be
            in [0, 1].
        weights: Only meaningful for `mode="harmonic"`/`mode="geometric"`
            (the legacy flat three-way mean) -- `(w_mrc, w_v, w_h)`, must sum
            to 1 within floating-point tolerance; defaults to equal thirds.
            Must be left `None` for the default `mode="nested"`, which has no
            weight parameter by construction (a plain, equal, unweighted
            geometric mean of `MRC` and `V`) --
            passing `weights` with `mode="nested"` raises `ValueError` rather
            than silently ignoring it.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal
            weights.
        mode: Aggregation structure. One of:
            - `"nested"` (default, 2026-09-29): `stop_score = sqrt(MRC * V) * H`.
            - `"harmonic"`: the pre-2026-09-29 flat weighted harmonic mean
              (`p=-1`) of all three sub-scores -- kept for comparison/
              back-compatibility.
            - `"geometric"`: the flat weighted geometric mean (`p=0`) of all
              three sub-scores -- the form before the harmonic-mean revision,
              also kept for comparison/back-compatibility.
        p: Deprecated escape hatch for `mode in ("harmonic", "geometric")`
            only: an explicit generalized-mean exponent overriding the
            mode's default (-1.0 / 0.0 respectively), for callers that need
            some other point on the flat `M_p` family for comparison
            purposes. Ignored (must be `None`) for `mode="nested"`.

    Returns:
        A float in [0, 1]: `stop_score` under the requested aggregation
        structure, hard-capped at 1.0 (see module docstring's "2026-09-29
        hard cap" note -- `V(speed)` itself may still exceed 1.0, but the
        composite `stop_score` never does).

    Raises:
        ValueError: If mrc_score/frequency_score is outside [0, 1],
            speed_score is negative, `mode` is unrecognized, `weights`/`p`
            are passed with `mode="nested"`, or (for the legacy modes) the
            weights do not sum to 1 within tolerance.
    """
    if not 0.0 <= mrc_score <= 1.0:
        raise ValueError(f"mrc_score must be in [0, 1], got {mrc_score}")
    if speed_score < 0.0:
        raise ValueError(f"speed_score must be >= 0, got {speed_score}")
    if not 0.0 <= frequency_score <= 1.0:
        raise ValueError(f"frequency_score must be in [0, 1], got {frequency_score}")

    warn_if_unvalidated_region(region)

    if mode == "nested":
        if weights is not None:
            raise ValueError(
                "weights is not meaningful for mode='nested' (a plain, equal, "
                "unweighted geometric mean of MRC and V by construction) -- "
                "pass mode='harmonic' or mode='geometric' to use custom weights"
            )
        if p is not None:
            raise ValueError("p is not meaningful for mode='nested' -- pass mode='harmonic'/'geometric' instead")
        if mrc_score == 0.0 or speed_score == 0.0 or frequency_score == 0.0:
            return 0.0
        return min(1.0, math.sqrt(mrc_score * speed_score) * frequency_score)

    if mode not in _LEGACY_MODE_EXPONENTS:
        raise ValueError(f"mode must be one of 'nested', 'harmonic', 'geometric', got {mode!r}")

    w = weights if weights is not None else REGIONS[region].weights
    p_eff = _LEGACY_MODE_EXPONENTS[mode] if p is None else p
    return min(1.0, _flat_generalized_mean(mrc_score, speed_score, frequency_score, w, p_eff))
