"""Combines MRC, speed, and frequency sub-scores into one stop-level score.

**2026-09-29 aggregation-STRUCTURE revision (this revision).** `stop_score` is no longer a
flat (generalized-power-)mean of all three sub-scores. It is now a **nested/hierarchical**
combination -- scoring.md Section 1.2:

    stop_score = sqrt(MRC * V) * H

`MRC` (mode/reliability/comfort) and `V` (speed) are combined via a plain, equal,
UNWEIGHTED geometric mean between just the two of them (`sqrt(MRC * V)`) -- they are
treated as genuinely substitutable with one another, both describing "how good is the
ride once it arrives" (a faster vehicle can compensate for a slightly lower mode
category, and vice versa). `H` (headway/frequency) is then applied as a separate
multiplicative GATE on top of that sub-mean, not pooled into the same mean -- headway is
treated as categorically non-substitutable with the other two: a fast, comfortable
vehicle that essentially never comes is not rescued by being fast and comfortable. See
scoring.md Section 1.2 ("Why nested: MRC/V pooled, H gated separately") for the full
literature discussion (TfL PTAL's own Equivalent Doorstep Frequency formula structurally
dominated by frequency, mode only a minor additive adjustment; Liu, Deng & Zhang 2011's
conjoint-analysis weights, reliability highest; the broader frequency/reliability-
dominance elasticity literature) and the explicit honesty framing on what is
structurally-grounded (the qualitative case for treating H differently from MRC/V) vs.
this project's own reasoned partition choice (pooling MRC with V specifically, rather
than some other grouping -- no single source tests this exact partition).

**Important non-obvious property**: unlike a true generalized mean, this nested form is
NOT idempotent -- when `MRC = V = H = x`, `stop_score = x**2`, not `x`. A uniformly
"good" stop at `x=0.90` now scores `0.81`, not `0.90`. This is an honest, reported
consequence of gating (multiplying) rather than averaging H in; see scoring.md Section
1.2's verification table. It does not change H(.), V(.), MRC(.), their internal
constants, or make MRC/V any less bounded -- only the aggregation step changes.

**Prior forms kept for comparison/back-compatibility** (`mode="harmonic"` /
`mode="geometric"`), reproducing the pre-2026-09-29 flat generalized-power-mean of all
three sub-scores at `p=-1` (weighted harmonic mean, the immediately-prior default) and
`p=0` (plain geometric mean, the form before that) respectively, weights included. These
are not simply legacy cruft: scoring.md's own verification table for this revision
directly compares the new nested form against both, so the ability to reproduce their
exact prior numbers is load-bearing for that before/after/before-that accounting, and for
any future re-litigation of this choice.

**2026-09-29 aggregation-form revision (prior, superseded by the structure above)**: `p`
moved from `0` (plain geometric mean) to `-1` (harmonic mean) for the flat three-factor
mean, in response to a real production stop (mode=bus, MRC=0.800, V=0.8401,
headway=285.73 min so H=0.1312) that the geometric mean scored 0.4451 ("mediocre")
despite the ~4.75-hour headway; `p=-1` gave 0.2981 (-33%). See `mode="harmonic"` below
for how to reproduce this. That flat-mean fix is now itself superseded as the *default*
by the nested structure above, which handles the same motivating example even more
sharply (see scoring.md Section 1.2's verification table) by taking `H` out of the mean
entirely rather than only making the mean stricter.

**2026-09-28 model rewrite**: this is a genuine three-factor score, not the prior
version's four-factor extension (separate `mode_score` and `reliability_score` alongside
`speed_score`/`frequency_score`). Mode and reliability/comfort are now one combined
`MRC(.)` axis (see `mrc.py`'s module docstring for why no clean literature-backed *mode
premium net of reliability* exists), restoring scoring.md's original three-term
structure.

Equal (unweighted) combination of `MRC` and `V` is this project's own reasoned choice
(no literature reviewed differentially weights mode/comfort against speed specifically);
equal weights among all three sub-scores under the legacy `mode="harmonic"`/`"geometric"`
paths are, as before, the stated result of three independent literature searches
(VoT-ratio, MCDA/AHP, conjoint/BWS) that found no adoptable differential-weighting scheme
(scoring.md Section 1.2) -- not an unexamined placeholder.

**2026-09-29 hard cap at 1.0 (this revision, explicit user request: "no stop score can be
above 1").** `V(speed)` itself is still allowed to exceed 1.0 (up to 1.15, `speed.py`) --
a fast regional/commuter-rail line genuinely gets credited above the 25 km/h "excellent"
anchor, per `speed.py`'s own reasoning. But `stop_score` -- the thing actually consumed
downstream (aggregation weights, the walk-decay coupling `t0(stop_score)`, and every
display/UI surface) -- is a composite index, and this project's own convention (matching
`MRC`/`H`, both individually bounded to [0, 1]) is that the *composite* score reads on a
clean, comparable [0, 1]/percentage scale: `min(1.0, ...)` is applied to every
`stop_score` aggregation mode's output here, not left to spill over to ~1.05-1.07 just
because one input sub-score's own uncapped ceiling happened to exceed 1.0. This does NOT
change `V(speed)`'s own formula/ceiling, or any other sub-score -- only the final
composite. Before this cap: nested's uncapped ceiling was `sqrt(1.0*1.15)*1.0 ~= 1.0724`;
the legacy harmonic mean's was ~1.0455; the legacy geometric mean's was ~1.0477 -- all
three now clip to exactly 1.0 at that point instead.
"""

from __future__ import annotations

import math

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region

_WEIGHT_SUM_TOLERANCE = 1e-6

#: Default aggregation structure, scoring.md Section 1.2 (2026-09-29 structure revision).
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
    # non-compensatory boundary behavior scoring.md Section 1.2 requires. Handled
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
            geometric mean of `MRC` and `V`, per scoring.md Section 1.2) --
            passing `weights` with `mode="nested"` raises `ValueError` rather
            than silently ignoring it.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal
            weights.
        mode: Aggregation structure, scoring.md Section 1.2. One of:
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
