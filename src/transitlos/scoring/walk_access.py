"""Walk-time decay function D(walk_time) used to discount stop_score by walk access.

**2026-08-12 rewrite, at the user's explicit request** ("this decay is way
too fast" / "I was thinking on multiplying stop score by distance score"):
replaces the previous `stop_score`-scaled offset-power form with a plain,
stop-independent decay curve:

    D(t) = 1                                for t <= t_sat
    D(t) = (1 + (t - t_sat) / t0) ^ (-p)     for t > t_sat

and `accessibility_score = stop_score * D(t)` (see `accessibility_score.py`)
is now a simple product of two independent factors, matching the classic
Hansen/gravity-model separation of "opportunity attractiveness" (stop_score)
from "impedance" (D(t)) -- see
`report/scoring/parameters/11_walk_time_valuation_and_decay.md` §1's own
foundational framing (Handy & Niemeier 1997).

Two changes from the 2026-08-11 design:

1. **Decoupled from `stop_score`.** The previous design scaled the decay
   curve's own scale parameter `t0` by `stop_score` (`t0 = t0_base * (1 + k
   * stop_score)`), so a high-quality stop got both a higher `stop_score`
   *and* a more generous walk-tolerance curve -- double-counting stop
   quality, since `stop_score` already folds in mode/speed/frequency/
   reliability (`stop_score.py`). `D(t)` is now a single curve shared by
   every stop regardless of quality; only the final multiplication by
   `stop_score` differentiates a great stop from a mediocre one at any
   given walk time. This also means `accessibility_score` is now genuinely
   bounded to `[0, stop_score]` (and since `stop_score` is itself capped at
   1.0, `accessibility_score` is bounded to `[0, 1]`), rather than
   depending on the interaction of two stop-quality-dependent terms.
2. **Recalibrated shape** (file 11 §11). The previous `t0_base=4.5 min`,
   `k=0.8`, `p=4.0` combination collapsed to roughly 10-25% of full value
   within a single 250 m step past the saturation plateau -- much steeper
   than either the directly-measured walking-distance literature or
   published real-world precedent (Walk Score/Transit Score's own decay
   benchmark: full value to ~400 m, ~12% remaining at 1600 m, ~0% by
   2400 m). The new default (`t0_base=9.0 min`, `p=1.2`) is calibrated so
   `D(800 m) ~= 0.5` -- TCRP 95's own 800 m *rail* planning threshold
   (deliberately used over the shorter 400 m bus threshold, at the user's
   explicit direction: since the curve is now mode-agnostic and shared, and
   a bus stop's typically lower `stop_score` already discounts it on its
   own, anchoring to the *rail* threshold avoids also penalizing rail/metro
   stops -- which riders demonstrably walk much farther to reach, file 11
   §9.2 -- for distances their real catchment comfortably covers).

Still an offset-power form, not plain negative-exponential: this is
literature-grounded via Kanafani (1983, quoted in Iacono/Krizek/
El-Geneidy 2010's peer-reviewed JTG paper) and Chen (2015, *Chaos, Solitons
& Fractals*), and directly corroborated by real data: El-Geneidy et al.
(2014, Montreal, N=16,014 trips) found negative exponential fits real bus
walking-distance data well (R²=0.95) but fits rail/metro data poorly
(R²=0.43) -- rail/metro riders tolerate a flatter, slower-thinning falloff
than a constant-hazard exponential can represent (file 11 §9.2). The
offset-power form's hazard rate `p/(t0+t)` *declines* with distance --
steep loss of willingness near the stop, progressively gentler loss for
already-long walks -- matching that finding.

`t0_base` and `p` are exposed engineering parameters (never silently
hardcoded), defaulted per-region from `parameters.py` -- but per the user's
earlier explicit request, every region defaults to the **global** params
for this decay shape (only `walking_speed_mps` and the soft
walk-distance-threshold reference point vary by region; see
`parameters.py`'s region docstrings), since no region-specific decay-rate
literature exists (file 11 §3).

The flat plateau up to `t_sat` (walk time corresponding to
`walk_distance_saturation_m`, default 250 m) is unchanged: a "close enough"
zone where D(t) = 1.0 exactly, matching the same saturating-plateau-then-
decay shape used by `frequency.py` and `speed.py` for consistency across
the model's saturating sub-scores.
"""

from __future__ import annotations

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def walk_decay(
    walk_time_minutes: float,
    t0_base_minutes: float | None = None,
    p: float | None = None,
    region: str = "global",
    saturation_minutes: float | None = None,
) -> float:
    """Compute the walk-time decay factor D(t).

    Flat at 1.0 up to `saturation_minutes`, then
    `(1 + (t - saturation_minutes) / t0) ^ (-p)` beyond it -- see module
    docstring. Independent of stop quality: multiply the result by
    `stop_score` yourself (or use `accessibility_score.accessibility_score`,
    which does this) to get the walk-discounted accessibility score.

    Args:
        walk_time_minutes: Walking time to the stop, in minutes. Must be
            >= 0.
        t0_base_minutes: Scale parameter (minutes) of the offset-power
            curve. If None, defaults to the region's `walk_t0_base_minutes`
            (see `parameters.py`) -- calibrated so `D(800 m) ~= 0.5`, TCRP
            95's rail planning threshold (file 11 §9.3/§11).
        p: Shape parameter controlling the offset-power curve's hazard-rate
            decline. If None, defaults to the region's
            `walk_decay_shape_p`. See file 11 §8.1/§11.
        region: Region key into `parameters.REGIONS` used to select default
            `t0_base_minutes`/`p`/`saturation_minutes` if not explicitly
            overridden. One of "global", "europe", "north_america",
            "global_south". Every region defaults to the same decay-shape
            parameters as "global" unless explicitly passed otherwise (see
            module docstring) -- only `walking_speed_mps` varies the
            effective `saturation_minutes` by region.
        saturation_minutes: Walk time at/below which D(t) = 1.0 exactly. If
            None, derived from the region's `walk_distance_saturation_m`
            and `walking_speed_mps` (default 250 m -> ~3.2 min at 1.3 m/s).

    Returns:
        A float in (0, 1]: exactly 1.0 for walk_time_minutes <=
        saturation_minutes, monotonically decreasing and asymptotically
        approaching (but never reaching) 0 as walk_time_minutes grows
        beyond it.

    Raises:
        ValueError: If walk_time_minutes is negative.
    """
    if walk_time_minutes < 0:
        raise ValueError(f"walk_time_minutes must be >= 0, got {walk_time_minutes}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    t0 = t0_base_minutes if t0_base_minutes is not None else params.walk_t0_base_minutes
    pp = p if p is not None else params.walk_decay_shape_p
    t_sat = (
        saturation_minutes if saturation_minutes is not None
        else params.walk_distance_saturation_m / params.walking_speed_mps / 60.0
    )

    if walk_time_minutes <= t_sat:
        return 1.0
    return (1.0 + (walk_time_minutes - t_sat) / t0) ** (-pp)


def walk_time_for_decay(
    target_decay: float,
    t0_base_minutes: float | None = None,
    p: float | None = None,
    region: str = "global",
    saturation_minutes: float | None = None,
) -> float:
    """Invert `walk_decay`: the walk time at which D(t) == target_decay.

    Closed-form inverse of `walk_decay`'s offset-power formula, kept
    alongside it so the two never drift out of sync. Used by
    `distance_matrix.build_intelligent_distance_matrix` to choose distance
    thresholds that land exactly on desired output access-score values,
    rather than an arbitrary flat convention (SCORING.md's "intelligent
    discretization" step).

    Args:
        target_decay: Desired `D(t)` value, in (0, 1]. `1.0` returns
            `saturation_minutes` (the start of the plateau).
        t0_base_minutes, p, region, saturation_minutes: Same meaning as in
            `walk_decay` -- must match the call whose curve is being
            inverted.

    Returns:
        The walk time (minutes) at which `walk_decay(t) == target_decay`.

    Raises:
        ValueError: If target_decay is not in (0, 1].
    """
    if not 0.0 < target_decay <= 1.0:
        raise ValueError(f"target_decay must be in (0, 1], got {target_decay}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    t0 = t0_base_minutes if t0_base_minutes is not None else params.walk_t0_base_minutes
    pp = p if p is not None else params.walk_decay_shape_p
    t_sat = (
        saturation_minutes if saturation_minutes is not None
        else params.walk_distance_saturation_m / params.walking_speed_mps / 60.0
    )

    if target_decay >= 1.0:
        return t_sat
    return t_sat + t0 * (target_decay ** (-1.0 / pp) - 1.0)
