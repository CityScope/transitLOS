"""Walk-access decay `D(t; stop_score)`: discounts a stop's score by the walking time to it.

    D(t; stop_score) = 1                                              for 0 <= t <= t_plateau
    D(t; stop_score) = (1 + (t - t_plateau) / t0(stop_score))^(-p)    for t > t_plateau
    t0(stop_score)   = t0_base(mode) * (1 + k * stop_score)

A better stop decays more slowly (riders walk further to it), so `PTLOS = stop_score * D(t; stop_score)` is one joint function of
walk time and stop quality. `t0_base` depends on the mode (bus, rail, tram; see `mrc.mode_category`); the plateau (200 m, ~2.56 min
at 1.3 m/s) and `p = 1` are the same for every mode. Defaults: `parameters.RegionParameters`.
"""

from __future__ import annotations

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def t0_of_stop_score(
    mode: str,
    stop_score: float,
    region: str = "global",
    t0_base_minutes: float | None = None,
    k: float | None = None,
) -> float:
    """Compute `t0(stop_score) = t0_base(mode) * (1 + k * stop_score)`.

    Args:
        mode: One of `"bus"`, `"rail"`, `"tram"` (see `mrc.mode_category`).
        stop_score: The stop-level quality score. Must be >= 0 (may exceed
            1.0 up to ~1.048, see `stop_score.py`'s module docstring).
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal
            parameters.
        t0_base_minutes: Per-mode base scale override. If None, looked up
            from the region's `t0_base_minutes[mode]`.
        k: Quality-coupling constant override. If None, defaults to the
            region's `k_quality` (2.5).

    Returns:
        `t0`, in minutes.

    Raises:
        ValueError: If stop_score is negative, or mode is not recognized
            and no explicit `t0_base_minutes` override is given.
    """
    if stop_score < 0.0:
        raise ValueError(f"stop_score must be >= 0, got {stop_score}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    if t0_base_minutes is None:
        if mode not in params.t0_base_minutes:
            raise ValueError(f"mode must be one of {sorted(params.t0_base_minutes)}, got {mode!r}")
        t0_base_minutes = params.t0_base_minutes[mode]
    kk = k if k is not None else params.k_quality

    return t0_base_minutes * (1.0 + kk * stop_score)


def walk_decay(
    walk_time_minutes: float,
    stop_score: float,
    mode: str,
    region: str = "global",
    t_plateau_minutes: float | None = None,
    p: float | None = None,
    t0_base_minutes: float | None = None,
    k: float | None = None,
) -> float:
    """Compute the walk-access decay factor `D(t; stop_score)`.

    Flat at 1.0 up to `t_plateau_minutes`, then
    `(1 + (t - t_plateau) / t0(stop_score)) ^ (-p)` beyond it, where
    `t0(stop_score) = t0_base(mode) * (1 + k * stop_score)` -- see module
    docstring. `PTLOS = stop_score * walk_decay(...)`
    (`accessibility_score.accessibility_score` computes this product).

    Args:
        walk_time_minutes: Walking time to the stop, in minutes. Must be
            >= 0.
        stop_score: The stop-level quality score (from `stop_score.stop_score`).
            Must be >= 0.
        mode: One of `"bus"`, `"rail"`, `"tram"` -- selects `t0_base` (see
            `mrc.mode_category`). Ignored if `t0_base_minutes` is passed
            explicitly.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region shares identical, universal
            parameters.
        t_plateau_minutes: Walk time at/below which D(t)=1.0 exactly. If
            None, derived from the region's `t_plateau_distance_m` and
            `walking_speed_mps` (200 m -> ~2.56 min at 1.3 m/s).
        p: Shape parameter. If None, defaults to the region's `p_shape`
            (1.0).
        t0_base_minutes: Per-mode base scale override, forwarded to
            `t0_of_stop_score`.
        k: Quality-coupling constant override, forwarded to
            `t0_of_stop_score`.

    Returns:
        A float in (0, 1]: exactly 1.0 for walk_time_minutes <=
        t_plateau_minutes, monotonically decreasing and asymptotically
        approaching (but never reaching) 0 as walk_time_minutes grows
        beyond it.

    Raises:
        ValueError: If walk_time_minutes or stop_score is negative.
    """
    if walk_time_minutes < 0:
        raise ValueError(f"walk_time_minutes must be >= 0, got {walk_time_minutes}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    pp = p if p is not None else params.p_shape
    t_plateau = (
        t_plateau_minutes if t_plateau_minutes is not None
        else params.t_plateau_distance_m / params.walking_speed_mps / 60.0
    )

    if walk_time_minutes <= t_plateau:
        return 1.0

    t0 = t0_of_stop_score(
        mode, stop_score, region=region, t0_base_minutes=t0_base_minutes, k=k
    )
    return (1.0 + (walk_time_minutes - t_plateau) / t0) ** (-pp)


def walk_time_for_decay(
    target_decay: float,
    stop_score: float,
    mode: str,
    region: str = "global",
    t_plateau_minutes: float | None = None,
    p: float | None = None,
    t0_base_minutes: float | None = None,
    k: float | None = None,
) -> float:
    """Invert `walk_decay`: the walk time at which D(t; stop_score) == target_decay.

    Since `p` is fixed at 1.0 by default, this remains a closed-form inverse
    for the default shape parameter (and any other `p`, since the
    offset-power form `(1+x)^(-p)` inverts in closed form for any `p != 0`);
    `t0` itself already depends on `stop_score`/`mode`, which are required
    inputs here rather than something to additionally solve for. Kept
    alongside `walk_decay` so the two never drift out of sync. Used by
    `distance_matrix.build_intelligent_distance_matrix` to choose distance
    thresholds that land exactly on desired output access-score values.

    Args:
        target_decay: Desired `D(t; stop_score)` value, in (0, 1]. `1.0`
            returns `t_plateau_minutes` (the start of the plateau).
        stop_score: The stop-level quality score this curve is being
            inverted for (must match the `walk_decay` call being inverted).
        mode: One of `"bus"`, `"rail"`, `"tram"` (must match the `walk_decay`
            call being inverted).
        region, t_plateau_minutes, p, t0_base_minutes, k: Same meaning as in
            `walk_decay` -- must match the call whose curve is being
            inverted.

    Returns:
        The walk time (minutes) at which
        `walk_decay(t, stop_score, mode) == target_decay`.

    Raises:
        ValueError: If target_decay is not in (0, 1], or stop_score is
            negative.
    """
    if not 0.0 < target_decay <= 1.0:
        raise ValueError(f"target_decay must be in (0, 1], got {target_decay}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    pp = p if p is not None else params.p_shape
    t_plateau = (
        t_plateau_minutes if t_plateau_minutes is not None
        else params.t_plateau_distance_m / params.walking_speed_mps / 60.0
    )

    if target_decay >= 1.0:
        return t_plateau

    t0 = t0_of_stop_score(
        mode, stop_score, region=region, t0_base_minutes=t0_base_minutes, k=k
    )
    return t_plateau + t0 * (target_decay ** (-1.0 / pp) - 1.0)
