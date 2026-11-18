"""PTLOS of an origin: a stop's quality score times its mode- and quality-coupled walk access.

    PTLOS(stop_score, t) = stop_score * D(t; stop_score)

`stop_score` comes from `stop_score.stop_score`, `D` from `walk_access.walk_decay`. It measures service quality from an origin,
not full accessibility to destinations; the name `accessibility_score` is kept for existing callers.
"""

from __future__ import annotations

from .walk_access import walk_decay
from ._warnings import warn_if_unvalidated_region


def accessibility_score(
    stop_score: float,
    walk_time_minutes: float,
    mode: str,
    region: str = "global",
    t0_base_minutes: float | None = None,
    p: float | None = None,
    k: float | None = None,
) -> float:
    """Compute PTLOS = stop_score * D(walk_time; stop_score).

    Args:
        stop_score: The stop-level quality score, typically from
            `stop_score.stop_score`. Must be >= 0 (may exceed 1.0 up to
            ~1.048, see `stop_score.py`'s module docstring).
        walk_time_minutes: Walking time to the stop, in minutes. Must be
            >= 0.
        mode: One of `"bus"`, `"rail"`, `"tram"` -- selects the per-mode
            `t0_base` used by `walk_decay`'s `D(t; stop_score)` (see
            `mrc.mode_category`).
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region now shares identical, universal
            parameters (see `parameters.py`'s module docstring).
        t0_base_minutes: Forwarded to `walk_access.walk_decay`. If None,
            uses the region's per-mode default.
        p: Forwarded to `walk_access.walk_decay`. If None, uses the region's
            default.
        k: Quality-coupling constant, forwarded to `walk_access.walk_decay`.
            If None, uses the region's default (2.5).

    Returns:
        A float: the walk-discounted stop score (PTLOS). Non-negative;
        bounded above by `stop_score` itself, which itself may reach
        slightly above 1.0 (see `stop_score.py`).

    Raises:
        ValueError: If stop_score or walk_time_minutes is negative.
    """
    if stop_score < 0.0:
        raise ValueError(f"stop_score must be >= 0, got {stop_score}")

    warn_if_unvalidated_region(region)
    decay = walk_decay(
        walk_time_minutes,
        stop_score,
        mode,
        region=region,
        t0_base_minutes=t0_base_minutes,
        p=p,
        k=k,
    )
    return stop_score * decay
