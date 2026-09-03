"""Top-level accessibility_score combining a stop's quality score with walk access.

Implements SCORING.md §1.2:

    accessibility_score = stop_score * D(walk_time)

where `stop_score` comes from `stop_score.stop_score` and `D(·)` is the
walk-time decay function in `walk_access.walk_decay`. **2026-08-12, at the
user's explicit request**: this is now a plain product of two independent
factors -- `stop_score` no longer feeds into `D(·)` itself (see
`walk_access.py`'s module docstring for why the earlier stop_score-scaled
decay double-counted stop quality). `stop_score` is also now genuinely
bounded to [0, 1] (`speed_score`'s saturation cap, `speed.py`), so
`accessibility_score` is bounded to [0, 1] too. This is the final per-stop/
per-trip metric the rest of the pipeline is expected to consume.
"""

from __future__ import annotations

from .walk_access import walk_decay
from ._warnings import warn_if_unvalidated_region


def accessibility_score(
    stop_score: float,
    walk_time_minutes: float,
    region: str = "global",
    t0_base_minutes: float | None = None,
    p: float | None = None,
) -> float:
    """Compute accessibility_score = stop_score * D(walk_time).

    Args:
        stop_score: The stop-level quality score, typically from
            `stop_score.stop_score`. Must be in [0, 1].
        walk_time_minutes: Walking time to the stop, in minutes. Must be
            >= 0.
        region: Region key into `parameters.REGIONS`. One of "global",
            "europe", "north_america", "global_south". Passing
            "global_south" emits a UserWarning per SCORING.md §3.3 (no
            adequate regional literature base for these parameters there).
        t0_base_minutes: Forwarded to `walk_access.walk_decay`. If None,
            uses the region's default.
        p: Forwarded to `walk_access.walk_decay`. If None, uses the region's
            default.

    Returns:
        A float in [0, 1]: the walk-discounted stop score.

    Raises:
        ValueError: If stop_score is outside [0, 1], or walk_time_minutes
            is negative.
    """
    if not 0.0 <= stop_score <= 1.0:
        raise ValueError(f"stop_score must be in [0, 1], got {stop_score}")

    warn_if_unvalidated_region(region)
    decay = walk_decay(
        walk_time_minutes,
        region=region,
        t0_base_minutes=t0_base_minutes,
        p=p,
    )
    return stop_score * decay
