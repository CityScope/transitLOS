"""Shared helper for flagging use of the Global-South/LMIC region as unvalidated.

SCORING.md §3.3 is explicit that the Global-South/LMIC region reuses the
global defaults not because they have been shown to hold there, but because
no adequate regional literature base exists (aside from one value-of-time
study). This module centralizes the runtime warning so every scoring
function surfaces that caveat consistently rather than re-implementing it.
"""

from __future__ import annotations

import warnings

from .parameters import UNVALIDATED_REGIONS


def warn_if_unvalidated_region(region: str) -> None:
    """Emit a UserWarning if `region` is flagged as lacking a validated parameter set.

    Args:
        region: Region key as passed to a scoring function (e.g. "global",
            "europe", "north_america", "global_south").

    Returns:
        None. Emits a `warnings.warn` call as a side effect when applicable.
    """
    if region in UNVALIDATED_REGIONS:
        warnings.warn(
            f"region={region!r} reuses global default scoring parameters as an "
            "unvalidated extrapolation -- SCORING.md §3.3 found no adequate "
            "region-specific literature base for walk-distance thresholds, "
            "walking speed, headway elasticity, or decay parameters in this "
            "region. Treat results with caution.",
            UserWarning,
            stacklevel=3,
        )
