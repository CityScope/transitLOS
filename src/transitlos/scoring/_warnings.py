"""Shared helper for flagging use of the Global-South/LMIC region as unvalidated.

A region listed in `parameters.UNVALIDATED_REGIONS` reuses the global defaults without having been checked there; this module
centralizes the runtime warning so every scoring function raises it the same way (the set is currently empty).
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
            "unvalidated extrapolation (walk-distance thresholds, walking speed, "
            "headway elasticity and decay parameters were not checked for this "
            "region). Treat results with caution.",
            UserWarning,
            stacklevel=3,
        )
