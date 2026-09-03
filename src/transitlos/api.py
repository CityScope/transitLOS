"""Consolidated public surface for transitLOS.

Re-exports every user-facing function so callers can do
`from transitlos import api` (or `import transitlos`) and get everything
from one place, instead of having to know which submodule each function
lives in.
"""

from __future__ import annotations

from .equity import equity_statistics
from .h3_resample import level_of_service_to_h3
from .level_of_service import compute_level_of_service
from .network import prepare_street_network
from .population import cross_with_population
from .stop_scores import compute_stop_scores
from .stops import download_and_prepare_stops

__all__ = [
    "download_and_prepare_stops",
    "compute_stop_scores",
    "prepare_street_network",
    "compute_level_of_service",
    "cross_with_population",
    "level_of_service_to_h3",
    "equity_statistics",
]
