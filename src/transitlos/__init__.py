"""transitLOS: transit level-of-service accessibility scoring.

Ties street-network accessibility (`UrbanAccessAnalyzer`), transit stops
(`pyGTFSHandler`), and population/census data (`pycensus`/`geohierarchy`)
together into a single transit level-of-service accessibility score. See
`report/scoring/SCORING.md` for the underlying scoring methodology
(`transitlos.scoring`) and `transitlos.api` for the consolidated public
surface re-exported below.
"""

from .api import (
    level_of_service_to_h3,
    compute_level_of_service,
    compute_stop_scores,
    cross_with_population,
    download_and_prepare_stops,
    equity_statistics,
    prepare_street_network,
)

__all__ = [
    "download_and_prepare_stops",
    "compute_stop_scores",
    "prepare_street_network",
    "compute_level_of_service",
    "cross_with_population",
    "level_of_service_to_h3",
    "equity_statistics",
]
