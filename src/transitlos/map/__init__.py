"""Lightweight, tile-based folium map builder -- reusable across any transitLOS study.

`build_city_map` is column-agnostic: every field it renders (the score
column, circle-size/opacity candidate fields, census-level GeoDataFrames,
stats-panel data) is supplied by the caller, not hardcoded to any specific
study's schema (e.g. a particular census provider's column names). A study
with a different set of census columns, a different scoring column name, or
no census data at all works without changing this package -- only the
caller's own column lists change.
"""

from .build import (
    BACKGROUND_MAPS,
    DEFAULT_BACKGROUND,
    RDYLGN_HEX,
    build_city_map,
    build_download_panel_block,
    build_place_rank_panel_block,
)
from .routes import MODE_FALLBACK_COLOR, build_route_lines

__all__ = [
    "build_city_map",
    "build_route_lines",
    "build_place_rank_panel_block",
    "build_download_panel_block",
    "MODE_FALLBACK_COLOR",
    "RDYLGN_HEX",
    "BACKGROUND_MAPS",
    "DEFAULT_BACKGROUND",
]
