"""The h3 grid and population rules shared by the studies (city science network, US, Spain). Each study holds its own `GridRules`
instance (city: `StudyParams.grid`; US / Spain: `code/study_params.py` `GRID`) so every value can be changed per study.

Grid
    A cell of the finest resolution exists when its centre is within `street_max_dist_m` of a street (scored or not; the street need not
    touch the cell). Its level of service is the best (Max) score of the street edges within reach of its centre, the nearest street as
    fallback. Coarser resolutions and census geometries aggregate the finest cells (counts summed, level of service population-weighted).
Inhabited cells
    A cell of the finest resolution is kept when its residents + jobs (any active source) reach `occupancy_min_per_km2` per km2 of the
    cell; other finest cells are uninhabited and dropped. Coarser cells and census geometries have no such rule.
Census -> cells
    A cell takes data from the one polygon holding its centre; the polygon's counts are split over its cells by built floor area
    (residential for residents' columns, non-residential for jobs; total, then equal split as fallbacks), the cells' floor areas rescaled to
    the polygon's (pycensus.dasymetric). Floor area: national cadasters (pycensus.cadaster), else GlobalBuildingAtlas.
Rescaling
    People and jobs of a source that land on no kept cell (off-road, or in an uninhabited cell) are put back with ONE factor per source and
    processed area (city AOI, US county, Spanish province): every kept cell is multiplied by total / kept. Raster sources (WorldPop, the
    built-surface estimate) are resampled onto the cells and scaled this way. Vector sources (census polygons, blocks) first spread each
    polygon over its own street cells when `within_source_polygon` (only a polygon without any street cell feeds the area-wide factor).
    Every rescale is logged ("[roadless] ..."), as a WARNING when more than `rescale_warn_share` of the source is affected.
Census polygons without a cell
    (city studies) a census polygon containing no cell centre goes whole to the nearest cell within `census_fallback_max_m`.
Built surface (GlobalBuildingAtlas, `pycensus.census.worldwide.gba`)
    built floor area = footprint x height / `floor_height_m`. The built-surface population keeps WorldPop where WorldPop >=
    `built_pop_min_ratio` x (people per built m2) x built m2, and uses the rate x built m2 elsewhere; outside the census the rate is the
    built-area-weighted rate of the `built_pop_k_nearest` nearest census polygons.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridRules:
    """The rules that decide which H3 cells exist and how census and grid sources are rescaled onto them (distances,
    occupancy thresholds, rescale warnings, fallbacks).
    """
    street_max_dist_m: float = 125.0       # finest cells exist within this distance (m) of a street; 125 = the old res-11 reach
    occupancy_min: float = 1.0             # a cell is inhabited with >= this many residents or jobs of any active source
    occupancy_min_per_km2: float = 10.0    # the finest cells only: inhabited with >= this many residents + jobs per km2 (coarser cells / census: no rule)
    rescale_warn_share: float = 0.05       # WARNING when more than this share of a source's people / jobs is rescaled
    within_source_polygon: bool = False    # True: vector sources spread each polygon over its own street cells before the area-wide factor
    census_fallback_max_m: float = 600.0   # city studies: a census polygon without a cell centre goes to the nearest cell within this
    floor_height_m: float = 3.0            # built floor area = footprint x height / this
    built_pop_min_ratio: float = 0.25      # WorldPop below this share of the built-surface expectation counts as missing
    built_pop_k_nearest: int = 5           # census polygons whose rates calibrate a pixel outside the census


DEFAULT = GridRules()
