"""Lightweight, tile-based folium map builder shared by every city.

Uses `geohierarchy.maps.folium.render.HierarchyMap`/`MultiHierarchyMap` as the
tile-based rendering backbone (never raw embedded GeoJSON): one zoom-banded
level per H3 resolution (or, for US/census cities, per census geometry level)
so only one resolution's tiles are ever fetched at a time.

Three kinds of layer:
  - Shape (mutually exclusive, native Leaflet radio via `MultiHierarchyMap`'s
    base-layer groups): `"hexagons"`, `"circles"`, and -- only for cities
    with census data -- `"census"` (itself a multi-level hierarchy from the
    coarsest to the finest available census geometry).
  - Streets (independent checkbox overlay): always rendered above the shape
    layer, togglable on its own regardless of which shape is active.
  - Development opportunity (independent checkbox overlay, US/census cities
    only): h3 hexagon borders only, shown above streets when active.

A small custom control panel (opacity-by-field, circle-size-by-field,
background-map choice, B&W basemap) and a legend are injected as extra
Leaflet controls stacked below the native layer control. All level_of_service
coloring uses a fixed red (bad)-yellow-green (good) scale.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import warnings
import json
import os
import re
from html import escape
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd


def _trim_malloc() -> None:
    """`gc.collect()` + glibc `malloc_trim(0)`.

    Real 2026-08-26 Boston OOM fix, part 2: freeing a shape's
    `GeoHierarchy` data (`gc.collect()`, including the one already inside
    `HierarchyMap.build(free_level_data=True)`) drops the Python-level
    references, but glibc's allocator does not generally hand freed heap
    memory back to the OS on its own -- fragmented arenas from millions of
    small Polars/numpy/shapely allocations stay resident as RSS regardless.
    Confirmed live: after switching `build_city_map` to build+free each
    shape (hexagons/circles/census/streets) one at a time instead of all
    at once, Boston's RSS climb became smoother/more gradual but still
    monotonically climbed to the same ~14-15GB peak and got OOM-killed --
    i.e. `gc.collect()` alone was freeing Python objects without actually
    shrinking the process's memory footprint. `ctypes.CDLL(None).malloc_trim(0)`
    (glibc-only; a no-op ImportError/AttributeError anywhere else, e.g. a
    non-glibc libc, is swallowed) asks the allocator to release contiguous
    free pages back to the OS, which is exactly what's needed between
    shapes here. Safe to call unconditionally -- it never frees live data,
    only unused arena pages.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
import shapely

from .routes import MODE_FALLBACK_COLOR
from .pop_chunks import GEOID_CHUNK_DIRNAME, POP_CHUNK_DEG, POP_CHUNK_DIRNAME

# Item 6 (2026-08-16): which census level gets the post-Compute live-recolor
# treatment. Block group is the finest/most-common "census geometries" level
# and the one `transitlos.map.pop_chunks.build_geoid_chunks_from_h3_and_census`
# is meant to be run against; tract/county are a documented followup (they'd
# need their own GEOID<->h3-cell chunks, or a rollup of this level's mapping).
CENSUS_LIVE_RECOLOR_LEVEL = "blockgroup"

# 9-stop ColorBrewer RdYlGn (bad=red -> good=green), matching level_of_service's
# semantics directly. Hardcoded (not looked up via branca at build time) so
# the exact same stops are usable both server-side (matplotlib-free) and
# baked into the client-side JS interpolation function below.
RDYLGN_HEX = [
    "#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
    "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850",
]

# Fields never offered in the circle-size / opacity field dropdowns even
# though they may be numeric (identifiers, the score itself -- which already
# drives color, and the development-opportunity flag -- which isn't numeric).
_FIELD_EXCLUDE = {
    "h3_cell", "edge_id", "geo_id", "geometry", "equity_flag",
    # Real, but excluded from ANOVA/regression/opacity/circle-size
    # specifically: a handful of census sources (CBS/Israel, Eustat/Euskadi)
    # publish their OWN per-locality "population density" field alongside
    # this pipeline's own `pop_density` (population/area_km2, computed
    # uniformly per h3 cell from the same population source used
    # everywhere else on the map). Both are real, but they measure the
    # same concept at different granularities (one real, published,
    # locality-broadcast value vs. one per-cell dasymetric value) and
    # humanize to near-identical labels ("Population density" vs "Pop
    # density") -- real bug report: Beersheba's ANOVA showed both as
    # separate, confusingly-similar rows. `pop_density` is always present
    # and is the one every other part of this pipeline (regression,
    # equity-flag split) already uses, so it's the one kept eligible here.
    "cbs_populationDensity", "eustat_populationDensity",
}

# Every country-prefixed population column this pipeline (or CS_transitLOS's
# `_apply_map_field_policy`) may leave on a level's GeoDataFrame -- mirrors
# `__popupPopulationValue`'s JS candidate list below (which itself mirrors
# `_population_column` in CS_transitLOS's pipeline.py) and pipeline.py's own
# `CENSUS_COLUMN_PREFIXES`. Used by `_popup_fields` to guarantee population
# always reaches the tile's properties even when `_FIELD_EXCLUDE` has grown
# "population" (see that constant's docstring below and
# `_apply_map_field_policy`'s docstring in pipeline.py) -- real bug
# (2026-08-30, live Hamburg map): `_apply_map_field_policy` only promised to
# remove `population` from the four dropdown *selectors*, but `_popup_fields`
# (which decides which columns are even written into the level's vector
# tiles at all, not just which are offered as dropdown choices) reused the
# same `_numeric_field_candidates`/`_FIELD_EXCLUDE`-filtered list, so
# `population` silently never reached the tile at all -- every popup showed
# a real `pop_density` (computed from population before the exclusion) next
# to a null "Population" row, since the raw count itself was never baked
# into the tile to look up.
_POPULATION_COLUMN_CANDIDATES = (
    "population", "acs5_population", "acs3_population", "acs1_population",
    "dhc_population", "inegi_population", "ine_population", "ine_cl_population",
    "cbs_population", "eustat_population", "statcan_population", "moi_population",
    "estadisticaad_population", "destatis_population", "ba_population",
    "worldpop_population",
)

# A handful of well-known, no-API-key-required tile sources for the
# background-map dropdown.
BACKGROUND_MAPS = {
    "positron": {
        "label": "CartoDB Positron (light)",
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "dark_matter": {
        "label": "CartoDB Dark Matter",
        "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    },
    "osm": {
        "label": "OpenStreetMap",
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "&copy; OpenStreetMap contributors",
    },
    "esri_imagery": {
        "label": "Esri World Imagery (satellite)",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri, Maxar, Earthstar Geographics",
    },
    "google_roads": {
        "label": "Google Maps",
        "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "attribution": "&copy; Google",
    },
    "google_satellite": {
        "label": "Google Satellite",
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attribution": "&copy; Google",
    },
    "google_hybrid": {
        "label": "Google Hybrid",
        "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attribution": "&copy; Google",
    },
}
DEFAULT_BACKGROUND = "positron"

# Zoom at or above which each mode's transit lines become visible. Rail is
# the coarsest, city-shaping network so it appears first; bus is by far the
# densest and would turn the map into spaghetti at anything but close zoom.
ROUTE_MODE_MIN_ZOOM = {"rail": 12, "tram": 13, "bus": 14}

# Zoom at or above which the optional per-stop text annotations ("stop names"
# / "routes" checkboxes) render. Deliberately high: these are per-stop DOM
# labels, and below this zoom a real city's stop density makes them both
# unreadable and slow.
STOP_ANNOTATION_MIN_ZOOM = 16

# Line weight per mode -- rail heaviest, bus lightest, so the network
# hierarchy reads at a glance even when all three modes are on.
_ROUTE_MODE_WEIGHT = {"rail": 4.0, "tram": 3.0, "bus": 2.0}


def _color_interp_js() -> str:
    """JS function `rdylgn(t)`: linear-interpolate `t` in `[0,1]` across `RDYLGN_HEX`."""
    stops = json.dumps(RDYLGN_HEX)
    return (
        "function rdylgn(t) {\n"
        f"  var stops = {stops};\n"
        "  t = Math.max(0, Math.min(1, isFinite(t) ? t : 0));\n"
        "  var scaled = t * (stops.length - 1);\n"
        "  var i = Math.floor(scaled);\n"
        "  var frac = scaled - i;\n"
        "  if (i >= stops.length - 1) return stops[stops.length - 1];\n"
        "  function hex2rgb(h) { h = h.replace('#',''); return [parseInt(h.substr(0,2),16), parseInt(h.substr(2,2),16), parseInt(h.substr(4,2),16)]; }\n"
        "  var a = hex2rgb(stops[i]), b = hex2rgb(stops[i+1]);\n"
        "  var rgb = [0,1,2].map(function(k) { return Math.round(a[k] + (b[k]-a[k])*frac); });\n"
        "  return 'rgb(' + rgb.join(',') + ')';\n"
        "}"
    )


# --- Client-side field derivation -------------------------------------------
#
# Every fill-styled layer reads a feature's value for the selected field
# straight off its vector-tile properties. That breaks in two real,
# separately-observed ways (both reproduced headless on 2026-08-13):
#
#  1. CENSUS geometries carry only the RAW census counts (`acs_workers`,
#     `acs_households`, ...) -- the pipeline's derived share/rate columns are
#     materialized onto the H3 grid only (`prepare_grid_for_map`), never onto
#     the census levels. So "Opacity by acs_transit_share" found
#     `properties['acs_transit_share'] == null` on every census polygon and
#     fell back to the flat base opacity: the control looked completely dead
#     on that shape while working fine on hexagons/circles.
#  2. UNIT DRIFT between the tiles and the build-time domains: `pop_density`
#     is people/km2 in the current pipeline but people/m2 in tiles baked
#     before that change. A ~1e6 mismatch against the Python-computed
#     percentile domain clamps EVERY feature to the same end of the ramp --
#     which is exactly why "Circle size by" looked like it did nothing (its
#     default field is a density, so every circle sat at the minimum radius).
#
# Deriving the value from the raw inputs the feature *does* carry fixes both
# at once and is unit-stable by construction: the formula here produces the
# same number the pipeline's own column does, so it agrees with the
# build-time domain no matter what an older tile happens to have baked.
# A derivation wins over the stored column whenever its inputs are present
# (that is the point in case 2); when they aren't, the stored value is used
# unchanged, and fields with no derivation are unaffected.
#
# Each entry is `field -> (numerator alternatives, denominator alternatives,
# scale)`; the first present alternative wins.
_DERIVED_FIELDS: Dict[str, tuple] = {
    # people per km2 (area_m2 -> km2 is the 1e6 factor)
    "pop_density": (("population", "acs_population"), ("area_m2",), 1e6),
    "acs_renter_rate": (("acs_households_renter",), ("acs_households",), 1.0),
    "acs_owner_rate": (("acs_households_owner",), ("acs_households",), 1.0),
    "acs_rent_burden_rate": (("acs_rent_burdened",), ("acs_rent_burden_population",), 1.0),
    "acs_poverty_rate": (("acs_poverty_below_1_00",), ("acs_poverty_population",), 1.0),
    "acs_labor_force_rate": (("acs_labor_force",), ("acs_population_16plus",), 1.0),
    "acs_unemployment_rate": (("acs_unemployed",), ("acs_labor_force",), 1.0),
    "acs_worker_rate": (("acs_workers",), ("acs_population",), 1.0),
    "acs_transit_share": (("acs_workers_transit",), ("acs_workers",), 1.0),
    "acs_car_commute_share": (("acs_workers_car",), ("acs_workers",), 1.0),
    "acs_walk_commute_share": (("acs_workers_walk",), ("acs_workers",), 1.0),
    "acs_bike_commute_share": (("acs_workers_bike",), ("acs_workers",), 1.0),
    "acs_car_free_rate": (("acs_vehicles_households_0",), ("acs_households",), 1.0),
    "acs_bachelors_rate": (("acs_education_bachelors_plus",), ("acs_population_25plus",), 1.0),
    "acs_less_than_hs_rate": (("acs_education_less_than_hs",), ("acs_population_25plus",), 1.0),
    "dhc_white_share": (("dhc_population_white",), ("dhc_population",), 1.0),
    "dhc_black_share": (("dhc_population_black",), ("dhc_population",), 1.0),
    "dhc_asian_share": (("dhc_population_asian",), ("dhc_population",), 1.0),
    "dhc_native_share": (("dhc_population_native",), ("dhc_population",), 1.0),
    "dhc_other_race_share": (("dhc_population_other_race",), ("dhc_population",), 1.0),
    "dhc_hispanic_share": (("dhc_population_hispanic",), ("dhc_population",), 1.0),
    "dhc_female_share": (("dhc_population_female",), ("dhc_population",), 1.0),
    "dhc_male_share": (("dhc_population_male",), ("dhc_population",), 1.0),
    "dhc_children_share": (("dhc_population_under_18",), ("dhc_population",), 1.0),
    "dhc_elderly_share": (("dhc_population_over_65",), ("dhc_population",), 1.0),
}


# Bug fix (user, verbatim spec, 2026-08-30): "I want always the absolute
# value and in ( ) the relative value except where it does not make sense".
# `_DERIVED_FIELDS` above only covers US ACS/DHC columns (~23 pairs) -- every
# non-US country's real share columns (Andorra's `foreign_born_share`,
# WorldPop's `worldpop_male_share`, Mexico's `inegi_catholic_share`, Israel's
# `cbs_jewish_share`, ...) and even several US ones (`education_*_share`,
# `car_ownership_rate`, ...) had no absolute-numerator pairing at all, so
# they always rendered as their own disconnected row instead of being paired
# with their real absolute count column -- the exact bug the user reported
# live (Andorra popup: "Population 10.5" / "Foreign born share (%) 57.4%" /
# "Foreign born population 6.1" all as separate unpaired rows). This is the
# `_shape_popup_js` popup's OWN copy (absolute-column -> relative-field-key)
# of `code.pipeline.SHARE_COLUMNS`' numerator/relative-field pairing --
# transitLOS can't import that CS_transitLOS-application module (wrong
# dependency direction: pipeline.py imports FROM this package, not the
# reverse), so this is a parallel constant, same pattern as every other
# CS_transitLOS-derived vocabulary list already duplicated this session
# (`combined_map.py`'s `WEIGHT_COLUMN_CANDIDATES`/`ANOVA_REGRESSION_COLUMNS`)
# -- generated by walking `SHARE_COLUMNS`' every `(numerator, denominator)`
# candidate per relative key and keeping numerator -> key (first-seen wins
# when the same numerator name is a candidate for more than one relative
# key, which does not currently happen but is a safe tie-break if it ever
# does). `RATE_SOURCE_COLUMNS` entries (a rate already published directly by
# the source, e.g. `cbs_academic_degree_rate`) are deliberately NOT included
# here -- there is no absolute-count column to pair them with at all, which
# is exactly the user's own "except where it does not make sense" carve-out;
# those rows correctly stay standalone.
_POPUP_ABS_TO_REL: Dict[str, str] = {
    "acs5_householdsRenter": "acs_renter_rate",
    "acs5_householdsOwner": "acs_owner_rate",
    "acs5_rentBurdened": "acs_rent_burden_rate",
    "acs5_vehiclesHouseholds0": "acs_car_free_rate",
    "acs5_povertyBelow100": "acs_poverty_rate",
    "laborForce": "labor_force_rate",
    "unemployedResidents": "unemployment_rate",
    "employedResidents": "employment_rate",
    "acs5_workers": "acs_worker_rate",
    "publicTransportCommuters": "transit_share",
    "carCommuters": "car_commute_share",
    "walkCommuters": "walk_commute_share",
    "bikeCommuters": "bike_commute_share",
    "ba_pendler_einpendler": "commuter_in_share",
    "ba_pendler_sameGemeindeWorkers": "local_worker_share",
    "acs5_educationBachelorsPlus": "acs_bachelors_rate",
    "acs5_educationLessThanHs": "acs_less_than_hs_rate",
    "acs5_educationHighSchool": "acs_high_school_rate",
    "acs5_educationSomeCollege": "acs_some_college_rate",
    "educationPrimaryOrLessPopulation": "education_primary_or_less_share",
    "educationSecondaryPopulation": "education_secondary_share",
    "educationUniversityPopulation": "education_university_share",
    "educationOtherTertiaryPopulation": "education_other_tertiary_share",
    "foreignBornPopulation": "foreign_born_share",
    "eustat_establishments": "eustat_establishments_rate",
    "eustat_birthsPeriod2021_2025": "eustat_birth_rate_2021_2025",
    "eustat_deathsPeriod2021_2025": "eustat_death_rate_2021_2025",
    "statcan_renterHouseholds": "statcan_renter_rate",
    "statcan_ownerHouseholds": "statcan_owner_rate",
    "moi_minorityEthnicityPopulation": "moi_minority_ethnicity_share",
    "dhc_whitePopulation": "dhc_white_share",
    "dhc_blackPopulation": "dhc_black_share",
    "dhc_asianPopulation": "dhc_asian_share",
    "dhc_nativePopulation": "dhc_native_share",
    "dhc_otherRacePopulation": "dhc_other_race_share",
    "dhc_hispanicPopulation": "dhc_hispanic_share",
    "dhc_nonwhitePopulation": "dhc_nonwhite_share",
    "dhc_age18to64Population": "dhc_working_age_share",
    "femalePopulation": "female_share",
    "malePopulation": "male_share",
    "under18Population": "children_share",
    "over65Population": "elderly_share",
    "adultPopulation": "adult_share",
    "minorityEthnicityPopulation": "minority_ethnicity_share",
    "religiousPopulation": "religious_share",
    "inegi_populationCatholic": "inegi_catholic_share",
    "inegi_populationNoReligion": "inegi_no_religion_share",
    "inegi_populationEvangelicalProtestant": "inegi_evangelical_share",
    "inegi_populationOtherReligion": "inegi_other_religion_share",
    "inegi_populationWithHealthcare": "inegi_healthcare_coverage_rate",
    "inegi_populationDisability": "inegi_disability_rate",
    "inegi_dwellingsInternet": "inegi_internet_rate",
    "inegi_dwellingsComputer": "inegi_computer_rate",
    "inegi_dwellingsCellphone": "inegi_cellphone_rate",
    "inegi_occupiedDwellings": "inegi_occupancy_rate",
    "carHouseholds": "car_ownership_rate",
    "ine_cl_urbanPopulation": "ine_cl_urban_share",
    "ine_cl_ruralPopulation": "ine_cl_rural_share",
    "cbs_jewishPopulation": "cbs_jewish_share",
    "cbs_muslimPopulation": "cbs_muslim_share",
    "cbs_christianPopulation": "cbs_christian_share",
    "cbs_druzePopulation": "cbs_druze_share",
    "cbs_otherReligionPopulation": "cbs_other_religion_share",
    "cbs_secularPopulation": "cbs_secular_share",
    "cbs_traditionalReligiosityPopulation": "cbs_traditional_religiosity_share",
    "cbs_religiousObservantPopulation": "cbs_religious_observant_share",
    "cbs_haredimPopulation": "cbs_haredim_share",
    "cbs_otherReligiosityPopulation": "cbs_other_religiosity_share",
    "worldpop_malePopulation": "worldpop_male_share",
    "worldpop_femalePopulation": "worldpop_female_share",
    "worldpop_under18Population": "worldpop_children_share",
    "worldpop_over65Population": "worldpop_elderly_share",
    "worldpop_adultPopulation": "worldpop_adult_share",
    "worldpop_urbanPopulation": "worldpop_urban_share",
    "urbanPopulation": "urban_share",
}


def _field_value_js() -> str:
    """JS `fieldValue(properties, field)`: the ONE way any layer reads a field off a feature.

    See `_DERIVED_FIELDS` for why a derivation is preferred over the stored
    column. Emitted as a top-level `function` declaration (a real global,
    same mechanism as `rdylgn`) so every independently-injected script can
    call it.
    """
    table = json.dumps({k: [list(v[0]), list(v[1]), v[2]] for k, v in _DERIVED_FIELDS.items()})
    return (
        f"var __DERIVED_FIELDS = {table};\n"
        "function __firstNumProp(properties, names) {\n"
        "  for (var i = 0; i < names.length; i++) {\n"
        "    var v = properties[names[i]];\n"
        "    if (v != null && !isNaN(v)) return Number(v);\n"
        "  }\n"
        "  return null;\n"
        "}\n"
        "function fieldValue(properties, field) {\n"
        "  if (!properties || !field) return null;\n"
        "  var d = __DERIVED_FIELDS[field];\n"
        "  if (d) {\n"
        "    var num = __firstNumProp(properties, d[0]), den = __firstNumProp(properties, d[1]);\n"
        "    if (num != null && den != null && den > 0) return num / den * d[2];\n"
        "  }\n"
        "  var v = properties[field];\n"
        "  return (v == null || isNaN(v)) ? null : Number(v);\n"
        "}"
    )


def _vectorgrid_click_polyfill_js() -> str:
    """Restore `L.DomEvent.fakeStop`, without which NO vector-tile click ever fires.

    Leaflet.VectorGrid 1.3.0 (the pinned CDN build every layer here renders
    through) ends `L.Canvas.Tile._onClick` with::

        L.DomEvent.fakeStop(e);
        this._fireEvent([clickedLayer], e);

    `fakeStop` was a Leaflet <=1.5 API and no longer exists in the Leaflet
    1.9 build folium ships, so that line throws `TypeError:
    L.DomEvent.fakeStop is not a function` *before* `_fireEvent` runs. The
    hit test itself succeeds -- verified headless on 2026-08-13 by patching
    `_onClick` and logging: `containsHits: 1` on every click, immediately
    followed by the TypeError -- so the symptom is not "clicks miss the
    polygon" but "the click is found and then silently discarded inside a
    DOM event handler". That is why clicking a hexagon/circle/census shape
    did nothing at all, no matter which popup mechanism was configured.

    The modern equivalent is still present as the private `_fakeStop`
    (it just sets `e._stopped`), so aliasing it restores the exact old
    behavior. Emitted before any layer is created, and idempotent.
    """
    return (
        "if (window.L && L.DomEvent && !L.DomEvent.fakeStop) {\n"
        "  L.DomEvent.fakeStop = L.DomEvent._fakeStop || function(e) { e._stopped = true; };\n"
        "}"
    )


def _shape_popup_js() -> str:
    """JS `__shapePopupHtml(properties)`: the click popup for hexagons/circles/census.

    Wired in via geohierarchy's `popup_js` hook (which replaces its default
    field table) on all three shape hierarchies, so one implementation serves
    whichever shape the "Shape" dropdown currently shows. `popup_fields` is
    still passed alongside it -- that list is what decides which properties
    get baked into the vector tiles in the first place; `popup_js` only
    decides how they are DISPLAYED.

    Layout mirrors the stop popup's conventions (bold title, label/value
    table rows, same fonts/colors): level_of_service and population are pulled
    out as a fixed header pair, and every remaining variable goes into a
    height-capped, scrollable table with its own slider, so a geometry
    carrying 40+ census columns cannot grow the popup off-screen.
    """
    return (
        # Item 8: a share/rate/ratio/pct field is stored as a raw 0..1
        # decimal (see `_DERIVED_FIELDS`) but that is not how a person reads
        # it -- shown as a percentage instead, with a "%" unit, wherever a
        # key name marks it as one. `density` is deliberately excluded: it is
        # a relative field by the same naming convention but is NOT a 0..1
        # share (see `is_relative_field`/`_RELATIVE_SUFFIXES`).
        "function __isPercentField(key) {\n"
        "  var k = String(key || '').toLowerCase();\n"
        "  return /(share|rate|ratio|pct|percent)$/.test(k);\n"
        "}\n"
        # Item 11: mirrors `field_unit_suffix` in build.py -- a density
        # field's LABEL gets " (pop/km²)" (its value stays a raw number,
        # unlike a percent field, whose value itself is rescaled).
        "function __fieldUnitSuffix(key) {\n"
        "  var k = String(key || '').toLowerCase();\n"
        "  if (k.indexOf('density') >= 0) return ' (pop/km\\u00b2)';\n"
        "  if (__isPercentField(key)) return ' (%)';\n"
        "  if (k.indexOf('kmh') >= 0 || k.indexOf('km_h') >= 0) return ' (km/h)';\n"
        "  if (k.indexOf('minutes') >= 0) return ' (min)';\n"
        "  return '';\n"
        "}\n"
        "function __shapePopupNum(v, key) {\n"
        "  if (v == null || v === '' || isNaN(v)) return String(v == null ? '-' : v);\n"
        "  if (key && __isPercentField(key)) return (Number(v) * 100).toFixed(1) + '%';\n"
        "  return window.__fmtLegendNum ? window.__fmtLegendNum(Number(v)) : String(v);\n"
        "}\n"
        # Mirrors `strip_source_prefix`/`_humanize_field_name`/`display_field_name`
        # in build.py (Python side, used for the server-rendered dropdown
        # `<option>` labels) -- this is the client-side equivalent for labels
        # built at render time (popup rows, regression/ANOVA/distribution
        # axis labels), so a field reads the same human, cross-country
        # -comparable name everywhere on the page, not just in dropdowns.
        "var __SOURCE_PREFIXES = ['acs5_','acs3_','acs1_','dhc_','lodes_wac_','lodes_rac_',\n"
        "  'inegi_','ine_cl_','ine_','cbs_','eustat_','statcan_','moi_','estadisticaad_'];\n"
        "function __stripSourcePrefix(key) {\n"
        "  var k = String(key || '');\n"
        "  for (var i = 0; i < __SOURCE_PREFIXES.length; i++) {\n"
        "    if (k.indexOf(__SOURCE_PREFIXES[i]) === 0) return k.slice(__SOURCE_PREFIXES[i].length);\n"
        "  }\n"
        "  return k;\n"
        "}\n"
        "function __humanizeFieldName(name) {\n"
        # Mirrors `_FIELD_LABEL_OVERRIDES` (Python side, see
        # `_humanize_field_name`'s docstring) -- see the identical fix's
        # comment on the second copy of this function, below.
        "  var __FIELD_LABEL_OVERRIDES = {'educationMeanGrade': 'Average years of schooling'};\n"
        "  if (Object.prototype.hasOwnProperty.call(__FIELD_LABEL_OVERRIDES, name)) return __FIELD_LABEL_OVERRIDES[name];\n"
        "  var words = (String(name).replace(/_/g, ' ').match(/[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\\d+/g)) || [];\n"
        "  if (!words.length) return name;\n"
        "  var last = words[words.length - 1].toLowerCase();\n"
        "  if (words.length > 1 && last === 'households') {\n"
        "    return 'Households with ' + words.slice(0, -1).map(function(w) { return w.toLowerCase(); }).join(' ');\n"
        "  }\n"
        "  if (words.length > 1 && last === 'population') {\n"
        "    var head = words.slice(0, -1).map(function(w) { return w.toLowerCase(); }).join(' ');\n"
        "    return head.charAt(0).toUpperCase() + head.slice(1) + ' population';\n"
        "  }\n"
        "  var joined = words.map(function(w) { return w.toLowerCase(); }).join(' ');\n"
        "  return joined.charAt(0).toUpperCase() + joined.slice(1);\n"
        "}\n"
        "function __fieldLabel(key) {\n"
        "  return __humanizeFieldName(__stripSourcePrefix(key)) + __fieldUnitSuffix(key);\n"
        "}\n"
        "function __shapePopupRow(label, value, strong) {\n"
        "  return '<tr><td style=\"font-weight:600;color:#555;padding:2px 8px 2px 0;white-space:nowrap;\">' + label +\n"
        "    '</td><td style=\"padding:2px 0;text-align:right;\">' +\n"
        "    (strong ? '<b>' + value + '</b>' : value) + '</td></tr>';\n"
        "}\n"
        "function __popupPopulationValue(properties) {\n"
        # Mirrors `_population_column` in pipeline.py -- the popup head row
        # special-cases population regardless of which country's source
        # prefixed it, same reasoning as that function's docstring (a
        # stale-hardcoded-column-list would silently show '-' for every
        # non-US city).
        # Real bug (2026-08-28, live Hamburg/Taipei maps): this list mirrored
        # only PART of pipeline.py's `CENSUS_COLUMN_PREFIXES` -- missing
        # 'destatis_' (Germany), 'ba_' (Germany jobs source, carries no
        # population but harmless to include), 'dhc_' (US decennial),
        # 'acs3_'/'acs1_' (other ACS vintages) and 'estadisticaad_'
        # (Andorra's other source). Any city whose only matching candidate
        # was missing here showed population as '-' in every popup even
        # though the real value was sitting right there in `properties`
        # under its (unrenamed -- 'population' collides with the h3 grid's
        # own WorldPop column, see `_RENAME_EXCLUDED_CANONICAL_NAMES` in
        # pipeline.py) source-prefixed key. Kept as an explicit list (not
        # derived from a shared constant) since this is a separate
        # JS-vs-Python copy with no import path between them -- see
        # `_population_column` in pipeline.py for the canonical Python list
        # this must stay in sync with.\n"
        "  var candidates = ['population', 'acs5_population', 'acs3_population', 'acs1_population',\n"
        "    'dhc_population', 'inegi_population', 'ine_population', 'ine_cl_population', 'cbs_population',\n"
        "    'eustat_population', 'statcan_population', 'moi_population', 'estadisticaad_population',\n"
        "    'destatis_population', 'ba_population', 'worldpop_population'];\n"
        "  for (var i = 0; i < candidates.length; i++) {\n"
        "    if (properties[candidates[i]] != null) return properties[candidates[i]];\n"
        "  }\n"
        "  return null;\n"
        "}\n"
        # Bug fix (user, verbatim spec): "All stats panels of those popups
        # should always include all absolute columns with the relative
        # values in parenthesis ... and always include population and
        # level of service." The header pair already guarantees the second half;
        # this table used to show every absolute AND relative column as its
        # own separate row (e.g. a `acs_workers_transit` row AND an unrelated
        # `acs_transit_share` row elsewhere in the list) instead of pairing
        # them into one "1,234 (12.3%)" row the way the spec asks. Pair them
        # here using `_POPUP_ABS_TO_REL` (absolute/numerator column name ->
        # the one relative field it feeds) -- a comprehensive, all-countries
        # mapping (see that constant's own docstring for why it exists as a
        # standalone constant rather than reusing `_DERIVED_FIELDS`, which
        # only covers ~23 US-only pairs and was the root cause of a live bug
        # report: Andorra's popup showed "Foreign born share (%)"/"Worldpop
        # male share (%)"/etc as disconnected standalone rows instead of
        # paired with their real absolute counts).
        f"  var __ABS_TO_REL = {json.dumps(_POPUP_ABS_TO_REL)};\n"
        "function __shapePopupHtml(properties) {\n"
        "  var skip = {geometry: 1, h3_cell: 1, geo_id: 1, GEOID: 1, geoid: 1, level_of_service: 1, population: 1,\n"
        "    acs5_population: 1, acs3_population: 1, acs1_population: 1, dhc_population: 1,\n"
        "    inegi_population: 1, ine_population: 1, ine_cl_population: 1,\n"
        "    cbs_population: 1, eustat_population: 1, statcan_population: 1, moi_population: 1,\n"
        "    estadisticaad_population: 1, destatis_population: 1, ba_population: 1, worldpop_population: 1};\n"
        "  var head = '<table style=\"border-collapse:collapse;width:100%;\">' +\n"
        "    __shapePopupRow('Level of service', __shapePopupNum(properties.level_of_service), true) +\n"
        "    __shapePopupRow('Population', __shapePopupNum(__popupPopulationValue(properties)), true) +\n"
        "    '</table>';\n"
        # First pass: for every absolute column paired with a relative one
        # (`__ABS_TO_REL`), if BOTH are present on this feature, mark the
        # relative field "consumed" so the second pass doesn't also show it
        # as its own standalone row -- it appears in parentheses on the
        # absolute row instead.
        "  var consumedRelative = {};\n"
        "  Object.keys(__ABS_TO_REL).forEach(function(absKey) {\n"
        "    var relKey = __ABS_TO_REL[absKey];\n"
        "    if (properties[absKey] != null && properties[relKey] != null) consumedRelative[relKey] = true;\n"
        "  });\n"
        "  var rows = '', n = 0;\n"
        "  Object.keys(properties).forEach(function(k) {\n"
        # Internal tile bookkeeping keys (geohierarchy prefixes the parent
        # id column with an underscore) are never shown.
        "    if (skip[k] || k.charAt(0) === '_' || consumedRelative[k]) return;\n"
        "    n++;\n"
        "    var relKey = __ABS_TO_REL[k];\n"
        "    var valueHtml = __shapePopupNum(properties[k], k);\n"
        "    if (relKey && properties[relKey] != null) {\n"
        "      valueHtml += ' <span style=\"color:#888;\">(' + __shapePopupNum(properties[relKey], relKey) + ')</span>';\n"
        "    }\n"
        "    rows += __shapePopupRow(__fieldLabel(k), valueHtml, false);\n"
        "  });\n"
        "  if (!n) return '<div style=\"font:12px sans-serif;\"><div style=\"font-weight:700;margin-bottom:4px;\">Area</div>' + head + '</div>';\n"
        # Item (user, 2026-08-23): the full variables table used to render
        # open by default, which meant every single popup -- even a quick
        # glance at the level of service -- paid for a 40+-row scrollable table.
        # Now it starts collapsed behind a small stats-emoji toggle button
        # next to the header pair; clicking it expands/collapses the table
        # in place. IDs are suffixed with a random token so multiple popups
        # opened across a session (Leaflet keeps old popup DOM around) never
        # collide and toggle each other's tables.
        "  var uid = 'p' + Math.random().toString(36).slice(2);\n"
        "  var tid = '__shapePopupTable_' + uid, bid = '__shapePopupToggle_' + uid;\n"
        # Real bug (user, 2026-08-28): the table div below already scrolls
        # natively (`max-height` + `overflow-y:auto`, a real browser
        # scrollbar) -- the custom `<input type=range>` slider that used to
        # sit under it was a second, redundant scroll control driving the
        # exact same `scrollTop`, so every expanded table showed two stacked
        # scroll widgets for one scrollable area. Dropped the slider
        # entirely and kept only the native scrollbar the `overflow-y:auto`
        # div already provides -- one control, not two. `sid` (the scroll
        # div's own id) is no longer needed without a slider to target it.
        # `max-width` widened from the implicit popup default so the table
        # isn't as cramped, without ballooning into a full redesign. Widened
        # again 2026-08-29 (user, live map): 320px was still cramped enough
        # that column values wrapped/truncated in the table; 400px gives the
        # table meaningfully more room. Widened again 2026-08-30 (user,
        # explicit ask on the same popup) to 480px -- still well clear of
        # running off a typical (>=1024px) viewport -- MapLibre's own popup
        # positioning already re-anchors near a map edge (it flips the
        # anchor side rather than letting the popup run off-screen), so no
        # separate clamping logic is needed here. Widened again 2026-09-01
        # (user, live map, alongside dropping the redundant `pop_density`
        # popup row) to 560px, together with raising the MapLibre
        # `maplibregl.Popup`'s own `maxWidth` (see
        # `geohierarchy.maps.maplibre.render`'s `interaction_js`, which was
        # separately clamping this same popup at MapLibre's 240px default
        # regardless of this inner CSS -- both had to move together).
        "  var toggle = '<button id=\"' + bid + '\" type=\"button\" title=\"Show all ' + n + ' census variables\" ' +\n"
        "    'style=\"font-size:14px;line-height:1;border:1px solid #e5e8eb;border-radius:4px;background:#f7f8f9;' +\n"
        "    'cursor:pointer;padding:2px 6px;margin-left:6px;\" ' +\n"
        "    'onclick=\"var w=document.getElementById(\\'' + tid + '\\');' +\n"
        "    'var open=w.style.display!==\\'none\\';w.style.display=open?\\'none\\':\\'\\';' +\n"
        "    'this.title=(open?\\'Show\\':\\'Hide\\')+\\' all ' + n + ' census variables\\';\">\\ud83d\\udcca ' + n + '</button>';\n"
        "  return '<div style=\"font:12px sans-serif;max-width:560px;\">' +\n"
        "    '<div style=\"font-weight:700;margin-bottom:4px;display:flex;align-items:center;\">Area' + toggle + '</div>' + head +\n"
        "    '<div id=\"' + tid + '\" style=\"display:none;margin-top:6px;\">' +\n"
        "    '<div style=\"max-height:220px;overflow-y:auto;border:1px solid #e5e8eb;border-radius:4px;padding:2px 6px;\">' +\n"
        "    '<table style=\"border-collapse:collapse;width:100%;\">' + rows + '</table></div></div></div>';\n"
        "}"
    )


def _opacity_helper_js() -> str:
    """JS `opacityFromField(properties, domains, base)`: the ONE opacity-by-field rule.

    Emitted once as a page-level function declaration (same mechanism as
    `rdylgn` -- a top-level `function` in a plain `<script>` is a real
    global) and called by *every* fill-styled layer: h3 hexagons, centroid
    circles and census geometries. Having a single implementation is the
    point: these three previously carried near-duplicate inline copies of
    the formula, which is exactly how they drifted out of sync.

    Shape of the mapping:
      * `norm` = the feature's value, normalized against the field's
        percentile-clipped domain (p15/p85, see `_percentile_domain`) and
        clamped to [0, 1].
      * The contrast slider `window.__opacityContrast` (0..1) controls how
        far apart low and high values are allowed to land:
          - at contrast 0 the layer is uniform (every feature at `base`
            opacity) -- i.e. the control is off, matching prior behavior;
          - at contrast 1 the low end drops to ~0.05 (all but invisible)
            while the high end stays at full 1.0.
      * `norm` is additionally gamma-shaped by `1 + 2*contrast` before the
        interpolation. A purely linear ramp between those endpoints still
        read as washed-out because most real fields put the bulk of their
        features mid-domain; the gamma pushes everything but the genuinely
        high values toward the transparent end, which is what makes high
        contrast look dramatic rather than merely dim.
    Returns `base` unchanged when no field is selected or the feature has
    no value for it.
    """
    return (
        "function opacityFromField(properties, domains, base) {\n"
        "  var shape = (window.__shapeOpacity != null) ? window.__shapeOpacity : 1;\n"
        "  var field = window.__opacityField;\n"
        # `fieldValue`, not `properties[field]` -- census geometries carry
        # only raw counts, and older tiles can carry a differently-scaled
        # density (see `_DERIVED_FIELDS`).
        "  var value = fieldValue(properties, field);\n"
        "  if (!field || field === 'none' || !domains[field] || value == null) {\n"
        "    return Math.max(0, Math.min(1, base * shape));\n"
        "  }\n"
        "  var d = domains[field];\n"
        "  var norm = (value - d[0]) / Math.max(d[1] - d[0], 1e-9);\n"
        "  norm = Math.max(0, Math.min(1, norm));\n"
        "  var c = (window.__opacityContrast != null) ? window.__opacityContrast : 0.5;\n"
        "  c = Math.max(0, Math.min(1, c));\n"
        "  var minOp = 1 - 0.95 * c, maxOp = 1;\n"
        "  var shaped = Math.pow(norm, 1 + 2 * c);\n"
        "  var opacity = (minOp + (maxOp - minOp) * shaped) * shape;\n"
        "  return Math.max(0, Math.min(1, opacity));\n"
        "}"
    )


def _numeric_field_candidates(gdf: gpd.GeoDataFrame, extra_exclude: Sequence[str] = ()) -> List[str]:
    """Numeric, non-identifier columns eligible for the circle-size/opacity dropdowns.

    This is the ONE shared source every field-offering surface in this
    module builds from -- circle-size/opacity dropdowns, the click popup
    (`_popup_fields`), and every stats tab (ANOVA/Regression/Distribution,
    via `_stats_field_candidates`) -- so a fix made here (rather than in
    each of those call sites separately) applies everywhere at once.
    """
    exclude = _FIELD_EXCLUDE | set(extra_exclude)
    fields = []
    for col in gdf.columns:
        if col in exclude:
            continue
        try:
            if np.issubdtype(gdf[col].dtype, np.number):
                fields.append(col)
        except TypeError:
            continue
    # 2026-09-01 (live user report -- "many maps with Population density
    # (+0.44) and pop density in ANOVA and many other places, this doesn't
    # make any sense... only one population column and one worldpop
    # population column"): `pop_density` is the exact same people-per-km2
    # figure as whichever of `population_density` (census-source) /
    # `worldpop_population_density` (WorldPop) its underlying
    # `_population_column` happened to resolve to (see
    # `code.pipeline._add_derived_density_columns`'s docstring) -- with
    # either or both of those real, source-labeled columns also present,
    # `pop_density` is a third, exactly-duplicate entry rather than new
    # information. Only drop it when there's a real named-by-source column
    # to replace it with (an older/coarser resample, or a country with no
    # census join and somehow no WorldPop join either, still needs
    # `pop_density` as its only density figure). This mirrors the same fix
    # already applied to the click popup specifically
    # (`_column_metadata_rows`'s docstring) -- doing it here instead makes
    # it apply to every dropdown/tab at once, not just the popup.
    if "pop_density" in fields and (
        "population_density" in fields or "worldpop_population_density" in fields
    ):
        fields = [f for f in fields if f != "pop_density"]
    # 2026-09-02 (explicit user request, flagged from the same "too many
    # duplicate columns" family as the fix above): `jobs_and_population_density`
    # is a literal alias of `pop_jobs_density` -- see
    # `code.pipeline._add_derived_density_columns`'s own comment ("a
    # same-named alias of the pre-existing `pop_jobs_density`... this
    # reuses that computation rather than re-deriving it") -- the two are
    # never present without each other (whichever grid has one has the
    # other, copied verbatim), so showing both is a pure duplicate, never
    # two different real measurements the way e.g. census vs. WorldPop
    # population density are. Keep the more descriptive/newer name.
    if "pop_jobs_density" in fields and "jobs_and_population_density" in fields:
        fields = [f for f in fields if f != "pop_jobs_density"]
    return fields


def _preferred_density_field(fields: Sequence[str]) -> Optional[str]:
    """The best available population-density field in `fields`, for default circle-size/etc.

    Priority (verbatim user request: "jobs and population density instead
    of population density or worldpop density... if not jobs then
    population density instead of worldpop density"):
      1. the combined population+jobs density (`pop_jobs_density`, aliased
         `jobs_and_population_density` -- see `_numeric_field_candidates`'s
         dedup, which keeps only one of the two names per city).
      2. real census-source `population_density`, then the legacy generic
         `pop_density`.
      3. `worldpop_population_density` -- last resort only, never preferred
         over a real census/jobs-based density.
    `None` if none of these are offered at all.
    """
    for cand in (
        "pop_jobs_density", "jobs_and_population_density",
        "population_density", "pop_density",
        "worldpop_population_density",
    ):
        if cand in fields:
            return cand
    return None


# --- Relative-vs-absolute column classification -----------------------------
#
# Every numeric column produced by the pipeline is either an ABSOLUTE
# extensive quantity (a raw count that scales with the size of the geometry
# it was aggregated over: `population`, `acs_households`, `acs_workers_transit`)
# or a RELATIVE intensive one (a share, rate, density, median or mean, which
# is comparable across geometries of different size: `acs_transit_share`,
# `acs_poverty_rate`, `pop_density`, `acs_income_median_household`).
#
# That distinction decides which dropdown a column may appear in:
#   * RELATIVE only -> "Opacity by", "Circle size by", regression, ANOVA.
#     Mapping or correlating a raw count is meaningless here, because a
#     hexagon's count is mostly a function of how much land it covers.
#   * ABSOLUTE only -> the distribution tab's overlay bar, which SUMS the
#     column within each access-score bin; summing rates/medians is
#     nonsense, summing people/jobs/households is exactly the intent.
#
# The rule is a NAMING CONVENTION match, deliberately not a hardcoded column
# list: the census/jobs pipeline gains columns over time (e.g. a new
# `jobs_density` or `pop_jobs_density`) and they must classify correctly
# without this module being told about them. Anything not matching a
# relative marker is treated as absolute, which is the safe default because
# the pipeline's raw-count columns are the ones with no such suffix.
_RELATIVE_MARKERS = (
    "share", "rate", "ratio", "density", "pct", "percent",
    "median", "mean", "avg", "average", "index", "score",
    "per_capita", "per_km", "per_ha", "per_1000", "_per_",
)


def is_relative_field(col: str) -> bool:
    """True if `col`'s name marks it as a share/rate/density/median-style intensive value."""
    name = str(col).lower()
    return any(marker in name for marker in _RELATIVE_MARKERS)


def relative_fields(fields: Sequence[str]) -> List[str]:
    """`fields` filtered to relative (share/rate/density/...) columns, order preserved."""
    return [f for f in fields if is_relative_field(f)]


# Item 11: units on every field name shown to a person -- dropdown options,
# legends, popups -- not just the popup table (which already special-cases
# share/rate/pct/ratio fields, see `__isPercentField` in `_shape_popup_js`).
# Naming-convention match, same spirit as `_RELATIVE_MARKERS`: a suffix on
# the column name decides the unit, so newly-added pipeline columns pick up
# the right one automatically.
def field_unit_suffix(col: str) -> str:
    """`" (pop/km²)"`, `" (%)"`, `" (min)"`, `" (km/h)"`, or `""` -- appended to a field's display label."""
    name = str(col).lower()
    if "density" in name:
        return " (pop/km²)"
    if any(marker in name for marker in ("share", "rate", "pct", "percent", "ratio")):
        return " (%)"
    if "kmh" in name or "km_h" in name:
        return " (km/h)"
    if "minutes" in name:
        return " (min)"
    return ""


# Every country's census source prefixes its columns with a real
# `pycensus.<country>.<source>.schema.SCHEMA(prefix=...)` (see e.g.
# `pycensus.countries.mexico.inegi.schema`) so `inegi_population`/`acs5_population`/
# `cbs_population` never collide inside `pycensus`'s own multi-source
# registry. On the map, a single study only ever joins ONE country's data,
# so that collision risk doesn't exist here -- and per the user's explicit
# ask, stripping the prefix (in display text only, never in the underlying
# column key) surfaces the field's real *global_schema.json* name whenever
# one exists, since this project's whole cross-country renaming effort (see
# `pycensus/IMPLEMENTING_A_COUNTRY.md`) already made e.g. `inegi_population`/
# `acs5_population`/`cbs_population` share the exact same logical name
# (`population`) -- the prefix was the only thing making them look
# different. A field with no canonical global_schema equivalent (e.g.
# `inegi_populationCatholic`) still gets its prefix stripped and its
# camelCase humanized, just without claiming a cross-country-comparable
# name it doesn't have.
_SOURCE_PREFIXES = (
    "acs5_", "acs3_", "acs1_", "dhc_", "lodes_wac_", "lodes_rac_",
    "inegi_", "ine_cl_", "ine_", "cbs_", "eustat_", "statcan_", "moi_",
    "estadisticaad_",
)


def strip_source_prefix(col: str) -> str:
    """`col` with its known census-source prefix removed, or `col` unchanged if none matches."""
    for prefix in _SOURCE_PREFIXES:
        if col.startswith(prefix):
            return col[len(prefix):]
    return col


# --- Metadata tab: which agency/dataset each source prefix comes from and ---
# the highest real resolution level that source publishes at. A superset of
# `_SOURCE_PREFIXES` (adds `destatis_`/`ba_`/`worldpop_`, which that table
# doesn't need since it only drives prefix-STRIPPED display names, not
# source attribution) -- see `code/pipeline.py`'s `CENSUS_COLUMN_PREFIXES`
# for the same prefix roster this mirrors. Levels are each source's real
# documented finest publication unit (ACS5 never publishes below block
# group; DHC/LODES are real decennial/LEHD block-level products; WorldPop is
# a continuous raster, not a discrete geography, so its "level" is its pixel
# size rather than a polygon level).
_METADATA_SOURCE_INFO: Dict[str, Tuple[str, str]] = {
    "acs5_": ("US Census Bureau, ACS 5-year", "block group"),
    "acs3_": ("US Census Bureau, ACS 3-year", "block group"),
    "acs1_": ("US Census Bureau, ACS 1-year", "block group"),
    "dhc_": ("US Census Bureau, Decennial DHC", "block"),
    "lodes_wac_": ("US Census Bureau, LEHD LODES (workplace)", "block"),
    "lodes_rac_": ("US Census Bureau, LEHD LODES (residence)", "block"),
    "inegi_": ("INEGI (Mexico)", "AGEB"),
    "ine_cl_": ("INE (Chile)", "block"),
    "ine_": ("INE (Spain)", "census section"),
    "cbs_": ("CBS (Israel)", "locality"),
    "eustat_": ("Eustat (Basque Country)", "census section"),
    "statcan_": ("Statistics Canada", "dissemination block"),
    "moi_": ("Ministry of Interior", "locality"),
    "estadisticaad_": ("Departament d'Estadistica (Andorra)", "parish"),
    "destatis_": ("Destatis (Germany)", "municipality"),
    "ba_": ("Bundesagentur fuer Arbeit (Germany)", "municipality"),
    "worldpop_": ("WorldPop", "100m grid"),
}


def _metadata_source_for(col: str) -> Tuple[str, str]:
    """`(source label, highest real resolution level)` for a real census/WorldPop column, else `("", "")`."""
    for prefix, info in _METADATA_SOURCE_INFO.items():
        if col.startswith(prefix):
            return info
    if col == "population":
        # Bare `population` is always the WorldPop-derived count on every
        # path this pipeline builds -- never a renamed census column, since
        # `population` is deliberately excluded from the census-source
        # rename (see `code/pipeline.py`'s `_RENAME_EXCLUDED_CANONICAL_NAMES`).
        return _METADATA_SOURCE_INFO["worldpop_"]
    return ("", "")


_GLOBAL_SCHEMA_DEFINITIONS_CACHE: Optional[Dict[str, str]] = None


def _global_schema_definitions() -> Dict[str, str]:
    """`{feature_name: definition}` from pyCensus's `global_schema.json`, memoized.

    Reused (not re-typed) for the Metadata tab's description column, per the
    project's own anti-fabrication rule (`pyCensus/IMPLEMENTING_A_COUNTRY.md`):
    a description should come from the one real schema file that already
    documents each canonical field, not a second hand-maintained copy that
    can drift from it. Returns `{}` (never raises) if `pycensus` isn't
    importable from wherever this module is running -- `build.py` is shared
    across studies and some don't depend on it.
    """
    global _GLOBAL_SCHEMA_DEFINITIONS_CACHE
    if _GLOBAL_SCHEMA_DEFINITIONS_CACHE is None:
        try:
            import pycensus

            schema_path = os.path.join(os.path.dirname(pycensus.__file__), "schemas", "global_schema.json")
            with open(schema_path, encoding="utf-8") as f:
                entries = json.load(f)
            _GLOBAL_SCHEMA_DEFINITIONS_CACHE = {e["feature_name"]: e.get("definition", "") for e in entries}
        except Exception:
            _GLOBAL_SCHEMA_DEFINITIONS_CACHE = {}
    return _GLOBAL_SCHEMA_DEFINITIONS_CACHE


def _column_metadata_rows(gdf, share_source_map: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """One metadata row per real census/WorldPop column on `gdf`, for the stats panel's "Metadata" tab.

    Only real source columns (recognized by `_metadata_source_for`) are
    included -- not derived/bookkeeping fields like `level_of_service`,
    `equity_flag`, `h3_cell`, `geometry`, or the pipeline's own
    `pop_density`/`pop_jobs_density`/`jobs_share`/... derived columns (those
    are combinations of real source columns, not a source column
    themselves; the real inputs they're built from already get their own
    row).

    `total`: the column's plain sum, shown only for absolute (count-like)
    fields -- per explicit user request, summing a rate/share/density/
    median/mean field is not a meaningful figure, so those rows leave
    `total` blank. Uses the same `is_relative_field` naming-convention
    classifier every other selector in this module already uses.

    `average`: population-weighted mean -- `Mean(weight_column="population")`
    is this whole pipeline's own standing convention for aggregating any
    per-cell/per-polygon field (see `code/pipeline.py`'s
    `_census_geometries_with_score` and `_relative_column_stats_for_grid`),
    reused here rather than inventing a second one. Falls back to a plain
    unweighted mean when `gdf` has no `population` column at all (a study
    with no census/WorldPop population join, for whatever reason).

    `description`: pulled from pyCensus's own `global_schema.json`
    `definition` field for whichever columns have a real canonical
    (de-prefixed) match there; a column with no canonical equivalent (e.g.
    `inegi_populationCatholic`) falls back to its humanized display name
    rather than a blank cell.

    `share_source_map` (2026-09-01, per explicit user request: "do not
    separate share and total columns... men and total, average, share, so
    we do not need a separate share column"): maps a relative share
    column's name (e.g. `male_share`) to the absolute numerator column it
    was actually computed from (e.g. `malePopulation`) -- see
    `code.pipeline._share_column_sources`, the CS_transitLOS-level function
    that derives this from `SHARE_COLUMNS`/the same numerator/denominator
    pairs `_add_share_columns` used to materialize the share column in the
    first place. When given, a share column no longer gets its own row;
    instead its population-weighted average (already a 0-1 fraction) is
    folded into its numerator's row as a `share` percentage, so "Male
    population" carries Total + Average + Share together instead of
    "Male population" and "Male share" appearing as two disconnected rows.
    A share column with no present numerator row (e.g. a rate with no
    absolute-count counterpart on this grid, like `poverty_share`) still
    gets its own row, as before -- nothing is silently dropped.
    """
    weight_col = "population" if "population" in gdf.columns else None
    weights = gdf[weight_col].to_numpy(dtype=float) if weight_col else None
    definitions = _global_schema_definitions()
    rows_by_col: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for col in gdf.columns:
        source, level = _metadata_source_for(col)
        if not source:
            continue
        try:
            if not np.issubdtype(gdf[col].dtype, np.number):
                continue
        except TypeError:
            continue
        values = gdf[col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        total_str = "" if is_relative_field(col) else f"{np.nansum(values[finite]):,.0f}"
        if weights is not None:
            w = np.where(finite, weights, 0.0)
            w = np.where(np.isfinite(w), w, 0.0)
            wsum = float(np.sum(w))
            avg = (float(np.sum(np.where(finite, values, 0.0) * w) / wsum) if wsum > 0
                   else float(np.mean(values[finite])))
        else:
            avg = float(np.mean(values[finite]))
        logical = strip_source_prefix(col)
        description = definitions.get(logical) or display_field_name(col)
        order.append(col)
        rows_by_col[col] = {
            "name": field_label(col),
            "source": source,
            "level": level,
            "total": total_str,
            "average": f"{avg:,.2f}",
            "share": "",
            "unit": (field_unit_suffix(col).strip(" ()")) or ("count" if not is_relative_field(col) else ""),
            "weight_column": weight_col or "(unweighted)",
            "description": description,
            "_share_avg_fraction": avg if is_relative_field(col) else None,
        }

    if share_source_map:
        for share_col, numerator_col in share_source_map.items():
            share_row = rows_by_col.get(share_col)
            numerator_row = rows_by_col.get(numerator_col)
            if share_row is None or numerator_row is None:
                continue
            frac = share_row["_share_avg_fraction"]
            if frac is not None:
                numerator_row["share"] = f"{frac * 100:,.1f}%"
            del rows_by_col[share_col]
            order.remove(share_col)

    # 2026-09-07 bug fix (live report, Beersheba): a source-prefixed real
    # census column (e.g. `cbs_population`) and an unrelated bare column
    # (e.g. `population`, always WorldPop -- see `_metadata_source_for`)
    # can independently strip/humanize to the exact SAME display name
    # ("Population") even though they're different columns with different
    # real values -- `_rename_canonical_columns` in `code/pipeline.py`
    # already refuses to MERGE them (keeps `cbs_population` prefixed
    # specifically because bare `population` already exists), but nothing
    # here accounted for that same collision when building each row's
    # display label, so both rows silently rendered under one identical
    # name with no way to tell which was which. Disambiguate generically
    # (any country/column pair, not just this one) by appending the
    # source whenever two or more rows share a name.
    name_counts: Dict[str, int] = {}
    for col in order:
        name_counts[rows_by_col[col]["name"]] = name_counts.get(rows_by_col[col]["name"], 0) + 1
    for col in order:
        row = rows_by_col[col]
        if name_counts[row["name"]] > 1:
            row["name"] = f"{row['name']} ({row['source'] or col})"

    return [
        {k: v for k, v in rows_by_col[col].items() if not k.startswith("_")}
        for col in order
    ]


def _humanize_field_name(name: str) -> str:
    """CamelCase/snake_case field name -> a readable phrase, with a couple of targeted rewrites.

    Generic camelCase/snake_case-to-words for the common case, plus two
    naming-convention-based rewrites (same spirit as `_RELATIVE_MARKERS`/
    `field_unit_suffix`) for the two patterns this project's schemas
    actually use for compound concepts:
      - `...Households` (e.g. `motorcycleHouseholds`, `carHouseholds`) ->
        "Households with ..." (e.g. "Households with motorcycle").
      - `...Population` (e.g. `malePopulation`, `minorityEthnicityPopulation`)
        -> "... population" (e.g. "Male population").
    Anything else just gets space-separated and capitalized, e.g.
    `medianHouseholdIncome` -> "Median household income",
    `inegi_unemployment_rate` (already snake_case, from `SHARE_COLUMNS`) ->
    "Unemployment rate".
    """
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name.replace("_", " "))
    words = [w for w in words if w]
    if not words:
        return name
    if len(words) > 1 and words[-1].lower() == "households":
        return "Households with " + " ".join(w.lower() for w in words[:-1])
    if len(words) > 1 and words[-1].lower() == "population":
        return " ".join(w.lower() for w in words[:-1]).capitalize() + " population"
    return " ".join(w.lower() for w in words).capitalize()


# Targeted overrides for specific fields whose generic camelCase-to-words
# humanization reads awkwardly or ambiguously (e.g. "Education mean grade"
# for INEGI's `educationMeanGrade`, which is really "grado promedio de
# escolaridad" -- average completed YEARS of schooling, not a literal
# school "grade"). Keyed on the DE-PREFIXED logical name (checked before
# the generic humanizer), so it applies regardless of which country's
# source prefix the raw column carries.
_FIELD_LABEL_OVERRIDES = {
    "educationMeanGrade": "Average years of schooling",
}


def display_field_name(col: str) -> str:
    """`col`'s cross-country-comparable display name: source prefix stripped, then humanized."""
    logical = strip_source_prefix(col)
    if logical in _FIELD_LABEL_OVERRIDES:
        return _FIELD_LABEL_OVERRIDES[logical]
    return _humanize_field_name(logical)


def field_label(col: str) -> str:
    """A field's dropdown/legend display label, with its unit suffix (see `field_unit_suffix`)."""
    return f"{display_field_name(col)}{field_unit_suffix(col)}"


def absolute_fields(fields: Sequence[str]) -> List[str]:
    """`fields` filtered to absolute raw-count columns -- the exact complement of `relative_fields`."""
    return [f for f in fields if not is_relative_field(f)]


def _percentile_domain(series, lo: float = 5, hi: float = 95) -> Tuple[float, float]:
    """`(p5, p95)` of `series` (falls back to true min/max if that range is degenerate).

    Used for the opacity-by-field mapping instead of true min/max: a field
    with a long tail (income, population) would otherwise map almost every
    cell into a narrow sliver of the 0-1 opacity range near one end,
    leaving no visible contrast. Clipping to the 5th/95th percentile before
    normalizing spreads the bulk of the data across the full opacity range;
    values outside it simply clamp to fully faint/opaque.

    Zero/missing values are excluded before computing percentiles -- for
    most of these fields (income, counts, ...) `0`/null means "no data",
    not "the true value is zero", and leaving them in skews the percentile
    bounds toward the low end.
    """
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return (0.0, 1.0)
    p_lo, p_hi = float(np.percentile(values, lo)), float(np.percentile(values, hi))
    if p_hi - p_lo < 1e-12:
        return _finite_domain(series)
    return (p_lo, p_hi)


# Item 5 (2026-08-15): after the scenario editor's Compute recomputes
# per-H3-cell level_of_service for the population chunks its buffer touched
# (`window.__h3AccessOverrides`, set by `computeAccess()`), the base map's
# own hexagon/circle fill color should reflect that too, not just the stats
# panel. Both levels are keyed by `h3_cell` at build time (`gh_hex`/`gh_circ`
# in `_build_hierarchy_maps` both use `id_col="h3_cell"`), so a feature's own
# id is comparable against the override map's child-cell ids -- EXACTLY the
# same "roll the touched children's population-weighted gain up to this
# feature's own resolution" identity `getStatsData` already uses for the
# stats panel (see `_stats_panel_js`'s `__h3ToParent`/`__h3Resolution`,
# published onto `window` for reuse here since this style function runs in a
# different injected `<script>`).
#
# Census levels are NOT covered: they're keyed by `GEOID`, a wholly different
# id space with no H3 relationship, so `properties.h3_cell` is simply absent
# on those features and this whole block is a no-op for them -- the census
# layer keeps showing the static baked-in level_of_service. Making census
# recolor too would need a real GEOID<->H3 spatial join, out of scope here;
# best-effort is hexagons + circles only, per spec.
def _h3_override_lookup_js(res: Optional[int] = None) -> str:
    """Build the `_H3_OVERRIDE_LOOKUP_JS` body for one H3 resolution.

    The per-feature id on real vector-tile data is NOT a flat `h3_cell` --
    `HierarchyMap`/`geojson-vt` namespace each level's own id column by the
    level name to keep sibling levels' properties from colliding on the same
    tile, so a resolution-9 hexagon's id key is `_h3_9_h3_cell`, not
    `h3_cell` (a bare `properties.h3_cell` is always `undefined` on real
    tiles, which made this whole override block permanently dead). Since
    `_polygon_style_js`/`_circle_style_js` are invoked once per resolution
    (see the `for res in resolutions:` loops in `_build_hierarchy_maps`),
    `res` here is a plain Python int and the namespaced key can be baked in
    as a literal at build time -- no runtime detection needed.

    `res=None` is for callers with no single resolution to bake in (the
    census levels, which share `_polygon_style_js`'s body across levels
    keyed by `GEOID` and never carry an `h3_cell` at all -- see the module
    comment above this function). That falls back to the old flat
    `properties.h3_cell`, which stays `undefined` there, so the override
    block is a correct no-op for census exactly as before.

    Also fixes: these tiles never carry a raw `population` field, only
    `acs_population`/`dhc_population`/etc, so `properties.population > 0`
    was equally always false. Falls back to `properties.acs_population`
    wherever `population` is used, both in the guard and in the final
    population-weighted blend.
    """
    id_key_js = json.dumps(f"_h3_{res}_h3_cell") if res is not None else None
    id_expr = f"properties[{id_key_js}]" if id_key_js is not None else "properties.h3_cell"
    pop_guard = "(properties.population > 0 || properties.acs_population > 0)"
    pop_value = "(properties.population > 0 ? properties.population : properties.acs_population)"
    return (
        "  var __sc = properties.level_of_service || 0;\n"
        "  var __ov = window.__h3AccessOverrides;\n"
        f"  if (__ov && {id_expr} != null && {pop_guard} &&\n"
        "      window.__h3ToParent && window.__h3Resolution) {\n"
        f"    var __res = window.__h3Resolution({id_expr});\n"
        "    var __deltaPop = 0;\n"
        "    for (var __childId in __ov) {\n"
        "      var __o = __ov[__childId];\n"
        "      if (__o.access == null || __o.gainBase == null) continue;\n"
        "      var __parentId = window.__h3ToParent(__childId, __res);\n"
        f"      if (__parentId !== {id_expr}) continue;\n"
        "      var __pop = (__o.population != null && __o.population >= 0) ? __o.population : 0;\n"
        "      __deltaPop += (__o.access - __o.gainBase) * __pop;\n"
        "    }\n"
        "    if (__deltaPop !== 0) {\n"
        f"      __sc = Math.max(0, Math.min(1, __sc + __deltaPop / {pop_value}));\n"
        "    }\n"
        "  }\n"
    )


def _census_geoid_override_lookup_js(id_expr: str) -> str:
    """Census-polygon counterpart of `_h3_override_lookup_js` (2026-08-16, item 6).

    Census features are keyed by GEOID, with no per-feature H3 id, so the
    hexagon/circle override lookup (which walks `window.__h3AccessOverrides`
    by h3-parent identity) is structurally a no-op here. Instead this reads
    `window.__geoidAccessOverrides` -- a GEOID -> population-weighted delta
    SUM (not a final score), computed client-side in `computeAccess()` from
    only the member h3 cells that were actually touched by the scenario's
    buffer (see `transitlos.map.pop_chunks.build_geoid_chunks_from_h3_and_census`
    for how the GEOID<->h3-cell correspondence reaches the client). Dividing
    that sum by the polygon's own total population and adding it to the
    already-baked `level_of_service` reproduces the exact same population-weighted
    mean the build-time aggregation used, updated only for the cells that
    changed -- untouched member cells implicitly keep contributing their
    original (baked) share, since the delta sum only ever includes terms for
    cells present in `window.__h3AccessOverrides`.
    """
    return (
        "  var __sc = properties.level_of_service || 0;\n"
        "  var __gov = window.__geoidAccessOverrides;\n"
        f"  if (__gov && {id_expr} != null) {{\n"
        f"    var __gd = __gov[{id_expr}];\n"
        "    if (__gd != null) {\n"
        "      var __gpop = (properties.population > 0 ? properties.population : properties.acs_population) || 0;\n"
        "      if (__gpop > 0) __sc = Math.max(0, Math.min(1, __sc + __gd / __gpop));\n"
        "    }\n"
        "  }\n"
    )


def _circle_radius_expr(field: str, domain: tuple) -> List[Any]:
    """`["interpolate", ["linear"], ["get", field], d0, 2, d1, 18], ...]`-shaped
    MapLibre expression matching `_circle_style_js`'s own `2 + 16 * norm`
    Folium/Leaflet radius formula (same 2-18px range, same field/domain
    inputs) -- kept as a small helper so both the static per-layer paint
    (`_score_maplibre_paint`) and any future live field-switch can build the
    identical expression from a `(field, domain)` pair.
    """
    d0, d1 = domain
    d1 = d1 if d1 > d0 else d0 + 1e-9
    # Item 8: same `coalesce` null-guard as `_score_maplibre_paint`'s color
    # expression -- a feature missing `field` would otherwise throw.
    return ["interpolate", ["linear"], ["coalesce", ["get", field], d0], d0, 2, d1, 18]


def _score_maplibre_paint(
    kind: str,
    score_col: str,
    score_col_domain: Tuple[float, float],
    radius_field_domains: Optional[Dict[str, tuple]] = None,
    default_circle_field: Optional[str] = None,
    no_outline: bool = False,
) -> Dict[str, Any]:
    """MapLibre `paint` dict approximating `_polygon_style_js`/`_circle_style_js`'s
    red-yellow-green score coloring, via `MapLayer.maplibre_paint` (see that
    field's docstring in geohierarchy).

    Static (not zoom/property-control-reactive like the Folium/JS version --
    no live `__h3AccessOverrides` recolor, no opacity-by-field dropdown) --
    an `["interpolate", ["linear"], ["get", score_col], ...]` fill/circle/
    line-color expression evaluated once against the same `RDYLGN_HEX` stops
    and `score_col_domain` every level already shares. Good enough for
    MapLibre's base-layer-parity goal (correct, data-driven coloring) without
    porting the editor's live-recolor machinery to MapLibre.

    `radius_field_domains`/`default_circle_field` (only meaningful when
    `kind == "circle"`): Folium's circle layer already sizes each point by a
    selectable numeric field (`_circle_style_js`, "Circle size by" dropdown,
    percentile-clipped domain, 3-18px range). MapLibre had no equivalent --
    every circle rendered at a flat `circle-radius: 5` regardless of the
    underlying data. When a domain is available for `default_circle_field`,
    this emits the matching `_circle_radius_expr` interpolate expression
    instead of the flat radius, so circle size again correlates with real
    column values (statically, at the field Folium would default the
    dropdown to -- no live field-switch UI on the MapLibre page yet).
    """
    score_min, score_max = score_col_domain
    span = max(score_max - score_min, 1e-9)
    n = len(RDYLGN_HEX)
    stops: List[Any] = []
    for i, hex_color in enumerate(RDYLGN_HEX):
        stops.append(score_min + span * i / (n - 1))
        stops.append(hex_color)
    # Item 8 (this round, verbatim user request): fixes a real reported
    # console error -- "Expected value to be of type number, but found null
    # instead." -- MapLibre `interpolate` throws (not just misrenders) the
    # instant it evaluates against a `null` property, and `["get", score_col]`
    # returns exactly `null` for any feature missing that column (e.g. an h3
    # cell/census polygon with no computed `level_of_service`, common at a study
    # area's edge or for a city where `score_col` is a field like
    # `reliability` that's frequently absent -- see the stops popup's own
    # `reliability`-is-often-null handling for the same underlying gap).
    # `coalesce` picks the domain's own minimum as a safe fallback so the
    # expression always evaluates to a real number.
    color_expr = ["interpolate", ["linear"], ["coalesce", ["get", score_col], score_min], *stops]
    if kind == "circle":
        radius: Any = 5
        if default_circle_field and radius_field_domains and default_circle_field in radius_field_domains:
            radius = _circle_radius_expr(default_circle_field, radius_field_domains[default_circle_field])
        return {"circle-color": color_expr, "circle-radius": radius, "circle-opacity": 0.75}
    if kind == "line":
        return {"line-color": color_expr, "line-width": 2}
    # Item 6 (verbatim user request): H3 hexagon cells get NO border/outline
    # at all -- `rgba(0,0,0,0)` (fully transparent), not just a lighter
    # color, so no seam is visible between adjacent hexagons. `no_outline`
    # is only passed `True` for the hexagons level (see its `configure_level`
    # call site) -- census polygons keep their outline, which the user did
    # not ask to remove and which still helps distinguish adjacent
    # geographies at that level.
    outline = "rgba(0,0,0,0)" if no_outline else "#333333"
    return {"fill-color": color_expr, "fill-opacity": 0.75, "fill-outline-color": outline}


def _polygon_style_js(
    score_col_domain: Tuple[float, float],
    opacity_field_domains: Dict[str, tuple],
    res: Optional[int] = None,
    census_id_expr: Optional[str] = None,
) -> str:
    """Style function body for a polygon level (h3 hexagons or census geometries).

    Reads the page-level `window.__opacityField` global (set by the control
    panel) so a single style function serves every resolution/level without
    rebuilding tiles when that control changes -- only `.redraw()` (see
    `_control_panel_js`) is needed. Opacity domains are percentile-clipped
    (see `_percentile_domain`) so the mapping keeps visible contrast
    regardless of the field's actual distribution.

    Also applies the scenario editor's live `window.__h3AccessOverrides`, if
    any -- see `_H3_OVERRIDE_LOOKUP_JS`'s docstring-comment above (item 5).

    `census_id_expr`, if given, is a JS property-access expression (e.g.
    `"properties['_census_blockgroup_GEOID']"`) selecting this level's own
    namespaced GEOID; when present it takes over from the h3-keyed override
    lookup entirely and uses `_census_geoid_override_lookup_js` instead (item
    6) -- the two lookups are mutually exclusive per level, matching how a
    census feature never carries an `h3_cell` property.
    """
    score_min, score_max = score_col_domain
    domains_json = json.dumps({k: list(v) for k, v in opacity_field_domains.items()})
    override_js = (
        _census_geoid_override_lookup_js(census_id_expr)
        if census_id_expr is not None
        else _h3_override_lookup_js(res)
    )
    return (
        "function(properties) {\n"
        + override_js
        + f"  var t = (__sc - {score_min}) / {max(score_max - score_min, 1e-9)};\n"
        "  var fill = rdylgn(t);\n"
        f"  var domains = {domains_json};\n"
        "  var opacity = opacityFromField(properties, domains, 0.75);\n"
        "  return {color: fill, weight: 0, opacity: 0, fill: true, fillColor: fill, fillOpacity: opacity};\n"
        "}"
    )


def _circle_style_js(
    score_col_domain: Tuple[float, float],
    radius_field_domains: Dict[str, tuple],
    opacity_field_domains: Dict[str, tuple],
    res: Optional[int] = None,
) -> str:
    """Style function body for a `"circles"` centroid-point level.

    Radius is driven by whichever field `window.__circleField` selects (any
    numeric column offered in the "Circle size by" dropdown); opacity by the
    same percentile-normalized, field-driven logic as `_polygon_style_js`
    (so the opacity dropdown affects circles too); color by the same RdYlGn
    level_of_service scale as every other layer.
    """
    score_min, score_max = score_col_domain
    radius_domains_json = json.dumps({k: list(v) for k, v in radius_field_domains.items()})
    opacity_domains_json = json.dumps({k: list(v) for k, v in opacity_field_domains.items()})
    default_field = next(iter(radius_field_domains), None)
    return (
        "function(properties) {\n"
        + _h3_override_lookup_js(res)
        + f"  var t = (__sc - {score_min}) / {max(score_max - score_min, 1e-9)};\n"
        "  var fill = rdylgn(t);\n"
        f"  var radiusDomains = {radius_domains_json};\n"
        f"  var field = window.__circleField || {json.dumps(default_field)};\n"
        "  var radius = 6;\n"
        # Same `fieldValue` indirection as the opacity mapping: reading the
        # raw property here is what made every circle render at the minimum
        # radius whenever the selected field was a density (unit drift
        # between the baked tiles and this build's domains).
        "  var value = fieldValue(properties, field);\n"
        "  if (field && radiusDomains[field] && value != null) {\n"
        "    var d = radiusDomains[field];\n"
        "    var norm = Math.max(0, Math.min(1, (value - d[0]) / Math.max(d[1] - d[0], 1e-9)));\n"
        "    radius = 2 + 16 * norm;\n"
        "  }\n"
        f"  var opacityDomains = {opacity_domains_json};\n"
        "  var opacity = opacityFromField(properties, opacityDomains, 0.8);\n"
        "  return {radius: radius, color: fill, weight: 0, opacity: 0, fill: true, fillColor: fill, fillOpacity: opacity};\n"
        "}"
    )


def _edges_style_js(score_col_domain: Tuple[float, float], street_min_zoom: int) -> str:
    """Style function body for the street-edges level (colored by level_of_service, no fill).

    Whether streets are shown at all is the native Leaflet overlay checkbox
    (streets are their own `MultiHierarchyMap` overlay layer) -- but that
    single-level layer's tiles must cover the *full* zoom range to satisfy
    `HierarchyMap`'s zoom-partition validation (see `_level_zoom_bands`), so
    "only show streets once zoomed in" is enforced here instead, by reading
    the live map's current zoom via `window.__leafletMapRef` (set in
    `_control_panel_js`). No opacity-by-field effect either -- opacity
    intentionally never affects streets, only hexagons/circles/census
    geometries.
    """
    score_min, score_max = score_col_domain
    return (
        "function(properties) {\n"
        f"  if (window.__leafletMapRef && window.__leafletMapRef.getZoom() < {street_min_zoom}) {{\n"
        "    return {opacity: 0, weight: 0, fill: false};\n"
        "  }\n"
        f"  var t = ((properties.level_of_service || 0) - {score_min}) / {max(score_max - score_min, 1e-9)};\n"
        "  var c = rdylgn(t);\n"
        "  return {color: c, weight: 1, opacity: 0.9, fill: false};\n"
        "}"
    )


def _development_style_js() -> str:
    """Style function body for the development-opportunity overlay: border-only, transparent otherwise.

    `equity_flag`'s `"more_transit"` category is now itself the population-
    weighted dense-and-under-served split (`stats.development_priority_flag`)
    -- there is no longer a separate `development_priority` boolean/layer to
    give precedence over it, just the two `equity_flag` categories.

    Border weight scales down as the map zooms in, reading the live zoom off
    `window.__leafletMapRef` (set in `_control_panel_js`) the same way
    `_edges_style_js` gates street visibility by zoom -- a fixed pixel weight
    that looks right zoomed out swamps the polygon once zoomed in far enough
    that a cell only spans a few dozen screen pixels.

    2026-08-15: the original `6 - (z-12)*0.6` curve (floor 1.5) was still
    visibly thick in real use at high zoom (z=16 -> 3.6px, z=18 -> 2.4px,
    never actually reaching the 1.5 floor at any on-screen zoom). Lowered
    both the base weight and the floor, and steepened the falloff, so the
    border is genuinely thin (<=1px) by z=16 instead of merely "less thick
    than before": z=12 -> 4px, z=14 -> 1.6px, z=16 -> floor 0.8px.
    """
    return (
        "function(properties) {\n"
        "  var z = (window.__leafletMapRef && window.__leafletMapRef.getZoom()) || 12;\n"
        "  var w = Math.max(0.8, Math.min(4, 4 - (z - 12) * 1.2));\n"
        "  if (properties.equity_flag === 'more_transit') {\n"
        "    return {color: '#ff8c00', weight: w, opacity: 1, fill: false};\n"
        "  }\n"
        "  if (properties.equity_flag === 'more_housing') {\n"
        "    return {color: '#1e6fd9', weight: w, opacity: 1, fill: false};\n"
        "  }\n"
        "  return {opacity: 0, weight: 0, fill: false};\n"
        "}"
    )


# The client-side badge (see `_stops_layer_block`'s `svgFor`) now draws its
# own small line-art SVG per mode directly in JS -- no server-side glyph
# table is needed for it any more.

# Blue sequential scale (light -> dark) for stop_score, distinct from the
# RdYlGn scale used everywhere else -- stops are a genuinely different kind
# of marker (mode emoji, not a filled shape) so a different palette keeps
# them visually distinguishable from the hexagon/circle/census fill.
STOP_BLUE_HEX = ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#084594"]


def _extract_layer_control_var(html: str) -> Optional[str]:
    """Find the JS variable name folium assigned the `L.control.layers(...)` instance.

    Declared `let layer_control_XXXX = L.control.layers(...)` at the top
    level of a plain `<script>` tag -- **unlike `_extract_map_var`'s `var
    map_XXXXXXXX`, a top-level `let` does NOT become a `window` property**
    (real bug, found 2026-08-12 via headless-browser reproduction: every
    stop-layer-checkbox registration silently no-op'd because
    `window["layer_control_XXXX"]` was `undefined`, so the `if (controlRef
    && controlRef.addOverlay)` guard downstream always failed). A top-level
    `let`/`const` *is* still reachable by bare identifier from a later
    top-level `<script>` tag in the same document (classic scripts share one
    global lexical environment, `let`/`const` bindings included) -- just not
    via `window[...]` string lookup. `_inject_controls_into_saved_html`
    exploits this: a tiny script emitted right after folium's own,
    referencing the name *as a literal identifier* (not a string), copies it
    onto an explicit `window.__layerControlRef` that every later script can
    then reach by ordinary property lookup.
    """
    m = re.search(r"(?:let|var)\s+(layer_control_[0-9a-fA-F]+)\s*=\s*L\.control\.layers\(", html)
    return m.group(1) if m else None


def _extract_base_layer_vars(html: str) -> Dict[str, str]:
    """Map display name -> JS var name for each `base_layers` entry folium's `L.control.layers` was given.

    e.g. ``{"hexagons": "feature_group_7a75...", "circles": "feature_group_6515...", ...}``.
    Unlike `layer_control_XXXX` itself, these `feature_group_XXXX` vars are
    declared `var` at the top level (see the `L.featureGroup(...)` assignment
    folium emits), so they're reachable as `window["feature_group_XXXX"]`
    from any later script -- no bridging needed (contrast
    `_extract_layer_control_var`'s docstring, which is the `let`-scoped
    exception).
    """
    result: Dict[str, str] = {}
    for block_name in ("base_layers", "overlays"):
        m = re.search(block_name + r"\s*:\s*\{(.*?)\}", html, re.S)
        if not m:
            continue
        result.update(re.findall(r'"([^"]+)"\s*:\s*(feature_group_[0-9a-fA-F]+)', m.group(1)))
    return result


def _stops_json_data(stops_gdf: gpd.GeoDataFrame, score_domain: Tuple[float, float]) -> str:
    """Column-oriented JSON for the stops layer: lon/lat + mode/score + popup detail fields.

    Each source row is really a (physical stop, route) pair collapsed to one row per
    stop by `transitlos.stops.download_and_prepare_stops`'s `how="best"` headway
    aggregation, so `route_label`/`route_id` on a given row is only that stop's single
    busiest route. `all_routes` (built there from raw GTFS `stop_times`/`trips`,
    independent of that collapse) lists *every* route actually serving the physical
    stop, and is preferred here when present.
    """
    stops_wgs84 = stops_gdf.to_crs(4326) if stops_gdf.crs is not None else stops_gdf
    lon = stops_wgs84.geometry.x.to_numpy()
    lat = stops_wgs84.geometry.y.to_numpy()
    mode = (
        stops_wgs84["mode_category"].to_numpy() if "mode_category" in stops_wgs84.columns
        else np.full(len(stops_wgs84), "bus")
    )
    route_col = "all_routes" if "all_routes" in stops_wgs84.columns else (
        "route_label" if "route_label" in stops_wgs84.columns else (
            "route_id" if "route_id" in stops_wgs84.columns else None
        )
    )

    def _num(v):
        if v is None:
            return None
        try:
            fv = float(v)
            return None if not np.isfinite(fv) else round(fv, 4)
        except (TypeError, ValueError):
            return None

    records = []
    for i in range(len(stops_wgs84)):
        # A single non-finite coordinate breaks the whole client-side `forEach` loop
        # (`L.marker([NaN, NaN])` throws), silently dropping every stop marker --
        # so stops without a usable point geometry are skipped here instead.
        if not (np.isfinite(lon[i]) and np.isfinite(lat[i])):
            continue
        row = stops_wgs84.iloc[i]
        route_val = row.get(route_col) if route_col else None
        # `stop_id`/`parent_station`: real GTFS identity fields, threaded
        # through from `transitlos.stops.download_and_prepare_stops` (real
        # linkage where the agency provides it, else a `stop_group_distance`
        # proximity fallback -- see that module's docstring / round-19's
        # memory note). `parent_station` is never null on this pipeline's
        # output (falls back to the stop's own `stop_id` when ungrouped), so
        # the client can always tell "this platform" from "this station".
        stop_id_val = row.get("stop_id")
        parent_val = row.get("parent_station")
        # `stop_routes`: routes serving this exact `stop_id` (a real,
        # per-platform GTFS aggregation -- see `stops.py`'s
        # `per_stop_id_route_list`), distinct from `all_routes`/`route`
        # (station-wide, `at="parent_station"` grouped) so the popup can show
        # "routes at this stop" separately from "other routes via this
        # station" without fabricating anything.
        stop_routes_val = row.get("stop_routes") if "stop_routes" in row else None
        all_routes_val = row.get("all_routes") if "all_routes" in row else route_val
        # `reliability_score` is NOT a measured metric on this pipeline --
        # `transitlos.stop_scores.compute_stop_scores` documents that
        # `headway_cv` (headway variance/regularity) is never available from
        # pyGTFSHandler today, so it substitutes an assumed default
        # (`default_headway_cv=0.0`, i.e. "assume perfectly regular service")
        # and always flags the row `reliability_score_is_assumed=True`. The
        # resulting `reliability_score` is therefore always a flat 1.0 for
        # every stop on every real dataset checked -- not real reliability
        # data. Only pass a real value through when a future pipeline change
        # actually measures it (`reliability_score_is_assumed is False`);
        # otherwise the client shows "N/A" rather than a fabricated number.
        reliability_assumed = bool(row.get("reliability_score_is_assumed", True))
        reliability_val = None if reliability_assumed else _num(row.get("reliability_score"))
        rec = {
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "mode": str(mode[i]),
            "stop_id": str(stop_id_val) if pd.notna(stop_id_val) else None,
            "parent_station": str(parent_val) if pd.notna(parent_val) else None,
            "parent_station_name": str(row["parent_station_name"]) if "parent_station_name" in row and pd.notna(row["parent_station_name"]) else None,
            # Real set of mode classes present at this station (e.g. "bus +
            # rail"), from `stops.py`'s cheap `station_modes` -- see its
            # comment for why this replaced an earlier, memory-unsafe
            # attempt to also recompute a mode-restricted headway/speed.
            "station_modes": str(row["station_modes"]) if "station_modes" in row and pd.notna(row["station_modes"]) else None,
            "stop_name": str(row["stop_name"]) if "stop_name" in row and pd.notna(row["stop_name"]) else None,
            "route": str(route_val) if pd.notna(route_val) else None,
            "stop_routes": str(stop_routes_val) if pd.notna(stop_routes_val) else None,
            "all_routes": str(all_routes_val) if pd.notna(all_routes_val) else None,
            "headway_minutes": _num(row.get("headway_minutes")),
            "avg_speed_kmh": _num(row.get("avg_speed_kmh")),
            "mode_score": _num(row.get("mode_score")),
            "speed_score": _num(row.get("speed_score")),
            "frequency_score": _num(row.get("frequency_score")),
            "stop_score": _num(row.get("stop_score")),
            "reliability": reliability_val,
        }
        records.append(rec)
    return json.dumps({"records": records, "score_domain": list(score_domain), "blue_scale": STOP_BLUE_HEX})


def _stops_layer_block(stops_gdf: gpd.GeoDataFrame, score_domain: Tuple[float, float], map_var: Optional[str], layer_control_var: Optional[str]) -> str:
    """Assemble the stops-layer data + Leaflet.markercluster wiring for injection into the saved map.

    A genuinely different rendering path from every other layer here: real
    Leaflet `L.marker`s with `L.divIcon` content (an emoji, no background
    shape, per the request) inside a `Leaflet.markercluster` group --
    vector-tile canvas styling has no notion of "cluster nearby features
    into a bubble at low zoom" or arbitrary per-feature HTML content, both
    of which this needs.
    """
    map_ref = f'window["{map_var}"]' if map_var else "null"
    # NOT `window["<name>"]` -- the folium variable is `let`-scoped, which
    # never becomes a `window` property (see `_extract_layer_control_var`'s
    # docstring); `_inject_controls_into_saved_html` bridges it onto this
    # explicit global instead.
    control_ref = "window.__layerControlRef" if layer_control_var else "null"
    data_json = _stops_json_data(stops_gdf, score_domain)
    mode_fallback_json = json.dumps(MODE_FALLBACK_COLOR)
    anno_min_zoom = STOP_ANNOTATION_MIN_ZOOM

    js = f"""
window.__stopsData = {data_json};
(function() {{
  var mapRef = {map_ref};
  var controlRef = {control_ref};
  if (!mapRef || !window.L || !window.L.markerClusterGroup) return;

  var blueScale = window.__stopsData.blue_scale;
  var sMin = window.__stopsData.score_domain[0], sMax = window.__stopsData.score_domain[1];

  function blueFor(t) {{
    t = Math.max(0, Math.min(1, isFinite(t) ? t : 0));
    var scaled = t * (blueScale.length - 1);
    var i = Math.floor(scaled), frac = scaled - i;
    if (i >= blueScale.length - 1) return blueScale[blueScale.length - 1];
    function hex2rgb(h) {{ h = h.replace('#', ''); return [parseInt(h.substr(0,2),16), parseInt(h.substr(2,2),16), parseInt(h.substr(4,2),16)]; }}
    var a = hex2rgb(blueScale[i]), b = hex2rgb(blueScale[i + 1]);
    var rgb = [0, 1, 2].map(function(k) {{ return Math.round(a[k] + (b[k] - a[k]) * frac); }});
    return 'rgb(' + rgb.join(',') + ')';
  }}

  // A plain colored-circle + Font Awesome vehicle icon instead of a
  // full-color vehicle emoji -- renders identically everywhere (no emoji
  // font dependency), and trivially recolorable with plain CSS `color`
  // (unlike emoji, which needed the old SVG mask-image workaround to tint
  // at all).
  //
  // ONLY Font Awesome *Free* classes may be used here: the page loads the
  // free CDN build (see the `<link>` in `_inject_controls_into_saved_html`),
  // in which a Pro-only class silently renders NOTHING (its `::before`
  // `content` computes to `none`, leaving an empty circle). `fa-train-simple`
  // / `fa-tram-simple` were exactly that bug. Verified-free replacements:
  //   rail -> `fa-train`      (locomotive front; reads as heavy rail)
  //   tram -> `fa-train-tram` (streetcar with overhead pole; visually
  //                            distinct from `fa-train` at badge size)
  //   bus  -> `fa-bus-simple` (free despite the `-simple` suffix)
  // Note `fa-tram` is an ALIAS of `fa-cable-car` in FA6 and draws a gondola,
  // not a streetcar -- do not use it for the tram mode.
  function svgFor(mode, px) {{
    var cls = mode === 'rail' ? 'fa-train' : (mode === 'tram' ? 'fa-train-tram' : 'fa-bus-simple');
    return '<i class="fa-solid ' + cls + '" style="font-size:' + Math.round(px * 0.62) + 'px;color:#fff;"></i>';
  }}

  function modeBadgeHtml(mode, tint, sizePx) {{
    var iconPx = Math.round(sizePx * 0.66);
    return '<div style="width:' + sizePx + 'px;height:' + sizePx + 'px;border-radius:50%;' +
      'background:' + tint + ';' +
      'display:flex;align-items:center;justify-content:center;' +
      'box-shadow:0 0 0 1.5px rgba(255,255,255,0.9);line-height:1;">' + svgFor(mode, iconPx) + '</div>';
  }}

  // Small inline badge for use next to text (e.g. in the popup's "mode" row),
  // sized/positioned to sit on the text baseline rather than as a standalone marker.
  function modeBadgeInline(mode, tint) {{
    return '<span style="display:inline-flex;vertical-align:middle;margin-right:4px;">' +
      modeBadgeHtml(mode, tint, 15) + '</span>';
  }}

  // The same "white circle behind the number" treatment already used for the
  // cluster count badge, applied to any numeric score label shown directly on
  // a marker (as opposed to inside the click popup) so it stays legible
  // against any basemap color.
  function scoreBadgeHtml(text, color) {{
    // Text is always black (#000) regardless of the passed-in `color` -- the
    // white pill background is meant to guarantee legibility against any
    // basemap, and tinted (light-colored) text defeated that on light tints.
    return '<div style="font-size:9px;line-height:11px;font-weight:700;color:#000;' +
      'background:rgba(255,255,255,0.9);border-radius:7px;padding:0 4px;display:inline-block;' +
      'margin-top:1px;">' + text + '</div>';
  }}

  // --- Route badges -------------------------------------------------------
  // A stop's `all_routes` is a comma-separated list of route names; each one
  // renders as a small pill filled with that route's real GTFS `route_color`
  // (text in its `route_text_color`), falling back to the mode's default
  // color / white text when the feed gives no color or the transit-lines
  // overlay wasn't built at all. Lookup is via `window.__routeStyles`
  // (published by `_routes_layer_block`), keyed by normalized name -- see
  // `_normalize_route_label` for why normalization is needed.
  var modeFallbackColor = {mode_fallback_json};
  function normRouteName(s) {{ return String(s).replace(/[_\\s]+/g, ' ').trim().toLowerCase(); }}
  function routeStyleFor(name, fallbackMode) {{
    var styles = window.__routeStyles || {{}};
    var hit = styles[normRouteName(name)];
    if (hit) return {{color: hit[0], text: hit[1], mode: hit[2]}};
    var mode = fallbackMode || 'bus';
    return {{color: modeFallbackColor[mode] || modeFallbackColor.bus, text: '#FFFFFF', mode: mode}};
  }}
  function splitRoutes(routeStr) {{
    if (!routeStr) return [];
    return String(routeStr).split(',').map(function(s) {{ return s.trim(); }}).filter(function(s) {{ return s.length > 0; }});
  }}
  function routeBadgeHtml(name, fallbackMode, fontPx) {{
    var st = routeStyleFor(name, fallbackMode);
    var display = String(name).replace(/_/g, ' ');
    return '<span style="display:inline-block;background:' + st.color + ';color:' + st.text + ';' +
      'font:700 ' + (fontPx || 10) + 'px sans-serif;line-height:1.35;padding:1px 5px;border-radius:7px;' +
      'margin:1px 3px 1px 0;white-space:nowrap;box-shadow:0 0 0 1px rgba(0,0,0,0.18);">' + display + '</span>';
  }}
  window.__routeBadgeHtml = routeBadgeHtml;
  window.__splitRoutes = splitRoutes;
  // Item 6/7: shared with the scenario editor (`_editor_js`) so a route's
  // station markers -- both the actively-edited route and every other
  // user-drawn route -- render the SAME mode-specific rail/tram/bus icon
  // real GTFS stops use, everywhere on the page, instead of a generic dot.
  window.__modeBadgeHtml = modeBadgeHtml;
  // Item 4: same publication, for the same reason -- the scenario editor's
  // user-drawn station markers should tint by stop_score exactly like real
  // GTFS stops do (`blueFor(t)` where `t` is normalized against this map's
  // own score_domain), not a flat/route color. `scoreBadgeHtml` too, so a
  // freshly-drawn station can show its own "score: X.XX" pill once Computed,
  // matching a real stop's marker in every visual respect.
  window.__blueFor = blueFor;
  window.__scoreBadgeHtml = scoreBadgeHtml;
  window.__stopScoreDomain = [sMin, sMax];

  // --- Mode visibility (the bus/tram/rail checkboxes) ---------------------
  // Those checkboxes gate the route LINES layer via `window.__routeModeVisible`
  // (published by `_routes_layer_block`); the same flag has to gate everything
  // stop-side too, or unchecking "bus" leaves every bus stop, its popup's bus
  // route badges and its on-map route annotations fully visible. Resolution is
  // PER ROUTE, not per stop: a stop's own `mode` is only that of its single
  // busiest route, so a mixed rail+bus stop must keep showing its rail badge
  // while dropping the bus one -- which means going through
  // `routeStyleFor(...).mode` (the real GTFS mode of each named route, from
  // the transit-lines data) for every entry in `all_routes`, falling back to
  // the stop's own mode for a route the lines layer doesn't know about.
  function modeVisible(mode) {{
    var vis = window.__routeModeVisible;
    return !vis || vis[mode] !== false;
  }}
  function visibleRouteNames(r) {{
    return splitRoutes(r.route).filter(function(n) {{
      return modeVisible(routeStyleFor(n, r.mode).mode);
    }});
  }}
  // A stop with no route list at all can only be judged by its own mode.
  function stopIsVisible(r) {{
    var names = splitRoutes(r.route);
    if (!names.length) return modeVisible(r.mode);
    return visibleRouteNames(r).length > 0;
  }}

  // Stop names arrive with spaces replaced by underscores from the GTFS
  // handler's own normalization ("Beacon_St_opp_Walnut_St"); restore them
  // for anything shown to a reader.
  function prettyName(s) {{ return s ? String(s).replace(/_/g, ' ') : s; }}

  function fmt2(v) {{ return (v == null) ? null : v.toFixed(2); }}
  // Sub-scores (mode_score/frequency_score/speed_score) are already on a
  // [0,1] scale, same as stop_score -- reuse the same blue scale to color the
  // "score: X.XX" text itself, so a glance at the color communicates quality
  // consistently with the rest of the map's visual language (marker tint,
  // stop-score legend), not just the number.
  function scoreSpan(s) {{
    if (s == null) return '';
    return ' <span style="color:' + blueFor(s) + ';font-weight:600;">score: ' + fmt2(s) + '</span>';
  }}

  // Cluster icons deliberately show the BEST-scoring member's mode symbol
  // and tint (via iconCreateFunction below), not a plain count circle --
  // at the user's explicit request, so zooming out still communicates "the
  // best thing reachable near here" rather than hiding it behind an
  // undifferentiated cluster blob.
  var cluster = L.markerClusterGroup({{
    maxClusterRadius: 20, disableClusteringAtZoom: 17,
    iconCreateFunction: function(clusterObj) {{
      var members = clusterObj.getAllChildMarkers();
      var best = null;
      members.forEach(function(m) {{
        var meta = m.__stopMeta;
        if (!meta) return;
        if (!best || (meta.stop_score != null && (best.stop_score == null || meta.stop_score > best.stop_score))) {{
          best = meta;
        }}
      }});
      var mode = best ? best.mode : 'bus';
      var tint = best ? best.tint : blueFor(0);
      var count = clusterObj.getChildCount();
      var html = '<div style="text-align:center;">' +
        modeBadgeHtml(mode, tint, 20) +
        scoreBadgeHtml(count, '#333') +
        '</div>';
      return L.divIcon({{html: html, className: 'stop-cluster-icon', iconSize: [32, 36]}});
    }},
  }});
  var stopMarkers = [];
  window.__stopsData.records.forEach(function(r) {{
    var t = (r.stop_score != null) ? (r.stop_score - sMin) / Math.max(sMax - sMin, 1e-9) : 0;
    var tint = blueFor(t);
    var html = '<div style="text-align:center;">' +
      modeBadgeHtml(r.mode, tint, 19) +
      (r.stop_score != null ? scoreBadgeHtml(fmt2(r.stop_score), tint) : '') +
      '</div>';
    var icon = L.divIcon({{html: html, className: 'stop-marker-icon', iconSize: [22, 29]}});
    var marker = L.marker([r.lat, r.lon], {{icon: icon}});
    // Read by iconCreateFunction above to pick each cluster's best member.
    marker.__stopMeta = {{mode: r.mode, stop_score: r.stop_score, tint: tint}};

    // Table-style popup: label | value [score: X.XX], mirroring the
    // "place-chmnl (rail) / route ... / mode rail score: 1.00 / ..." layout
    // requested, with the small mode badge shown next to the mode value too.
    //
    // Built lazily (a function passed to `bindPopup`, which Leaflet calls on
    // open) rather than eagerly per stop: the route badges need
    // `window.__routeStyles` from the separately-injected transit-lines
    // script, and deferring to open time removes any dependence on which of
    // the two `load` listeners happens to run first -- besides not building
    // thousands of popup HTML strings the user will never open.
    function popupHtml() {{
    var rows = '';
    var routeNames = visibleRouteNames(r);
    var routeCell = routeNames.length
      ? routeNames.map(function(n) {{ return routeBadgeHtml(n, r.mode, 10); }}).join('')
      : '-';
    rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;vertical-align:top;">route</td>' +
      '<td style="padding:2px 0;">' + routeCell + '</td></tr>';
    rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">mode</td><td style="padding:2px 0;">' +
      modeBadgeInline(r.mode, tint) + r.mode + scoreSpan(r.mode_score) + '</td></tr>';
    if (r.headway_minutes != null) {{
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">headway</td><td style="padding:2px 0;">' + fmt2(r.headway_minutes) + ' min' + scoreSpan(r.frequency_score) + '</td></tr>';
    }}
    if (r.avg_speed_kmh != null) {{
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">speed</td><td style="padding:2px 0;">' + fmt2(r.avg_speed_kmh) + ' km/h' + scoreSpan(r.speed_score) + '</td></tr>';
    }}
    rows += '<tr><td style="font-weight:700;padding:4px 8px 0 0;border-top:1px solid #eee;white-space:nowrap;">stop score</td>' +
      '<td style="padding:4px 0 0;border-top:1px solid #eee;font-weight:700;white-space:nowrap;color:' + (r.stop_score != null ? blueFor(t) : '#333') + ';">' +
      (r.stop_score != null ? fmt2(r.stop_score) : '-') + '</td></tr>';
    // `minWidth`/`maxWidth` bindPopup options (not just CSS on the inner
    // div) -- Leaflet's popup wrapper otherwise sizes to its own default
    // max-width (300px) first and can still wrap a row's content before the
    // inner div's `min-width` has any effect, which is what put "stop
    // score"'s label+value pair on two lines.
    return '<div style="font:12px sans-serif;">' +
      '<div style="font-weight:700;margin-bottom:4px;white-space:nowrap;">' + prettyName(r.stop_name || 'Stop') + ' (' + r.mode + ')</div>' +
      '<table style="border-collapse:collapse;width:100%;">' + rows + '</table></div>';
    }}
    // A FUNCTION, not a prebuilt string: Leaflet re-invokes it on every
    // open, so the badge list automatically reflects whatever modes are
    // checked at the moment the user clicks -- no popup refresh wiring.
    marker.bindPopup(popupHtml, {{minWidth: 210, maxWidth: 260}});
    stopMarkers.push({{marker: marker, record: r}});
    cluster.addLayer(marker);
  }});

  // Live per-mode stop visibility: a stop whose every serving route belongs
  // to an unchecked mode is removed from the cluster group outright (rather
  // than left on the map with an empty popup), and comes back the moment any
  // of its modes is re-checked.
  window.__updateStopVisibility = function() {{
    stopMarkers.forEach(function(entry) {{
      var want = stopIsVisible(entry.record);
      var has = cluster.hasLayer(entry.marker);
      if (want && !has) cluster.addLayer(entry.marker);
      else if (!want && has) cluster.removeLayer(entry.marker);
    }});
    if (window.__updateStopAnnotations) window.__updateStopAnnotations();
  }};

  cluster.addTo(mapRef);
  // Exposed so the consolidated control panel's "stops" checkbox (item H)
  // can add/remove this marker-cluster group directly, the same way it
  // drives the streets/development FeatureGroups -- this layer isn't a
  // folium-generated `feature_group_XXXX` var (it's built dynamically here
  // in JS), so it needs its own explicit global instead of going through
  // `_extract_base_layer_vars`.
  window.__stopsClusterLayer = cluster;
  if (controlRef && controlRef.addOverlay) controlRef.addOverlay(cluster, 'stops');

  // --- On-map stop annotations: "stop names" + "routes" checkboxes --------
  // Both off by default, both gated to zoom >= STOP_ANNOTATION_MIN_ZOOM, and
  // both rendered only for the stops currently *in view*: a real city has
  // thousands of stops, and materializing a DOM label for every one of them
  // would cost far more than it's worth for labels that are only legible at
  // close zoom anyway. Rebuilt on every `moveend`/`zoomend` from the same
  // `window.__stopsData.records` the markers come from.
  var ANNO_MIN_ZOOM = {anno_min_zoom};
  var ROUTES_PER_PAGE = 4;
  var MAX_ANNOTATED = 400;
  window.__stopAnnoOpts = {{names: false, routes: false}};
  var namesLayer = L.layerGroup();
  var badgesLayer = L.layerGroup();
  // Per-stop paging position for the "routes" annotation, keyed by record
  // index, so a stop served by many routes shows a compact page of badges
  // plus small prev/next arrows instead of an unbounded wrapping blob.
  var routePage = {{}};

  function nameLabelHtml(r) {{
    return '<div style="position:absolute;left:15px;top:-7px;white-space:nowrap;' +
      'font:600 11px sans-serif;color:#1a1a1a;background:rgba(255,255,255,0.85);' +
      'padding:0 3px;border-radius:3px;pointer-events:none;">' + prettyName(r.stop_name || '') + '</div>';
  }}

  function badgesLabelHtml(r, idx) {{
    // Same per-route mode filter the popup uses, so the on-map badges and
    // the popup can never disagree about which routes are showing.
    var names = visibleRouteNames(r);
    if (!names.length) return null;
    var pages = Math.ceil(names.length / ROUTES_PER_PAGE);
    var page = Math.min(routePage[idx] || 0, pages - 1);
    var slice = names.slice(page * ROUTES_PER_PAGE, (page + 1) * ROUTES_PER_PAGE);
    var inner = slice.map(function(n) {{ return routeBadgeHtml(n, r.mode, 9); }}).join('');
    var pager = '';
    if (pages > 1) {{
      pager = '<span style="display:inline-block;white-space:nowrap;font:600 9px sans-serif;color:#333;">' +
        '<span class="stop-route-pg" data-dir="-1" style="cursor:pointer;padding:0 3px;">&#9664;</span>' +
        (page + 1) + '/' + pages +
        '<span class="stop-route-pg" data-dir="1" style="cursor:pointer;padding:0 3px;">&#9654;</span></span>';
    }}
    return '<div style="position:absolute;left:15px;top:6px;white-space:nowrap;' +
      'background:rgba(255,255,255,0.85);border-radius:4px;padding:1px 3px;">' + inner + pager + '</div>';
  }}

  function attachPager(marker, r, idx) {{
    var el = marker.getElement();
    if (!el) return;
    var buttons = el.querySelectorAll('.stop-route-pg');
    for (var i = 0; i < buttons.length; i++) {{
      (function(btn) {{
        L.DomEvent.on(btn, 'click', function(ev) {{
          L.DomEvent.stop(ev);
          var names = visibleRouteNames(r);
          var pages = Math.ceil(names.length / ROUTES_PER_PAGE);
          var dir = parseInt(btn.getAttribute('data-dir'), 10);
          routePage[idx] = ((routePage[idx] || 0) + dir + pages) % pages;
          marker.setIcon(L.divIcon({{
            html: badgesLabelHtml(r, idx), className: 'stop-routes-anno', iconSize: [0, 0],
          }}));
          // `setIcon` replaces the marker's DOM element outright, taking the
          // listeners above with it -- so they have to be re-bound to the
          // freshly created element, or the pager works exactly once.
          attachPager(marker, r, idx);
        }});
      }})(buttons[i]);
    }}
  }}

  window.__updateStopAnnotations = function() {{
    var opts = window.__stopAnnoOpts;
    var z = mapRef.getZoom();
    var showNames = !!opts.names && z >= ANNO_MIN_ZOOM;
    var showRoutes = !!opts.routes && z >= ANNO_MIN_ZOOM;
    namesLayer.clearLayers();
    badgesLayer.clearLayers();
    if (!showNames && mapRef.hasLayer(namesLayer)) mapRef.removeLayer(namesLayer);
    if (!showRoutes && mapRef.hasLayer(badgesLayer)) mapRef.removeLayer(badgesLayer);
    if (!showNames && !showRoutes) return;
    if (showNames && !mapRef.hasLayer(namesLayer)) namesLayer.addTo(mapRef);
    if (showRoutes && !mapRef.hasLayer(badgesLayer)) badgesLayer.addTo(mapRef);
    var bounds = mapRef.getBounds();
    var drawn = 0;
    window.__stopsData.records.forEach(function(r, idx) {{
      if (drawn >= MAX_ANNOTATED) return;
      if (!bounds.contains([r.lat, r.lon])) return;
      drawn++;
      if (showNames && r.stop_name) {{
        namesLayer.addLayer(L.marker([r.lat, r.lon], {{
          icon: L.divIcon({{html: nameLabelHtml(r), className: 'stop-name-anno', iconSize: [0, 0]}}),
          interactive: false, keyboard: false,
        }}));
      }}
      if (showRoutes) {{
        var html = badgesLabelHtml(r, idx);
        if (!html) return;
        var m = L.marker([r.lat, r.lon], {{
          icon: L.divIcon({{html: html, className: 'stop-routes-anno', iconSize: [0, 0]}}),
          keyboard: false,
        }});
        m.on('add', function() {{ attachPager(m, r, idx); }});
        badgesLayer.addLayer(m);
      }}
    }});
  }};
  mapRef.on('zoomend moveend', window.__updateStopAnnotations);
}})();
"""
    return (
        '<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css">\n'
        '<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css">\n'
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">\n'
        '<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>\n'
        "<script>\nwindow.addEventListener('load', function() {\n" + js + "\n});\n</script>\n"
    )


def _normalize_route_label(label: str) -> str:
    """Key routes are looked up by, tolerant of separator/case differences.

    A stop's `all_routes` string is built upstream (`transitlos.stops`) from
    the same GTFS `route_short_name`/`route_long_name` fields this module
    reads for the line overlay -- but it can arrive with spaces replaced by
    underscores (`"Orange_Line"` vs `"Orange Line"`) depending on the feed
    handler's own normalization. Collapsing both to lowercase
    single-space form is what lets a popup badge find its route's real GTFS
    color instead of silently falling back to the mode default.
    """
    return re.sub(r"[_\s]+", " ", str(label)).strip().lower()


def _route_styles_dict(routes_gdf: gpd.GeoDataFrame) -> Dict[str, List[str]]:
    """Route-badge style lookup shared by both renderers: name (normalized) -> [color, text_color, mode].

    Extracted from `_routes_json_data` (round 15) so the MapLibre-native
    `_routes_geojson_data` can publish the exact same `window.__routeStyles`
    Folium's route-badge chips already use, instead of duplicating the
    color/normalization logic.
    """
    styles: Dict[str, List[str]] = {}
    for _, row in routes_gdf.iterrows():
        mode = str(row.get("mode") or "bus")
        color = str(row.get("color") or MODE_FALLBACK_COLOR.get(mode, "#8b0000"))
        text_color = str(row.get("text_color") or "#FFFFFF")
        for key in (row.get("route_label"), row.get("route_short_name"), row.get("route_long_name"), row.get("route_id")):
            if key is None or (isinstance(key, float) and np.isnan(key)) or str(key).strip() == "":
                continue
            styles.setdefault(_normalize_route_label(key), [color, text_color, mode])
    return styles


def _routes_json_data(routes_gdf: gpd.GeoDataFrame) -> str:
    """Column-oriented JSON for the transit-lines overlay + the route-badge style lookup.

    Geometry is emitted as `[[lat, lon], ...]` arrays (Leaflet's own
    `L.polyline` order, not GeoJSON's lon/lat) rounded to 5 decimals (~1 m):
    these lines are a display overlay embedded inline in the HTML, so full
    float precision would multiply the page size for sub-meter detail no
    zoom level can show.
    """
    gdf = routes_gdf.to_crs(4326) if routes_gdf.crs is not None else routes_gdf

    def _parts(geom) -> List[List[List[float]]]:
        if geom is None or geom.is_empty:
            return []
        geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        out = []
        for line in geoms:
            coords = [[round(y, 5), round(x, 5)] for x, y in line.coords]
            if len(coords) >= 2:
                out.append(coords)
        return out

    routes = []
    styles: Dict[str, List[str]] = {}
    for _, row in gdf.iterrows():
        parts = _parts(row.geometry)
        mode = str(row.get("mode") or "bus")
        color = str(row.get("color") or MODE_FALLBACK_COLOR.get(mode, "#8b0000"))
        text_color = str(row.get("text_color") or "#FFFFFF")
        label = str(row.get("route_label") or row.get("route_id") or "")
        if parts:
            routes.append({"l": label, "m": mode, "c": color, "t": text_color, "g": parts})
        # The badge-style lookup is keyed by every name a stop's `all_routes`
        # string could plausibly use for this route (short name, long name,
        # raw id), so badge coloring doesn't depend on which one the stops
        # pipeline happened to pick.
        for key in (row.get("route_label"), row.get("route_short_name"), row.get("route_long_name"), row.get("route_id")):
            if key is None or (isinstance(key, float) and np.isnan(key)) or str(key).strip() == "":
                continue
            styles.setdefault(_normalize_route_label(key), [color, text_color, mode])
    return json.dumps({"routes": routes, "styles": styles, "min_zoom": ROUTE_MODE_MIN_ZOOM})


def _routes_layer_block(routes_gdf: gpd.GeoDataFrame, map_var: Optional[str]) -> str:
    """Assemble the transit-lines overlay: one Leaflet `L.polyline` set per mode, zoom-gated.

    A client-side GeoJSON-style overlay rather than a vector-tile layer, on
    purpose: there is one feature per `route_id` (hundreds, not the hundreds
    of thousands of features the h3/census/street layers carry), and the
    per-mode zoom gating + checkbox toggling is trivial with three
    `L.layerGroup`s but would need a tile rebuild to change with the
    tile-based path.

    Visibility follows the same swap-layers-in-on-`zoomend` pattern the
    zoom-banded vector-tile levels use: each mode's group is added to /
    removed from the map whenever the zoom crosses its
    `ROUTE_MODE_MIN_ZOOM` threshold or its checkbox changes.
    """
    map_ref = f'window["{map_var}"]' if map_var else "null"
    data_json = _routes_json_data(routes_gdf)
    weights_json = json.dumps(_ROUTE_MODE_WEIGHT)
    js = f"""
window.__routesData = {data_json};
window.__routeStyles = window.__routesData.styles;
(function() {{
  var mapRef = {map_ref};
  if (!mapRef || !window.L) return;
  var minZoom = window.__routesData.min_zoom;
  var weights = {weights_json};

  // One pane PER MODE, below markers (markerPane is 600) but above the
  // vector-tile canvases in the default tilePane (200) -- so lines never
  // hide a stop marker and are never hidden by a hexagon/census fill.
  //
  // Separate panes (rather than one shared pane) are what enforce the
  // requested stacking: rail on top of tram on top of bus. Layer *add*
  // order cannot do this here, because groups are attached/detached
  // dynamically on `zoomend` (see `__updateRouteLayers`), so whichever mode
  // happened to cross its zoom threshold last would otherwise end up on
  // top. Pane z-index is order-independent and therefore stable.
  var paneZ = {{bus: 448, tram: 449, rail: 450}};
  var panes = {{}};
  Object.keys(paneZ).forEach(function(m) {{
    var name = 'routeLinesPane_' + m;
    if (!mapRef.getPane(name)) {{
      mapRef.createPane(name);
      mapRef.getPane(name).style.zIndex = paneZ[m];
    }}
    panes[m] = name;
  }});

  var groups = {{}};
  Object.keys(minZoom).forEach(function(m) {{ groups[m] = L.layerGroup(); }});
  window.__routeModeVisible = {{bus: true, tram: true, rail: true}};
  // Master on/off for the whole transit-lines overlay (all modes at once),
  // independent of the per-mode bus/tram/rail checkboxes -- wired to the
  // consolidated panel's "Show routes" checkbox in `_control_panel_js`.
  // Defaults to visible so unchecked-by-default behavior never changes for
  // pages that don't wire the checkbox at all.
  if (window.__routesMasterVisible === undefined) window.__routesMasterVisible = true;

  window.__routesData.routes.forEach(function(r) {{
    var group = groups[r.m] || groups.bus;
    var weight = weights[r.m] || 2;
    r.g.forEach(function(part) {{
      var line = L.polyline(part, {{
        color: r.c, weight: weight, opacity: 0.85, pane: panes[r.m] || panes.bus,
        // No fill and a modest click tolerance: these sit under the stop
        // markers and must not swallow clicks meant for them.
        interactive: true, bubblingMouseEvents: false,
      }});
      line.bindTooltip(r.l, {{sticky: true, direction: 'top'}});
      group.addLayer(line);
    }});
  }});
  window.__routeLayerGroups = groups;

  // Assigned onto `window` (not a closure-local function): the consolidated
  // settings panel's bus/tram/rail checkboxes live in a completely separate
  // injected `<script>`'s own `load` closure, which can only reach helpers
  // published on `window` (see `__fmtLegendNum`'s comment for the bug this
  // convention exists to prevent).
  window.__updateRouteLayers = function() {{
    var z = mapRef.getZoom();
    Object.keys(groups).forEach(function(m) {{
      var visible = window.__routesMasterVisible !== false && !!window.__routeModeVisible[m] && z >= minZoom[m];
      var attached = mapRef.hasLayer(groups[m]);
      if (visible && !attached) groups[m].addTo(mapRef);
      else if (!visible && attached) mapRef.removeLayer(groups[m]);
    }});
  }};
  mapRef.on('zoomend', window.__updateRouteLayers);
  window.__updateRouteLayers();
}})();
"""
    return "<script>\nwindow.addEventListener('load', function() {\n" + js + "\n});\n</script>\n"


def _extract_map_var(html: str) -> Optional[str]:
    """Find the JS variable name folium assigned the Leaflet `L.map(...)` instance.

    It's declared `var map_XXXXXXXX = L.map(...)` at the top level of a
    plain (non-module) `<script>` tag, which makes it a `window` property --
    so once we know its name, `window["<name>"]` reaches the live map
    instance from our own injected `<script>` blocks (added after folium's,
    both deferred to `window`'s `load` event so ordering doesn't matter).
    """
    m = re.search(r"var\s+(map_[0-9a-fA-F]+)\s*=\s*L\.map\(", html)
    return m.group(1) if m else None


def _control_panel_html(circle_fields: List[str], opacity_fields: List[str], default_circle_field: Optional[str] = None) -> str:
    """Raw HTML for the ONE consolidated map-controls panel (background/shape/overlays/styling).

    Toggled open/closed by a small circular button, mirroring the
    `statsButton`/`statsPanel` pattern (see `_stats_panel_html`) -- this is
    the single home for every control that used to be split between the
    always-visible `mapControls` box and the native `L.control.layers`
    radio/checkbox widget (which is now hidden; see `_control_panel_js`).
    """
    circle_options = "".join(
        f'<option value="{f}"{" selected" if f == default_circle_field else ""}>{field_label(f)}</option>'
        for f in circle_fields
    )
    opacity_options = "".join(f'<option value="{f}">{field_label(f)}</option>' for f in opacity_fields)
    bg_options = "".join(
        f'<option value="{key}"{" selected" if key == DEFAULT_BACKGROUND else ""}>{spec["label"]}</option>'
        for key, spec in BACKGROUND_MAPS.items()
    )
    return f"""
<style>
  #mapSettingsButton {{ transition: transform 0.12s, box-shadow 0.12s; }}
  #mapSettingsButton:hover {{ transform: scale(1.08); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }}
  #mapControls label {{ display:block; font-weight:600; color:#444; margin-bottom:2px; font-size:11px; }}
  #mapControls select, #mapControls input[type=range] {{
    width:100%; padding:3px 5px; border:1px solid #d5d9de; border-radius:4px;
    background:#f9fafb; font:11.5px sans-serif; color:#222; box-sizing:border-box;
  }}
  #mapControls select:hover {{ border-color:#b7bec6; }}
  #mapControls .field-row {{ margin-bottom:5px; }}
  #mapControls .checkbox-row label {{
    display:flex; align-items:center; gap:5px; font-weight:normal; color:#333; cursor:pointer; font-size:11.5px;
  }}
  #mapControls .divider {{ margin:6px 0; border-top:1px solid #e5e8eb; }}
  /* Multiple related checkboxes on a single compact row (transit-line modes,
     stop annotations) -- same label styling as `.checkbox-row`, just laid
     out side by side so a 3-way or 2-way group costs one row, not three. */
  #mapControls .checkbox-grid {{ display:flex; gap:6px; }}
  #mapControls .checkbox-grid label {{
    display:flex; align-items:center; gap:3px; font-weight:normal; color:#333;
    cursor:pointer; font-size:11px; margin-bottom:0; white-space:nowrap;
  }}
  #mapControls .checkbox-grid input {{ margin:0; }}
</style>
<div id="mapSettingsButton" title="Map settings" style="position:absolute;bottom:20px;right:20px;z-index:1000;
  background:white;width:34px;height:34px;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,0.3);
  display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer;">🗺️</div>
<div id="mapControls" style="display:none;position:absolute;bottom:64px;right:20px;z-index:1000;background:white;
  padding:8px 10px;border-radius:6px;box-shadow:0 1px 6px rgba(0,0,0,0.35);font:11.5px sans-serif;
  width:230px;max-height:75vh;overflow-y:auto;box-sizing:border-box;">

  <div class="field-row">
    <label>Background map</label>
    <select id="backgroundSelect">
      {bg_options}
    </select>
  </div>
  <div class="field-row">
    <label>Basemap opacity <span id="basemapOpacityValue" style="font-weight:normal;color:#666;"></span></label>
    <input type="range" id="basemapOpacitySlider" min="0" max="100" value="60">
  </div>
  <div class="checkbox-row">
    <label><input type="checkbox" id="bwCheckbox"> Black &amp; white basemap</label>
  </div>

  <div class="divider"></div>
  <div class="field-row">
    <label>Shape</label>
    <select id="shapeSelect">
      <option value="hexagons">Hexagons</option>
      <option value="circles">Circles</option>
      <option value="census">Census</option>
      <option value="none">None</option>
    </select>
  </div>
  <div id="circleFieldRow" class="field-row" style="display:none;">
    <label>Circle size by</label>
    <select id="circleFieldSelect">
      {circle_options}
    </select>
  </div>
  <div class="field-row">
    <label>Opacity <span id="shapeOpacityValue" style="font-weight:normal;color:#666;"></span></label>
    <input type="range" id="shapeOpacitySlider" min="0" max="100" value="70">
  </div>

  <div class="divider"></div>
  <div class="checkbox-row" style="margin-bottom:4px;">
    <label><input type="checkbox" id="streetsCheckbox" checked> Streets</label>
  </div>
  <div class="checkbox-row" style="margin-bottom:4px;">
    <label><input type="checkbox" id="developmentCheckbox"> More transit/housing</label>
  </div>
  <div class="checkbox-row">
    <label><input type="checkbox" id="stopsCheckbox" checked> Stops</label>
  </div>

  <div class="divider"></div>
  <!-- Master toggle for the whole GTFS transit-lines overlay, independent
       of the per-mode bus/tram/rail checkboxes below (which stay checked
       even while this is off, so re-checking this restores exactly the
       mode selection the user had). -->
  <div class="checkbox-row" style="margin-bottom:4px;">
    <label><input type="checkbox" id="showRoutesCheckbox" checked> Show routes</label>
  </div>
  <!-- Transit lines, all three modes on one compact row (each is
       additionally zoom-gated: rail >= 12, tram >= 13, bus >= 14). -->
  <div class="checkbox-grid" style="margin-bottom:4px;">
    <label><input type="checkbox" id="busLinesCheckbox" checked> bus</label>
    <label><input type="checkbox" id="tramLinesCheckbox" checked> tram</label>
    <label><input type="checkbox" id="railLinesCheckbox" checked> rail</label>
  </div>
  <!-- Per-stop on-map text annotations, both off by default and only
       rendered at zoom >= {STOP_ANNOTATION_MIN_ZOOM}. -->
  <div class="checkbox-grid">
    <label><input type="checkbox" id="stopNamesCheckbox"> stop names</label>
    <label><input type="checkbox" id="stopRoutesCheckbox"> Route labels</label>
  </div>

  <div class="divider"></div>
  <div class="field-row">
    <label>Opacity by</label>
    <select id="opacitySelect">
      <option value="none">None (fixed)</option>
      {opacity_options}
    </select>
  </div>
  <div class="field-row">
    <label>Contrast <span id="opacityContrastValue" style="font-weight:normal;color:#666;"></span></label>
    <input type="range" id="opacityContrastSlider" min="0" max="100" value="50">
  </div>
</div>
"""


def _legend_html(score_domain: Tuple[float, float], show_development: bool) -> str:
    """Raw HTML for the legend control (color scale, circle size, opacity contrast, development flags)."""
    score_min, score_max = score_domain
    gradient = f"linear-gradient(to right, {', '.join(RDYLGN_HEX)})"
    dev_legend = ""
    if show_development:
        dev_legend = """
<div id="devLegend" style="display:none;margin-top:8px;">
  <div style="font-weight:600;">Development opportunity</div>
  <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #ff8c00;margin-right:5px;"></span>More transit needed</div>
  <div><span style="display:inline-block;width:12px;height:12px;border:2px solid #1e6fd9;margin-right:5px;"></span>More housing needed</div>
</div>"""
    # A clean 6-label set every 0.2 (0.0-1.0) instead of every 0.1 (10 labels) --
    # fewer, less-crowded labels that still mark the full range unambiguously.
    score_ticks = "".join(f'<span style="flex:1;text-align:center;">{v:.1f}</span>' for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    stop_gradient = f"linear-gradient(to right, {', '.join(STOP_BLUE_HEX)})"
    stop_score_ticks = "".join(
        f'<span style="flex:1;text-align:center;">{score_min + (score_max - score_min) * v:.1f}</span>'
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    return f"""
<div id="mapLegend" style="background:white;padding:8px 14px;border-radius:6px;
  box-shadow:0 1px 4px rgba(0,0,0,0.3);font:12px sans-serif;width:min(210px,80vw);margin-top:6px;
  box-sizing:border-box;">
  <div style="font-weight:600;">Level of service</div>
  <div style="height:10px;border-radius:2px;background:{gradient};margin:5px 0 3px;"></div>
  <div style="display:flex;font-size:9.5px;">
    {score_ticks}
  </div>
  <div style="font-weight:600;margin-top:8px;">Stop score</div>
  <div style="height:10px;border-radius:2px;background:{stop_gradient};margin:5px 0 3px;"></div>
  <div style="display:flex;font-size:9.5px;">
    {stop_score_ticks}
  </div>
  <div id="circleLegend" style="display:none;margin-top:8px;">
    <div style="font-weight:600;">Circle size (<span id="circleLegendField"></span>)</div>
    <div style="display:flex;align-items:flex-end;gap:10px;margin-top:6px;padding-left:2px;">
      <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#888;"></div>
        <span id="circleLegendMin" style="font-size:9.5px;margin-top:4px;"></span>
      </div>
      <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
        <div style="width:18px;height:18px;border-radius:50%;background:#888;"></div>
        <span id="circleLegendMid" style="font-size:9.5px;margin-top:4px;"></span>
      </div>
      <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
        <div style="width:30px;height:30px;border-radius:50%;background:#888;"></div>
        <span id="circleLegendMax" style="font-size:9.5px;margin-top:4px;"></span>
      </div>
    </div>
  </div>
  <div id="opacityLegend" style="display:none;margin-top:8px;">
    <div style="font-weight:600;">Opacity (<span id="opacityLegendField"></span>)</div>
    <div style="height:10px;border-radius:2px;margin:3px 0;
      background:linear-gradient(to right, rgba(80,80,80,0.12), rgba(80,80,80,0.92));"></div>
    <div style="display:flex;justify-content:space-between;margin-top:3px;">
      <span id="opacityLegendMin"></span><span id="opacityLegendMax"></span>
    </div>
  </div>
  {dev_legend}
</div>
"""


def _control_panel_js(
    map_var: Optional[str],
    default_circle_field: Optional[str],
    radius_field_domains: Dict[str, tuple],
    opacity_field_domains: Dict[str, tuple],
    base_layer_vars: Optional[Dict[str, str]] = None,
    default_shape: str = "hexagons",
    radius_field_domains_by_res: Optional[Dict[int, Dict[str, tuple]]] = None,
    circle_zoom_bands: Optional[Dict[int, tuple]] = None,
) -> str:
    """JS wiring the consolidated control panel + legend to the map.

    The shape choice (hexagons/circles/census/none) and the streets/
    development overlay checkboxes are driven directly via
    `map.addLayer()`/`removeLayer()` on the underlying `feature_group_XXXX`
    Leaflet `FeatureGroup`s folium created (looked up by `base_layer_vars`,
    see `_extract_base_layer_vars`) -- NOT by clicking the native
    `L.control.layers` radio/checkbox inputs, since that widget has no
    concept of a base-layer "None" (item F) and this panel replaces it
    entirely (it's hidden via CSS below, kept in the DOM only because its
    internal `layeradd`/`layerremove` listener is what still fires the
    `baselayerchange`/`overlayadd`/`overlayremove` events this file's other
    logic (`__reorderLayers`, the circle-field row toggle, the development
    legend toggle) depends on -- those fire for ANY layer registered with
    the control being added/removed, whether via its own UI or plain
    `map.addLayer()`, so hiding rather than removing it keeps everything
    downstream working unchanged.
    """
    map_ref = f'window["{map_var}"]' if map_var else "null"
    radius_domains_json = json.dumps({k: list(v) for k, v in radius_field_domains.items()})
    opacity_domains_json = json.dumps({k: list(v) for k, v in opacity_field_domains.items()})
    # Per-resolution radius domains + the zoom band each resolution is
    # actually shown at (2026-09-04, "circle size legend should update with
    # zoom") -- lets the legend text pick the domain matching whichever
    # resolution the map is currently displaying, instead of always reading
    # off the flat (finest-resolution) `radius_field_domains` above.
    radius_domains_by_res_json = json.dumps(
        {str(res): {k: list(v) for k, v in domains.items()} for res, domains in (radius_field_domains_by_res or {}).items()}
    )
    circle_zoom_bands_json = json.dumps(
        {str(res): list(band) for res, band in (circle_zoom_bands or {}).items()}
    )
    base_layer_vars = base_layer_vars or {}
    shape_layers_json = json.dumps(base_layer_vars)
    default_shape_json = json.dumps(default_shape if default_shape in base_layer_vars else "hexagons")
    return f"""
window.__opacityField = 'none';
window.__opacityContrast = 0.5;
// Per-shape-type opacity defaults (issue 7, verbatim user request): census
// and hexagons default to 70%, circles to 80% -- circles get their own
// independently-defaulted value even though all three shapes still share
// the single `shapeOpacitySlider` control (only one shape is ever active at
// a time, so the slider just needs to load/store the value for whichever
// shape is currently selected; see the `shapeSelect` change handler below).
window.__shapeOpacityByType = {{hexagons: 0.7, circles: 0.8, census: 0.7}};
window.__shapeOpacity = 1;
window.__circleField = {json.dumps(default_circle_field)};
var __leafletMap = {map_ref};
window.__leafletMapRef = __leafletMap;

// The native layer control (radio buttons for hexagons/circles/census,
// checkboxes for streets/development/stops) is now fully superseded by this
// panel's shape dropdown + overlay checkboxes -- hidden, not removed (see
// this function's docstring for why removal would break other logic).
var __hideLayerControlStyle = document.createElement('style');
__hideLayerControlStyle.textContent = '.leaflet-control-layers {{ display: none !important; }}';
document.head.appendChild(__hideLayerControlStyle);

// "Opacity by <field>" mapping: `norm` is the field's p15-p85-normalized
// [0,1] value (see `_percentile_domain(..., 15, 85)`). The Contrast slider
// (`window.__opacityContrast`, 0..1) sets both the achievable opacity range
// ([1 - 0.95*c, 1]) and a gamma shaping of `norm` (exponent `1 + 2*c`).
// At contrast=0 every feature renders at opacity 1 regardless of its value
// (control effectively off); at contrast=1 low values fall to ~0.05 while
// high values stay fully opaque. The single implementation lives in
// `opacityFromField` (see `_opacity_helper_js`) and is shared verbatim by
// the hexagon, circle AND census layers.

function __redrawMatching(pred) {{
  if (!window.__vgLayers) return;
  Object.keys(window.__vgLayers).forEach(function(k) {{
    if (pred(k) && window.__vgLayers[k].redraw) window.__vgLayers[k].redraw();
  }});
}}
function __redrawAll() {{ __redrawMatching(function() {{ return true; }}); }}

// The streets overlay is deliberately drawn ABOVE the active shape (see
// `__reorderLayers`), and Leaflet.VectorGrid gives every interactive tile
// canvas `pointer-events: auto`. Those two together meant the streets canvas
// swallowed every click meant for the hexagon/circle/census fill underneath
// it -- whichever canvas is topmost at a pixel receives the DOM click, and a
// canvas that finds no feature there simply discards it rather than passing
// it down to the canvas below. Streets carry no popup worth opening (their
// only property is level_of_service, already shown by the shape popup), so they
// opt out of hit-testing entirely and let clicks through to the shape below.
// Re-applied on `overlayadd` because toggling the checkbox re-attaches a
// fresh container.
// NOTE it has to be set on every TILE canvas, not once on the layer's
// container div: VectorGrid writes `pointerEvents = 'auto'` onto each canvas
// element itself (`L.Canvas.Tile.initialize`), and an explicit value on the
// child beats an inherited `none` from the parent -- setting only the
// container looked right and changed nothing (verified headless: the streets
// canvas was still `elementFromPoint`'s answer). Tiles are created lazily as
// you pan/zoom, so new ones are caught via `tileload` as well.
function __makeStreetsClickThrough() {{
  if (!window.__vgLayers) return;
  Object.keys(window.__vgLayers).forEach(function(k) {{
    if (k.indexOf('streets:') !== 0) return;
    var lyr = window.__vgLayers[k];
    var tiles = lyr._tiles || {{}};
    Object.keys(tiles).forEach(function(tk) {{
      if (tiles[tk].el && tiles[tk].el.style) tiles[tk].el.style.pointerEvents = 'none';
    }});
    if (!lyr.__clickThroughBound) {{
      lyr.__clickThroughBound = true;
      lyr.on('tileload', function(e) {{
        if (e.tile && e.tile.style) e.tile.style.pointerEvents = 'none';
      }});
    }}
  }});
}}

function __reorderLayers() {{
  // Streets, then development-opportunity borders (in that order, so development ends
  // up on top of both), always rendered above whichever shape (hexagons/circles/
  // census) is currently active. Switching the active shape (a `baselayerchange`)
  // freshly attaches that shape's canvas to the map, which can otherwise land above
  // the already-attached streets/development canvases -- so this has to re-run on
  // every shape change too, not just when streets/development are toggled.
  if (!window.__vgLayers) return;
  Object.keys(window.__vgLayers).forEach(function(k) {{
    if (k.indexOf('streets:') === 0 && window.__vgLayers[k].bringToFront) window.__vgLayers[k].bringToFront();
  }});
  Object.keys(window.__vgLayers).forEach(function(k) {{
    if (k.indexOf('development:') === 0 && window.__vgLayers[k].bringToFront) window.__vgLayers[k].bringToFront();
  }});
}}

if (__leafletMap) {{
  __leafletMap.on('overlayadd overlayremove baselayerchange', __reorderLayers);
  __leafletMap.on('overlayadd baselayerchange', __makeStreetsClickThrough);
  __leafletMap.on('zoomend', function() {{
    // Streets re-render for the zoom-gated visibility cutoff (`street_min_zoom`,
    // see `_edges_style_js`); development borders re-render for their
    // zoom-responsive stroke weight (see `_development_style_js`). Both style
    // functions already read the live zoom off `window.__leafletMapRef`, but
    // VectorGrid only re-invokes a level's style function when that level's
    // canvas is told to `.redraw()` -- it does not re-run styling on its own
    // just because the map zoomed, so without this the already-rendered tile
    // is left showing whatever weight/opacity it was drawn with.
    __redrawMatching(function(k) {{ return k.indexOf('streets:') === 0 || k.indexOf('development:') === 0; }});
    __updateCircleLegend(window.__circleField);
  }});
  __leafletMap.on('overlayadd', function(e) {{
    if (e.name === 'development') document.getElementById('devLegend').style.display = 'block';
  }});
  __leafletMap.on('overlayremove', function(e) {{
    if (e.name === 'development') document.getElementById('devLegend').style.display = 'none';
  }});
  __leafletMap.on('baselayerchange', function(e) {{
    var isCircles = e.name === 'circles';
    document.getElementById('circleFieldRow').style.display = isCircles ? 'block' : 'none';
    document.getElementById('circleLegend').style.display = isCircles ? 'block' : 'none';
  }});
}}

var __radiusFieldDomains = {radius_domains_json};
var __opacityFieldDomains = {opacity_domains_json};
var __radiusFieldDomainsByRes = {radius_domains_by_res_json};
var __circleZoomBands = {circle_zoom_bands_json};
// Picks the resolution whose zoom band contains `z`, for legend text that
// tracks whichever resolution is actually being displayed at the current
// zoom. Falls back to the finest (highest-numbered) resolution if `z` is
// past every band's upper end (shouldn't happen -- bands partition the
// whole [0,25] range -- but keeps this defensive rather than returning
// undefined).
function __resForZoom(z) {{
  var best = null;
  Object.keys(__circleZoomBands).forEach(function(resKey) {{
    var band = __circleZoomBands[resKey];
    if (z >= band[0] && z <= band[1]) best = resKey;
  }});
  if (best != null) return best;
  var keys = Object.keys(__circleZoomBands).map(Number);
  if (!keys.length) return null;
  // 2026-09-06 bug fix (live report -- circle legend showing a tiny max
  // value like "10" at very low zoom, when the huge res-5 circles actually
  // on screen hold thousands of people): `z` past every band means EITHER
  // zoomed in further than the finest resolution's band (rare, should keep
  // showing the finest res) OR zoomed OUT further than the coarsest
  // resolution's band lower bound (the common case at low zoom -- H3
  // resolution numbers increase with granularity, so the coarsest
  // resolution is the LOWEST key, not the highest). Picking `Math.max`
  // unconditionally always answered "finest resolution", so a low-zoom
  // legend was reading the finest resolution's tiny per-cell domain
  // instead of the coarse resolution's actual (huge) one.
  var minKey = Math.min.apply(null, keys), maxKey = Math.max.apply(null, keys);
  var belowAll = z < __circleZoomBands[String(minKey)][0];
  return String(belowAll ? minKey : maxKey);
}}
// Assigned onto `window`, not a plain top-level `function` declaration --
// this whole block runs inside its own `window.addEventListener('load',
// function() {{...}})` closure, and `_stats_panel_js`'s `renderRegression`
// (a completely separate injected `<script>` with its own `load` closure)
// calls this same helper. A closure-local `function __fmtLegendNum(){{}}`
// is invisible outside its own closure, throwing "__fmtLegendNum is not
// defined" the moment `renderRegression` reached its axis-tick-label code
// -- silently aborting *after* `regSummary`'s R^2 text was already set but
// before the SVG plot ever rendered (exactly the "shows R^2, no graph" bug,
// found 2026-08-12 via headless-browser reproduction). `window` is shared
// across every closure, so this makes the helper reachable from any of
// this page's independently-injected scripts.
// 2026-09-06, explicit user request ("numbers with k or M to try avoiding
// numbers with more than 3 significant digits. No decimals if some of the
// legend values are above 100"): `refAbs` is the largest magnitude among a
// legend's related values (e.g. a circle-size legend's min/mid/max shown
// together) -- when given and >= 100, EVERY value in that group drops
// decimals for a consistent look, even a small one that would otherwise
// still show a decimal (a legend reading "0.5 / 50 / 12k" mixes precision
// levels the moment any sibling value is large). Single-value call sites
// (regression axis ticks, etc.) omit `refAbs`, keeping this function's
// original per-value behavior for them.
window.__fmtLegendNum = function(v, refAbs) {{
  if (v == null || isNaN(v)) return '';
  var abs = Math.abs(v);
  if (abs === 0) return '0';
  var ref = (refAbs != null && isFinite(refAbs)) ? Math.abs(refAbs) : abs;
  var noDecimals = ref >= 100;
  // >=4-digit values get a k/M suffix instead of a long run of digits (e.g.
  // 12345 -> "12.3k", not "12,345") -- chosen from the VALUE's own
  // magnitude, not the group's, so a small sibling in a mixed-magnitude
  // legend (e.g. min=50 next to max=10050) still reads "50", not a
  // group-scale "0k". Only the decimals-or-not policy above is shared
  // across the group.
  var scale = 1, suffix = '';
  if (abs >= 1e6) {{ scale = 1e6; suffix = 'M'; }}
  else if (abs >= 1e3) {{ scale = 1e3; suffix = 'k'; }}
  if (scale > 1) {{
    var scaled = v / scale;
    if (noDecimals) {{
      var rounded = Math.round(scaled);
      // A k-value that rounds up to a full 1000 belongs in the next
      // suffix tier (e.g. 999,999 -> "1M", not the confusing "1,000k").
      if (suffix === 'k' && Math.abs(rounded) >= 1000) return Math.round(v / 1e6) + 'M';
      return rounded.toLocaleString() + suffix;
    }}
    // 3 significant figures: 2 decimals under 10, 1 decimal under 100, none above.
    var scaledAbs = Math.abs(scaled);
    var dec = scaledAbs >= 100 ? 0 : (scaledAbs >= 10 ? 1 : 2);
    var s = scaled.toFixed(dec);
    if (dec > 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
    return s + suffix;
  }}
  if (noDecimals) return Math.round(v).toLocaleString();
  if (abs >= 1) {{
    // 1 decimal place, unless that rounds to a whole number (or the field is
    // already an integer) -- then show no decimal at all.
    var rounded1 = Math.round(v * 10) / 10;
    if (Math.abs(rounded1 - Math.round(rounded1)) < 1e-9) return Math.round(rounded1).toLocaleString();
    return rounded1.toFixed(1);
  }}
  // abs < 1: a flat 1-decimal-place rule rounds anything under 0.05 to "0"
  // -- real bug found 2026-08-12, e.g. pop_density's true per-cell domain
  // sits in the 1e-6..1e-2 range, so its legend always read "0/0/0" no
  // matter the actual (correctly computed) min/mid/max. Scale the decimal
  // count to the value's own magnitude instead, keeping ~3 significant
  // figures at any scale, then trim trailing zeros.
  var decimals = Math.min(2, Math.max(1, 2 - Math.floor(Math.log10(abs))));
  var s = v.toFixed(decimals);
  s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s === '' || s === '-' ? '0' : s;
}};

function __updateCircleLegend(field) {{
  // Item 1: mirror the opacity legend's unit-label treatment -- prefer the
  // circle-size <select>'s own selected <option> text (which already carries
  // the field's unit suffix via `field_label`/`field_unit_suffix`) over the
  // bare field name, falling back to the bare name if the select/option
  // can't be found for some reason.
  var circleSelect = document.getElementById('circleFieldSelect');
  var circleLabel = field || '';
  if (circleSelect) {{
    for (var oi = 0; oi < circleSelect.options.length; oi++) {{
      if (circleSelect.options[oi].value === field) {{ circleLabel = circleSelect.options[oi].textContent; break; }}
    }}
  }}
  document.getElementById('circleLegendField').textContent = circleLabel;
  var domains = __radiusFieldDomains;
  if (window.__leafletMapRef && Object.keys(__radiusFieldDomainsByRes).length) {{
    var res = __resForZoom(window.__leafletMapRef.getZoom());
    if (res != null && __radiusFieldDomainsByRes[res]) domains = __radiusFieldDomainsByRes[res];
  }}
  var d = domains[field];
  // Shared `refAbs` (the group's own max) so min/mid/max always agree on
  // one k/M scale and one decimals-or-not policy -- see `__fmtLegendNum`'s
  // comment on why a per-value decision looks inconsistent across a legend.
  var refAbs = d ? Math.max(Math.abs(d[0]), Math.abs(d[1])) : null;
  document.getElementById('circleLegendMin').textContent = d ? __fmtLegendNum(d[0], refAbs) : '';
  document.getElementById('circleLegendMid').textContent = d ? __fmtLegendNum((d[0] + d[1]) / 2, refAbs) : '';
  document.getElementById('circleLegendMax').textContent = d ? __fmtLegendNum(d[1], refAbs) : '';
}}

document.getElementById('circleFieldSelect').addEventListener('change', function(e) {{
  window.__circleField = e.target.value;
  __updateCircleLegend(e.target.value);
  __redrawMatching(function(k) {{ return k.indexOf('circles:') === 0; }});
}});
__updateCircleLegend(window.__circleField);

document.getElementById('opacitySelect').addEventListener('change', function(e) {{
  window.__opacityField = e.target.value;
  var legend = document.getElementById('opacityLegend');
  if (e.target.value === 'none') {{
    legend.style.display = 'none';
  }} else {{
    legend.style.display = 'block';
    // Item 11: the dropdown's own <option> text already carries the field's
    // unit suffix (see `field_label`/`field_unit_suffix`) -- reuse it here
    // instead of the bare field name so the legend shows units too.
    var selOpt = e.target.options[e.target.selectedIndex];
    document.getElementById('opacityLegendField').textContent = selOpt ? selOpt.textContent : e.target.value;
    var d = __opacityFieldDomains[e.target.value];
    document.getElementById('opacityLegendMin').textContent = d ? __fmtLegendNum(d[0]) : '';
    document.getElementById('opacityLegendMax').textContent = d ? __fmtLegendNum(d[1]) : '';
  }}
  __redrawAll();
}});

document.getElementById('opacityContrastSlider').addEventListener('input', function(e) {{
  var pct = parseInt(e.target.value, 10);
  window.__opacityContrast = pct / 100;
  document.getElementById('opacityContrastValue').textContent = pct + '%';
  if (window.__opacityField && window.__opacityField !== 'none') {{ __redrawAll(); }}
}});
document.getElementById('opacityContrastValue').textContent = '50%';

// Shared opacity slider for whichever shape (hexagons/circles/census) is
// currently active -- on top of, not instead of, the data-driven "Opacity
// by" field above (that one maps a field's value to opacity; this one is a
// simple global boost/fade).
document.getElementById('shapeOpacitySlider').addEventListener('input', function(e) {{
  var pct = parseInt(e.target.value, 10);
  window.__shapeOpacity = pct / 100;
  // Remember this value against whichever shape is active, so switching
  // shapes and switching back restores what the user set, rather than
  // reverting to the type's original default.
  if (window.__shapeOpacityByType.hasOwnProperty(__activeShape)) {{
    window.__shapeOpacityByType[__activeShape] = window.__shapeOpacity;
  }}
  document.getElementById('shapeOpacityValue').textContent = pct + '%';
  __redrawMatching(function(k) {{ return k.indexOf('streets:') !== 0 && k.indexOf('development:') !== 0; }});
}});

// --- Shape dropdown (item F): hexagons/circles/census/none -----------------
var __shapeLayers = {shape_layers_json};
// Whichever shape `MultiHierarchyMap` was told to show by default (census
// where the study has census geometries, hexagons otherwise) -- this is the
// layer folium already attached, so the dropdown only has to AGREE with it,
// never swap anything at load time.
var __activeShape = {default_shape_json};
document.getElementById('shapeSelect').value = __activeShape;
// Load the newly-active shape's own remembered/default opacity into the
// shared slider (issue 7: circles default to 80%, hexagons/census to 70%).
function __syncShapeOpacitySlider() {{
  var op = window.__shapeOpacityByType.hasOwnProperty(__activeShape)
    ? window.__shapeOpacityByType[__activeShape] : 0.7;
  window.__shapeOpacity = op;
  var pct = Math.round(op * 100);
  var slider = document.getElementById('shapeOpacitySlider');
  if (slider) slider.value = pct;
  document.getElementById('shapeOpacityValue').textContent = pct + '%';
}}
__syncShapeOpacitySlider();
document.getElementById('shapeSelect').addEventListener('change', function(e) {{
  if (!__leafletMap) return;
  var oldVar = __shapeLayers[__activeShape];
  if (oldVar && window[oldVar]) __leafletMap.removeLayer(window[oldVar]);
  __activeShape = e.target.value;
  __syncShapeOpacitySlider();
  var newVar = __shapeLayers[__activeShape];
  if (__activeShape !== 'none' && newVar && window[newVar]) __leafletMap.addLayer(window[newVar]);
  // Explicit, not solely reliant on the `baselayerchange` event (which only
  // fires when a layer is actually added) -- selecting "none" adds nothing,
  // so the circle-field controls need to be hidden here directly too.
  var isCircles = __activeShape === 'circles';
  document.getElementById('circleFieldRow').style.display = isCircles ? 'block' : 'none';
  document.getElementById('circleLegend').style.display = isCircles ? 'block' : 'none';
  __redrawMatching(function(k) {{ return k.indexOf('streets:') !== 0 && k.indexOf('development:') !== 0; }});
}});

// --- Overlay checkboxes (streets/development/stops), driven the same way --
var __streetsVar = {json.dumps(base_layer_vars.get("streets"))};
document.getElementById('streetsCheckbox').addEventListener('change', function(e) {{
  if (!__leafletMap || !__streetsVar || !window[__streetsVar]) return;
  if (e.target.checked) __leafletMap.addLayer(window[__streetsVar]);
  else __leafletMap.removeLayer(window[__streetsVar]);
}});

var __developmentVar = {json.dumps(base_layer_vars.get("development"))};
document.getElementById('developmentCheckbox').addEventListener('change', function(e) {{
  if (!__leafletMap || !__developmentVar || !window[__developmentVar]) return;
  if (e.target.checked) __leafletMap.addLayer(window[__developmentVar]);
  else __leafletMap.removeLayer(window[__developmentVar]);
}});

document.getElementById('stopsCheckbox').addEventListener('change', function(e) {{
  if (!__leafletMap || !window.__stopsClusterLayer) return;
  if (e.target.checked) __leafletMap.addLayer(window.__stopsClusterLayer);
  else __leafletMap.removeLayer(window.__stopsClusterLayer);
}});

// --- Master "Show routes" checkbox (whole transit-lines overlay) ----------
// Independent of the per-mode bus/tram/rail checkboxes just below: this
// gates `window.__routesMasterVisible`, which `__updateRouteLayers` ANDs in
// alongside the per-mode flag and zoom threshold, so toggling it off/on
// never disturbs which individual modes are checked.
document.getElementById('showRoutesCheckbox').addEventListener('change', function(e) {{
  window.__routesMasterVisible = e.target.checked;
  if (window.__updateRouteLayers) window.__updateRouteLayers();
}});

// --- Transit-line mode checkboxes (bus/tram/rail) -------------------------
// These only record *intent*; actual visibility is the AND of this flag and
// the mode's own zoom threshold, which is why both this handler and the
// map's `zoomend` go through the same `window.__updateRouteLayers`
// (published by `_routes_layer_block` -- a separate `load` closure, hence
// the `window.` indirection).
[['busLinesCheckbox', 'bus'], ['tramLinesCheckbox', 'tram'], ['railLinesCheckbox', 'rail']].forEach(function(pair) {{
  var el = document.getElementById(pair[0]);
  if (!el) return;
  el.addEventListener('change', function(e) {{
    if (!window.__routeModeVisible) return;
    window.__routeModeVisible[pair[1]] = e.target.checked;
    if (window.__updateRouteLayers) window.__updateRouteLayers();
    // The same flag gates the stop markers, their popup route badges and the
    // on-map route annotations (see `_stops_layer_block`'s `stopIsVisible`).
    if (window.__updateStopVisibility) window.__updateStopVisibility();
    else if (window.__updateStopAnnotations) window.__updateStopAnnotations();
  }});
}});

// --- On-map stop annotations (stop names / route badges) ------------------
[['stopNamesCheckbox', 'names'], ['stopRoutesCheckbox', 'routes']].forEach(function(pair) {{
  var el = document.getElementById(pair[0]);
  if (!el) return;
  el.addEventListener('change', function(e) {{
    if (!window.__stopAnnoOpts) return;
    window.__stopAnnoOpts[pair[1]] = e.target.checked;
    if (window.__updateStopAnnotations) window.__updateStopAnnotations();
  }});
}});

// --- Scale bar ------------------------------------------------------------
// Bottom-left, where Leaflet's own control container keeps it flush to the
// corner. The stats button shares this corner but is offset upward
// (`bottom:56px`, see `_stats_panel_html`) so the two stack rather than
// overlap.
if (__leafletMap && L.control.scale) {{
  L.control.scale({{position: 'bottomleft', metric: true, imperial: true, maxWidth: 140}}).addTo(__leafletMap);
}}

// --- Settings panel toggle (mirrors #statsButton/#statsPanel) -------------
document.getElementById('mapSettingsButton').addEventListener('click', function() {{
  var panel = document.getElementById('mapControls');
  panel.style.display = (panel.style.display === 'none') ? 'block' : 'none';
}});

// --- Basemap opacity slider (item C) --------------------------------------
document.getElementById('basemapOpacitySlider').addEventListener('input', function(e) {{
  if (!__leafletMap) return;
  var pct = parseInt(e.target.value, 10);
  document.getElementById('basemapOpacityValue').textContent = pct + '%';
  var op = pct / 100;
  // ONLY the dedicated `basemapPane` (the raster background tile layer) --
  // NOT Leaflet's default `tilePane`, which is where the hexagon/circle/
  // census/streets/development VectorGrid canvases render too (they don't
  // specify their own `pane` option), so setting opacity there faded
  // everything but the stops layer (which uses `markerPane`), a real bug.
  var basemapPane = __leafletMap.getPane('basemapPane');
  if (basemapPane) basemapPane.style.opacity = op;
}});
document.getElementById('basemapOpacityValue').textContent = '60%';

// A persistent CSS rule (toggled via a class on the map container), not a
// one-off `img.style.filter` write to whatever tile <img> elements happen to
// exist at the moment the checkbox is clicked. The old approach reset to
// color on every zoom/pan: Leaflet's TileLayer creates brand-new <img>
// elements for every newly-visible tile coordinate as you zoom, and those
// never got the greyscale filter applied at all -- only the *already
// rendered* tiles at click-time did. A CSS class targets the selector
// itself, so it applies uniformly to every tile <img>, present or future,
// without any per-tile JS bookkeeping.
var __bwStyleTag = document.createElement('style');
__bwStyleTag.textContent =
  '.bw-basemap-active .leaflet-tile-pane img.leaflet-tile {{ filter: grayscale(100%); }}';
document.head.appendChild(__bwStyleTag);
document.getElementById('bwCheckbox').addEventListener('change', function(e) {{
  if (!__leafletMap) return;
  var container = __leafletMap.getContainer();
  container.classList.toggle('bw-basemap-active', e.target.checked);
}});

// Background (raster basemap) tile layers always render in their own pane,
// pinned *below* Leaflet's default 'tilePane' (zIndex 200) where the
// VectorGrid hexagon/circle/census/streets/development canvases live.
// Without this, switching basemap appended the newly-added raster
// L.tileLayer after the already-attached vector canvases in the DOM (later
// DOM siblings paint on top within the same pane's stacking context), so it
// visually covered every shape layer until the user re-picked a shape and
// forced its canvas to re-attach on top again -- the exact "polygons/
// hexagons disappear until reselected" bug. A dedicated lower-zIndex pane
// makes the basemap render underneath regardless of add/DOM order, so this
// can never regress no matter what order layers get (re)added in.
if (__leafletMap && !__leafletMap.getPane('basemapPane')) {{
  __leafletMap.createPane('basemapPane');
  __leafletMap.getPane('basemapPane').style.zIndex = 150;
}}

var __backgroundLayers = {{}};
var __backgroundSpecs = {json.dumps(BACKGROUND_MAPS)};
var __activeBackground = {json.dumps(DEFAULT_BACKGROUND)};
var __defaultBasemapSwapped = false;
function __setBackground(key) {{
  if (!__leafletMap) return;
  if (!__backgroundLayers[key]) {{
    var spec = __backgroundSpecs[key];
    __backgroundLayers[key] = L.tileLayer(spec.url, {{attribution: spec.attribution, maxZoom: 20, pane: 'basemapPane'}});
  }}
  if (__defaultBasemapSwapped && __backgroundLayers[__activeBackground]) {{
    __leafletMap.removeLayer(__backgroundLayers[__activeBackground]);
  }} else {{
    // The initial basemap was created by folium/geohierarchy directly in
    // Leaflet's default 'tilePane', not our dedicated 'basemapPane' -- find
    // it via the map's own layer registry and remove it, whether this is the
    // first user-driven switch OR the one-time swap-in below (so the basemap
    // opacity slider works on the very first drag, not only after the user
    // has changed the background dropdown at least once).
    __leafletMap.eachLayer(function(layer) {{ if (layer instanceof L.TileLayer) __leafletMap.removeLayer(layer); }});
    __defaultBasemapSwapped = true;
  }}
  __backgroundLayers[key].addTo(__leafletMap);
  __activeBackground = key;
}}
document.getElementById('backgroundSelect').addEventListener('change', function(e) {{ __setBackground(e.target.value); }});
// Swap the folium-created default basemap into 'basemapPane' immediately on
// load (not lazily on first dropdown change) so the basemap-opacity slider
// (which only touches 'basemapPane') works right away.
__setBackground(__activeBackground);
// Apply the slider's own default (60%, issue 7) to the actual pane -- just
// setting the label text above doesn't touch the pane's real CSS opacity,
// which otherwise stays at the browser default (fully opaque) until the
// user first drags the slider.
(function() {{
  var basemapPane = __leafletMap && __leafletMap.getPane('basemapPane');
  if (basemapPane) basemapPane.style.opacity = 0.6;
}})();

__reorderLayers();
__makeStreetsClickThrough();
"""


def build_city_map(
    h3_by_resolution: Dict[int, gpd.GeoDataFrame],
    edges_gdf: gpd.GeoDataFrame,
    tiles_dir: str,
    out_html: str,
    score_col: str = "level_of_service",
    street_min_zoom: int = 14,
    census_by_level: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    circle_fields: Optional[List[str]] = None,
    opacity_fields: Optional[List[str]] = None,
    stats_by_area: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    is_us: bool = False,
    stops_gdf: Optional[gpd.GeoDataFrame] = None,
    routes_gdf: Optional[gpd.GeoDataFrame] = None,
    region: str = "global",
    skip_tile_build: bool = False,
    development_gdf: Optional[gpd.GeoDataFrame] = None,
    use_pmtiles: bool = True,
    renderer: str = "maplibre",
    enable_place_comparison: bool = False,
    downloads_manifest: Optional[dict] = None,
    share_source_map: Optional[Dict[str, str]] = None,
) -> str:
    """Build and save one city's transit-LOS map.

    Args:
        h3_by_resolution: Mapping of H3 resolution -> GeoDataFrame (h3
            polygon geometry + `score_col`, optionally `equity_flag` and any
            number of other numeric columns). One `HierarchyMap` level
            (`"h3_{res}"`) is created per entry, zoom-banded so only one
            resolution renders at a time. Every column present is shown in
            the click popup (all census variables + the access value).
        edges_gdf: Street-edges GeoDataFrame with `score_col`, its own
            independently togglable overlay layer, rendered only at
            `street_min_zoom` and above and always above the active shape
            layer.
        tiles_dir: Directory vector tiles are written to.
        out_html: Output map HTML path.
        score_col: Column name colored (red-yellow-green) on every level.
        street_min_zoom: Minimum zoom streets become visible at.
        census_by_level: Optional mapping of census level name -> GeoDataFrame
            with the same geometry/score/field conventions as
            `h3_by_resolution`, in coarse-to-fine order (e.g. `county`,
            `tract`, `blockgroup`, `block` -- there is no true national
            boundary in the available Census geometry schema, so `county`
            stands in as the coarsest/"country"-scale level). Only passed
            for cities with census data; when given, a third `"census"`
            shape option (itself multi-resolution, zoom-banded exactly like
            the h3 levels) is exposed. GeoDataFrames should carry a unique
            per-row identifier; if none of `("geo_id", "GEOID", "geoid")` is
            present the DataFrame index is used.
        circle_fields: Numeric columns selectable in the "Circle size by"
            dropdown. Defaults to every numeric, non-identifier column found
            on the finest H3 resolution's GeoDataFrame.
        opacity_fields: Numeric columns selectable in the "Opacity by"
            dropdown (affects hexagons/circles/census, never streets).
            Defaults to the same list as `circle_fields`.
        stats_by_area: Optional `{"metro": gdf, "core": gdf}` (the
            `stats_h3_resolution` grid, already split) powering the
            bottom-left stats panel (distribution / regression / ANOVA
            tabs). Omit to skip the stats panel entirely.
        is_us: Whether `stats_by_area`/`h3_by_resolution` grids carry US
            Census `acs_*` columns.
        development_gdf: Optional polygon GeoDataFrame (e.g. census block
            groups) with an `equity_flag` column to draw the development-
            opportunity overlay on directly, instead of the H3-hexagon
            fallback built from `h3_by_resolution`. Any per-row identifier
            column named `GEOID`/`geo_id`/`geoid` is used as the tile
            feature key; the DataFrame index is used otherwise. Building
            straight from this (when the caller has it) instead of building
            H3 tiles and overwriting them afterward avoids writing millions
            of tiles that get immediately discarded -- see this function's
            own development-overlay section for why that two-phase dance
            existed and why it's no longer needed.
        stops_gdf: Optional stop-level GeoDataFrame (point geometry) with a
            `mode_category` column (`"bus"`/`"tram"`/`"rail"`) and a
            `stop_score` column, plus any other columns to show in each
            stop's click popup. Rendered as clustered, emoji markers
            (clustering at low zoom, individual stops once zoomed in),
            colored by `stop_score` on a blue scale -- a genuinely different
            rendering path from every other layer (real Leaflet markers via
            Leaflet.markercluster, not vector tiles), since clustering and
            arbitrary marker content aren't something vector-tile canvas
            styling can do. Omit to skip the stops layer entirely.
        routes_gdf: Optional route-level GeoDataFrame -- one Line/
            MultiLineString feature per route -- as produced by
            `transitlos.map.routes.build_route_lines`, with `route_label`,
            `mode` (`"bus"`/`"tram"`/`"rail"`), `color` and `text_color`
            columns. Rendered as a client-side polyline overlay (not vector
            tiles: it's hundreds of features, and its per-mode zoom gating
            and checkbox toggling would otherwise need a tile rebuild to
            change), one togglable `L.layerGroup` per mode, each visible
            only at/above `ROUTE_MODE_MIN_ZOOM[mode]`. Also supplies the
            per-route color/text-color lookup that stop popups and the
            on-map "routes" annotation use for their route badges; without
            it those badges fall back to `MODE_FALLBACK_COLOR`. Omit to skip
            the transit-lines overlay entirely.
        region: `transitlos.scoring.REGIONS` key whose parameters power the
            stats panel's "Discretization" tab (speed_score/frequency_score/
            reliability_score/walk_decay curve shapes) -- should match
            whatever region was actually used to compute this city's
            `stop_score`/`level_of_service` columns, so the tab reflects the
            parameters genuinely behind the map's own data, not a mismatched
            default. Only affects that one tab; every other layer/score is
            unaffected by this value.
        skip_tile_build: If True, skip regenerating vector tiles entirely
            and reuse whatever already exists on disk under `tiles_dir`
            (added 2026-08-12, for fast HTML/JS/CSS-only iteration). Tile
            generation is by far the most expensive part of this function
            (tens of minutes on a real city); nothing about a stop popup's
            HTML, a legend's CSS, an opacity slider, or the stats panel's JS
            depends on the tile *contents* (geometry/properties baked into
            the `.pbf` files) at all -- only on this function's own
            Python-generated wrapper HTML/JS, which this always regenerates
            regardless of this flag. Only set this when you know the
            underlying data (`h3_by_resolution`/`edges_gdf`/`census_by_level`
            /styling *fields*) hasn't changed since tiles were last built --
            if it has, the map will render stale or missing data using the
            new wrapper's styling logic.
        enable_place_comparison: Whether the stats panel's "Place rank" tab
            (cross-city, population-weighted median level_of_service, ranked
            across the `CS_transitLOS` multi-city roster -- see
            `CS_PLACE_MANIFEST` in `_stats_panel_js`) is shown at all.
            Defaults to False: it only makes sense for a map that is
            genuinely one of that roster's cities, where the sibling
            `../<city>/results/summary.json` fetches it performs actually
            resolve. A standalone study map outside `CS_transitLOS/` (e.g.
            `TransitLOSStudies/boston_city`) isn't in the manifest, so
            leaving this on there just fires a batch of always-404 fetches
            on open with no way for the tab to ever populate -- pass True
            only from a caller that knows this map lives at
            `CS_transitLOS/<city>/map.html`. When False, the tab (and its
            button) don't render at all; "Distribution" becomes the default
            active tab instead. Applies to both renderers.
        use_pmtiles: If True (default), generate and use PMTiles (.pmtiles)
            format for vector tiles. PMTiles are single-file archives that are
            faster to generate and easier to host. If False, use legacy XYZ
            PBF directory structure for backward compatibility.
        renderer: "maplibre" (default) or "folium". MapLibre reads the
            .pmtiles written by the tile build directly -- no XYZ/pbf
            extraction step -- and gives fast base-layer parity (per-level
            coloring, popups, hover, layer switcher) but NOT the scenario
            editor or stats panel. Pass "folium" to get the full
            Leaflet-based editor (route drawing, grade-separation paint,
            live recolor, Distribution/Regression/ANOVA tabs) -- Folium's
            editor still requires XYZ PBF tiles internally (extracted from
            the same .pmtiles by `HierarchyMap.build()`), so it remains the
            slower path; a fully PMTiles-native Folium editor (via
            protomaps-leaflet) is not yet implemented.

    Returns:
        Absolute path to the saved HTML file.

    Note:
        `HierarchyMap` embeds `tiles_dir` verbatim into the browser-side
        Leaflet tile URL, so an absolute filesystem path would produce a
        `map.html` that only loads correctly if a web server happened to be
        rooted at `/`. `tiles_dir` is converted to a path *relative to
        `out_html`'s directory*, and tile-writing is done with the working
        directory temporarily switched to that same directory so on-disk
        tiles land exactly where the relative URL expects them.
    """
    from geohierarchy import Max
    from geohierarchy.core import GeoHierarchy
    from geohierarchy.maps.folium.render import HierarchyMap, MultiHierarchyMap

    out_html_abs = os.path.abspath(out_html)
    out_dir = os.path.dirname(out_html_abs) or "."
    os.makedirs(out_dir, exist_ok=True)
    tiles_dir_abs = os.path.abspath(tiles_dir)
    os.makedirs(tiles_dir_abs, exist_ok=True)
    tiles_dir_rel = os.path.relpath(tiles_dir_abs, start=out_dir)

    resolutions = sorted(h3_by_resolution)
    finest = h3_by_resolution[resolutions[-1]]
    score_domain = _finite_domain(finest[score_col])
    has_census = bool(census_by_level)
    has_development_gdf = development_gdf is not None and not development_gdf.empty and "equity_flag" in development_gdf.columns
    show_development = has_development_gdf or (has_census and "equity_flag" in finest.columns)

    # ONE canonical (count_fields, share_fields) split, computed once from
    # `finest` and reused verbatim by every count-type selector
    # (circle-size, distribution main/overlay) and every share-type selector
    # (opacity-by, regression, ANOVA) via `_stats_field_candidates`/
    # `_stats_count_fields` below, which are now thin wrappers around this
    # same split rather than independent re-derivations. Real bug fixed
    # 2026-09-07 (live report, Beersheba): `circleFieldSelect` vs.
    # `distMainSelect`, and `opacitySelect` vs. `regFieldSelect`, each built
    # their own field list from scratch -- close but not identical, since
    # e.g. CS_transitLOS's `_apply_map_field_policy` used to patch
    # `_stats_count_fields` alone to sneak `population` back in for the
    # distribution tab without doing the same for `circle_fields`, and
    # `opacity_fields` applied an extra census-polygon-presence filter (see
    # its own comment below) that `regression_fields` never did. Building
    # this ONE pair up front removes the possibility of the two drifting.
    _all_candidates = _numeric_field_candidates(finest, extra_exclude=[score_col])
    _canonical_count_fields = absolute_fields(_all_candidates)
    _canonical_share_fields = relative_fields(_all_candidates)
    if circle_fields is None:
        # 2026-09-05, explicit user request: "circle size by only allow
        # count columns" -- a circle's AREA already encodes magnitude
        # (bigger circle = more of X), so sizing it by a count (population,
        # jobs, households, ...) is the natural fit; a relative/density
        # field is better expressed by opacity/color instead (see
        # `opacity_fields` below), which is what this swaps to. See
        # `is_relative_field`/`absolute_fields` for the naming-convention
        # rule this filters on.
        circle_fields = list(_canonical_count_fields)
        if not circle_fields and "population" in finest.columns:
            circle_fields = ["population"]
    if opacity_fields is None:
        # 2026-09-05, explicit user request: "opacity by only allow
        # relative or density columns" -- the complement of `circle_fields`
        # above, computed independently (not derived FROM circle_fields
        # anymore, since that now holds counts instead of relative fields).
        opacity_fields = list(_canonical_share_fields)
        # 2026-08-25 -> 2026-09-07: this dropdown used to be further
        # restricted to fields present on EVERY census level's own
        # GeoDataFrame (not just the h3 grid), because `opacitySelect` is
        # ONE GLOBAL dropdown shared by every shape (hexagons, circles, AND
        # census polygons -- see `__shapeLayerIdTypes` in
        # `geohierarchy.maps.maplibre.render`) and a field absent from a
        # census polygon's own tiled properties makes `['get', field]`
        # return `null` there, so `coalesce` (see `__shapeOpacityExpr`'s own
        # null-safety comment) falls back to the domain minimum for every
        # census feature -- a uniform, value-independent opacity on that
        # shape only, even though the SAME field genuinely varies on
        # hexagons/circles.
        #
        # Per explicit user request ("in all share column selectors such as
        # regression, opacity by show exactly the same column list"),
        # `opacity_fields` is now the same canonical `share_fields` list
        # `regression_fields` uses -- no census-presence filtering. The
        # tradeoff above still applies (a handful of h3-only share/density
        # fields degrade to a flat minimum opacity specifically while the
        # census shape is active), but it degrades gracefully (documented
        # coalesce fallback, not an error) and is preferred over the
        # dropdowns silently disagreeing with each other.

    # Percentile-clipped (not true min/max) for the same reason as
    # `opacity_field_domains` below: a few extreme outliers otherwise stretch
    # the domain so far that every "typical" value maps into a narrow sliver
    # of the 3-18px radius range, killing visible size contrast between them.
    radius_field_domains = {f: _percentile_domain(finest[f]) for f in circle_fields if f in finest.columns}
    # Per-resolution domains (2026-09-04, user report: "the circle size
    # legend should update with zoom"). `radius_field_domains` above is
    # computed ONCE from `finest` and reused identically at every zoom level
    # -- a coarse resolution's per-cell values (e.g. population summed over
    # a huge res-5 hexagon) are on a completely different scale than the
    # finest resolution's, so every coarse-zoom circle rendered near the
    # domain's max (always maxed-out-looking) regardless of its real
    # relative size at that zoom. Each resolution gets its own
    # percentile-clipped domain from its OWN data here, used both for the
    # actual per-resolution circle-radius paint (Folium's per-level
    # `_circle_style_js` closure and MapLibre's per-level static
    # `_score_maplibre_paint`, both already configured per resolution in the
    # loop below) and for the live legend text (`_control_panel_js`'s
    # `__updateCircleLegend`, which picks the domain matching the map's
    # current zoom via `circle_zoom_bands`).
    radius_field_domains_by_res: Dict[int, Dict[str, tuple]] = {
        res: {f: _percentile_domain(h3_by_resolution[res][f]) for f in circle_fields if f in h3_by_resolution[res].columns}
        for res in resolutions
    }
    opacity_field_domains = {f: _percentile_domain(finest[f], 15, 85) for f in opacity_fields if f in finest.columns}
    # Same default-field choice `_inject_controls_into_saved_html` uses for
    # Folium's "Circle size by" dropdown -- computed here too so MapLibre's
    # static circle-radius paint (`_score_maplibre_paint`) can size circles
    # by the same field instead of a flat radius.
    default_circle_field = _preferred_density_field(circle_fields) or (circle_fields[0] if circle_fields else None)

    edges_wgs84 = edges_gdf.to_crs(4326) if edges_gdf.crs is not None else edges_gdf.copy()
    if "edge_id" not in edges_wgs84.columns:
        edges_wgs84 = edges_wgs84.reset_index(drop=True)
        edges_wgs84["edge_id"] = edges_wgs84.index

    def _popup_fields(gdf) -> list[str]:
        # Every numeric column (all census variables + the access value) --
        # not just circle_fields -- so clicking any hexagon/circle/census
        # shape shows the full picture, per the "click to see all census
        # variables and the access value" requirement.
        fields = [score_col]
        for extra in _numeric_field_candidates(gdf):
            if extra not in fields:
                fields.append(extra)
        # Force population back into the tile's own property set regardless
        # of `_FIELD_EXCLUDE` -- see `_POPULATION_COLUMN_CANDIDATES`'s
        # docstring. The popup's fixed header row always needs a real
        # population figure to sit next to `pop_density`; without this, a
        # study that excludes `population` from the four dropdown selectors
        # (CS_transitLOS's `_apply_map_field_policy`) also silently drops it
        # from every tile, leaving `pop_density` real but `population` null
        # in every popup.
        for cand in _POPULATION_COLUMN_CANDIDATES:
            if cand in gdf.columns and cand not in fields:
                fields.append(cand)
        # `pop_density`'s duplicate-of-a-real-source-column removal (2026-09-01
        # user report) now happens once, centrally, in
        # `_numeric_field_candidates` itself -- inherited here for free since
        # `fields` above is seeded from it, rather than re-applied per call site.
        return fields

    # Delegates to the page-level `__shapePopupHtml` (see `_shape_popup_js`)
    # rather than inlining the popup markup into every level's tile config:
    # the popup is pure presentation, so keeping it in the injected script
    # means it can be changed without touching (or rebuilding) tiles.
    shape_popup_js = (
        "function(properties) { return window.__shapePopupHtml ? "
        "window.__shapePopupHtml(properties) : ''; }"
    )
    # Census levels have no single H3 resolution to bake into the override
    # lookup (they're keyed by GEOID, not h3_cell) -- res=None here, which
    # makes `_h3_override_lookup_js` fall back to a bare `properties.h3_cell`
    # that's correctly always absent/no-op on census features. The h3/circle
    # hexagon levels below build their own per-resolution style string so the
    # override lookup's namespaced id key (`_h3_<res>_h3_cell`) matches the
    # actual property key on that level's tiles.
    poly_style = _polygon_style_js(score_domain, opacity_field_domains)
    edges_style = _edges_style_js(score_domain, street_min_zoom)

    def _id_col(gdf: gpd.GeoDataFrame) -> str:
        for cand in ("geo_id", "GEOID", "geoid"):
            if cand in gdf.columns:
                return cand
        return "geo_id"

    def _with_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if any(c in gdf.columns for c in ("geo_id", "GEOID", "geoid")):
            return gdf
        gdf = gdf.reset_index(drop=True).copy()
        gdf["geo_id"] = gdf.index
        return gdf

    resolutions_levels = [f"h3_{res}" for res in resolutions]
    # Each shape/overlay is now an independently built HierarchyMap (streets
    # and development are their own layers, not baked into every shape), so
    # each one partitions the *full* [0, 25] zoom range on its own.
    zoom_bands = _level_zoom_bands(resolutions_levels)

    def _apply_zoom_bands(hmap, levels: List[str], bands: Dict[Hashable, tuple]) -> None:
        # Explicit bands for every level (not auto-assignment): `HierarchyMap`'s
        # zoom-banding requires all levels to jointly *partition* [0, 25] with zero
        # gaps/overlaps, and its auto-assignment doesn't correctly account for
        # multiple independently-built HierarchyMaps sharing the same zoom range.
        for level, (lo, hi) in bands.items():
            if level in levels:
                hmap.set_resolution(level, min_zoom=lo, max_zoom=hi)

    # Coarser map zoom-band resolutions (h3_5/6/8/9/10) are deliberately built
    # from a resample that only keeps population/level_of_service (see
    # `_resample_h3(..., sum_cols=[])` in `code.pipeline`) -- Census columns
    # native only to the finest resolution can't "reach" those coarser levels,
    # which is expected (the map never reads Census fields off a coarse hex),
    # not a missing-config bug, so geohierarchy's warning about it is muted
    # for this whole hierarchy-construction + tile-build section.
    _warn_ctx = warnings.catch_warnings()
    _warn_ctx.__enter__()
    warnings.filterwarnings("ignore", category=UserWarning, module="geohierarchy.*")

    def _h3_cell_centroids(h3_cells) -> np.ndarray:
        """Analytic H3-cell centroids (EPSG:4326 shapely Points), no polygon math.

        Every row here is one H3 cell (its `h3_cell` id column), so the true
        centroid is exactly `h3.cell_to_latlng` -- cheaper than materializing
        the hexagon polygon and running shapely's polygon-centroid algorithm
        over potentially millions of rows. H3's lat/lng is always WGS84, and
        these grids are already built/consumed in EPSG:4326 (`GeoHierarchy(crs=4326)`
        above), so no reprojection is needed either. Prefers `h3ronpy`'s
        Arrow-native vectorized path when available (mirrors
        `UrbanAccessAnalyzer.h3_ops._cell_polygons`), falling back to the
        always-available `h3` package's per-cell loop.
        """
        cells = pd.Series(h3_cells).astype(str)
        try:
            import h3ronpy
            import h3ronpy.vector as h3ronpy_vector
            import polars as pl
            import pyarrow as pa

            wkb = h3ronpy_vector.cells_to_wkb_points(h3ronpy.cells_parse(pa.array(cells)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return shapely.from_wkb(pl.Series(wkb).to_numpy())
        except ImportError:
            lats_lngs = [h3.cell_to_latlng(cell) for cell in cells]
            return shapely.points([(lng, lat) for lat, lng in lats_lngs])

    def _level_frame(gdf: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
        """The frame a base-shape level is handed to `GeoHierarchy.add_level`.

        Normally the level's full attribute table -- every column is baked
        into that level's vector tiles as feature properties.

        On the `skip_tile_build` fast path, though, no tiles are generated
        for these layers at all, and the saved page needs nothing from the
        hierarchy but each level's *extent*: the popup field lists, style
        functions and zoom bands are all computed from the caller's own
        GeoDataFrames above and embedded in the HTML as JS. `add_level`
        nonetheless stores a full Polars copy of every attribute column it
        is handed -- on Boston's 1.6M-cell resolution 11 that is ~0.9 GB per
        hierarchy, and this grid is handed to two of them (hexagons and
        circles). Passing just the id + geometry there leaves the saved HTML
        byte-identical while removing several GB from the fast path's peak.

        Deliberately keyed on `skip_tile_build` rather than always applied:
        a real tile build reads exactly these attribute columns.
        """
        if not skip_tile_build:
            return gdf
        if gdf.geometry.name not in gdf.columns:
            return gdf
        keep = [c for c in (id_col, gdf.geometry.name) if c in gdf.columns]
        # `.copy()` because `add_level` mutates the frame it is given (it
        # appends its own id/area/length/bounds/centroid columns); handing it
        # a bare slice of the caller's grid would write through to it.
        return gdf[keep].copy()

    # --- "hexagons" shape: h3 polygons colored/opacity'd by poly_style ---
    gh_hex = GeoHierarchy(crs=4326)
    for res in resolutions:
        gh_hex.add_level(f"h3_{res}", _level_frame(h3_by_resolution[res], "h3_cell"), id_col="h3_cell", agg=Max())
    hex_map = HierarchyMap(gh_hex, levels=resolutions_levels, tiles_dir=str(os.path.join(tiles_dir_rel, "hexagons")), use_pmtiles=use_pmtiles, extract_xyz=(renderer == "folium"))
    for res in resolutions:
        hex_map.configure_level(
            f"h3_{res}",
            style_js=_polygon_style_js(score_domain, opacity_field_domains, res=res),
            popup_fields=_popup_fields(h3_by_resolution[res]),
            popup_js=shape_popup_js,
            maplibre_paint=_score_maplibre_paint("polygon", score_col, score_domain, no_outline=True),
        )
    _apply_zoom_bands(hex_map, resolutions_levels, zoom_bands)
    if not skip_tile_build:
        # Build + free hexagons' tiles NOW, before "circles" duplicates the
        # same per-resolution attribute data (just with centroid geometry
        # instead of polygons) into a second `GeoHierarchy`. Real 2026-08-26
        # Boston OOM fix: this function used to construct EVERY shape's full
        # `GeoHierarchy` (hexagons, circles, census, streets -- each a full
        # Polars copy of every attribute column, per level) before tiling
        # ANY of them (the old single `multi.build()` call at the bottom),
        # so peak memory was the SUM of every shape's full data, not the max
        # of any one -- and Boston's res-11 grid alone is ~0.9GB per copy,
        # doubled immediately by "circles" duplicating it. Tiling and
        # freeing (`free_level_data=True`, see `HierarchyMap.build`'s
        # docstring) each shape as soon as it's configured, before the next
        # shape's construction even starts, keeps at most one shape's data
        # resident at a time instead of all of them.
        with contextlib.chdir(out_dir):
            hex_map.build(free_level_data=True)
            _trim_malloc()

    # --- "circles" shape: same resolutions, centroid points sized by a selectable field ---
    gh_circ = GeoHierarchy(crs=4326)
    for res in resolutions:
        centroid_gdf = _level_frame(h3_by_resolution[res], "h3_cell")
        if centroid_gdf is h3_by_resolution[res]:
            centroid_gdf = centroid_gdf.copy()
        centroid_gdf["geometry"] = _h3_cell_centroids(centroid_gdf["h3_cell"])
        gh_circ.add_level(f"h3_{res}", centroid_gdf, id_col="h3_cell", agg=Max())
    circ_map = HierarchyMap(
        gh_circ, levels=resolutions_levels, tiles_dir=str(os.path.join(tiles_dir_rel, "circles")), use_pmtiles=use_pmtiles,
        extract_xyz=(renderer == "folium"),
    )
    for res in resolutions:
        _res_radius_domains = radius_field_domains_by_res.get(res) or radius_field_domains
        circ_map.configure_level(
            f"h3_{res}",
            style_js=_circle_style_js(score_domain, _res_radius_domains, opacity_field_domains, res=res),
            popup_fields=_popup_fields(h3_by_resolution[res]),
            popup_js=shape_popup_js,
            # Circles render at a fixed pixel radius (up to 18px) client-side, unlike a
            # polygon's geometry -- the default 2% tile buffer can be far fewer pixels
            # than that at high zoom, clipping circles right at tile edges. A much
            # bigger buffer keeps a copy of near-edge points in neighboring tiles too.
            buffer_frac=0.15,
            # Point geometry -- must be "circle" kind, or MapLibre's default
            # "polygon"/fill kind silently renders nothing (fill layers only
            # draw Polygon/MultiPolygon features).
            layer_type="circle",
            maplibre_paint=_score_maplibre_paint(
                "circle", score_col, score_domain,
                radius_field_domains=_res_radius_domains,
                default_circle_field=default_circle_field,
            ),
        )
    _apply_zoom_bands(circ_map, resolutions_levels, zoom_bands)
    if not skip_tile_build:
        with contextlib.chdir(out_dir):
            circ_map.build(free_level_data=True)
            _trim_malloc()

    named_maps = {"hexagons": hex_map, "circles": circ_map}

    # --- "census" shape (only for cities with census data): coarsest-to-finest census geometry ---
    if has_census:
        census_levels_sorted = list(census_by_level)  # caller-provided coarse -> fine order

        # No synthetic gap-fill (removed 2026-08-30 per explicit user
        # request: "the intended behaviour is just census geometries and no
        # weird rasterization for the census option" -- the coarsest
        # level's geometry used to be replaced with `coarsest -
        # union(finer)`, a SYNTHETIC residual shape covering the whole AOI,
        # always rendered). That stays gone: nothing here fabricates new
        # geometry.
        #
        # 2026-09-01 (live user report, Guadalajara): "sometimes there is
        # block data but blockgroups do not appear. Make sure as much as
        # possible of the AOI is covered with block or blockgroup data" --
        # Mexico's INEGI (like the US ACS) only publishes block/ageb
        # boundaries for urban localities (see
        # `pycensus.countries.mexico.inegi.loader`'s module docstring), so a
        # real coverage hole in the FINEST level is common and, at the
        # finest level's own zoom band, used to just show the basemap even
        # where the next-coarser level's REAL polygon does cover that spot.
        # Restoring `set_fallback_level` (not the geometry substitution
        # removed above) fixes this without synthesizing anything: it just
        # widens the second-finest real level's own zoom range down to 0
        # so it renders as a backdrop underneath every other level at every
        # zoom, including the finest level's band -- MapLibre's painter's
        # -algorithm z-order means it is only ever VISIBLE where a finer
        # level has no feature to draw on top of it, i.e. exactly a real
        # coverage gap. `census_level_names_real[-2]` (one step coarser
        # than finest) is the generic "block/blockgroup"-equivalent level
        # for any country -- ageb for Mexico, blockgroup for the US, etc.
        # Only set when there are at least 2 levels (n==1 has no coarser
        # level to fall back to).

        census_level_names_real = [f"census_{name}" for name in census_levels_sorted]
        census_level_names = census_level_names_real
        census_zoom_bands = _level_zoom_bands(census_level_names_real)
        # Germany's "grid" level is the real Zensus 2022 100m raster, rendered
        # as its own native 100m x 100m rectangles (no H3 resampling --
        # 2026-08-30, see `_germany_grid_census_loader` in pipeline.py) --
        # dense enough to read well before the zoom
        # `_level_zoom_bands`'s linear weighting reserves for it as the single
        # finest census level (its min_zoom otherwise lands in the low-to-mid
        # 20s). Pull its band a few zoom levels earlier, taking the difference
        # from the next-coarsest level's band so the [0, 25] partition this
        # `HierarchyMap` requires stays gapless (see `_level_zoom_bands`'s
        # docstring). No-op for every other country (no "census_grid" level).
        if "census_grid" in census_zoom_bands and len(census_level_names_real) > 1:
            grid_lo, grid_hi = census_zoom_bands["census_grid"]
            earlier_lo = max(grid_lo - 4, 0)
            prev_level = census_level_names_real[census_level_names_real.index("census_grid") - 1]
            prev_lo, prev_hi = census_zoom_bands[prev_level]
            if earlier_lo > prev_lo:  # keep at least one zoom level for prev_level
                census_zoom_bands[prev_level] = (prev_lo, earlier_lo - 1)
                census_zoom_bands["census_grid"] = (earlier_lo, grid_hi)

        def _level_key(census_level_name: str) -> str:
            return census_level_name[len("census_"):]

        gh_census = GeoHierarchy(crs=4326)
        for name in census_level_names:
            gdf = _with_id(census_by_level[_level_key(name)].to_crs(4326))
            gh_census.add_level(
                name, _level_frame(gdf, _id_col(gdf)), id_col=_id_col(gdf), agg=Max()
            )
        census_map = HierarchyMap(
            gh_census, levels=census_level_names, tiles_dir=str(os.path.join(tiles_dir_rel, "census")), use_pmtiles=use_pmtiles,
            extract_xyz=(renderer == "folium"),
        )
        for name in census_level_names:
            key = _level_key(name)
            gdf = census_by_level[key]
            # Item 6 (2026-08-16): only `CENSUS_LIVE_RECOLOR_LEVEL` ships the
            # GEOID<->h3-cell chunks (`transitlos.map.pop_chunks`) a live
            # style needs -- other levels (tract/county) keep the old no-op
            # `poly_style`, documented followup in that function's docstring.
            if key == CENSUS_LIVE_RECOLOR_LEVEL:
                id_col_this = _id_col(_with_id(gdf.to_crs(4326)))
                id_key_js = json.dumps(f"_census_{key}_{id_col_this}")
                level_style = _polygon_style_js(
                    score_domain, opacity_field_domains,
                    census_id_expr=f"properties[{id_key_js}]",
                )
            else:
                level_style = poly_style
            census_map.configure_level(
                name, style_js=level_style, popup_fields=_popup_fields(gdf),
                popup_js=shape_popup_js,
                maplibre_paint=_score_maplibre_paint("polygon", score_col, score_domain),
            )
        _apply_zoom_bands(census_map, census_level_names_real, census_zoom_bands)
        # 2026-09-01: widens the second-finest level's zoom band down to
        # render under the finest level at EVERY zoom -- wanted for a real
        # coverage hole (Mexico: block data exists but blockgroup polygons
        # don't, so blockgroup fills the gap). 2026-09-07 bug fix (live
        # Hamburg report): when the finest level is `census_grid` (a real,
        # wall-to-wall 100m Zensus raster -- see `_germany_grid_census_loader`),
        # any area it doesn't cover is genuinely no-data, not a polygon
        # coverage gap -- applying the same fallback there made municipality
        # (Gemeinde) polygons visibly show through underneath the grid at
        # its own high-zoom band, which is the opposite of what a raster's
        # "no data" should look like.
        if len(census_level_names_real) >= 2 and census_level_names_real[-1] != "census_grid":
            census_map.set_fallback_level(census_level_names_real[-2])
        if not skip_tile_build:
            with contextlib.chdir(out_dir):
                census_map.build(free_level_data=True)
                _trim_malloc()

        named_maps["census"] = census_map

    # --- "streets" overlay: independent checkbox, always above the active shape layer ---
    # A single-level HierarchyMap must still cover the *full* [0, 25] zoom
    # range on its own (see `_level_zoom_bands`'s docstring). Streets are
    # invisible below `street_min_zoom` anyway (`edges_style_js` hides them
    # client-side), so tiling the *entire* metro's edge network (hundreds of
    # thousands of features) at every zoom down to 0 was pure waste -- it was
    # the dominant cost of the whole map build. A second, genuinely empty
    # level covers the low-zoom band instead (an empty GeoDataFrame tiles to
    # nothing almost instantly), and the real edge geometry is only ever
    # tiled for the zoom band it's actually shown at.
    gh_streets = GeoHierarchy(crs=4326)
    # A genuinely 0-row GeoDataFrame trips a geopandas edge case where
    # `.area`/`.length` on an empty GeoSeries come back with a `geometry`
    # dtype instead of float (breaking `GeoHierarchy.add_level`'s polars
    # conversion) -- a single real row sidesteps that while still being
    # negligible to tile (at most one tile touched per low zoom).
    edges_empty = edges_wgs84.iloc[:1].copy()
    gh_streets.add_level("edges_hidden", edges_empty, id_col="edge_id", agg=Max())
    gh_streets.add_level("edges", _level_frame(edges_wgs84, "edge_id"), id_col="edge_id", agg=Max())
    streets_map = HierarchyMap(
        gh_streets, levels=["edges_hidden", "edges"], tiles_dir=str(os.path.join(tiles_dir_rel, "streets")), use_pmtiles=use_pmtiles,
        extract_xyz=(renderer == "folium"),
    )
    # `layer_type="street_overlay"` (registry kind="line") -- without it,
    # MapLibre defaults every level to the "polygon"/fill kind, which
    # silently renders nothing for LineString edge geometry (a "fill" GL
    # layer only draws Polygon/MultiPolygon features).
    streets_map.configure_level(
        "edges_hidden", style_js=edges_style, popup_fields=[score_col],
        layer_type="street_overlay",
        maplibre_paint=_score_maplibre_paint("line", score_col, score_domain),
    )
    streets_map.configure_level(
        "edges", style_js=edges_style, popup_fields=[score_col],
        layer_type="street_overlay",
        maplibre_paint=_score_maplibre_paint("line", score_col, score_domain),
    )
    streets_map.set_resolution("edges_hidden", min_zoom=0, max_zoom=street_min_zoom - 1)
    streets_map.set_resolution("edges", min_zoom=street_min_zoom, max_zoom=25)
    if not skip_tile_build:
        with contextlib.chdir(out_dir):
            streets_map.build(free_level_data=True)
            _trim_malloc()

    overlay_maps = {"streets": streets_map}
    overlay_show = {"streets": True}

    # --- "development" overlay: independent checkbox, always covers the full [0, 25] zoom range ---
    # Fixed at a single level/resolution -- not multi-resolution like the base
    # shape layers -- since this overlay's purpose is a stable, always-legible
    # signal at a consistent granularity, not a zoom-adaptive level of detail.
    #
    # Two sources, in priority order:
    #  1. `development_gdf` (e.g. census block groups), when the caller has
    #     one -- the real geometry the equity classification was computed
    #     on, and the ONLY tiles ever written for this overlay in that case.
    #  2. An H3-hexagon fallback built from `h3_by_resolution`, for studies
    #     with no better geometry available (non-US, or census fetch failed).
    #
    # Earlier (2026-08-12 to 2026-08-14) this overlay was ALWAYS built from
    # H3 first -- even for census studies -- specifically because
    # `skip_tile_build=True` reused stale on-disk tiles from a prior run that
    # predated `equity_flag` being added to `popup_fields` (see git history
    # for that bug's forensics). A separate script then redrew it a second
    # time from census polygons after `build_city_map` returned. In practice
    # that meant every fast rebuild wrote H3 tiles for the whole metro
    # (millions of tiny files, tens of minutes) purely to immediately
    # overwrite them -- and if that second step was ever interrupted or
    # raced against a second concurrent build, the H3 tiles it meant to
    # discard were what survived (this happened twice in one session).
    # Accepting `development_gdf` directly here removes the wasted first
    # pass and the race entirely: a census study's dev tiles are written
    # once, correctly, in this same function call.
    if show_development:
        if has_development_gdf:
            dev_gdf = development_gdf.to_crs(4326).reset_index(drop=True).copy()
            id_col = next((c for c in ("GEOID", "geo_id", "geoid") if c in dev_gdf.columns), None)
            dev_gdf["h3_cell"] = dev_gdf[id_col].astype(str) if id_col else dev_gdf.index.astype(str)
            dev_level = "geo"
            resolve_dev_tiles = True
        else:
            # `equity_flag` is only ever computed on the *native* h3_resolution
            # grid upstream (it's a categorical column, not something
            # `_resample_h3` propagates onto the coarser resampled resolutions
            # built for the other map zoom bands) -- so naively picking
            # whichever resolution is numerically closest to 9 can (and did)
            # land on a resampled resolution with no `equity_flag` column at
            # all, silently rendering nothing (every feature's
            # `properties.equity_flag` is `undefined`, matching neither branch
            # in `_development_style_js`, so every hex is fully transparent).
            # Restrict the candidate set to resolutions that actually carry
            # the column, then -- per "h3 resolution 9 or less" -- prefer the
            # finest candidate that is <= 9, falling back to the coarsest
            # available candidate above 9 only if nothing <= 9 has the data.
            dev_candidates = [r for r in resolutions if "equity_flag" in h3_by_resolution[r].columns]
            if not dev_candidates:
                dev_candidates = resolutions
            le9 = [r for r in dev_candidates if r <= 9]
            dev_res = max(le9) if le9 else min(dev_candidates)
            dev_gdf = h3_by_resolution[dev_res]
            dev_level = f"h3_{dev_res}"
            resolve_dev_tiles = not skip_tile_build

        gh_dev = GeoHierarchy(crs=4326)
        gh_dev.add_level(dev_level, dev_gdf, id_col="h3_cell", agg=Max())
        dev_map = HierarchyMap(
            gh_dev, levels=[dev_level], tiles_dir=str(os.path.join(tiles_dir_rel, "development")), use_pmtiles=use_pmtiles,
            extract_xyz=(renderer == "folium"),
        )
        dev_style = _development_style_js()
        # `native_zoom_range=(0, 12)`, not the default (native max capped
        # only at `MAX_NATIVE_TILE_ZOOM`=18): this overlay is a stable,
        # always-visible border -- content doesn't meaningfully change
        # between zoom 12 and 18 the way a genuinely detailed layer would --
        # so tiling it independently at every native zoom up to 18 was pure
        # waste (measured: 1,170,681 tile files for one 3,661-feature
        # census-blockgroup level). Leaflet.VectorGrid reuses zoom-12's
        # tiles (scaled) for every deeper display zoom, cutting native
        # zoom passes from 19 to 13 and -- more importantly -- removing
        # exactly the deepest zooms, which dominate total tile count (tile
        # count roughly quadruples per zoom level for full-AOI coverage).
        # The native MIN stays 0, unchanged: raising it would do the
        # opposite of what's wanted here -- a display zoom below the native
        # range forces Leaflet to fetch `4^(native_min - display_z)` tiles
        # to cover one screen, the exact blowup this session already
        # evaluated and rejected for the wide-band shape layers (see this
        # function's zoom-banding docstring). Narrowing only the max is the
        # safe direction: zooming in past native max reuses/upscales one
        # tile, it never multiplies fetches.
        dev_popup_fields = ["equity_flag", score_col]
        dev_map.configure_level(
            dev_level, style_js=dev_style, popup_fields=dev_popup_fields,
            native_zoom_range=(0, 12),
        )
        dev_map.set_resolution(dev_level, min_zoom=0, max_zoom=25)
        overlay_maps["development"] = dev_map
        overlay_show["development"] = False
        # With `development_gdf` supplied, always (re)build -- it's real
        # census-derived geometry, cheap relative to the multi-resolution
        # hex/circle/census/streets build, and needs to reflect whatever data
        # the caller just computed even on a `skip_tile_build=True` fast
        # rebuild. Without it (H3 fallback), only build when the main
        # multi-resolution build is also running -- this path used to force
        # a rebuild unconditionally to dodge the stale-`equity_flag` bug
        # above, but that bug was in the *properties baked into the tile*,
        # which `development_gdf`'s single-pass build (or a real
        # `skip_tile_build=False` run) both already produce correctly.
        if resolve_dev_tiles:
            with contextlib.chdir(out_dir):
                dev_map.build()

    # 2026-09-05, explicit user request: "on all maps by default on startup
    # activate the circles view" -- "circles" (centroid points sized/colored
    # by a selectable field) is always built regardless of census
    # availability, so it's always a safe default (unlike the old
    # census-when-available/hexagons-otherwise choice below, kept only as
    # dead documentation of the prior behavior).
    default_shape = "circles"
    multi = MultiHierarchyMap(
        named_maps, default=default_shape, tiles_dir=None,
        overlay_hierarchy_maps=overlay_maps, overlay_show=overlay_show,
    )

    with contextlib.chdir(out_dir):
        # Real 2026-08-26 Boston OOM fix: each of hexagons/circles/census/
        # streets is now already built (`.build(free_level_data=True)`,
        # see `HierarchyMap.build`'s docstring in geohierarchy) right after
        # its own `GeoHierarchy` is configured above, one shape at a time,
        # instead of a single `multi.build()` call here after EVERY shape's
        # full attribute data was already constructed and held resident
        # simultaneously (the actual peak-memory driver -- Boston's res-11
        # grid alone is ~0.9GB per copy, and this function builds four such
        # groups). `multi` only needs each child's already-written tiles +
        # bounds/id_cols from here on, both still intact after
        # `free_level_data=True` trimmed the rest.
        _warn_ctx.__exit__(None, None, None)
        if renderer == "maplibre":
            # MapLibre reads the same .pmtiles written above directly (no
            # XYZ/pbf extraction, no separate tile-build step) -- base-layer
            # parity only: correct per-level coloring, click popups, hover,
            # and the same named/overlay group radio+checkbox switcher as
            # Folium's LayerControl. The scenario editor (route drawing,
            # grade-separation paint, live __h3AccessOverrides recolor) and
            # the Distribution/Regression/ANOVA stats panel remain
            # Folium-only -- pass renderer="folium" to get those.
            # Item 3/4 (round 11): "Circle size by"/"Opacity by"/"Contrast"
            # live controls, ported from Folium's `_control_panel_html`. Same
            # `circle_fields`/`opacity_fields`/`radius_field_domains`/
            # `opacity_field_domains`/`default_circle_field` this function
            # already computes for the Folium branch's `_control_panel_html`
            # call below and for `_score_maplibre_paint`'s static default
            # (round 5) -- now also threaded to the live MapLibre dropdown.
            multi.save_maplibre(
                os.path.basename(out_html_abs),
                radius_field_domains=radius_field_domains,
                opacity_field_domains=opacity_field_domains,
                circle_fields=circle_fields,
                opacity_fields=opacity_fields,
                default_circle_field=default_circle_field,
                # `render.py` (geohierarchy) is renderer-agnostic and knows
                # nothing about census-source naming conventions, so it never
                # derives a display label itself -- `field_label` (this
                # module's cross-country humanizer) is computed here and
                # handed over as a plain lookup table.
                field_labels={f: field_label(f) for f in set(circle_fields) | set(opacity_fields)},
                # 2026-09-05 real bug fix ("circle legend does not change
                # with zoom"): `save_multi_maplibre`'s own live "Circle
                # size by" JS used to apply ONE flat domain to every
                # resolution's circle layer on page load, silently
                # overwriting the per-resolution static paint computed
                # above (`_score_maplibre_paint` calls inside the circles
                # loop) -- see `save_multi_maplibre`'s docstring for the
                # full story. These let it pick the right domain per
                # layer/zoom instead.
                radius_field_domains_by_res=radius_field_domains_by_res,
                circle_zoom_bands={res: zoom_bands[f"h3_{res}"] for res in resolutions},
            )
            # Item 5 (user feedback, 2026-08-19): stops/routes/labels/popups
            # were entirely absent from the MapLibre page -- `stops_gdf`/
            # `routes_gdf` were accepted by this function but only ever
            # wired into the Folium (`else`) branch below via
            # `_inject_controls_into_saved_html`. See
            # `_maplibre_stops_routes_js`'s docstring for what this
            # MapLibre-native (not ported-from-Leaflet) equivalent covers.
            _inject_maplibre_stops_routes_into_saved_html(
                os.path.basename(out_html_abs),
                stops_gdf=stops_gdf,
                routes_gdf=routes_gdf,
                score_domain=score_domain,
            )
            if stats_by_area:
                # computeAccess() port (item 2 of the session's port) --
                # needs the baseline-access data + scoring params/pop-chunk
                # config the Folium editor also reads. See
                # `_maplibre_compute_access_js`'s docstring for what this
                # scoped port does and doesn't cover.
                _editor_finest = next(iter(stats_by_area.values()))
                _editor_fields = _stats_field_candidates(_editor_finest)
                _editor_count_fields = circle_fields if circle_fields else _stats_count_fields(_editor_fields)
                stats_data_json = _stats_json_data(
                    stats_by_area, list(dict.fromkeys(_editor_fields + _editor_count_fields))
                )
                _inject_maplibre_editor_into_saved_html(
                    os.path.basename(out_html_abs),
                    params=load_edit_params(),
                    pop_chunk_cfg={"dir": POP_CHUNK_DIRNAME, "deg": POP_CHUNK_DEG, "res": 11},
                    stats_data_json=stats_data_json,
                    region=region,
                )
                # Stats panel port (deferred item from this session's
                # MapLibre migration). `_stats_panel_block` is renderer-
                # agnostic -- its HTML is plain absolutely-positioned divs
                # (no Leaflet control), and its JS reads only
                # `window.__statsData`/`window.__editParams` (both already
                # set above by `_inject_maplibre_editor_into_saved_html` /
                # `_maplibre_compute_access_js`) plus the optional
                # `window.__editor` object, which MapLibre's scoped-down
                # editor deliberately doesn't provide -- every "Compare with
                # scenario" lookup wrapped in try/catch there falls back to
                # baseline data, so the panel degrades gracefully to
                # "None"/"Current network (baseline)" only, matching
                # MapLibre's single-current-route scope (see
                # `_maplibre_compute_access_js`'s docstring).
                _inject_maplibre_stats_panel_into_saved_html(
                    os.path.basename(out_html_abs),
                    stats_by_area=stats_by_area,
                    is_us=is_us,
                    region=region,
                    enable_place_comparison=enable_place_comparison,
                    share_source_map=share_source_map,
                )
                # Item C (user feedback, round 13): top-center place/
                # scenario/results bar. Needs `window.__editor` (set by
                # `_inject_maplibre_editor_into_saved_html` just above) and
                # `window.__enablePlaceComparison`/`.comparePlaceSelect`
                # wiring (set by the stats-panel injection just above), so
                # it must run after both and before the legend injection's
                # bare `</body>` splice closes off the `</script>\n</body>`
                # marker every one of these needs.
                _inject_maplibre_topbar_into_saved_html(
                    os.path.basename(out_html_abs),
                    enable_place_comparison=enable_place_comparison,
                )
            # Real reported gap (user testing feedback): MapLibre had no
            # legend at all -- see `_inject_maplibre_legend_into_saved_html`'s
            # docstring. Run LAST: it inserts before the bare `</body>`
            # marker, which every earlier injection above still needs to
            # find as `</script>\n</body>` (script-scope splicing) -- doing
            # this first would break those.
            _inject_maplibre_legend_into_saved_html(
                os.path.basename(out_html_abs),
                score_domain=score_domain,
                show_development=show_development,
            )
            if downloads_manifest:
                # "Download data" panel (user-requested): offers real
                # checkboxes/pickers for what this specific city actually
                # has on disk (baked in as JSON by the caller, same pattern
                # as `_stats_json_data`), and downloads the corresponding
                # `results/{metro,core}/downloads/...` parquet file(s) via
                # synthesized `<a download>` clicks. Independent of every
                # other injector above (own markers, own JS namespace), so
                # order relative to them doesn't matter -- placed last only
                # to mirror the legend's own "runs last" convention.
                _inject_maplibre_download_panel_into_saved_html(
                    os.path.basename(out_html_abs),
                    downloads_manifest=downloads_manifest,
                )
        else:
            multi.save(os.path.basename(out_html_abs))
            _inject_controls_into_saved_html(
                os.path.basename(out_html_abs),
                circle_fields=circle_fields,
                opacity_fields=opacity_fields,
                show_development=show_development,
                score_domain=score_domain,
                radius_field_domains=radius_field_domains,
                opacity_field_domains=opacity_field_domains,
                stats_by_area=stats_by_area,
                is_us=is_us,
                stops_gdf=stops_gdf,
                routes_gdf=routes_gdf,
                region=region,
                default_shape=default_shape,
                enable_place_comparison=enable_place_comparison,
                share_source_map=share_source_map,
                radius_field_domains_by_res=radius_field_domains_by_res,
                circle_zoom_bands={res: zoom_bands[f"h3_{res}"] for res in resolutions},
            )

    if stats_by_area:
        write_stats_data_json(out_html_abs, stats_by_area)

    return out_html_abs


def write_stats_data_json(out_html_abs: str, stats_by_area: Dict[str, gpd.GeoDataFrame]) -> str:
    """Write `results/stats_data.json` next to `results/*.parquet`: the same payload baked into `map.html`.

    `map.html` already inlines this exact payload via `_stats_json_data`
    (baked into `window.__statsData` so the in-page stats panel needs no
    network round-trip). This sibling file re-derives the identical
    column-oriented `{"metro": {...}, "core": {...}}` JSON and writes it to
    `<city_dir>/results/stats_data.json` so *other* pages -- the combined
    multi-city map, or another city's stats panel doing a cross-city
    comparison -- can `fetch()` just this small JSON instead of loading the
    whole (tile-heavy) `map.html`.

    Uses the same field selection as the inlined payload
    (`_stats_field_candidates`/`_stats_count_fields` over the finest area's
    columns) so the two stay in sync.
    """
    finest = next(iter(stats_by_area.values()))
    fields = _stats_field_candidates(finest)
    count_fields = _stats_count_fields(fields)
    payload_json = _stats_json_data(stats_by_area, list(dict.fromkeys(fields + count_fields)))
    # `out_html_abs` is `<city_dir>/map.html`; the parquet outputs already
    # live in the sibling `<city_dir>/results/` directory.
    out_dir = os.path.dirname(out_html_abs)
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "stats_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload_json)
    return out_path


def _stats_field_candidates(gdf: gpd.GeoDataFrame) -> List[str]:
    """Numeric, non-identifier, non-score columns available for regression/ANOVA."""
    return _numeric_field_candidates(gdf, extra_exclude=["level_of_service"])


def _stats_count_fields(fields: List[str]) -> List[str]:
    """Subset of `fields` usable as the distribution tab's overlay bar (absolute counts)."""
    return absolute_fields(fields)


def _stats_json_data(stats_by_area: Dict[str, gpd.GeoDataFrame], fields: List[str]) -> str:
    """Column-oriented JSON: `{"metro": {"level_of_service": [...], "population": [...], ...}, "core": {...}}`.

    Column-oriented (not one object per row) keeps the embedded payload
    small and lets every stats-tab computation (weighted mean/median,
    binning, regression) run as flat array loops in JS with no per-row
    object overhead.
    """
    payload: Dict[str, Dict[str, list]] = {}
    for area, gdf in stats_by_area.items():
        cols = ["level_of_service"] + [f for f in fields if f in gdf.columns]
        area_data: Dict[str, list] = {}
        for col in cols:
            values = gdf[col].to_numpy(dtype=float)
            area_data[col] = [None if not np.isfinite(v) else round(float(v), 6) for v in values]
        # `h3_cell` is a join key, not a regression/ANOVA variable -- it never
        # appears in `regression_fields`/`count_fields` (dropdowns), but is
        # included whenever present so the client can overlay recomputed
        # per-cell level_of_service values (from the scenario editor's Compute)
        # onto the right rows. See `getStatsData` in `_stats_panel_js`.
        if "h3_cell" in gdf.columns:
            area_data["h3_cell"] = gdf["h3_cell"].astype(str).tolist()
        payload[area] = area_data
    return json.dumps(payload)


def _compare_with_control_html(
    prefix: str,
    secondary_scenario_options: str,
    include_column: bool,
    overlay_options: str,
    has_multi_area: bool,
    enable_place_comparison: bool,
    place_options: str = "",
) -> str:
    """One "Compare with" button + popover for a stats-panel tab.

    Redesign (user feedback, 2026-08-19): a tab used to expose its comparison
    axes as N separate flat `<select>`s sitting side by side (e.g.
    Distribution had "Compare with column" AND "Compare with scenario" as two
    equally-weighted dropdowns). The user asked for these to instead read as
    ONE "Compare with" control that opens a small panel containing whichever
    of the (up to 4) sub-dimensions actually apply to this tab/map:
      - column: only meaningful for Distribution's histogram overlay -- every
        other tab already has a fixed field (level_of_service) so a column picker
        would be redundant there.
      - scenario: always offered when the editor's named-scenario system
        exists (`window.__editor.state.scenarios`).
      - area (city core / metro area): only offered when this map's
        `stats_by_area` genuinely has more than one area (`has_multi_area`)
        -- a single-area map has nothing to toggle.
      - place: only offered when the map was built with
        `enable_place_comparison=True` (multi-place rosters like
        CS_transitLOS; off for single-place maps like boston_city).
    All still-relevant `<select>` element ids are unchanged
    (`{prefix}SecondaryScenarioSelect`, `{prefix}OverlaySelect`) so the
    existing render/read functions keep working; two NEW ids are added when
    applicable: `{prefix}CompareAreaSelect`, `{prefix}ComparePlaceSelect`.
    """
    rows = []
    if include_column:
        rows.append(f"""
        <div style="margin-bottom:6px;">
          <label>Column</label>
          <select id="{prefix}OverlaySelect" style="width:100%;">
            <option value="none" selected>None</option>
            {overlay_options}
          </select>
        </div>""")
    rows.append(f"""
        <div style="margin-bottom:6px;">
          <label>Scenario</label>
          <select id="{prefix}SecondaryScenarioSelect" style="width:100%;">
            {secondary_scenario_options}
          </select>
        </div>""")
    if has_multi_area:
        rows.append(f"""
        <div style="margin-bottom:6px;">
          <label>City core / metro area</label>
          <select id="{prefix}CompareAreaSelect" style="width:100%;">
            <option value="metro">Metro area</option>
            <option value="core">City core</option>
          </select>
        </div>""")
    # Item 4 (verbatim user request, 2026-08-19): the per-tab "Place"
    # cross-place select was REMOVED here at the time -- that functionality
    # was folded entirely into the stats panel's "Place rank" tab (a
    # column-weighted median used to RANK places), not a per-tab "compare
    # this chart against another place" control.
    #
    # 2026-09-02 follow-up (explicit user request, verbatim: "on global map
    # compare with should allow to select city core/metro, scenario and
    # place so that cities can be compared between them in regression anova
    # distribution etc"): re-added. `enable_place_comparison` alone used to
    # gate the tab; now also requires real `place_options` (built from
    # `place_manifest` -- empty/`None` for a caller with no cross-place
    # roster, e.g. a standalone per-city `map.html`) so the row never shows
    # an empty, useless dropdown.
    if enable_place_comparison and place_options:
        rows.append(f"""
        <div style="margin-bottom:6px;">
          <label>Place</label>
          <select id="{prefix}ComparePlaceSelect" style="width:100%;">
            <option value="" selected>None (this place)</option>
            {place_options}
          </select>
        </div>""")
    return f"""
    <div class="compareWithWrap" style="position:relative;margin-bottom:6px;">
      <button type="button" class="compareWithBtn" data-target="{prefix}ComparePopover"
        style="width:100%;text-align:left;padding:5px 8px;border:1px solid #d5d9de;border-radius:5px;
        background:#f5f6f8;cursor:pointer;font:12px sans-serif;color:#333;">
        Compare with <span style="float:right;color:#888;">&#9662;</span>
      </button>
      <div id="{prefix}ComparePopover" class="compareWithPopover" style="display:none;position:absolute;
        top:100%;left:0;right:0;z-index:1001;background:white;border:1px solid #d5d9de;border-radius:5px;
        box-shadow:0 2px 8px rgba(0,0,0,0.2);padding:8px;margin-top:2px;">
        {"".join(rows)}
      </div>
    </div>"""


def _stats_panel_html(
    regression_fields: List[str],
    count_fields: List[str],
    enable_place_comparison: bool = True,
    has_multi_area: bool = True,
    place_rank_only: bool = False,
    metadata_rows: Optional[List[Dict[str, str]]] = None,
    place_manifest: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Raw HTML for the bottom-left stats button + panel skeleton (tabs filled in by JS).

    `place_manifest` (2026-09-02, re-added per explicit user request: "on
    global map compare with should allow to select city core/metro,
    scenario and place so that cities can be compared between them in
    regression anova distribution etc" -- a per-tab "Place" row in the
    "Compare with" popover existed before and was deliberately removed
    2026-08-19 in favor of the Place-rank tab alone; this reinstates it,
    the user's later request taking precedence) builds the `<option>` list
    for each tab's new place `<select>` -- same `[{"key": ..., "label":
    ...}]` shape `_stats_panel_js`'s own `place_manifest` param already
    takes for the Place-rank tab, so a caller (`code.combined_map`) passes
    the exact same value to both. `None`/empty means no place options to
    offer, so the row is omitted entirely (a per-city `map.html` doesn't
    currently pass one -- see that call site's own comment).
    """
    reg_options = "".join(f'<option value="{f}">{field_label(f)}</option>' for f in regression_fields)
    overlay_options = "".join(f'<option value="{f}">{field_label(f)}</option>' for f in count_fields)
    # Place-rank tab's own weight-column options: same field list as
    # `overlay_options`, but with `population` pre-selected (user's explicit
    # ask -- ranking should default to population-weighted, not the plain
    # unweighted median) and "Unweighted" only falling back to `selected`
    # when `population` isn't even offered. Kept separate from
    # `overlay_options` itself, which stays un-preselected -- it's also
    # reused by the Distribution tab's "Compare with" popover, where nothing
    # should be pre-selected by design (see that popover's own comment).
    _population_in_counts = "population" in count_fields
    place_rank_weight_options = "".join(
        f'<option value="{f}"{" selected" if f == "population" else ""}>{field_label(f)}</option>'
        for f in count_fields
    )
    # "None + field" variants for the item-2/item-3 optional second-column
    # dropdowns: distribution's main-bucket column (defaults to population,
    # so behavior is unchanged unless the user picks something else) and the
    # regression/ANOVA "compare with" dropdowns (default "none" == today's
    # fixed level_of_service behavior).
    # Item 3 (verbatim user request): "on distribution by default activate
    # population and jobs and if not population and jobs activate
    # population" -- prefer the combined population+jobs count column
    # (`pop_jobs_total`, only present for cities with a jobs source joined),
    # falling back to plain `population` (always present) when it isn't.
    _dist_default_field = "pop_jobs_total" if "pop_jobs_total" in count_fields else "population"
    main_options = "".join(
        f'<option value="{f}"{" selected" if f == _dist_default_field else ""}>{field_label(f)}</option>'
        for f in count_fields
    )
    # Shared static markup for every "Compare with" popover's "Scenario"
    # select (distribution/regression/ANOVA each own one, item 3's fix -- no
    # more single shared dropdown sitting above every tab). User feedback
    # (2026-08-19): default to "Current network (baseline)" -- the closest
    # thing this codebase has to a "default scenario" (the reserved/baseline
    # network state every map starts on before any user scenario is
    # created/activated, see `__statsReservedName()`/`getStatsData`) -- rather
    # than "None", so opening a tab shows a real baseline comparison out of
    # the box instead of nothing. "None" is still offered (now second) for
    # users who want single-series view. `__populateSecondaryScenarioSelect`
    # regenerates this same option list client-side once real scenarios exist,
    # preserving whichever value was already selected.
    # Follow-up fix (verbatim user request): nothing in any "Compare with"
    # popover should be pre-selected -- opening a tab should show the plain/
    # unweighted single-series view until the user explicitly picks a
    # comparison, not a baseline-vs-baseline comparison nobody asked for.
    secondary_scenario_options = (
        '<option value="none" selected>None</option>'
        '<option value="">Current network (baseline)</option>'
    )
    # Re-added "Compare with place" row's own `<option>`s -- see this
    # function's docstring. "None" first/selected, matching every other
    # "Compare with" sub-dimension's own "nothing selected by default" rule.
    place_options = "".join(
        f'<option value="{p["key"]}">{p["label"]}</option>' for p in (place_manifest or [])
    )
    # Item 1 fix (user feedback, 2026-08-19): city-rank cross-city comparison
    # only makes sense for maps built as part of the CS_transitLOS multi-city
    # roster -- standalone study maps (e.g. TransitLOSStudies/boston_city)
    # aren't in `CS_PLACE_MANIFEST` and previously still showed the tab as the
    # default/active one, firing a batch of always-404 cross-city fetches
    # on open. `enable_place_comparison=False` drops the tab (and its button)
    # entirely and falls back to "Distribution" as the default active tab,
    # instead of rendering a tab that can never populate.
    # `place_rank_only` (combined_map.py's use case: it has no per-place h3
    # stats of its own, only the cross-place ranking, so the other four tabs
    # -- Distribution/Regression/ANOVA/Discretization -- would render broken
    # empty panels with nothing to feed them) forces the place-comparison tab
    # on and every other tab off, regardless of `enable_place_comparison`.
    if place_rank_only:
        enable_place_comparison = True
    placerank_tab_btn = (
        '<button class="statsTabBtn active" data-tab="placerank">Place rank</button>'
        if enable_place_comparison else ""
    )
    other_tab_btns = "" if place_rank_only else (
        f'<button class="statsTabBtn{" active" if not enable_place_comparison else ""}" data-tab="distribution">Distribution</button>\n'
        '    <button class="statsTabBtn" data-tab="regression">Regression</button>\n'
        '    <button class="statsTabBtn" data-tab="anova">ANOVA</button>\n'
        '    <button class="statsTabBtn" data-tab="discretization">Discretization</button>'
    )
    # Metadata tab: offered whenever the caller passes ANY `metadata_rows`
    # value other than `None` -- `is not None` (not truthy) so an explicit
    # `[]` still renders the tab's shell (header + empty `<tbody
    # id="metadataTabBody">`). 2026-09-02: `combined_map.py` (the
    # `place_rank_only` case referenced below) now uses exactly that --
    # it has no per-h3-cell GeoDataFrame of its own at PYTHON build time
    # (a real per-city map does), so it passes `[]` for the static shell
    # and populates `#metadataTabBody` live in JS from `window.__statsData`
    # instead (see its own `__renderMetadataTab`/`window.__renderMetadataTab`
    # hook, and `__renderActiveTab`'s optional call to it below).
    metadata_tab_btn = (
        '<button class="statsTabBtn" data-tab="metadata">Metadata</button>'
        if metadata_rows is not None else ""
    )
    default_tab = "placerank" if enable_place_comparison else "distribution"
    return f"""
<style>
  #statsButton {{ transition: transform 0.12s, box-shadow 0.12s; }}
  #statsButton:hover {{ transform: scale(1.08); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }}
</style>
<!-- Bottom-left corner, STACKED ABOVE the scale bar rather than sharing its
     spot: `L.control.scale` (added in `_control_panel_js`) sits flush in
     that corner inside Leaflet's own bottom-left control container, whose
     two lines (metric + imperial) plus Leaflet's 10px corner padding
     occupy roughly the bottom 45px. `bottom:56px` clears that with a
     visible gap, so the button reads as a second item in the same corner
     stack instead of overlapping the scale. -->
<div id="statsButton" title="Statistics" style="position:absolute;bottom:56px;left:10px;z-index:1000;
  background:white;width:34px;height:34px;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,0.3);
  display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer;">📊</div>
<div id="statsPanel" style="display:none;position:absolute;bottom:100px;left:10px;z-index:1000;background:white;
  width:460px;max-height:70vh;overflow-y:auto;border-radius:6px;box-shadow:0 1px 6px rgba(0,0,0,0.35);
  font:12px sans-serif;padding:10px;">
  <style>
    .statsTabBtn, .statsAreaBtn, .statsParamBtn {{
      flex: 1; padding: 6px 4px; border: 1px solid #d5d9de; background: #f5f6f8;
      color: #333; border-radius: 5px; cursor: pointer; font: 12px sans-serif;
      transition: background 0.12s, color 0.12s, border-color 0.12s;
    }}
    .statsTabBtn:hover, .statsAreaBtn:hover, .statsParamBtn:hover {{ background: #e9edf2; }}
    .statsTabBtn.active, .statsAreaBtn.active, .statsParamBtn.active {{
      background: #3a6ea8; border-color: #3a6ea8; color: white; font-weight: 600;
    }}
  </style>
  <div style="display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap;">
    <!-- "Place rank" is deliberately FIRST (before Distribution too): it's a
         cross-city overview, not a per-area breakdown of THIS city, so it
         reads as the entry point rather than one more per-city tab. See
         renderPlaceRank/CS_PLACE_MANIFEST below -- it fetches every CS city's
         `results/summary.json` over HTTP (relative sibling paths under
         CS_transitLOS/), so it only populates when this map.html is served
         over http(s), not opened via file://, and only lists CS_transitLOS
         cities (never TransitLOSStudies/boston_city, which is a separate
         standalone study outside CS_transitLOS/). -->
    {placerank_tab_btn}
    {other_tab_btns}
    {metadata_tab_btn}
  </div>
  <div id="statsAreaBtnRow" style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px;">
    <div style="display:flex;gap:4px;">
      <button class="statsAreaBtn active" data-area="metro">Metro area</button>
      <button class="statsAreaBtn" data-area="core">City core</button>
    </div>
  </div>
  <!-- The MAIN column/series of every tab below always follows whichever
       scenario is currently active in the editor panel (exactly as it did
       before the optional secondary-scenario feature existed) -- there is no
       "make the whole panel show scenario X" control here.

       Each tab below (distribution/regression/ANOVA) owns its OWN "Compare
       with" popover (`_compare_with_control_html`, opened from a single
       button rather than N flat side-by-side selects) bundling up to 4
       independent sub-dimensions, each only shown when it actually applies
       to this tab/map:
         - column (`{{prefix}}OverlaySelect`): Distribution ONLY -- its
           histogram overlay is the one place a second data column makes
           sense; Regression/ANOVA already have a fixed target (level_of_service).
         - scenario (`{{prefix}}SecondaryScenarioSelect`): always present when
           the editor's scenario system exists. Defaults to "Current network
           (baseline)" -- the closest thing to a "default scenario" this
           codebase has -- rather than "None", per user feedback (2026-08-19).
         - area (`{{prefix}}CompareAreaSelect`): only shown when this map's
           `stats_by_area` genuinely has more than one area (core vs. metro);
           lets the secondary series be pulled from the OTHER area than the
           one the main panel (`window.__statsArea`) is currently showing.
         - place (`{{prefix}}ComparePlaceSelect`): only shown when this map
           was built with `enable_place_comparison=True` (multi-place rosters
           like CS_transitLOS -- off for single-place maps like boston_city).
           Fetches that other place's `results/stats_data.json` (same
           relative-sibling-path pattern as the Place-rank tab's fetch) and
           uses it as the secondary series' data source instead of a scenario.
       For regression/ANOVA the scenario/place dimensions add a whole second
       bar/line series (same field, same level_of_service target, just sourced
       from elsewhere). For distribution they add a second HISTOGRAM of the
       same "Distribute by" field. Kept in sync as the user creates/computes
       scenarios by `__populateSecondaryScenarioSelect`, see below. -->

  {"" if not enable_place_comparison else f'''<div id="statsTab_placerank" class="statsTabPanel">
    <div style="margin-bottom:6px;color:#666;">
      Column-weighted median level of service across CS_transitLOS places, used
      to RANK them (best to worst). Pick a weight column below -- left at
      "Unweighted", every h3 cell counts equally. Requires this map to be
      served over http(s) -- opening map.html directly via file:// leaves
      this list empty.
    </div>
    <div style="margin-bottom:6px;">
      <label>Weight column</label>
      <select id="placeRankWeightSelect" style="width:100%;">
        <option value="none"{" selected" if not _population_in_counts else ""}>Unweighted (plain median)</option>
        {place_rank_weight_options}
      </select>
    </div>
    <div id="placeRankList"></div>
  </div>'''}

  {"" if place_rank_only else f'''
  <div id="statsTab_distribution" class="statsTabPanel" style="display:{'none' if default_tab != 'distribution' else 'block'};">
    <!-- Item 3 fix: "Distribute by" (the MAIN column the histogram is
         computed over) is the TOP control, on its own row -- it used to
         visually sit BELOW the shared "Compare with scenario" dropdown
         (which lived above every tab), making the panel read as if
         "Compare with scenario" were the main/top control for Distribution.
         The two comparison dropdowns now live in their own row below,
         clearly secondary to "Distribute by". -->
    <div style="margin-bottom:6px;">
      <label>Distribute by</label>
      <select id="distMainSelect" style="width:100%;">
        {main_options}
      </select>
    </div>
    {_compare_with_control_html("dist", secondary_scenario_options, True, overlay_options, has_multi_area, enable_place_comparison, place_options)}
    <div id="distSummary" style="font-weight:600;margin-bottom:8px;"></div>
    <div id="distBars"></div>
  </div>

  <div id="statsTab_regression" class="statsTabPanel" style="display:none;">
    <div style="margin-bottom:6px;">
      <label>Compare access with</label>
      <select id="regFieldSelect" style="width:100%;">
        {reg_options}
      </select>
    </div>
    <!-- No column dimension in this tab's "Compare with" popover (item 1
         correction) -- regression is always level_of_service vs. the field
         above; only Distribution's "Compare with" popover offers a column
         sub-selector. -->
    {_compare_with_control_html("reg", secondary_scenario_options, False, "", has_multi_area, enable_place_comparison, place_options)}
    <div id="regSummary" style="font-weight:600;margin-bottom:6px;"></div>
    <div id="regPlot"></div>
    <!-- Every candidate variable's R^2 against level_of_service, ANOVA-bar
         styled, so "which variable explains access best" is readable at a
         glance instead of requiring the dropdown to be walked one by one. -->
    <div style="font-weight:600;margin:10px 0 4px;">R&sup2; by variable</div>
    <div id="regR2Bars"></div>
  </div>

  <div id="statsTab_anova" class="statsTabPanel" style="display:none;">
    <!-- No column dimension in this tab's "Compare with" popover (item 1
         correction) -- ANOVA is always level_of_service split by each variable's
         own median. -->
    {_compare_with_control_html("anova", secondary_scenario_options, False, "", has_multi_area, enable_place_comparison, place_options)}
    <div id="anovaBars"></div>
  </div>

  <div id="statsTab_discretization" class="statsTabPanel" style="display:none;">
    <div style="margin-bottom:6px;color:#666;">
      Shows the scoring formula's own shape -- x is the raw input, y is the
      resulting [0,1] sub-score -- using this map's own default parameters,
      not live per-stop data.
    </div>
    <div style="display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap;">
      <button class="statsParamBtn active" data-param="speed">Speed</button>
      <button class="statsParamBtn" data-param="frequency">Frequency</button>
      <button class="statsParamBtn" data-param="reliability">Reliability</button>
      <button class="statsParamBtn" data-param="distance">Walk distance</button>
      <!-- "Level of service" is deliberately LAST: it is the composed, top-level
           result (stop_score x walk decay), so the row reads left-to-right
           from raw sub-scores through stop_score to the final score. -->
      <button class="statsParamBtn" data-param="stop_score">Stop score</button>
      <button class="statsParamBtn" data-param="access">Level of service</button>
    </div>
    <!-- Only meaningful for the "Stop score" curve: one curve per mode
         (mode_score is fixed per curve), x is whichever of speed/headway the
         dropdown selects, and the OTHER one is held fixed at the slider's
         value (reliability_score's headway_cv input is always held fixed
         here too, at a representative default -- it isn't one of the two
         axes the user asked to toggle between). -->
    <div id="discStopScoreRow" style="display:none;margin-bottom:8px;">
      <label style="display:block;font-weight:600;color:#444;margin-bottom:2px;font-size:11px;">
        X axis
      </label>
      <select id="discStopScoreAxisSelect" style="width:100%;margin-bottom:6px;">
        <option value="headway" selected>Headway</option>
        <option value="speed">Speed</option>
      </select>
      <label style="display:block;font-weight:600;color:#444;margin-bottom:2px;font-size:11px;">
        <span id="discStopScoreSliderLabel"></span>
        <span id="discStopScoreSliderValue" style="font-weight:normal;color:#666;"></span>
      </label>
      <input type="range" id="discStopScoreSlider" style="width:100%;">
    </div>
    <!-- Only meaningful for the "Level of service" curve (level_of_service =
         stop_score * D(walk_time)): exactly the same axis-toggle + fixed-
         value-slider mechanism as the "Stop score" row above, with its two
         inputs (stop_score, walk_time) instead of (speed, headway). It
         replaces the old walk-time-only slider outright -- one control
         pattern for both composed curves, not two different ones. -->
    <div id="discAccessRow" style="display:none;margin-bottom:8px;">
      <label style="display:block;font-weight:600;color:#444;margin-bottom:2px;font-size:11px;">
        X axis
      </label>
      <select id="discAccessAxisSelect" style="width:100%;margin-bottom:6px;">
        <option value="stop_score" selected>Stop score</option>
        <option value="walk_time">Walk time</option>
      </select>
      <label style="display:block;font-weight:600;color:#444;margin-bottom:2px;font-size:11px;">
        <span id="discAccessSliderLabel"></span>
        <span id="discAccessSliderValue" style="font-weight:normal;color:#666;"></span>
      </label>
      <input type="range" id="discAccessSlider" style="width:100%;">
    </div>
    <div id="discSummary" style="font-weight:600;margin-bottom:6px;"></div>
    <div id="discPlot"></div>
  </div>
'''}
{_metadata_tab_html(metadata_rows) if metadata_rows is not None else ""}
</div>
"""


def _metadata_tab_html(metadata_rows: List[Dict[str, str]]) -> str:
    """Static HTML table for the "Metadata" tab: one row per real census/WorldPop column.

    Rendered directly in Python (unlike Distribution/Regression/ANOVA, which
    recompute live from `window.__statsData` in JS) because this data --
    source, resolution level, total, weighted average, unit, weight column,
    description -- is fixed at build time for a real per-city map and never
    varies with the metro/core area toggle or a scenario edit the way a
    live per-cell statistic would; see `_column_metadata_rows` for how each
    field is derived.
    """
    header = (
        "<tr>"
        "<th style='text-align:left;'>Column</th>"
        "<th style='text-align:left;'>Source</th>"
        "<th style='text-align:left;'>Highest level</th>"
        "<th style='text-align:right;'>Total</th>"
        "<th style='text-align:right;'>Average</th>"
        "<th style='text-align:right;'>Share</th>"
        "<th style='text-align:left;'>Unit</th>"
        "<th style='text-align:left;'>Weight column</th>"
        "<th style='text-align:left;'>Description</th>"
        "</tr>"
    )
    body = "".join(
        "<tr style='border-top:1px solid #eee;'>"
        f"<td style='font-weight:600;'>{escape(str(r['name']))}</td>"
        f"<td>{escape(str(r['source']))}</td>"
        f"<td>{escape(str(r['level']))}</td>"
        f"<td style='text-align:right;'>{escape(str(r['total']))}</td>"
        f"<td style='text-align:right;'>{escape(str(r['average']))}</td>"
        f"<td style='text-align:right;'>{escape(str(r.get('share', '')))}</td>"
        f"<td>{escape(str(r['unit']))}</td>"
        f"<td>{escape(str(r['weight_column']))}</td>"
        f"<td style='color:#555;'>{escape(str(r['description']))}</td>"
        "</tr>"
        for r in metadata_rows
    )
    return f"""
  <div id="statsTab_metadata" class="statsTabPanel" style="display:none;">
    <div style="margin-bottom:6px;color:#666;">
      Every real census / WorldPop column this map carries -- source, the
      finest real geography it's published at, this place's total and
      population-weighted average, and its unit. Densities/shares/rates/
      medians never show a "Total" (summing them is not a meaningful
      figure); "Average" is population-weighted whenever a population
      column exists on this map.
    </div>
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead>{header}</thead>
        <tbody id="metadataTabBody">{body}</tbody>
      </table>
    </div>
  </div>
"""


def _discretization_params_js(region: str) -> str:
    """Bake this map's `transitlos.scoring.REGIONS[region]` params into JS constants.

    Powers the Discretization tab's formula mirrors (`__jsSpeedScore` etc.)
    -- kept in sync with `transitlos.scoring`'s own Python formulas by
    reading the *actual* dataclass values used to build this map, not a
    second hardcoded copy of the numbers (only the formula *shape* is
    duplicated in JS, per this file's existing pattern -- see the module
    docstring's note on regression running as "flat array loops in JS with
    no per-row [library]").
    """
    from ..scoring.parameters import REGIONS

    p = REGIONS[region]
    consts = {
        "weights": list(p.weights),
        "headwaySaturationMinutes": p.headway_saturation_minutes,
        "headwayDecayScaleMinutes": p.headway_decay_scale_minutes,
        "maxUsefulHeadwayMinutes": p.max_useful_headway_minutes,
        "walkingSpeedMps": p.walking_speed_mps,
        "walkT0BaseMinutes": p.walk_t0_base_minutes,
        "walkDecayShapeP": p.walk_decay_shape_p,
        "walkDistanceSaturationM": p.walk_distance_saturation_m,
        "modeSpeedBreakpointsKmh": list(p.mode_speed_breakpoints_kmh),
        "speedSaturationKmh": p.speed_saturation_kmh,
        "reliabilityCvScale": p.reliability_cv_scale,
    }
    return f"window.__disc = {json.dumps(consts)};\n"


# Fallback CS_transitLOS place roster for the Place-rank tab, used only when
# a caller doesn't pass its own `place_manifest` -- i.e. every EXISTING
# per-city `build_city_map` call site (`CS_transitLOS/code/pipeline.py`),
# none of which know the full CS_transitLOS roster themselves. Kept here
# (rather than importing `CS_transitLOS/code/combined_map.py`'s
# `CITY_MANIFEST`, an outside project this package has no dependency on) as
# a hand-maintained mirror -- see the historical comment this replaced,
# still true of this fallback: kept in sync by hand, no cross-file import.
# `combined_map.py` itself, which DOES know its own real, live roster (from
# `_available_cities()`), passes `place_manifest` explicitly and never
# touches this fallback at all.
_DEFAULT_CS_PLACE_MANIFEST: List[Dict[str, str]] = [
    {"key": "andorra", "label": "Andorra"},
    {"key": "beerseba", "label": "Beer Sheva"},
    {"key": "gipuzkoa", "label": "Gipuzkoa"},
    {"key": "guadalajara", "label": "Guadalajara"},
    {"key": "hamburg", "label": "Hamburg"},
    {"key": "shanghai", "label": "Shanghai"},
    {"key": "taipei", "label": "Taipei"},
    {"key": "toronto", "label": "Toronto"},
    {"key": "boston", "label": "Boston"},
    {"key": "san_francisco", "label": "San Francisco"},
]


def _stats_panel_js(
    regression_fields: List[str],
    is_us: bool,
    region: str = "global",
    enable_place_comparison: bool = True,
    has_multi_area: bool = True,
    place_rank_only: bool = False,
    place_rank_fetch_prefix: str = "../",
    place_manifest: Optional[List[Dict[str, str]]] = None,
) -> str:
    """JS implementing the four stats tabs entirely client-side from `window.__statsData`.

    `place_rank_only`/`place_rank_fetch_prefix` mirror `_stats_panel_html`'s
    params of the same name -- see that function's docstring. The fetch
    prefix defaults to `"../"` (a per-city `map.html`'s own path back up to
    its `CS_transitLOS/` siblings); `combined_map.py` -- itself living AT
    `CS_transitLOS/`, one level up from where a per-city map.html sits --
    passes `""` so the exact same `fetch('<prefix>' + key + '/results/...')`
    call resolves correctly from its own document location too.
    """
    if place_rank_only:
        enable_place_comparison = True
    place_manifest_json = json.dumps(
        place_manifest if place_manifest is not None else _DEFAULT_CS_PLACE_MANIFEST
    )
    # Item 3 (verbatim user request): "on regression population and jobs
    # density and if not population density" -- prefer the combined
    # population+jobs density column (`pop_jobs_density`, aliased
    # `jobs_and_population_density` -- see `_numeric_field_candidates`'s
    # dedup, which keeps only one of the two names per city), falling back
    # to plain `pop_density`, then whatever regression offers first.
    if "pop_jobs_density" in regression_fields:
        default_field = "pop_jobs_density"
    elif "jobs_and_population_density" in regression_fields:
        default_field = "jobs_and_population_density"
    elif "pop_density" in regression_fields:
        default_field = "pop_density"
    else:
        default_field = regression_fields[0] if regression_fields else "population"
    default_tab = "placerank" if enable_place_comparison else "distribution"
    enable_place_comparison_json = json.dumps(enable_place_comparison)
    has_multi_area_json = json.dumps(has_multi_area)
    return _discretization_params_js(region) + f"""
window.__enablePlaceComparison = {enable_place_comparison_json};
window.__hasMultiArea = {has_multi_area_json};
window.__statsArea = 'metro';
window.__statsTab = {json.dumps(default_tab)};
// The MAIN column always follows the editor's active scenario (i.e. plain
// `getStatsData(area)`, no override) -- there is no independent "main
// scenario" state here anymore.
//
// Each tab's OWN "Compare with scenario" <select>
// (distSecondaryScenarioSelect/regSecondaryScenarioSelect/anovaSecondaryScenarioSelect)
// is read directly at render time instead of through one shared global --
// 'none' (the default, item 4's fix) means "no secondary series at all", ''
// means the explicit baseline/current network, and any other string names a
// scenario to pull recomputed level_of_service values for. See getStatsData's
// `scenario` param and `__secondaryScenarioValue` below.

function __weightedMean(values, weights) {{
  var num = 0, den = 0;
  for (var i = 0; i < values.length; i++) {{
    var v = values[i], w = weights[i];
    if (v == null || w == null || isNaN(v) || isNaN(w) || w < 0) continue;
    num += v * w; den += w;
  }}
  return den > 0 ? num / den : NaN;
}}

function __weightedMedian(values, weights) {{
  var pairs = [];
  for (var i = 0; i < values.length; i++) {{
    var v = values[i], w = weights[i];
    if (v == null || w == null || isNaN(v) || isNaN(w) || w < 0) continue;
    pairs.push([v, w]);
  }}
  if (!pairs.length) return NaN;
  pairs.sort(function(a, b) {{ return a[0] - b[0]; }});
  var total = pairs.reduce(function(s, p) {{ return s + p[1]; }}, 0);
  var cum = 0, cutoff = total / 2;
  for (var j = 0; j < pairs.length; j++) {{
    cum += pairs[j][1];
    if (cum >= cutoff) return pairs[j][0];
  }}
  return pairs[pairs.length - 1][0];
}}

function __accessBin(s) {{
  if (s <= 1e-9) return 0;
  var upper = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001];
  for (var i = 0; i < upper.length; i++) {{ if (s <= upper[i] + 1e-9) return i + 1; }}
  return upper.length;
}}
var __binLabels = ['0', '0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1'];

// Regression/R^2/ANOVA read `window.__statsData[area]` through this helper
// instead of directly, so a scenario that has been Computed at least once
// can swap in recomputed per-cell level_of_service values without touching the
// baked-in baseline blob itself. equity_flag/development percentages are
// NOT read through this path (see `_development_style_js`'s consumer and
// the distribution tab, both of which still read `window.__statsData`
// directly) -- they stay static regardless of any scenario edit, per spec.
//
// A scenario counts as "active and computed" once its Compute has
// succeeded at least once; a scenario edited again without re-pressing
// Compute still shows the last computed numbers (same staleness tolerance
// the summary bar's own `stale` badge already communicates elsewhere in the
// UI) rather than silently reverting to the baseline blob.
// H3 index bit layout (v3/v4, unchanged across resolutions): bit63 reserved,
// bits62-59 mode, bits58-56 reserved, bits55-52 resolution, bits51-45 base
// cell, then 15 x 3-bit digit groups (res1's digit at bits44-42 down to
// res15's at bits2-0; digits past a cell's own resolution are already
// 0b111/"unused"). Converting a *child* cell to its *coarser* ancestor
// needs no lookup table -- overwrite the resolution field and force every
// digit group between the new (coarser) resolution and the old one to
// 0b111, leaving the shared path prefix untouched. This is exactly libh3's
// own `cellToParent` algorithm; ported here (rather than pulled in as a
// dependency) for the same reason the population-chunk partition key is a
// plain degree grid and not an H3 parent -- see PHASE 4's own comment in
// `_editor_js`.
function __h3Resolution(h3Hex) {{
  return Number((BigInt('0x' + h3Hex) >> 52n) & 0xFn);
}}
function __h3ToParent(h3Hex, parentRes) {{
  var v = BigInt('0x' + h3Hex);
  var childRes = Number((v >> 52n) & 0xFn);
  if (parentRes >= childRes) return h3Hex;
  v = v & ~(0xFn << 52n);
  v = v | (BigInt(parentRes) << 52n);
  for (var r = parentRes + 1; r <= childRes; r++) {{
    v = v | (0x7n << BigInt(45 - 3 * r));
  }}
  return v.toString(16);
}}

// Regression/R^2/ANOVA read `window.__statsData[area]` through this helper
// instead of directly, so a scenario that has been Computed at least once
// can swap in recomputed per-cell level_of_service values without touching the
// baked-in baseline blob itself. equity_flag/development percentages are
// NOT read through this path (see `_development_style_js`'s consumer and
// the distribution tab, both of which still read `window.__statsData`
// directly) -- they stay static regardless of any scenario edit, per spec.
//
// A scenario counts as "active and computed" once its Compute has
// succeeded at least once; a scenario edited again without re-pressing
// Compute still shows the last computed numbers (same staleness tolerance
// the summary bar's own `stale` badge already communicates elsewhere in the
// UI) rather than silently reverting to the baseline blob.
//
// The stats grid's own `h3_cell` rows are a much coarser H3 resolution than
// the recompute's per-cell overrides (population chunks are `PC.res`, e.g.
// 11, for a fine-grained what-if buffer; the stats grid is aggregated for a
// stable regression sample size) -- a direct id-for-id join would silently
// match nothing. Overrides are instead rolled up to each stats row's own
// resolution as a population-weighted mean before joining.
function __statsReservedName() {{
  return ((window.__editParams || {{}}).scenarios || {{}}).reserved_current_name || 'current';
}}

// `scenario` is an explicit override, used only when resolving the stats
// panel's OPTIONAL SECONDARY column (see `__secondaryScenarioData` below) --
// '' or the reserved name both mean "baseline", any other string names a
// scenario to pull recomputed level_of_service values for REGARDLESS of which
// scenario happens to be active in the editor panel right now. Omitting
// `scenario` entirely (undefined) preserves the original behavior of
// following the editor's own active scenario -- the MAIN column always
// calls `getStatsData(area)` with just one argument for exactly this reason.
function getStatsData(area, scenario) {{
  var base = window.__statsData[area];
  try {{
    var ed = window.__editor;
    if (!ed || !base || !base.h3_cell || !base.h3_cell.length) return base;
    var reserved = __statsReservedName();
    var st = ed.state;
    var explicit = (scenario !== undefined);
    var name = explicit ? scenario : (st && st.activeScenario);
    if (!name || name === reserved) return base;
    var lc;
    if (explicit && st && name !== st.activeScenario) {{
      // A scenario other than the editor's currently-active one: only its
      // own last successful Compute (if any) counts, via the per-scenario
      // cache -- `ed.lastCompute()` always reflects whichever scenario was
      // active when Compute was last pressed, which may be a different one.
      lc = ed.computeResultForScenario ? ed.computeResultForScenario(name) : null;
    }} else {{
      lc = ed.lastCompute && ed.lastCompute();
    }}
    if (!lc || !lc.ok || !lc.h3AccessOverrides) return base;
    var overrides = lc.h3AccessOverrides;    // childH3 -> {{access, population}}
    var baseRes = __h3Resolution(base.h3_cell[0]);

    // The overrides only cover the buffer around the scenario's stations --
    // a small, population-biased subset of each coarse parent cell's
    // children -- so REPLACING a parent's level_of_service with the touched
    // subset's own mean would be wrong (it silently drops every untouched
    // child, which can swing the parent's value either direction regardless
    // of whether the edit actually helped). Instead, roll up each touched
    // child's (new - old) *gain*, population-weighted, and add that as a
    // delta against the parent's own already-population-weighted baseline
    // level_of_service -- exactly the same "gain x population, divided by total
    // population" identity `computeAccess` already uses for the metro-wide
    // summary bar's `new_access`, just applied per stats-grid cell instead
    // of over the whole study area.
    var deltaPop = {{}};
    for (var childId in overrides) {{
      var ov = overrides[childId];
      var pop = (ov.population != null && isFinite(ov.population) && ov.population >= 0) ? ov.population : 0;
      if (ov.access == null || !isFinite(ov.access) || ov.gainBase == null || !isFinite(ov.gainBase)) continue;
      var parentId = __h3ToParent(childId, baseRes);
      deltaPop[parentId] = (deltaPop[parentId] || 0) + (ov.access - ov.gainBase) * pop;
    }}

    var ids = base.h3_cell, oldAccess = base.level_of_service, pops = base.population;
    var newAccess = null;
    for (var i = 0; i < ids.length; i++) {{
      var dp = deltaPop[ids[i]];
      var totalPop = pops ? pops[i] : null;
      if (dp && totalPop != null && isFinite(totalPop) && totalPop > 0) {{
        var rolled = Math.max(0, Math.min(1, oldAccess[i] + dp / totalPop));
        if (rolled !== oldAccess[i]) {{
          if (!newAccess) newAccess = oldAccess.slice();
          newAccess[i] = rolled;
        }}
      }}
    }}
    if (!newAccess) return base;
    var clone = {{}};
    for (var k in base) clone[k] = base[k];
    clone.level_of_service = newAccess;
    return clone;
  }} catch (e) {{
    return base;
  }}
}}

function __fmt(n) {{ return n.toLocaleString(undefined, {{maximumFractionDigits: 0}}); }}

// Reads a "Compare with scenario" <select>'s current value. 'none' (the
// default, item 4's fix) means "no secondary series" -- distinct from ''
// (baseline) and from any real scenario name -- so it's checked separately
// from just falsiness (`''` is a legitimate, selectable choice).
function __secondaryScenarioValue(selId) {{
  var sel = document.getElementById(selId);
  return sel ? sel.value : 'none';
}}

// Cross-place secondary-series cache: key -> fetched `results/stats_data.json`
// payload (`{{metro:{{...}}, core:{{...}}}}`), `null` while a fetch is pending,
// absent before the first request. Same relative-sibling-path fetch pattern
// as the Place-rank tab's `__fetchPlaceRank`, but per-place/on-demand instead
// of fetching the whole roster up front (a "Compare with" popover only ever
// needs the ONE place a user picks).
var __placeCompareCache = {{}};
function __placeCompareData(placeKey, area) {{
  if (!(placeKey in __placeCompareCache)) {{
    __placeCompareCache[placeKey] = null;
    // 2026-09-02 (re-adding the per-tab "Compare with place" control this
    // fetch already supported, per explicit user request -- see
    // `_compare_with_control_html`'s own comment): was hardcoded `'../'`,
    // correct for a per-city `map.html` (one level below its siblings) but
    // WRONG for `code.combined_map`'s page (lives at the CS_transitLOS
    // ROOT, directly alongside every city directory, no `../` needed) --
    // exactly the same city-root-vs-per-city-subdir distinction
    // `place_rank_fetch_prefix` already exists to handle for the Place-rank
    // tab's own fetches (`__fetchAllPlaceRankData`), reused here instead of
    // a second hardcoded assumption.
    fetch({json.dumps(place_rank_fetch_prefix)} + placeKey + '/results/stats_data.json')
      .then(function(resp) {{ return resp.ok ? resp.json() : null; }})
      .catch(function() {{ return null; }})
      .then(function(json) {{
        __placeCompareCache[placeKey] = json || false;
        __renderActiveTab();
      }});
    return null;
  }}
  var cached = __placeCompareCache[placeKey];
  if (!cached) return null;
  return cached[area] || cached.metro || null;
}}

// The prefix a "Compare with" popover's element ids share, e.g.
// 'distSecondaryScenarioSelect' -> 'dist'.
function __comparePrefix(selId) {{
  return selId.replace('SecondaryScenarioSelect', '');
}}

// Data for the OPTIONAL secondary series chosen in a tab's "Compare with"
// popover, or null when nothing applicable is selected yet. Precedence: an
// explicit "Place" selection wins over "Scenario" (a cross-place series is
// necessarily the OTHER place's baseline network, not a scenario on THIS
// place's editor) -- both read the popover's own "City core / metro area"
// sub-select when present, falling back to the main panel's current area.
function __secondaryScenarioData(selId) {{
  var prefix = __comparePrefix(selId);
  var areaSel = document.getElementById(prefix + 'CompareAreaSelect');
  var area = areaSel ? areaSel.value : window.__statsArea;
  var placeSel = document.getElementById(prefix + 'ComparePlaceSelect');
  var placeKey = placeSel ? placeSel.value : '';
  if (placeKey) return __placeCompareData(placeKey, area);
  var sel = __secondaryScenarioValue(selId);
  if (sel === 'none') return null;
  return getStatsData(area, sel);
}}

function renderDistribution() {{
  // Main column: always the active scenario, exactly like every other panel
  // that reads `window.__statsData` -- no override here.
  var data = getStatsData(window.__statsArea);
  var mainField = document.getElementById('distMainSelect').value;
  var overlayField = document.getElementById('distOverlaySelect').value;
  var scenSel = __secondaryScenarioValue('distSecondaryScenarioSelect');
  var score = data.level_of_service, pop = data[mainField] || [];
  var mainLabel = __fieldLabel(mainField);

  // "Compare with column" and "Compare with scenario" TOGETHER define at
  // most ONE secondary series, not two independent ones: the column picker
  // says which field the secondary series reads, the scenario picker says
  // which scenario it reads that field from, and each defaults
  // independently to "same as main" when left at None/none --
  //   - column left at None  -> secondary series uses mainField (so a
  //     scenario-only comparison shows the SAME column from a different
  //     scenario, matching how regression/ANOVA's secondary-scenario
  //     control works).
  //   - scenario left at none -> secondary series uses the active scenario
  //     (so a column-only comparison shows a different column from the
  //     SAME scenario's own level_of_service axis).
  // A secondary series exists iff at least one of the two pickers is set;
  // if BOTH are left at their defaults there is no secondary series at all.
  var distPlaceSel = document.getElementById('distComparePlaceSelect');
  var distPlaceKey = distPlaceSel ? distPlaceSel.value : '';
  var secField = overlayField !== 'none' ? overlayField : mainField;
  var hasSecondary = (overlayField !== 'none') || (scenSel !== 'none') || !!distPlaceKey;
  var secData = (scenSel !== 'none' || distPlaceKey) ? __secondaryScenarioData('distSecondaryScenarioSelect') : data;
  var secLabel = __fieldLabel(secField);
  if (distPlaceKey) secLabel += ' (' + distPlaceSel.options[distPlaceSel.selectedIndex].text + ')';
  else if (scenSel !== 'none') secLabel += ' (' + __secondaryScenarioLabel(scenSel) + ')';

  var popSums = new Array(11).fill(0);
  // `secData` can be null while a cross-place fetch (`__placeCompareData`) is
  // still in flight -- `__renderActiveTab()` re-runs once it resolves, so
  // treat "hasSecondary but no data yet" the same as "no secondary series"
  // for THIS render pass rather than throwing.
  var secSums = (hasSecondary && secData) ? new Array(11).fill(0) : null;
  var totalPop = 0, totalSec = 0;
  for (var i = 0; i < score.length; i++) {{
    var s = score[i], p = pop[i];
    if (s == null || p == null || isNaN(s) || isNaN(p)) continue;
    var b = __accessBin(s);
    popSums[b] += p; totalPop += p;
  }}
  if (secSums) {{
    var secScore = secData.level_of_service, secVals = secData[secField] || [];
    for (var j = 0; j < secScore.length; j++) {{
      var ss = secScore[j], sv = secVals[j];
      if (ss == null || sv == null || isNaN(ss) || isNaN(sv)) continue;
      secSums[__accessBin(ss)] += sv; totalSec += sv;
    }}
  }}
  // Item 4 (verbatim user request): every "weighted transit score" shown
  // anywhere in the stats panel is always a WEIGHTED MEDIAN by the
  // corresponding column, never a mean -- `__weightedMean` used to be used
  // here, which reads very differently for a skewed distribution (e.g. a
  // large population share sitting at exactly 0 access pulls a mean down
  // much further than the median most people actually experience).
  var weightedAccess = __weightedMedian(score, pop);
  var weightedAccessSec = secSums ? __weightedMedian(secData.level_of_service, secData[secField]) : null;
  // Item 3 (verbatim user request): show "Total <column> <n> weighted
  // transit level of service <value>" instead of the old bare "<column>
  // -weighted access: <value>".
  var summary = 'Total ' + mainLabel + ': <b>' + __fmt(totalPop) + '</b> &nbsp;|&nbsp; ' +
    mainLabel + '-weighted transit level of service: <b>' + weightedAccess.toFixed(2) + '</b>';
  if (weightedAccessSec != null) {{
    summary += ' &nbsp;|&nbsp; Total ' + secLabel + ': <b>' + __fmt(totalSec) + '</b> &nbsp;|&nbsp; ' +
      secLabel + '-weighted transit level of service: <b>' + weightedAccessSec.toFixed(2) + '</b>';
  }}

  // Rescale check: compare the RAW value ranges of mainField (from the main
  // scenario) and secField (from the secondary series' scenario) -- e.g.
  // population spans 0..hundreds of thousands while level_of_service spans 0..1.
  // If the secondary series' range is meaningfully larger (or smaller) than
  // the main column's, note it -- the bar WIDTH scale below is switched from
  // "each series independently stretched to its own peak" to "both series
  // share one peak-derived scale" specifically so a tightly-concentrated
  // column and a spread-out column don't draw with the same peak width
  // regardless of how differently shaped their real distributions are
  // (which is what would make the smaller-range column read as
  // flat/invisible next to the larger one). When the secondary series is
  // the SAME column (scenario-only comparison), the two ranges are close
  // enough in practice that this rarely fires, but it's still computed
  // against the secondary series' own data (not assumed identical) since a
  // different scenario's recomputed values can shift the range.
  var rescaled = false;
  if (secSums) {{
    var secVals2 = secData[secField] || [];
    var mainMin = Infinity, mainMax = -Infinity, secMin = Infinity, secMax = -Infinity;
    for (var r = 0; r < pop.length; r++) {{ var v = pop[r]; if (v != null && !isNaN(v)) {{ if (v < mainMin) mainMin = v; if (v > mainMax) mainMax = v; }} }}
    for (var r2 = 0; r2 < secVals2.length; r2++) {{ var v2 = secVals2[r2]; if (v2 != null && !isNaN(v2)) {{ if (v2 < secMin) secMin = v2; if (v2 > secMax) secMax = v2; }} }}
    var mainRange = isFinite(mainMax) ? (mainMax - mainMin) : 0;
    var secRange = isFinite(secMax) ? (secMax - secMin) : 0;
    // "Meaningfully different" = at least 1.5x the other column's span.
    rescaled = mainRange > 0 && secRange > 0 && (secRange > mainRange * 1.5 || mainRange > secRange * 1.5);
    if (rescaled) {{
      summary += ' &nbsp;<span style="font-weight:400;color:#888;" title="' +
        mainLabel + ' spans ' + __fmt(mainMin) + '\\u2013' + __fmt(mainMax) + '; ' +
        secLabel + ' spans ' + __fmt(secMin) + '\\u2013' + __fmt(secMax) +
        ' -- raw magnitudes are not comparable, so bars below are rescaled to a shared width scale instead.">(bars rescaled to a shared scale)</span>';
    }}
  }}
  document.getElementById('distSummary').innerHTML = summary;

  // Bars are normalized against the LARGEST bin in this chart, not against
  // the 0-100% of-total scale the printed percentages use: with 11 bins, a
  // realistic distribution peaks around 20-30% of total population, so an
  // absolute scale left every bar stuck in the left quarter of the panel and
  // wasted most of the width. Normalizing means the tallest bin always fills
  // the full width and the rest read as proportions of it, while the
  // printed "(x.x%)" labels still report the true share of the total.
  //
  // Exception: once `rescaled` is true (see above), popPct/ovPct are ALREADY
  // on a common axis (each is "% of that column's own total", which is
  // unit-free and directly comparable regardless of population being in the
  // hundreds-of-thousands and level_of_service summing to a handful) -- but
  // independently stretching each to its OWN peak would still erase any
  // real difference in how concentrated one distribution is vs. the other.
  // So when rescaled, both series share ONE peak-derived scale instead.
  var maxPopPct = 0, maxSecPct = 0;
  for (var m = 0; m <= 10; m++) {{
    if (totalPop > 0) maxPopPct = Math.max(maxPopPct, popSums[m] / totalPop * 100);
    if (secSums && totalSec > 0) maxSecPct = Math.max(maxSecPct, secSums[m] / totalSec * 100);
  }}
  if (rescaled) {{
    var sharedMaxPct = Math.max(maxPopPct, maxSecPct);
    maxPopPct = sharedMaxPct; maxSecPct = sharedMaxPct;
  }}

  var html = '';
  for (var k = 10; k >= 0; k--) {{
    var binColor = rdylgn(k / 10);
    var popPct = totalPop > 0 ? (popSums[k] / totalPop * 100) : 0;
    var popBarPct = maxPopPct > 0 ? (popPct / maxPopPct * 100) : 0;
    // Bin spacing vs. within-pair spacing: the bars of one bin are drawn
    // flush (1px apart, and the secondary series' count moved up into the
    // shared header line instead of sitting BETWEEN the bars, where it used
    // to push them apart), while the gap BETWEEN bins is larger than either
    // -- that contrast is what makes each bin's group read as a single unit.
    var secPct = 0, secBarPct = 0;
    if (secSums) {{
      secPct = totalSec > 0 ? (secSums[k] / totalSec * 100) : 0;
      secBarPct = maxSecPct > 0 ? (secPct / maxSecPct * 100) : 0;
    }}
    html += '<div style="margin-bottom:13px;">';
    html += '<div style="display:flex;justify-content:space-between;gap:8px;font-size:10.5px;color:#444;margin-bottom:2px;">' +
      '<span style="font-weight:600;">' + __binLabels[k] + '</span>' +
      '<span style="white-space:nowrap;">' + mainLabel + ': ' + __fmt(popSums[k]) + ' (' + popPct.toFixed(1) + '%)' +
      (secSums ? '<span style="color:#8833cc;"> &nbsp;|&nbsp; ' + secLabel + ': ' + __fmt(secSums[k]) + ' (' + secPct.toFixed(1) + '%)</span>' : '') +
      '</span></div>';
    html += '<div style="background:#eef1f4;border-radius:4px;height:11px;overflow:hidden;">' +
      '<div style="background:' + binColor + ';height:100%;border-radius:4px;width:' + popBarPct + '%;"></div></div>';
    if (secSums) {{
      html += '<div style="background:#f5eefc;border-radius:3px;height:8px;overflow:hidden;margin-top:1px;">' +
        '<div style="background:#8833cc;opacity:0.65;height:100%;border-radius:3px;width:' + secBarPct + '%;"></div></div>';
    }}
    html += '</div>';
  }}
  document.getElementById('distBars').innerHTML = html;
}}

function __weightedLinReg(xs, ys, ws) {{
  var sx = [], sy = [], sw = [];
  for (var i = 0; i < xs.length; i++) {{
    var x = xs[i], y = ys[i], w = ws ? ws[i] : 1;
    // Item 2 fix: `!x` already dropped a zero SELECTED-COLUMN (x) value from
    // both the fit and the plotted scatter (`fit.xs`/`fit.ys` feed
    // scatterAndFit directly) -- but `w <= 0` (was `w < 0`) is the actual
    // gap: a zero-population weight is a WorldPop/census nodata artifact,
    // not a legitimately-weighted real zero, and it used to slip through
    // (only strictly negative weights were rejected) contributing 0 to the
    // fit math while still inflating `n` and cluttering the scatter with a
    // point that has no real weight behind it.
    if (x == null || y == null || !x || isNaN(x) || isNaN(y) || w == null || isNaN(w) || w <= 0) continue;
    sx.push(x); sy.push(y); sw.push(w);
  }}
  var n = sx.length;
  if (n < 2) return {{slope: NaN, intercept: NaN, r2: NaN, n: n, xs: sx, ys: sy}};
  var W = sw.reduce(function(a, b) {{ return a + b; }}, 0);
  var xMean = 0, yMean = 0;
  for (i = 0; i < n; i++) {{ xMean += sw[i] * sx[i]; yMean += sw[i] * sy[i]; }}
  xMean /= W; yMean /= W;
  var sxx = 0, sxy = 0;
  for (i = 0; i < n; i++) {{ sxx += sw[i] * Math.pow(sx[i] - xMean, 2); sxy += sw[i] * (sx[i] - xMean) * (sy[i] - yMean); }}
  var slope = sxx > 0 ? sxy / sxx : NaN;
  var intercept = yMean - slope * xMean;
  var ssRes = 0, ssTot = 0;
  for (i = 0; i < n; i++) {{
    var pred = slope * sx[i] + intercept;
    ssRes += sw[i] * Math.pow(sy[i] - pred, 2);
    ssTot += sw[i] * Math.pow(sy[i] - yMean, 2);
  }}
  var r2 = ssTot > 0 ? 1 - ssRes / ssTot : NaN;
  return {{slope: slope, intercept: intercept, r2: r2, n: n, xs: sx, ys: sy}};
}}

function __niceTicks(min, max, n) {{
  if (max <= min) return [min];
  var ticks = [];
  for (var i = 0; i <= n; i++) ticks.push(min + (max - min) * i / n);
  return ticks;
}}

function __arrMin(arr) {{
  // NOT `Math.min.apply(null, arr)` -- that throws "Maximum call stack size
  // exceeded" once `arr` has more than a few tens of thousands of elements
  // (a real H3 grid easily does), which used to silently abort
  // renderRegression() right after regSummary's R^2 text was already set
  // but before the SVG plot was ever built -- exactly the "shows R^2, no
  // graph" bug this loop fixes (2026-08-12).
  var m = Infinity;
  for (var i = 0; i < arr.length; i++) {{ if (arr[i] < m) m = arr[i]; }}
  return m;
}}

function __arrMax(arr) {{
  var m = -Infinity;
  for (var i = 0; i < arr.length; i++) {{ if (arr[i] > m) m = arr[i]; }}
  return m;
}}

function __secondaryScenarioLabel(sel) {{
  return sel === '' ? 'baseline' : sel;
}}

// 2026-09-02 (fix alongside re-adding the "Compare with place" row --
// see `_compare_with_control_html`'s own comment): the secondary series'
// displayed label used to always be `__secondaryScenarioLabel(secSel)`,
// i.e. always describing a SCENARIO even when the secondary series is
// actually a whole different PLACE's baseline network -- summaries read as
// e.g. "none: R²=0.13" instead of "Concepción: R²=0.13".
// Mirrors `__secondaryScenarioData`'s own precedence (place wins over
// scenario) so the label always describes whichever source the number
// actually came from.
function __secondaryLabel(selId, sel) {{
  var prefix = __comparePrefix(selId);
  var placeSel = document.getElementById(prefix + 'ComparePlaceSelect');
  if (placeSel && placeSel.value) return placeSel.options[placeSel.selectedIndex].text;
  return __secondaryScenarioLabel(sel);
}}

function renderRegression() {{
  // Main series: x is the selected field, y is ALWAYS level_of_service, both
  // from the active scenario -- regression has no per-tab "against" column
  // picker anymore (item 1 correction). The only optional extra is a whole
  // SECOND SERIES from a different scenario (same field, same level_of_service
  // target), controlled by the shared "Compare with scenario" dropdown.
  var data = getStatsData(window.__statsArea);
  var field = document.getElementById('regFieldSelect').value;
  var rawX = data[field];
  var logX = rawX.map(function(v) {{ return (v != null && v > 0) ? Math.log(v) : null; }});

  // Try both the raw variable and its log; keep whichever fit has the better R^2
  // (log only wins if it's a genuinely better fit, not applied unconditionally).
  var rawFit = __weightedLinReg(rawX, data.level_of_service, data.population);
  var logFit = __weightedLinReg(logX, data.level_of_service, data.population);
  var useLog = isFinite(logFit.r2) && (!isFinite(rawFit.r2) || logFit.r2 > rawFit.r2);
  var fit = useLog ? logFit : rawFit;
  var fieldLabelUnit = __fieldLabel(field);
  var fieldLabel = useLog ? ('log(' + fieldLabelUnit + ')') : fieldLabelUnit;

  var secSel = __secondaryScenarioValue('regSecondaryScenarioSelect');
  var regPlaceSel = document.getElementById('regComparePlaceSelect');
  var regPlaceKey = regPlaceSel ? regPlaceSel.value : '';
  var hasSecondary = secSel !== 'none' || !!regPlaceKey;
  var secFit = null;
  if (hasSecondary) {{
    var secData = __secondaryScenarioData('regSecondaryScenarioSelect');
    var secRawX = secData && secData[field];
    if (secRawX) {{
      var secX = useLog ? secRawX.map(function(v) {{ return (v != null && v > 0) ? Math.log(v) : null; }}) : secRawX;
      secFit = __weightedLinReg(secX, secData.level_of_service, secData.population);
    }}
  }}

  var summaryText = 'level_of_service ~ ' + fieldLabel + '   R\\u00b2=' + (isFinite(fit.r2) ? fit.r2.toFixed(2) : 'n/a') + '   n=' + fit.n +
    (useLog ? '  (log fits better)' : '');
  if (hasSecondary && secFit) {{
    summaryText += '   |   ' + __secondaryLabel('regSecondaryScenarioSelect', secSel) + ': R\\u00b2=' +
      (isFinite(secFit.r2) ? secFit.r2.toFixed(2) : 'n/a') + '   n=' + secFit.n;
  }}
  document.getElementById('regSummary').textContent = summaryText;

  var w = 420, h = 280, padL = 55, padB = 34, padT = 12, padR = 12;
  var xs = fit.xs, ys = fit.ys;
  var xMin = __arrMin(xs), xMax = __arrMax(xs);
  var yMin = __arrMin(ys), yMax = __arrMax(ys);
  var haveSecPoints = hasSecondary && secFit && secFit.xs.length;
  if (haveSecPoints) {{
    xMin = Math.min(xMin, __arrMin(secFit.xs)); xMax = Math.max(xMax, __arrMax(secFit.xs));
    yMin = Math.min(yMin, __arrMin(secFit.ys)); yMax = Math.max(yMax, __arrMax(secFit.ys));
  }}
  var xr = xMax - xMin || 1, yr = yMax - yMin || 1;
  function px(x) {{ return padL + (x - xMin) / xr * (w - padL - padR); }}
  function py(y) {{ return h - padB - (y - yMin) / yr * (h - padT - padB); }}

  // Clip the fitted line to the plotted y-range box -- the line is
  // evaluated at x=xMin/xMax (always inside the x-domain by construction),
  // but its y-value there can fall outside [yMin, yMax] (e.g. a steep slope
  // near a domain edge), which used to draw the line poking out past the
  // axes/gridlines. Since x is already in-range, only the y-extent needs
  // clipping -- parametrize the segment and keep only the portion whose y
  // falls inside [yMin, yMax].
  function scatterAndFit(fitObj, pointColor, lineColor) {{
    var xsL = fitObj.xs, ysL = fitObj.ys;
    var maxPoints = 1500;
    var step = Math.max(1, Math.floor(xsL.length / maxPoints));
    var out = '';
    for (var i = 0; i < xsL.length; i += step) {{
      out += '<circle cx="' + px(xsL[i]).toFixed(1) + '" cy="' + py(ysL[i]).toFixed(1) + '" r="2" fill="' + pointColor + '" opacity="0.35"></circle>';
    }}
    if (isFinite(fitObj.slope)) {{
      var lx0 = xMin, ly0 = fitObj.slope * xMin + fitObj.intercept;
      var lx1 = xMax, ly1 = fitObj.slope * xMax + fitObj.intercept;
      var dy = ly1 - ly0;
      var t0 = 0, t1 = 1;
      if (dy !== 0) {{
        var tA = (yMin - ly0) / dy, tB = (yMax - ly0) / dy;
        var tlo = Math.min(tA, tB), thi = Math.max(tA, tB);
        t0 = Math.max(t0, tlo); t1 = Math.min(t1, thi);
      }} else if (ly0 < yMin || ly0 > yMax) {{
        t1 = -1; // flat line entirely outside the y-range: draw nothing
      }}
      if (t1 >= t0) {{
        var cx0 = lx0 + (lx1 - lx0) * t0, cy0 = ly0 + dy * t0;
        var cx1 = lx0 + (lx1 - lx0) * t1, cy1 = ly0 + dy * t1;
        out += '<line x1="' + px(cx0).toFixed(1) + '" y1="' + py(cy0).toFixed(1) + '" x2="' + px(cx1).toFixed(1) +
          '" y2="' + py(cy1).toFixed(1) + '" stroke="' + lineColor + '" stroke-width="2"></line>';
      }}
    }}
    return out;
  }}

  var mainDraw = scatterAndFit(fit, '#3366cc', '#cc3333');
  var secDraw = haveSecPoints ? scatterAndFit(secFit, '#ff9900', '#8833cc') : '';

  var xTicks = __niceTicks(xMin, xMax, 4), yTicks = __niceTicks(yMin, yMax, 4);
  var axes = '<line x1="' + padL + '" y1="' + (h - padB) + '" x2="' + (w - padR) + '" y2="' + (h - padB) + '" stroke="#999"></line>' +
    '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (h - padB) + '" stroke="#999"></line>';
  xTicks.forEach(function(t) {{
    var x = px(t);
    axes += '<line x1="' + x + '" y1="' + (h - padB) + '" x2="' + x + '" y2="' + (h - padB + 4) + '" stroke="#999"></line>';
    // Tick POSITIONS stay in log-space (geometry unchanged) when `useLog` --
    // only the LABEL text is converted back to real units via Math.exp, so
    // readers see actual field magnitudes instead of meaningless log values.
    // Same bug as the opacity/circle-size legends (`__fmtLeg` in
    // `geohierarchy.maps.maplibre.render`): a share/rate field is stored
    // as a raw 0..1 fraction, same as everywhere else on the grid -- axis
    // tick labels need the same percent rescale the click-popup already
    // applies (`__isPercentField`/`__shapePopupNum`), or "50%" reads as a
    // bare "0.5".
    var tickVal = useLog ? Math.exp(t) : t;
    var tickLabel = (window.__isPercentField && window.__isPercentField(field))
      ? (tickVal * 100).toFixed(1) + '%'
      : __fmtLegendNum(tickVal);
    axes += '<text x="' + x + '" y="' + (h - padB + 15) + '" font-size="9" text-anchor="middle">' + tickLabel + '</text>';
  }});
  yTicks.forEach(function(t) {{
    var y = py(t);
    axes += '<line x1="' + (padL - 4) + '" y1="' + y + '" x2="' + padL + '" y2="' + y + '"stroke="#999"></line>';
    axes += '<text x="' + (padL - 7) + '" y="' + (y + 3) + '" font-size="9" text-anchor="end">' + t.toFixed(2) + '</text>';
    axes += '<line x1="' + padL + '" y1="' + y + '" x2="' + (w - padR) + '" y2="' + y + '" stroke="#eee"></line>';
  }});
  var labels = '<text x="' + (padL + (w - padL - padR) / 2) + '" y="' + (h - 4) + '" font-size="10" text-anchor="middle" fill="#444">' + fieldLabel + '</text>' +
    '<text x="12" y="' + (padT + (h - padT - padB) / 2) + '" font-size="10" text-anchor="middle" fill="#444" transform="rotate(-90 12 ' + (padT + (h - padT - padB) / 2) + ')">level_of_service</text>';

  var legend = '';
  if (haveSecPoints) {{
    var lx = padL + 6, ly = padT + 10;
    legend += '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 16) + '" y2="' + ly + '" stroke="#cc3333" stroke-width="2"></line>' +
      '<text x="' + (lx + 20) + '" y="' + (ly + 3) + '" font-size="9" fill="#333">active scenario</text>';
    var ly2 = ly + 13;
    legend += '<line x1="' + lx + '" y1="' + ly2 + '" x2="' + (lx + 16) + '" y2="' + ly2 + '" stroke="#8833cc" stroke-width="2"></line>' +
      '<text x="' + (lx + 20) + '" y="' + (ly2 + 3) + '" font-size="9" fill="#333">' + __secondaryLabel('regSecondaryScenarioSelect', secSel) + '</text>';
  }}

  var svg = '<svg width="' + w + '" height="' + h + '" style="background:#fafafa;border:1px solid #ddd;">' +
    axes + mainDraw + secDraw + labels + legend +
    '</svg>';
  document.getElementById('regPlot').innerHTML = svg;
  renderR2Bars(field);
}}

// Best-of-{{raw, log}} R^2 for one field -- exactly the choice renderRegression
// makes for the plotted fit, so a bar can never disagree with the headline
// R^2 shown above for the same variable.
function __bestR2(data, field) {{
  var rawX = data[field];
  if (!rawX) return null;
  var rawFit = __weightedLinReg(rawX, data.level_of_service, data.population);
  var logFit = __weightedLinReg(
    rawX.map(function(v) {{ return (v != null && v > 0) ? Math.log(v) : null; }}),
    data.level_of_service, data.population);
  var useLog = isFinite(logFit.r2) && (!isFinite(rawFit.r2) || logFit.r2 > rawFit.r2);
  var fit = useLog ? logFit : rawFit;
  return isFinite(fit.r2) ? {{r2: fit.r2, useLog: useLog, n: fit.n}} : null;
}}

// Same bar visual language as renderAnova (label + value above a rounded
// track), minus the centered zero line: R^2 is one-directional, so the bar
// simply fills from the left. Non-positive fits are excluded outright rather
// than drawn as empty rows.
function renderR2Bars(selectedField) {{
  // Main series is always "every variable vs THIS (main/active-scenario)
  // column's level_of_service". The optional second bar per variable is the
  // SAME field's best-of-{{raw, log}} R^2 computed against a different
  // scenario's level_of_service, controlled by the shared "Compare with
  // scenario" dropdown -- reuses regSecondaryScenarioSelect since this bar
  // chart lives in the same regression tab/section as renderRegression's
  // scatter plot (see renderRegression above for the identical pattern).
  var data = getStatsData(window.__statsArea);
  var secSel = __secondaryScenarioValue('regSecondaryScenarioSelect');
  var r2PlaceSel = document.getElementById('regComparePlaceSelect');
  var r2PlaceKey = r2PlaceSel ? r2PlaceSel.value : '';
  var hasSecondary = secSel !== 'none' || !!r2PlaceKey;
  var secData = hasSecondary ? __secondaryScenarioData('regSecondaryScenarioSelect') : null;

  var results = [];
  __anovaFields.forEach(function(field) {{
    var hit = __bestR2(data, field);
    if (!(hit && hit.r2 > 0)) return;
    var secHit = hasSecondary && secData ? __bestR2(secData, field) : null;
    results.push({{
      field: field, r2: hit.r2, useLog: hit.useLog,
      secR2: (secHit && secHit.r2 > 0) ? secHit.r2 : null, secUseLog: secHit ? secHit.useLog : false
    }});
  }});
  results.sort(function(a, b) {{ return b.r2 - a.r2; }});
  var maxR2 = results.length ? results[0].r2 : 1;
  results.forEach(function(r) {{ if (r.secR2 != null && r.secR2 > maxR2) maxR2 = r.secR2; }});
  var html = '';
  results.forEach(function(r) {{
    var barWidth = (r.r2 / maxR2) * 100;
    var isSel = (r.field === selectedField);
    var color = isSel ? '#3a6ea8' : '#33aa55';
    html += '<div style="margin-bottom:6px;">';
    var rFieldLabel = __fieldLabel(r.field);
    html += '<div style="font-size:10.5px;color:#444;margin-bottom:2px;">' +
      (isSel ? '<b>' + rFieldLabel + '</b>' : rFieldLabel) + (r.useLog ? ' <span style="color:#888;">(log)</span>' : '') +
      '  <b>(' + r.r2.toFixed(2) + ')</b>' +
      (r.secR2 != null ? '  <span style="color:#8833cc;">' + __secondaryLabel('regSecondaryScenarioSelect', secSel) + ': ' +
        r.secR2.toFixed(2) + (r.secUseLog ? ' (log)' : '') + '</span>' : '') +
      '</div>';
    html += '<div style="position:relative;height:13px;background:#eef1f4;border-radius:3px;overflow:hidden;">';
    html += '<div style="position:absolute;left:0;top:0;bottom:0;width:' + barWidth + '%;background:' + color + ';border-radius:3px;"></div>';
    html += '</div>';
    if (r.secR2 != null) {{
      var secBarWidth = (r.secR2 / maxR2) * 100;
      html += '<div style="position:relative;height:13px;margin-top:2px;background:#eef1f4;border-radius:3px;overflow:hidden;">';
      html += '<div style="position:absolute;left:0;top:0;bottom:0;width:' + secBarWidth + '%;background:#8833cc;border-radius:3px;"></div>';
      html += '</div>';
    }}
    html += '</div>';
  }});
  document.getElementById('regR2Bars').innerHTML = html || '<div>No variables with a positive R&sup2;.</div>';
}}

var __anovaFields = {json.dumps(regression_fields)};

// Sanctioned way for an OUTSIDE script to change the Regression/ANOVA
// variable list after the page has loaded. `__anovaFields` above is a
// plain `var` scoped to this whole block's own enclosing closure (the
// `window.addEventListener('load', ...)` callback `_stats_panel_js`
// wraps everything in) -- despite the `var` keyword, that makes it
// function-local, NOT a `window` property, so an external caller assigning
// `window.__anovaFields = [...]` creates an unrelated same-named global
// that `renderAnova`/`renderRegression` never read (a real bug found live:
// `CS_transitLOS/code/combined_map.py`'s per-city dynamic field rebuild did
// exactly that and silently had no effect -- `window.__anovaFields` looked
// right when inspected, but the rendered ANOVA/Regression tabs kept using
// the original build-time list). This setter runs INSIDE the closure, so
// it can actually reassign the real variable, and also keeps
// `#regFieldSelect`'s `<option>`s (baked server-side at build time from
// the same original list) in sync with the new one.
function __setRegressionFields(fields) {{
  __anovaFields = fields;
  var sel = document.getElementById('regFieldSelect');
  if (sel) {{
    var prev = sel.value;
    sel.innerHTML = fields.map(function(f) {{ return '<option value="' + f + '">' + __fieldLabel(f) + '</option>'; }}).join('');
    sel.value = fields.indexOf(prev) !== -1 ? prev : (fields[0] || '');
  }}
}}
window.__setRegressionFields = __setRegressionFields;

// High vs. low half (split by `field`'s own median), difference in mean
// level_of_service, for whichever `data` blob is passed in -- shared by the main
// and optional secondary-scenario series so both are computed identically.
function __anovaDiff(data, field) {{
  if (!data[field]) return null;
  var xs = data[field], scores = data.level_of_service, ws = data.population;
  var pairsX = [], pairsS = [], pairsW = [];
  for (var i = 0; i < xs.length; i++) {{
    var x = xs[i], s = scores[i], w = ws[i];
    // Unlike `__weightedLinReg` (regression tab), ANOVA does NOT drop a
    // zero `x` value here: some real share/rate columns are genuinely
    // binary or zero-heavy (e.g. `cbs_jewish_share`/`cbs_muslim_share`,
    // derived from a per-locality dominant-classification code -- a
    // locality with 0% Muslim residents has a real, meaningful x=0, not a
    // nodata artifact). Treating every real zero as missing silently threw
    // away one whole side of exactly the comparison ANOVA exists to make,
    // and left ONLY x=1 rows for a variable like that -- which then made
    // the median split below degenerate to one empty group and the whole
    // row's diff evaluate to NaN (no bar rendered). `x == null`/`isNaN(x)`
    // below still correctly drops real missing data (an uncovered h3 cell
    // with no census join at all, `null`/`NaN`, not `0`).
    if (x == null || s == null || w == null || isNaN(x) || isNaN(s) || isNaN(w) || w <= 0) continue;
    pairsX.push(x); pairsS.push(s); pairsW.push(w);
  }}
  if (pairsX.length < 4) return null;
  var median = __weightedMedian(pairsX, pairsW);
  // A skewed/near-binary field (e.g. a locality-level classification like
  // `cbs_jewish_share`, which after the zero-drop above only ever holds
  // 1.0) can put the median AT the field's max (or min) value -- splitting
  // with a fixed `<=`/`>` boundary then puts every point in one group and
  // leaves the other empty, so `__weightedMean([], [])` divides 0/0 and
  // the whole ANOVA row comes back NaN (no bar renders for it). Try both
  // boundary conventions (`<` / `>=` first, then `<=` / `>`) and use
  // whichever actually produces two non-empty groups; only truly
  // degenerate data (every surviving value identical) fails both.
  function split(lowCmp) {{
    var lowS = [], lowW = [], highS = [], highW = [];
    for (var j = 0; j < pairsX.length; j++) {{
      if (lowCmp(pairsX[j], median)) {{ lowS.push(pairsS[j]); lowW.push(pairsW[j]); }}
      else {{ highS.push(pairsS[j]); highW.push(pairsW[j]); }}
    }}
    return (lowS.length && highS.length) ? {{lowS: lowS, lowW: lowW, highS: highS, highW: highW}} : null;
  }}
  var groups = split(function(x, m) {{ return x < m; }}) || split(function(x, m) {{ return x <= m; }});
  if (!groups) return null;
  var lowMean = __weightedMean(groups.lowS, groups.lowW), highMean = __weightedMean(groups.highS, groups.highW);
  return highMean - lowMean;
}}

function renderAnova() {{
  // Grouping variables (x) and the target (y, always level_of_service) both come
  // from the active-scenario main data -- ANOVA has no per-tab "compare
  // with" column picker anymore (item 1 correction). The only optional
  // extra is a whole SECOND SERIES (second bar per variable) from a
  // different scenario's level_of_service, controlled by the shared "Compare
  // with scenario" dropdown -- see renderRegression above for the same
  // pattern.
  var data = getStatsData(window.__statsArea);
  var secSel = __secondaryScenarioValue('anovaSecondaryScenarioSelect');
  var anovaPlaceSel = document.getElementById('anovaComparePlaceSelect');
  var anovaPlaceKey = anovaPlaceSel ? anovaPlaceSel.value : '';
  var hasSecondary = secSel !== 'none' || !!anovaPlaceKey;
  var secData = hasSecondary ? __secondaryScenarioData('anovaSecondaryScenarioSelect') : null;

  var results = [];
  __anovaFields.forEach(function(field) {{
    var diff = __anovaDiff(data, field);
    if (diff == null) return;
    var secDiff = (hasSecondary && secData) ? __anovaDiff(secData, field) : null;
    results.push({{field: field, diff: diff, secDiff: secDiff}});
  }});
  // Ordered by the *magnitude* of the main difference (biggest effect first,
  // whichever direction), even though the signed value is what's plotted.
  results.sort(function(a, b) {{ return Math.abs(b.diff) - Math.abs(a.diff); }});
  var allAbs = results.map(function(r) {{ return Math.abs(r.diff); }});
  results.forEach(function(r) {{ if (r.secDiff != null) allAbs.push(Math.abs(r.secDiff)); }});
  var maxAbs = Math.max.apply(null, allAbs.concat([1e-9]));

  function barRow(diff, color) {{
    var pct = (diff / maxAbs) * 50;
    var barLeft = diff < 0 ? (50 + pct) : 50;
    var barWidth = Math.abs(pct);
    return '<div style="position:relative;height:11px;background:#eef1f4;border-radius:3px;overflow:hidden;margin-bottom:2px;">' +
      '<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#9aa4ad;"></div>' +
      '<div style="position:absolute;left:' + barLeft + '%;top:0;bottom:0;width:' + barWidth + '%;background:' + color + ';border-radius:2px;"></div>' +
      '</div>';
  }}

  var html = '';
  results.forEach(function(r) {{
    html += '<div style="margin-bottom:6px;">';
    html += '<div style="font-size:10.5px;color:#444;margin-bottom:2px;">' + __fieldLabel(r.field) +
      '  <b>(' + (r.diff >= 0 ? '+' : '') + r.diff.toFixed(2) + ')</b>' +
      (r.secDiff != null ? '  <span style="color:#8833cc;">' + __secondaryLabel('anovaSecondaryScenarioSelect', secSel) + ': ' +
        (r.secDiff >= 0 ? '+' : '') + r.secDiff.toFixed(2) + '</span>' : '') +
      '</div>';
    html += barRow(r.diff, r.diff < 0 ? '#cc3333' : '#33aa55');
    if (r.secDiff != null) html += barRow(r.secDiff, r.secDiff < 0 ? '#aa7700' : '#8833cc');
    html += '</div>';
  }});
  var header = '<div style="color:#666;margin-bottom:6px;">High vs. low half, split by each variable\\'s own median -- difference in mean <b>level_of_service</b>' +
    (hasSecondary ? ' (top bar: active scenario, bottom bar: <b>' + __secondaryLabel('anovaSecondaryScenarioSelect', secSel) + '</b>)' : '') + '</div>';
  document.getElementById('anovaBars').innerHTML = header + (html || '<div>No variables available.</div>');
}}

// --- Discretization tab: pure formula-shape mirrors of transitlos.scoring's
// Python functions (speed_score, frequency_score, reliability_score,
// walk_decay), parameterized by window.__disc (this map's own REGIONS[region]
// defaults, baked in at build time -- see _discretization_params_js). These
// intentionally duplicate only the *shape*, matching this file's existing
// client-side-stats pattern (see renderRegression above) -- kept in sync by
// construction since the constants come from the real Python dataclass, not
// a second hand-copied set of numbers.

function __jsSpeedScore(avgSpeedKmh) {{
  // Mirrors speed.py's 2026-08-12 fix: the score reaches exactly 1.0 at
  // breakpoints[2] (ITDP's own "zero speed penalty" anchor) and stays
  // there -- `saturation_kmh` no longer extends the ramp, see that file's
  // module docstring for the discontinuity bug this replaced.
  var b = window.__disc.modeSpeedBreakpointsKmh;
  var b0 = b[0], b1 = b[1], b2 = b[2];
  if (avgSpeedKmh <= 0) return 0.0;
  if (avgSpeedKmh < b0) return 0.5 * (avgSpeedKmh / b0);
  if (avgSpeedKmh < b1) return 0.5 + 0.35 * (avgSpeedKmh - b0) / (b1 - b0);
  if (avgSpeedKmh < b2) return 0.85 + 0.15 * (avgSpeedKmh - b1) / (b2 - b1);
  return 1.0;
}}

function __jsFrequencyScore(headwayMinutes) {{
  var sat = window.__disc.headwaySaturationMinutes, scale = window.__disc.headwayDecayScaleMinutes;
  if (headwayMinutes <= sat) return 1.0;
  return Math.exp(-(headwayMinutes - sat) / scale);
}}

function __jsReliabilityScore(headwayCv) {{
  return Math.exp(-headwayCv / window.__disc.reliabilityCvScale);
}}

// Mirrors mode.py's MODE_SCORES -- fixed per-category constants, not a
// formula, so baked in directly rather than threaded through window.__disc.
var __MODE_SCORES = {{
  rail: 1.0,
  tram: Math.pow(1.0 * (1.0 / 1.15), 0.5),
  bus: 1.0 / 1.15,
}};
var __MODE_LINE_COLOR = {{ rail: '#006400', tram: '#8b4513', bus: '#8b0000' }};

// A fixed, representative headway_cv for reliability_score whenever it isn't
// one of the two axes the "Stop score" curve toggles between (speed and
// headway per the user's explicit request) -- reliability itself has no
// third slider here, it's just held steady so the two requested axes are
// the only things moving.
var __DISC_STOP_SCORE_DEFAULT_CV = 0.3;

// `headwayCv` is optional and defaults to `__DISC_STOP_SCORE_DEFAULT_CV`, so
// every existing Discretization-tab call site is unchanged. It exists for the
// scenario editor (Phase 4), which takes its CV from
// `default_edit_params.json`'s `access_recompute.headway_cv` rather than from
// this file's plot-only constant.
function __jsStopScore(mode, speedKmh, headwayMinutes, headwayCv) {{
  var w = window.__disc.weights;
  var m = __MODE_SCORES[mode];
  var s = __jsSpeedScore(speedKmh);
  var f = __jsFrequencyScore(headwayMinutes);
  var r = __jsReliabilityScore(headwayCv == null ? __DISC_STOP_SCORE_DEFAULT_CV : headwayCv);
  return Math.pow(m, w[0]) * Math.pow(s, w[1]) * Math.pow(f, w[2]) * Math.pow(r, w[3]);
}}

function __jsWalkDecay(walkTimeMinutes) {{
  var d = window.__disc;
  var tSat = d.walkDistanceSaturationM / d.walkingSpeedMps / 60.0;
  if (walkTimeMinutes <= tSat) return 1.0;
  return Math.pow(1.0 + (walkTimeMinutes - tSat) / d.walkT0BaseMinutes, -d.walkDecayShapeP);
}}

var __discParamDefs = {{
  speed: {{
    xLabel: {json.dumps(field_label("avg_speed_kmh"))}, yLabel: 'speed_score',
    // 50 km/h, not 90: the curve reaches its 1.0 plateau at the third mode
    // speed breakpoint well below that, so the extra 40 km/h was a flat
    // line squeezing the informative part of the shape into the left half.
    // (This is the standalone Speed tab only -- the Stop score tab's own
    // speed axis keeps its wider range, since there speed is one input of a
    // product that keeps varying past this point.)
    xMin: 0, xMax: 50, steps: 180,
    fn: function(x) {{ return __jsSpeedScore(x); }},
  }},
  frequency: {{
    xLabel: {json.dumps(field_label("headway_minutes"))}, yLabel: 'frequency_score',
    xMin: 0.5, xMax: 60, steps: 180,
    fn: function(x) {{ return __jsFrequencyScore(x); }},
  }},
  reliability: {{
    xLabel: 'headway_cv', yLabel: 'reliability_score',
    xMin: 0, xMax: 2, steps: 180,
    fn: function(x) {{ return __jsReliabilityScore(x); }},
  }},
  distance: {{
    xLabel: {json.dumps(field_label("walk_time_minutes"))}, yLabel: 'walk_decay (D)',
    xMin: 0, xMax: 30, steps: 180,
    fn: function(x) {{ return __jsWalkDecay(x); }},
  }},
  // level_of_service = stop_score * D(walk_time), per accessibility_score.py.
  // Its two inputs are symmetrical in exactly the way the stop_score curve's
  // speed/headway pair is, so it uses the SAME axis-toggle mechanism
  // (`axisToggle`): whichever of stop_score / walk_time the dropdown picks
  // becomes x, the other is held fixed at the slider's value. That replaced
  // the old walk-time-only slider, which could only ever show this curve
  // from one of its two sides.
  access: {{
    axisToggle: 'access', yLabel: 'level_of_service',
  }},
  // One curve per mode (mode_score fixed per curve, per __MODE_SCORES). X is
  // whichever of speed/headway `#discStopScoreAxisSelect` picks (default
  // headway, per explicit request); the other one is held fixed at
  // `#discStopScoreSlider`'s value. Handled specially in renderDiscretization
  // (multiCurve) rather than through the single-curve `fn` path every other
  // entry here uses.
  stop_score: {{
    multiCurve: true, axisToggle: 'stop_score',
    yLabel: 'stop_score',
  }},
}};

window.__discParam = 'speed';

// --- Axis-toggle curves ----------------------------------------------------
// Two curves (stop_score, level_of_service) are functions of two inputs each, and
// both use the identical UI: a dropdown picking which input is the x axis,
// plus a slider holding the OTHER one fixed. `__discAxisGroups` is the single
// registry of that pattern (element ids + per-axis config + live state) so the
// two share one implementation rather than each growing its own row/handlers.

// A slider whose raw track position is NOT the value it represents. The
// stop-score curve's fixed SPEED slider needs this: the interesting range is
// 0-30 km/h (where speed_score actually moves), but the axis must still reach
// 200 km/h for high-speed rail. A linear 0-200 track would compress every
// realistic bus/tram speed into its first 15%. Two linear segments instead --
// 0-30 km/h across the first 75% of the track, 30-200 km/h across the last
// 25% -- keep ordinary speeds precisely selectable while the tail stays
// reachable. `toRaw` is the exact inverse, needed to place the thumb when the
// value is set programmatically (axis switch / initial render).
var __SPEED_SLIDER_BREAK_RAW = 750, __SPEED_SLIDER_RAW_MAX = 1000;
var __SPEED_SLIDER_BREAK_KMH = 30, __SPEED_SLIDER_MAX_KMH = 200;
var __speedSliderRaw = {{
  min: 0, max: __SPEED_SLIDER_RAW_MAX, step: 1,
  toValue: function(raw) {{
    if (raw <= __SPEED_SLIDER_BREAK_RAW) return raw / __SPEED_SLIDER_BREAK_RAW * __SPEED_SLIDER_BREAK_KMH;
    return __SPEED_SLIDER_BREAK_KMH +
      (raw - __SPEED_SLIDER_BREAK_RAW) / (__SPEED_SLIDER_RAW_MAX - __SPEED_SLIDER_BREAK_RAW) *
      (__SPEED_SLIDER_MAX_KMH - __SPEED_SLIDER_BREAK_KMH);
  }},
  toRaw: function(v) {{
    if (v <= __SPEED_SLIDER_BREAK_KMH) return v / __SPEED_SLIDER_BREAK_KMH * __SPEED_SLIDER_BREAK_RAW;
    return __SPEED_SLIDER_BREAK_RAW +
      (v - __SPEED_SLIDER_BREAK_KMH) / (__SPEED_SLIDER_MAX_KMH - __SPEED_SLIDER_BREAK_KMH) *
      (__SPEED_SLIDER_RAW_MAX - __SPEED_SLIDER_BREAK_RAW);
  }},
}};

// Axis config for the two things the "Stop score" curve can plot on x
// (mirrors the frequency/speed entries above's own ranges, so switching
// between this curve and the standalone Frequency/Speed ones feels
// consistent).
var __discStopScoreAxisDefs = {{
  headway: {{ xLabel: {json.dumps(field_label("headway_minutes"))}, xMin: 0.5, xMax: 60, steps: 180,
    otherLabel: 'Speed (avg_speed_kmh)', otherUnit: ' km/h', otherDefault: 20,
    sliderRaw: __speedSliderRaw }},
  speed: {{ xLabel: {json.dumps(field_label("avg_speed_kmh"))}, xMin: 0, xMax: 90, steps: 180,
    otherLabel: 'Headway (headway_minutes)', otherUnit: ' min',
    otherMin: 0.5, otherMax: 60, otherStep: 0.5, otherDefault: 10 }},
}};

// The walk-time default is 2x the walk-decay saturation boundary -- the same
// representative, deliberately non-saturated value the access curve used
// before it had any slider, so the default view is unchanged. It depends on
// window.__disc, hence a function rather than a literal.
function __discDefaultWalkTime() {{
  var d = window.__disc;
  return 2 * (d.walkDistanceSaturationM / d.walkingSpeedMps / 60.0);
}}

var __discAccessAxisDefs = {{
  stop_score: {{ xLabel: 'stop_score', xMin: 0, xMax: 1, steps: 100,
    otherLabel: 'Walk time (minutes)', otherUnit: ' min',
    // step 0.1, not 0.5: the default is a computed value (~6.4 min) and a
    // coarser step would snap the thumb away from the number the label and
    // curve actually use.
    otherMin: 0, otherMax: 30, otherStep: 0.1, otherDefaultFn: __discDefaultWalkTime }},
  walk_time: {{ xLabel: {json.dumps(field_label("walk_time_minutes"))}, xMin: 0, xMax: 30, steps: 180,
    otherLabel: 'Stop score', otherUnit: '',
    otherMin: 0, otherMax: 1, otherStep: 0.01, otherDefault: 0.7 }},
}};

var __discAxisGroups = {{
  stop_score: {{
    defs: __discStopScoreAxisDefs, axis: 'headway', other: null,
    rowId: 'discStopScoreRow', selectId: 'discStopScoreAxisSelect', sliderId: 'discStopScoreSlider',
    labelId: 'discStopScoreSliderLabel', valueId: 'discStopScoreSliderValue',
  }},
  access: {{
    defs: __discAccessAxisDefs, axis: 'stop_score', other: null,
    rowId: 'discAccessRow', selectId: 'discAccessAxisSelect', sliderId: 'discAccessSlider',
    labelId: 'discAccessSliderLabel', valueId: 'discAccessSliderValue',
  }},
}};
// Kept as globals for backwards compatibility with anything (and anyone)
// reading the stop-score axis state by its original names.
window.__discStopScoreAxis = 'headway';
window.__discStopScoreOther = null;

function __discAxisDefault(axisDef) {{
  return axisDef.otherDefaultFn ? axisDef.otherDefaultFn() : axisDef.otherDefault;
}}

// Sync the group's row (dropdown + slider min/max/step/position + labels) to
// its current state, honoring a non-linear `sliderRaw` mapping when present.
function __discSyncAxisRow(groupKey) {{
  var g = __discAxisGroups[groupKey];
  var axisDef = g.defs[g.axis];
  if (g.other == null) g.other = __discAxisDefault(axisDef);
  var sel = document.getElementById(g.selectId);
  if (sel) sel.value = g.axis;
  var slider = document.getElementById(g.sliderId);
  if (slider) {{
    var raw = axisDef.sliderRaw;
    slider.min = raw ? raw.min : axisDef.otherMin;
    slider.max = raw ? raw.max : axisDef.otherMax;
    slider.step = raw ? raw.step : axisDef.otherStep;
    slider.value = raw ? raw.toRaw(g.other) : g.other;
  }}
  document.getElementById(g.labelId).textContent = axisDef.otherLabel;
  document.getElementById(g.valueId).textContent = g.other.toFixed(1) + (axisDef.otherUnit || '');
  if (groupKey === 'stop_score') {{
    window.__discStopScoreAxis = g.axis;
    window.__discStopScoreOther = g.other;
  }}
  return axisDef;
}}

function renderDiscretization() {{
  var def = __discParamDefs[window.__discParam];

  // Exactly one axis-toggle row can be visible at a time (the one belonging
  // to the selected curve, if it has one).
  Object.keys(__discAxisGroups).forEach(function(k) {{
    var row = document.getElementById(__discAxisGroups[k].rowId);
    if (row) row.style.display = (def.axisToggle === k) ? 'block' : 'none';
  }});
  var group = def.axisToggle ? __discAxisGroups[def.axisToggle] : null;
  var groupAxisDef = group ? __discSyncAxisRow(def.axisToggle) : null;
  var isMultiCurve = !!def.multiCurve;

  var w = 420, h = 280, padL = 45, padB = 34, padT = 12, padR = 12;
  var xMin, xMax, xLabel, yLabel = def.yLabel;
  var series = []; // [{{mode, xs, ys, color}}] for multiCurve, single entry otherwise
  var summaryText;

  if (isMultiCurve) {{
    var axisDef = groupAxisDef;
    xLabel = axisDef.xLabel;
    xMin = axisDef.xMin; xMax = axisDef.xMax;
    ['rail', 'tram', 'bus'].forEach(function(mode) {{
      var xs = [], ys = [];
      for (var i = 0; i <= axisDef.steps; i++) {{
        var x = xMin + (xMax - xMin) * i / axisDef.steps;
        var speedKmh = group.axis === 'speed' ? x : group.other;
        var headwayMin = group.axis === 'headway' ? x : group.other;
        xs.push(x); ys.push(__jsStopScore(mode, speedKmh, headwayMin));
      }}
      series.push({{mode: mode, xs: xs, ys: ys, color: __MODE_LINE_COLOR[mode]}});
    }});
    var otherLabelShort = group.axis === 'speed' ? 'headway' : 'avg_speed_kmh';
    summaryText = yLabel + ' vs ' + xLabel + ' by mode (this map\\'s default parameters, ' +
      otherLabelShort + ' = ' + group.other.toFixed(1) +
      ', headway_cv = ' + __DISC_STOP_SCORE_DEFAULT_CV.toFixed(2) + ')';
  }} else if (def.axisToggle === 'access') {{
    // level_of_service = stop_score * D(walk_time): whichever of the two is on
    // x sweeps, the other stays at the slider's value.
    var aDef = groupAxisDef;
    xLabel = aDef.xLabel;
    xMin = aDef.xMin; xMax = aDef.xMax;
    var axs = [], ays = [];
    for (var a = 0; a <= aDef.steps; a++) {{
      var ax = xMin + (xMax - xMin) * a / aDef.steps;
      var stopScoreVal = group.axis === 'stop_score' ? ax : group.other;
      var walkMinutes = group.axis === 'walk_time' ? ax : group.other;
      axs.push(ax); ays.push(stopScoreVal * __jsWalkDecay(walkMinutes));
    }}
    series.push({{mode: null, xs: axs, ys: ays, color: '#3a6ea8'}});
    if (group.axis === 'stop_score') {{
      summaryText = yLabel + ' vs stop_score (walk_time = ' + group.other.toFixed(1) +
        ' min, D = ' + __jsWalkDecay(group.other).toFixed(2) + ')';
    }} else {{
      summaryText = yLabel + ' vs walk_time_minutes (stop_score = ' + group.other.toFixed(2) + ')';
    }}
  }} else {{
    var xs = [], ys = [];
    for (var j = 0; j <= def.steps; j++) {{
      var xv = def.xMin + (def.xMax - def.xMin) * j / def.steps;
      xs.push(xv); ys.push(def.fn(xv));
    }}
    series.push({{mode: null, xs: xs, ys: ys, color: '#3a6ea8'}});
    xLabel = def.xLabel;
    xMin = def.xMin; xMax = def.xMax;
    summaryText = yLabel + ' vs ' + xLabel + ' (this map\\'s default parameters' +
      (def.summarySuffix ? ',' + def.summarySuffix() : '') + ')';
  }}

  document.getElementById('discSummary').textContent = summaryText;

  var allYs = [];
  series.forEach(function(s) {{ allYs = allYs.concat(s.ys); }});
  // The y-axis follows the DATA, it is not pinned to [0, 1]. It used to be
  // hardcoded `yMax = 1`, which silently truncated any curve that went
  // above 1 (the plotted polyline simply ran off the top of the SVG with no
  // indication). The sub-score formulas are all capped at 1.0 in Python
  // now, so in practice this usually still resolves to 1 -- the point is
  // that the chart no longer *asserts* that cap, so a formula change or an
  // out-of-range parameter (e.g. this stop_score curve, which is NOT
  // individually capped -- it's a geometric mean of sub-scores that are)
  // shows up as a visibly taller curve instead of being hidden. [0, 1] is
  // kept only as a MINIMUM extent so the ordinary curves keep their
  // familiar framing rather than zooming to fill.
  var dataMin = __arrMin(allYs), dataMax = __arrMax(allYs);
  var yMin = Math.min(0, isFinite(dataMin) ? dataMin : 0);
  var yMax = Math.max(1, isFinite(dataMax) ? dataMax * 1.02 : 1);
  var xr = xMax - xMin || 1, yr = yMax - yMin || 1;
  function px(x) {{ return padL + (x - xMin) / xr * (w - padL - padR); }}
  function py(y) {{ return h - padB - (y - yMin) / yr * (h - padT - padB); }}

  var curves = '';
  series.forEach(function(s) {{
    var points = '';
    for (var i = 0; i < s.xs.length; i++) {{
      points += px(s.xs[i]).toFixed(1) + ',' + py(s.ys[i]).toFixed(1) + ' ';
    }}
    curves += '<polyline points="' + points + '" fill="none" stroke="' + s.color + '" stroke-width="2"></polyline>';
  }});

  var xTicks = __niceTicks(xMin, xMax, 4), yTicks = __niceTicks(yMin, yMax, 4);
  var axes = '<line x1="' + padL + '" y1="' + (h - padB) + '" x2="' + (w - padR) + '" y2="' + (h - padB) + '" stroke="#999"></line>' +
    '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (h - padB) + '" stroke="#999"></line>';
  xTicks.forEach(function(t) {{
    var x = px(t);
    axes += '<line x1="' + x + '" y1="' + (h - padB) + '" x2="' + x + '" y2="' + (h - padB + 4) + '" stroke="#999"></line>';
    axes += '<text x="' + x + '" y="' + (h - padB + 15) + '" font-size="9" text-anchor="middle">' + t.toFixed(1) + '</text>';
  }});
  yTicks.forEach(function(t) {{
    var y = py(t);
    axes += '<line x1="' + (padL - 4) + '" y1="' + y + '" x2="' + padL + '" y2="' + y + '"stroke="#999"></line>';
    axes += '<text x="' + (padL - 7) + '" y="' + (y + 3) + '" font-size="9" text-anchor="end">' + t.toFixed(2) + '</text>';
    axes += '<line x1="' + padL + '" y1="' + y + '" x2="' + (w - padR) + '" y2="' + y + '" stroke="#eee"></line>';
  }});
  var labels = '<text x="' + (padL + (w - padL - padR) / 2) + '" y="' + (h - 4) + '" font-size="10" text-anchor="middle" fill="#444">' + xLabel + '</text>' +
    '<text x="10" y="' + (padT + (h - padT - padB) / 2) + '" font-size="10" text-anchor="middle" fill="#444" transform="rotate(-90 10 ' + (padT + (h - padT - padB) / 2) + ')">' + yLabel + '</text>';

  var legend = '';
  if (isMultiCurve) {{
    var lx = padL + 6, ly = padT + 12;
    series.forEach(function(s, idx) {{
      var yy = ly + idx * 14;
      legend += '<line x1="' + lx + '" y1="' + yy + '" x2="' + (lx + 16) + '" y2="' + yy + '" stroke="' + s.color + '" stroke-width="2"></line>' +
        '<text x="' + (lx + 20) + '" y="' + (yy + 3) + '" font-size="9" fill="#333">' + s.mode + '</text>';
    }});
  }}

  var svg = '<svg width="' + w + '" height="' + h + '" style="background:#fafafa;border:1px solid #ddd;">' +
    axes + curves + labels + legend +
    '</svg>';
  document.getElementById('discPlot').innerHTML = svg;
}}

// (Re)build ONE "Compare with scenario" dropdown's option list from the live
// editor state every time it might have changed -- 'none' (item 4's fix,
// meaning "no secondary series") always first/default, then the
// reserved/baseline entry, then every user scenario that has been Computed
// at least once (either it's the editor's currently active scenario and
// `lastCompute()` succeeded, or it has an entry in the per-scenario cache
// from a PRIOR Compute while it was active). The previously selected value
// is kept if it's still valid; otherwise falls back to 'none' -- selecting a
// scenario in one tab never silently carries over as a selection in another
// tab's own dropdown, since each is a fully independent <select>/element id.
function __populateSecondaryScenarioSelect(selId) {{
  var sel = document.getElementById(selId);
  if (!sel) return;
  var ed = window.__editor;
  var prev = sel.value;
  var reserved = __statsReservedName();
  var options = [
    {{value: 'none', label: 'None'}},
    {{value: '', label: 'Current network (baseline)'}}
  ];
  if (ed && ed.state && ed.state.scenarios) {{
    var active = ed.state.activeScenario;
    var lc = ed.lastCompute && ed.lastCompute();
    ed.state.scenarios.forEach(function(sc) {{
      if (!sc || !sc.name || sc.name === reserved) return;
      var computed = (sc.name === active && lc && lc.ok) ||
        (ed.computeResultForScenario && ed.computeResultForScenario(sc.name));
      if (computed) options.push({{value: sc.name, label: sc.name}});
    }});
  }}
  sel.innerHTML = options.map(function(o) {{
    return '<option value="' + o.value + '">' + o.label + '</option>';
  }}).join('');
  var stillValid = options.some(function(o) {{ return o.value === prev; }});
  sel.value = stillValid ? prev : 'none';
}}

function __populateAllSecondaryScenarioSelects() {{
  __populateSecondaryScenarioSelect('distSecondaryScenarioSelect');
  __populateSecondaryScenarioSelect('regSecondaryScenarioSelect');
  __populateSecondaryScenarioSelect('anovaSecondaryScenarioSelect');
  __populatePlaceSelects();
}}

// Fill every "Compare with" popover's "Place" <select> (`.comparePlaceSelect`,
// only rendered at all when `enable_place_comparison=True`) from
// `CS_PLACE_MANIFEST` -- same roster the Place-rank tab uses, minus this map's
// own place (comparing a place against itself is meaningless). Options
// beyond "This place only" are populated here rather than baked in at build
// time (`_compare_with_control_html`) because the manifest itself only
// exists client-side. Runs once on load and again wherever
// `__populateAllSecondaryScenarioSelects` already runs, matching the
// scenario select's own refresh pattern.
function __populatePlaceSelects() {{
  if (!window.__enablePlaceComparison) return;
  document.querySelectorAll('.comparePlaceSelect').forEach(function(sel) {{
    var prev = sel.value;
    var opts = '<option value="" selected>This place only</option>' +
      CS_PLACE_MANIFEST.map(function(c) {{ return '<option value="' + c.key + '">' + c.label + '</option>'; }}).join('');
    sel.innerHTML = opts;
    sel.value = prev || '';
  }});
}}
// This whole block runs inside `_stats_panel_js`'s own `window.addEventListener('load', ...)`
// closure, so a bare reference to this function from ANOTHER injected <script>'s closure
// (e.g. the scenario editor's `_maplibre_compute_access_js`, which calls this after every
// save/load-scenario so the stats panel's "Compare with scenario" dropdowns stay in sync)
// would throw a ReferenceError, not just silently no-op -- publish it on `window` like the
// other cross-script test/automation hooks in this file.
window.__populateAllSecondaryScenarioSelects = __populateAllSecondaryScenarioSelects;

// --- Place rank tab: population-weighted median level_of_service across every
// CS_transitLOS city, read from each city's OWN `results/summary.json`
// (written by `code.pipeline.write_city_summary` in CS_transitLOS/code/).
// This map.html is a standalone static file with no shared JS module
// system, so the manifest below is a plain copy of the CS city key list
// (kept in sync with CS_transitLOS/code/combined_map.py's CITY_MANIFEST
// by hand -- there is no cross-file import to share it through). It
// deliberately does NOT include `boston_city`
// (TransitLOSStudies/boston_city/), which is a separate standalone test
// study OUTSIDE CS_transitLOS/, not a "CS city".
//
// Every CS city lives as a sibling directory under CS_transitLOS/, so from
// ANY city's own `map.html` (itself at `CS_transitLOS/<key>/map.html`),
// `../<key>/results/summary.json` resolves correctly for every city in the
// manifest INCLUDING the current one (`../<own_key>/...` walks back up to
// the shared parent and into the same directory) -- one uniform path
// pattern, no special-casing "self" vs "other".
//
// Fetch only works when served over http(s) (fetch()/CORS blocks local
// `file://` reads); a city that 404s, isn't reachable, or hasn't been
// processed yet (no summary.json written) is shown as unranked/absent
// rather than erroring, same as the earlier combined-map version's
// handling of not-yet-processed cities.
var CS_PLACE_MANIFEST = {place_manifest_json};
// Item 4 (verbatim user request): the old cross-place "Compare with place"
// selects (per-tab, `.comparePlaceSelect`) are gone -- this ranking tab is
// now the ONE place cross-city comparison lives. Rather than the old fixed
// population-weighted median read straight off each city's small
// `results/summary.json` (computed once at build time, weight column fixed
// forever), this fetches each city's full `results/stats_data.json` (same
// per-h3-cell payload `__placeCompareData` already fetches for other
// purposes) so the median can be recomputed CLIENT-SIDE against whichever
// weight column the user picks in `#placeRankWeightSelect` -- "Unweighted"
// runs `__weightedMedian` with all-1 weights (a plain median), any other
// column runs it with that column's real per-cell values, exactly the
// weighting `code.stats.weighted_median` uses server-side for
// `metro_median_access`/`core_median_access` (verified equivalent -- see
// this round's audit note below).
var __placeRankFetching = {{}}; // key -> true while in flight
function __fetchAllPlaceRankData() {{
  if (!window.__enablePlaceComparison) return;
  CS_PLACE_MANIFEST.forEach(function(c) {{
    if (c.key in __placeCompareCache) return;
    if (__placeRankFetching[c.key]) return;
    __placeRankFetching[c.key] = true;
    fetch({json.dumps(place_rank_fetch_prefix)} + c.key + '/results/stats_data.json')
      .then(function(resp) {{ return resp.ok ? resp.json() : null; }})
      .catch(function() {{ return null; }})
      .then(function(json) {{
        __placeCompareCache[c.key] = json || false;
        __placeRankFetching[c.key] = false;
        if (window.__statsTab === 'placerank') renderPlaceRank();
      }});
  }});
}}
function renderPlaceRank() {{
  var el = document.getElementById('placeRankList');
  if (!el) return;
  __fetchAllPlaceRankData();
  var area = window.__statsArea === 'core' ? 'core' : 'metro';
  var weightSel = document.getElementById('placeRankWeightSelect');
  var weightCol = weightSel ? weightSel.value : 'none';
  var results = CS_PLACE_MANIFEST.map(function(c) {{
    var cached = __placeCompareCache[c.key];
    if (cached === undefined) return {{key: c.key, label: c.label, value: null, loading: true}};
    if (!cached) return {{key: c.key, label: c.label, value: null, loading: false}};
    var areaData = cached[area] || cached.metro;
    if (!areaData || !areaData.level_of_service) return {{key: c.key, label: c.label, value: null, loading: false}};
    // A place lacking the selected weight column entirely is EXCLUDED from
    // the ranking (shown as unranked "--"), not silently treated as
    // unweighted or given a 0 -- e.g. guadalajara has no census/jobs data,
    // so weighting by a census column would otherwise misrepresent it as
    // tied-for-last rather than genuinely lacking that data.
    if (weightCol !== 'none' && !areaData[weightCol]) return {{key: c.key, label: c.label, value: null, loading: false}};
    var score = areaData.level_of_service;
    var weights = (weightCol !== 'none') ? areaData[weightCol] : score.map(function() {{ return 1; }});
    var value = __weightedMedian(score, weights);
    return {{key: c.key, label: c.label, value: isNaN(value) ? null : value, loading: false}};
  }});
  var ranked = results.filter(function(c) {{ return c.value !== null; }})
    .slice().sort(function(a, b) {{ return b.value - a.value; }});
  var pending = results.filter(function(c) {{ return c.value === null && c.loading; }});
  var unranked = results.filter(function(c) {{ return c.value === null && !c.loading; }});
  var rows = '';
  for (var i = 0; i < ranked.length; i++) {{
    var c = ranked[i];
    rows += '<div style="display:flex;justify-content:space-between;padding:3px 0;' +
      (i % 2 === 1 ? 'background:#f7f9fc;' : '') + '">' +
      '<span><span style="color:#888;width:18px;display:inline-block;">' + (i + 1) + '</span>' + c.label + '</span>' +
      '<span style="font-weight:600;">' + c.value.toFixed(2) + '</span></div>';
  }}
  for (var p = 0; p < pending.length; p++) {{
    rows += '<div style="display:flex;justify-content:space-between;padding:3px 0;color:#aaa;">' +
      '<span><span style="width:18px;display:inline-block;">--</span>' + pending[p].label + '</span>' +
      '<span>loading...</span></div>';
  }}
  for (var k = 0; k < unranked.length; k++) {{
    var u = unranked[k];
    rows += '<div style="display:flex;justify-content:space-between;padding:3px 0;color:#aaa;">' +
      '<span><span style="width:18px;display:inline-block;">--</span>' + u.label + '</span>' +
      '<span>--</span></div>';
  }}
  el.innerHTML = rows;
  // Optional hook (2026-09-02, fixes a real user-reported inconsistency:
  // "the numbers and colors of the global map do not coincide with place
  // rank population weight"): `code.combined_map`'s overview-map circles
  // used to show `summary.json`'s `metro_median_access` -- baked in at the
  // last FULL pipeline run for that city, which can be well out of date
  // (e.g. after a `--map-only` rebuild that only refreshes
  // `stats_data.json`, not `summary.json` -- see
  // `code.pipeline.write_city_summary`'s own call site). This tab's
  // `results` (just computed above, from the SAME live `stats_data.json`
  // fetch this tab already does) is the single source of truth for "this
  // city's population-weighted median access" -- handing it to an external
  // hook lets a caller with its own per-city UI (marker colors/labels)
  // stay in sync with EXACTLY what this tab shows, rather than
  // maintaining a second, driftable computation. Only meaningful for the
  // real population weighting (`weightCol === 'population'`), since that's
  // the one fixed quantity an external caller like a world map wants --
  // an absent hook (a real per-city map.html never defines one) is a no-op.
  if (weightCol === 'population' && window.__onPlaceRankComputed) window.__onPlaceRankComputed(results);
}}
document.addEventListener('change', function(e) {{
  if (e.target && e.target.id === 'placeRankWeightSelect') renderPlaceRank();
}});
// Test/automation hook, same publication pattern as window.__getStatsData.
window.__renderPlaceRank = renderPlaceRank;

function __renderActiveTab() {{
  if (window.__statsTab === 'placerank') {{ renderPlaceRank(); return; }}
  // Metadata is normally a static table baked in at build time (see
  // `_metadata_tab_html` in build.py) -- nothing to compute/re-render here
  // for a real per-city map, it just needs to not fall through to
  // `renderAnova()` below. `window.__renderMetadataTab` (2026-09-02) is an
  // optional hook for a caller with no per-h3-cell GeoDataFrame at PYTHON
  // build time -- `code.combined_map`'s wrapper page, which instead
  // recomputes the table live from `window.__statsData` in JS, the same
  // pattern `__setRegressionFields`/`__rebuildStatsFieldSelectors` already
  // use for that page's Distribution/Regression/ANOVA field lists. Absent
  // (a real per-city map.html never defines it) is a no-op, same as before.
  if (window.__statsTab === 'metadata') {{
    if (window.__renderMetadataTab) window.__renderMetadataTab();
    return;
  }}
  __populateAllSecondaryScenarioSelects();
  if (window.__statsTab === 'distribution') renderDistribution();
  else if (window.__statsTab === 'regression') renderRegression();
  else if (window.__statsTab === 'discretization') renderDiscretization();
  else renderAnova();
}}

document.getElementById('statsButton').addEventListener('click', function() {{
  var panel = document.getElementById('statsPanel');
  var opening = panel.style.display === 'none';
  panel.style.display = opening ? 'block' : 'none';
  if (opening) __renderActiveTab();
}});

document.querySelectorAll('.statsTabBtn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    window.__statsTab = btn.dataset.tab;
    document.querySelectorAll('.statsTabBtn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    document.querySelectorAll('.statsTabPanel').forEach(function(p) {{ p.style.display = 'none'; }});
    document.getElementById('statsTab_' + window.__statsTab).style.display = 'block';
    // The Metro area/City core toggle only means anything for tabs backed
    // by live per-area data (distribution/regression/ANOVA) -- the
    // discretization tab plots a pure formula shape from this map's fixed
    // build-time parameters, unaffected by area, so showing an area toggle
    // there implied a non-existent effect.
    document.getElementById('statsAreaBtnRow').style.display =
      (window.__statsTab === 'discretization' || window.__statsTab === 'metadata') ? 'none' : 'flex';
    __renderActiveTab();
  }});
}});

document.querySelectorAll('.statsAreaBtn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    window.__statsArea = btn.dataset.area;
    document.querySelectorAll('.statsAreaBtn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    __renderActiveTab();
  }});
}});

document.querySelectorAll('.statsParamBtn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    window.__discParam = btn.dataset.param;
    document.querySelectorAll('.statsParamBtn').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    renderDiscretization();
  }});
}});

// One pair of handlers for BOTH axis-toggle curves (stop_score, access).
Object.keys(__discAxisGroups).forEach(function(key) {{
  var g = __discAxisGroups[key];
  document.getElementById(g.selectId).addEventListener('change', function(e) {{
    g.axis = e.target.value;
    // Switching axes swaps which variable is free vs fixed -- reset the
    // fixed-slider value to the new axis's own default rather than keeping a
    // now-differently-scaled number (e.g. a headway value of 45 makes no
    // sense as a leftover "speed" slider position after switching).
    g.other = __discAxisDefault(g.defs[g.axis]);
    renderDiscretization();
  }});
  document.getElementById(g.sliderId).addEventListener('input', function(e) {{
    var axisDef = g.defs[g.axis];
    var raw = parseFloat(e.target.value);
    // The slider's own value is a RAW track position when the axis declares
    // a non-linear mapping (see `__speedSliderRaw`), not the km/h the
    // formula takes -- so it always goes through `toValue` here.
    g.other = axisDef.sliderRaw ? axisDef.sliderRaw.toValue(raw) : raw;
    document.getElementById(g.valueId).textContent = g.other.toFixed(1) + (axisDef.otherUnit || '');
    renderDiscretization();
  }});
}});

document.getElementById('distOverlaySelect').addEventListener('change', renderDistribution);
document.getElementById('distMainSelect').addEventListener('change', renderDistribution);
document.getElementById('distSecondaryScenarioSelect').addEventListener('change', renderDistribution);
document.getElementById('regFieldSelect').addEventListener('change', renderRegression);
document.getElementById('regSecondaryScenarioSelect').addEventListener('change', renderRegression);
document.getElementById('anovaSecondaryScenarioSelect').addEventListener('change', renderAnova);

document.getElementById('regFieldSelect').value = {json.dumps(default_field)};

// --- "Compare with" popover wiring (item 1 redesign, 2026-08-19): each tab's
// popover button toggles ONLY its own popover (closing any other open one,
// so at most one is visible at a time); any change inside an OPEN popover's
// selects re-renders that tab exactly like the old flat selects did (the
// select element ids are unchanged, so this is on TOP of the listeners
// above, not a replacement for them -- the new area/place selects need their
// own listeners since they didn't exist before).
document.querySelectorAll('.compareWithBtn').forEach(function(btn) {{
  btn.addEventListener('click', function(e) {{
    e.stopPropagation();
    var targetId = btn.getAttribute('data-target');
    document.querySelectorAll('.compareWithPopover').forEach(function(p) {{
      p.style.display = (p.id === targetId && p.style.display === 'none') ? 'block' : 'none';
    }});
  }});
}});
document.addEventListener('click', function() {{
  document.querySelectorAll('.compareWithPopover').forEach(function(p) {{ p.style.display = 'none'; }});
}});
document.querySelectorAll('.compareWithPopover').forEach(function(p) {{
  p.addEventListener('click', function(e) {{ e.stopPropagation(); }});
}});
var __compareAreaHandlers = {{
  distCompareAreaSelect: renderDistribution,
  regCompareAreaSelect: renderRegression,
  anovaCompareAreaSelect: renderAnova
}};
var __comparePlaceHandlers = {{
  distComparePlaceSelect: renderDistribution,
  regComparePlaceSelect: renderRegression,
  anovaComparePlaceSelect: renderAnova
}};
Object.keys(__compareAreaHandlers).forEach(function(id) {{
  var el = document.getElementById(id);
  if (el) el.addEventListener('change', __compareAreaHandlers[id]);
}});
Object.keys(__comparePlaceHandlers).forEach(function(id) {{
  var el = document.getElementById(id);
  if (el) el.addEventListener('change', __comparePlaceHandlers[id]);
}});

// PHASE 4 (scenario editor): publish the formula mirrors + the weighted-mean
// helper as `window` properties. Everything in this file is emitted INSIDE a
// `window.addEventListener('load', function() {{ ... }})` closure, so these
// function declarations are local to it and are NOT reachable from the
// editor's own separate `load` closure (the same cross-script scoping trap
// `__fmtLegendNum` documents). The editor's access recompute reuses these
// exact functions rather than defining a second copy of the same formulas.
window.__jsSpeedScore = __jsSpeedScore;
window.__jsFrequencyScore = __jsFrequencyScore;
window.__jsReliabilityScore = __jsReliabilityScore;
window.__jsWalkDecay = __jsWalkDecay;
window.__jsStopScore = __jsStopScore;
window.__jsModeScores = __MODE_SCORES;
window.__jsWeightedMean = __weightedMean;
// Test/automation hook -- getStatsData is otherwise local to this closure
// (see the comment above), the same reason __jsStopScore etc. are published.
window.__getStatsData = getStatsData;
// Item 5: the hexagon/circle style functions (`_H3_OVERRIDE_LOOKUP_JS`, a
// different injected <script>) reuse these exact H3-parent helpers to roll
// the editor's per-cell overrides up to whatever resolution is currently on
// screen -- same cross-script `window` publication pattern as the rest of
// this block.
window.__h3ToParent = __h3ToParent;
window.__h3Resolution = __h3Resolution;
"""


def _stats_panel_helper_js() -> str:
    """Helper globals `_stats_panel_js` needs but doesn't define itself.

    Folium pages get these for free from OTHER Folium-only injected scripts
    (`_shape_popup_js()`'s `__fieldUnitSuffix`/`__isPercentField` for the
    click-popup table, `_color_interp_js()`'s `rdylgn` color ramp used by the
    distribution tab's bin coloring, `_control_panel_js`'s
    `window.__fmtLegendNum` number formatter used by the regression plot's
    axis ticks and popup values) -- a page that embeds `_stats_panel_html`/
    `_stats_panel_js` WITHOUT also including those other scripts (MapLibre's
    own page, via `_inject_maplibre_stats_panel_into_saved_html`; or
    `CS_transitLOS/code/combined_map.py`, which has none of this module's
    other injected scripts at all) needs this block too, or the stats panel
    throws "X is not defined" partway through rendering a tab (found live
    via headless-browser console errors, both when porting to MapLibre and
    again in `combined_map.py`, in each case as `__fieldLabel is not
    defined` since that's the first of these helpers `renderAnova`/
    `renderRegression`/`renderDistribution` actually call). Defined here
    verbatim (`rdylgn` reuses `_color_interp_js()` directly;
    `__fmtLegendNum`/`__fieldUnitSuffix`/`__isPercentField`/`__fieldLabel`
    copied from their Folium-only sources) rather than pulling in the whole
    popup/control-panel JS, to keep this scoped to just what the stats panel
    actually needs. Idempotent to include twice (plain `function`/`var`
    declarations, no side effects) -- a caller that already has these
    (nothing in this module currently does, but a future one might) is safe
    either way.
    """
    return (
        "<script>\n"
        + _color_interp_js() + "\n"
        # Kept in lockstep with `_control_panel_js`'s own `window.__fmtLegendNum`
        # (see that copy's 2026-09-06 comment on the `refAbs` k/M-suffix rule)
        # -- this string-literal copy exists only for pages with no circle-size
        # legend at all (see this function's own docstring), so it never needs
        # a `refAbs` argument, but keeps the same magnitude-scaling behavior.
        + "window.__fmtLegendNum = function(v, refAbs) {\n"
        "  if (v == null || isNaN(v)) return '';\n"
        "  var abs = Math.abs(v);\n"
        "  if (abs === 0) return '0';\n"
        "  var ref = (refAbs != null && isFinite(refAbs)) ? Math.abs(refAbs) : abs;\n"
        "  var noDecimals = ref >= 100;\n"
        "  var scale = 1, suffix = '';\n"
        "  if (abs >= 1e6) { scale = 1e6; suffix = 'M'; }\n"
        "  else if (abs >= 1e3) { scale = 1e3; suffix = 'k'; }\n"
        "  if (scale > 1) {\n"
        "    var scaled = v / scale;\n"
        "    if (noDecimals) {\n"
        "      var rounded = Math.round(scaled);\n"
        "      if (suffix === 'k' && Math.abs(rounded) >= 1000) return Math.round(v / 1e6) + 'M';\n"
        "      return rounded.toLocaleString() + suffix;\n"
        "    }\n"
        "    var scaledAbs = Math.abs(scaled);\n"
        "    var dec = scaledAbs >= 100 ? 0 : (scaledAbs >= 10 ? 1 : 2);\n"
        "    var s2 = scaled.toFixed(dec);\n"
        "    if (dec > 0) s2 = s2.replace(/0+$/, '').replace(/\\.$/, '');\n"
        "    return s2 + suffix;\n"
        "  }\n"
        "  if (noDecimals) return Math.round(v).toLocaleString();\n"
        "  if (abs >= 1) {\n"
        "    var rounded1 = Math.round(v * 10) / 10;\n"
        "    if (Math.abs(rounded1 - Math.round(rounded1)) < 1e-9) return Math.round(rounded1).toLocaleString();\n"
        "    return rounded1.toFixed(1);\n"
        "  }\n"
        "  var decimals = Math.min(2, Math.max(1, 2 - Math.floor(Math.log10(abs))));\n"
        "  var s = v.toFixed(decimals);\n"
        "  s = s.replace(/0+$/, '').replace(/\\.$/, '');\n"
        "  return s === '' || s === '-' ? '0' : s;\n"
        "};\n"
        "function __isPercentField(key) {\n"
        "  var k = String(key || '').toLowerCase();\n"
        "  return /(share|rate|ratio|pct|percent)$/.test(k);\n"
        "}\n"
        "function __fieldUnitSuffix(key) {\n"
        "  var k = String(key || '').toLowerCase();\n"
        "  if (k.indexOf('density') >= 0) return ' (pop/km\\u00b2)';\n"
        "  if (__isPercentField(key)) return ' (%)';\n"
        "  if (k.indexOf('kmh') >= 0 || k.indexOf('km_h') >= 0) return ' (km/h)';\n"
        "  if (k.indexOf('minutes') >= 0) return ' (min)';\n"
        "  return '';\n"
        "}\n"
        # Same cross-country display-name mirror as `_shape_popup_js`'s copy
        # (see that function's comment) -- the stats panel's regression/
        # ANOVA/distribution axis labels (`renderRegression`/`renderAnova`/
        # `renderDistribution` in `_stats_panel_js`) read raw `inegi_*`/
        # `acs5_*`/etc. column keys the same way the popup table does.
        "var __SOURCE_PREFIXES = ['acs5_','acs3_','acs1_','dhc_','lodes_wac_','lodes_rac_',\n"
        "  'inegi_','ine_cl_','ine_','cbs_','eustat_','statcan_','moi_','estadisticaad_'];\n"
        "function __stripSourcePrefix(key) {\n"
        "  var k = String(key || '');\n"
        "  for (var i = 0; i < __SOURCE_PREFIXES.length; i++) {\n"
        "    if (k.indexOf(__SOURCE_PREFIXES[i]) === 0) return k.slice(__SOURCE_PREFIXES[i].length);\n"
        "  }\n"
        "  return k;\n"
        "}\n"
        "function __humanizeFieldName(name) {\n"
        # Mirrors `_FIELD_LABEL_OVERRIDES` (Python side, see
        # `_humanize_field_name`'s docstring) -- this JS copy builds
        # client-side labels (regression/ANOVA/distribution axis labels,
        # popup rows) and was missing the override table entirely, so a
        # field like `inegi_educationMeanGrade` humanized generically to the
        # confusing "Education mean grade" here even though the
        # Python-rendered dropdowns already showed the clearer override.
        # Keyed on the already-de-prefixed name, same as the Python side.
        "  var __FIELD_LABEL_OVERRIDES = {'educationMeanGrade': 'Average years of schooling'};\n"
        "  if (Object.prototype.hasOwnProperty.call(__FIELD_LABEL_OVERRIDES, name)) return __FIELD_LABEL_OVERRIDES[name];\n"
        "  var words = (String(name).replace(/_/g, ' ').match(/[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\\d+/g)) || [];\n"
        "  if (!words.length) return name;\n"
        "  var last = words[words.length - 1].toLowerCase();\n"
        "  if (words.length > 1 && last === 'households') {\n"
        "    return 'Households with ' + words.slice(0, -1).map(function(w) { return w.toLowerCase(); }).join(' ');\n"
        "  }\n"
        "  if (words.length > 1 && last === 'population') {\n"
        "    var head = words.slice(0, -1).map(function(w) { return w.toLowerCase(); }).join(' ');\n"
        "    return head.charAt(0).toUpperCase() + head.slice(1) + ' population';\n"
        "  }\n"
        "  var joined = words.map(function(w) { return w.toLowerCase(); }).join(' ');\n"
        "  return joined.charAt(0).toUpperCase() + joined.slice(1);\n"
        "}\n"
        "function __fieldLabel(key) {\n"
        "  return __humanizeFieldName(__stripSourcePrefix(key)) + __fieldUnitSuffix(key);\n"
        "}\n"
        "</script>\n"
    )


def _stats_panel_block(
    stats_by_area: Dict[str, gpd.GeoDataFrame],
    is_us: bool,
    region: str = "global",
    enable_place_comparison: bool = True,
    share_source_map: Optional[Dict[str, str]] = None,
    count_fields: Optional[List[str]] = None,
    share_fields: Optional[List[str]] = None,
) -> str:
    """Assemble the stats-panel data script + HTML + wiring JS for injection into the saved map.

    `count_fields`/`share_fields`, when given, are the SAME canonical lists
    `build_city_map` already computed for the circle-size/opacity-by
    dropdowns (`circle_fields`/`opacity_fields`) -- passing them through
    here (rather than letting this function re-derive its own from
    `stats_by_area`'s own grid) is what keeps `distMainSelect`/`regFieldSelect`
    showing the EXACT same options as `circleFieldSelect`/`opacitySelect`,
    per explicit user request (2026-09-07). `None` (the default, used by
    `build_place_rank_panel_block`'s caller, which has no per-city grid of
    its own) falls back to deriving them independently from `stats_by_area`,
    same as before.
    """
    finest = next(iter(stats_by_area.values()))
    all_fields = _stats_field_candidates(finest)
    # Regression and ANOVA correlate a variable against level_of_service, which
    # only makes sense for intensive (relative) variables -- see
    # `is_relative_field`. The distribution tab's overlay bar sums its
    # variable per score bin, so it takes the exact complement.
    regression_fields = list(share_fields) if share_fields is not None else relative_fields(all_fields)
    if not regression_fields:
        regression_fields = ["pop_density"] if "pop_density" in finest.columns else []
    # Item 1 fix: this used to be `[] ` for non-US cities, which left the
    # distribution tab's main-column dropdown (`distMainSelect`) with ZERO
    # <option>s there -- its .value then read as "" and `data[""]` was
    # undefined, so the whole tab silently rendered empty. Non-US cities
    # have no census join at all, but they DO have a WorldPop-sourced
    # `population`/`pop_density` pair from the pipeline (see
    # `code/pipeline.py`), and `absolute_fields` naturally narrows to just
    # that (pop_density is relative, so it's excluded here) when no ACS
    # columns are present -- so this no longer needs to be US-gated.
    count_fields = list(count_fields) if count_fields is not None else _stats_count_fields(all_fields)
    data_json = _stats_json_data(stats_by_area, list(dict.fromkeys(regression_fields + count_fields)))
    # Item 1 (user feedback, 2026-08-19): the "Compare with" popover's area
    # (city core / metro) sub-dimension is only offered when this map
    # genuinely differentiates between them -- i.e. `stats_by_area` was built
    # with more than one key (boston_city passes `{"metro": ..., "core": ...}`,
    # so it gets the dimension; a single-area map wouldn't).
    has_multi_area = len(stats_by_area) > 1
    metadata_rows = _column_metadata_rows(finest, share_source_map=share_source_map)

    return (
        f"<script>\nwindow.__statsData = {data_json};\n</script>\n"
        + _stats_panel_html(regression_fields, count_fields, enable_place_comparison, has_multi_area, metadata_rows=metadata_rows)
        + "<script>\nwindow.addEventListener('load', function() {\n"
        + _stats_panel_js(regression_fields, is_us, region, enable_place_comparison, has_multi_area)
        + "\n});\n</script>\n"
    )


def build_place_rank_panel_block(
    place_manifest: List[Dict[str, str]],
    weight_columns: Sequence[str] = (),
    fetch_prefix: str = "",
) -> str:
    """The bottom-left 📊 stats popup, reduced to ONLY its "Place rank" tab.

    Public entry point for a caller with no per-place `stats_by_area`
    GeoDataFrame of its own -- concretely, `CS_transitLOS/code/combined_map.py`
    (a lightweight page generator with no `geopandas`/per-h3-cell data,
    assembling a page that iframes *other* already-built city `map.html`s
    rather than owning any map data itself). It reuses the exact same
    `_stats_panel_html`/`_stats_panel_js` a real per-city `build_city_map`
    call uses for its own "Place rank" tab -- one implementation, two
    call sites -- with the other four tabs (Distribution/Regression/ANOVA/
    Discretization) omitted entirely via `place_rank_only=True`, since this
    caller has no per-place h3 data to feed them.

    `place_manifest`: `[{"key": ..., "label": ...}, ...]` -- the live roster
    (e.g. from `combined_map.py`'s own `_available_cities()`), NOT this
    module's hand-maintained `_DEFAULT_CS_PLACE_MANIFEST` fallback.
    `weight_columns`: candidate weight-column keys for the tab's own
    weight-column <select> (e.g. `combined_map.py`'s `WEIGHT_COLUMN_CANDIDATES`
    keys) -- rendered through this module's own `field_label` so option text
    matches every other count-column label a real map's stats panel shows.
    `fetch_prefix`: prepended to each `fetch(...)` call's relative
    `results/stats_data.json` path -- `""` for a caller living at
    `CS_transitLOS/` itself (one level up from any per-city `map.html`,
    which uses this function's implicit `"../"` default via a direct
    `build_city_map` call instead of this one).
    """
    weight_columns = list(weight_columns)
    return (
        _stats_panel_html([], weight_columns, enable_place_comparison=True, has_multi_area=True, place_rank_only=True)
        + "<script>\nwindow.addEventListener('load', function() {\n"
        + _stats_panel_js(
            [], False, "global",
            enable_place_comparison=True, has_multi_area=True,
            place_rank_only=True, place_rank_fetch_prefix=fetch_prefix,
            place_manifest=place_manifest,
        )
        + "\n});\n</script>\n"
    )


# --- Scenario / route editor ("edit mode") ----------------------------------
#
# All 4 phases of this feature are now built. What lives here:
#   * a top-right "Edit" toggle button,
#   * entering/exiting edit mode (streets off + active shape opacity halved,
#     both restored on exit),
#   * scenario create/pick UI (with the reserved, never-editable "current"
#     scenario from `default_edit_params.json`),
#   * a route list with New/Edit/Delete for USER-CREATED routes only,
#   * a route metadata form (name / color / headway / mode),
#   * geometry editing on top of Leaflet-Geoman: draw, move node, add node,
#     delete node, extend-from-an-endpoint-only, add/delete station,
#   * an in-memory data model (`window.__editorState`) + JSON export/import,
#   * (Phase 3) per-segment grade separation + speed (with manual override)
#     and the live per-route cost, all driven from `default_edit_params.json`,
#   * (Phase 4) the purely client-side access-score recompute -- straight-line
#     buffers over the H3 res-11 population grid delivered as small on-demand
#     static chunk files (`transitlos.map.pop_chunks`) -- and the always-on
#     top-center summary bar (`#editSummaryBar`: scenario name / level of service
#     / cost / impact / impact-per-$M + the impact_score emoji).
#
# The Phase 4 recompute is deliberately a SIMPLIFICATION of the real pipeline,
# per explicit instruction: straight-line ("as the crow flies") distance from a
# res-11 cell centroid to the nearest new/edited stop, not a street-network
# walk. Everything else -- `stop_score`, `walk_decay`, the
# `accessibility_score = stop_score * D(t)` product, the walking speed and the
# 2 km outer cutoff -- mirrors `transitlos.scoring` exactly, by reusing the
# Discretization tab's own `__jsStopScore`/`__jsWalkDecay` mirrors rather than
# writing a second copy of those formulas.

EDIT_PARAMS_FILENAME = "default_edit_params.json"

# Pinned, NOT `@latest`: an unpinned CDN URL silently changes the drawing
# library under an already-published map. 2.20.0 is the current release
# (verified against the npm registry when this feature was built).
GEOMAN_VERSION = "2.20.0"
GEOMAN_JS = f"https://unpkg.com/@geoman-io/leaflet-geoman-free@{GEOMAN_VERSION}/dist/leaflet-geoman.min.js"
GEOMAN_CSS = f"https://unpkg.com/@geoman-io/leaflet-geoman-free@{GEOMAN_VERSION}/dist/leaflet-geoman.css"

# Fallback used only if `default_edit_params.json` can't be found -- keeps the
# editor functional (modes + reserved scenario name are all the Phase 2 UI
# actually reads) rather than throwing at map-build time.
_EDIT_PARAMS_FALLBACK = {
    "modes": {
        "rail": {"label": "Rail", "icon": "fa-train", "default_color": "#DA291C"},
        "tram": {"label": "Tram", "icon": "fa-train-tram", "default_color": "#00843D"},
        "brt": {"label": "BRT", "icon": "fa-bus-simple", "default_color": "#0072CE", "scores_as": "tram"},
        "bus": {"label": "Bus", "icon": "fa-bus-simple", "default_color": "#8b0000"},
    },
    "scenarios": {"reserved_current_name": "current"},
}


def load_edit_params(path: Optional[str] = None) -> dict:
    """Read `default_edit_params.json` (the editor's config) as a dict.

    Looked up relative to this file (``<repo>/transitLOS/default_edit_params.json``)
    unless an explicit path is given, so a study can pass its own copy in
    without touching code.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        # build.py -> map/ -> transitlos/ -> src/ -> transitLOS/
        candidate = os.path.join(here, "..", "..", "..", EDIT_PARAMS_FILENAME)
        path = os.path.normpath(candidate)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(_EDIT_PARAMS_FALLBACK)


def _editor_html(params: dict) -> str:
    """Static HTML/CSS for the edit button, the editor panel and the scenario modal."""
    modes = params.get("modes", {})
    mode_options = "".join(
        f'<option value="{k}">{v.get("label", k)}</option>' for k, v in modes.items()
    )
    return """
<style>
  /* Item 11: grouped with #statsButton (same bottom-left corner stack,
     same circular look) rather than living alone in the top-right.
     Stacked ABOVE #statsButton (same `left`, `bottom` increased by one
     button-height-plus-gap -- the 44px spacing #statsButton itself uses
     against the scale bar below it) rather than beside it, per live-
     testing feedback that a horizontal layout here felt cramped. */
  #editModeButton {
    position:absolute; bottom:100px; left:10px; z-index:1000;
    background:white; width:34px; height:34px; border-radius:50%; cursor:pointer;
    display:flex; align-items:center; justify-content:center; font-size:16px;
    box-shadow:0 1px 4px rgba(0,0,0,0.3); transition: transform 0.12s, box-shadow 0.12s;
  }
  #editModeButton:hover { transform: scale(1.08); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
  #editModeButton.active { background:#2b6cb0; color:white; }
  #editorPanel {
    display:none; position:absolute; top:10px; left:60px; z-index:1200; background:white;
    padding:8px 10px; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,0.35);
    font:11.5px sans-serif; width:262px; max-height:82vh; overflow-y:auto; box-sizing:border-box;
  }
  #editorPanel h4 { margin:0 0 6px; font-size:12.5px; color:#222; }
  #editorPanel .divider { margin:7px 0; border-top:1px solid #e5e8eb; }
  #editorPanel button {
    font:11px sans-serif; padding:3px 7px; margin:0 3px 3px 0; border:1px solid #c8cdd3;
    background:#f7f8fa; border-radius:4px; cursor:pointer; color:#222;
  }
  #editorPanel button:hover { background:#eceff3; }
  #editorPanel button.on { background:#2b6cb0; border-color:#2b6cb0; color:white; }
  #editorPanel button.primary { background:#2b6cb0; border-color:#2b6cb0; color:white; }
  /* Item 12: Compute is the single most important action in the panel --
     bigger, bolder, and visually separated from the other buttons. */
  #editorCompute {
    font-size:13px; font-weight:700; padding:6px 14px; margin:0 0 6px 0;
    display:block; width:100%; box-shadow:0 1px 4px rgba(43,108,176,0.5);
  }
  #editorTopButtonsRow { display:flex; gap:4px; }
  #editorTopButtonsRow button { flex:1; margin:0; }
  /* Item 12: grade separation is an important, cost-driving choice -- the
     button panel (`#editorGradeSepWrap`) is auto-shown while drawing/
     extending so it stays a visually prominent, always-live control
     (interactive even mid-draw), without a second dedicated dropdown. */
  #editorGradeSepWrap.default-mode {
    border:2px solid #2b6cb0; border-radius:6px; padding:5px 6px; margin:4px 0;
    background:#eef4fb;
  }
  /* Item: grade-separation "paint" mode (outside draw/extend) -- the armed
     button uses the same `.on` highlight as everywhere else; this wrap style
     just marks the whole panel as "live" the way default-mode does. */
  #editorGradeSepWrap.paint-mode {
    border:2px solid #7a3ea1; border-radius:6px; padding:5px 6px; margin:4px 0;
    background:#f6effa;
  }
  .gradesep-btn-speed { display:block; font-size:9.5px; font-weight:400; color:#666; }
  #editorGradeSepWrap.paint-mode .gradesep-btn-speed,
  #editorGradeSepWrap.default-mode .gradesep-btn-speed { color:#88909a; }
  #editorGradeSepButtons button.on .gradesep-btn-speed { color:#dbe8f7; }
  /* Item 2: "Draw" is the single action available on a brand-new route with
     no geometry yet -- give it the same visual weight `#editorCompute`
     already has, so the primary next step is unmistakable. */
  #editBtnDraw.draw-prominent {
    font-size:13px; font-weight:700; padding:7px 14px; margin:2px 0 6px;
    display:block; width:100%; box-sizing:border-box;
    background:#2b6cb0; border-color:#2b6cb0; color:#fff;
    box-shadow:0 1px 4px rgba(43,108,176,0.5);
  }
  #editBtnDraw.draw-prominent:hover { background:#245a92; }
  #editorPanel label { display:block; font-weight:600; color:#444; margin:4px 0 2px; font-size:11px; }
  #editorPanel input[type=text], #editorPanel input[type=number], #editorPanel select {
    width:100%; padding:3px 5px; border:1px solid #d5d9de; border-radius:4px;
    background:#f9fafb; font:11.5px sans-serif; color:#222; box-sizing:border-box;
  }
  #editorPanel input[type=color] { width:100%; height:24px; padding:0; border:1px solid #d5d9de; border-radius:4px; }
  .editor-route-row {
    display:flex; align-items:center; gap:4px; padding:2px 0; border-bottom:1px solid #f0f2f4;
  }
  .editor-route-row .name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .editor-route-row .swatch { width:10px; height:10px; border-radius:2px; flex:none; }
  .editor-route-row.readonly { color:#888; }
  #editorHint { color:#555; font-style:italic; min-height:13px; margin-top:4px; }
  #editorScenarioModal {
    display:none; position:absolute; top:0; left:0; right:0; bottom:0; z-index:1400;
    background:rgba(0,0,0,0.35); align-items:center; justify-content:center; font:12px sans-serif;
  }
  #editorScenarioModal .box {
    background:white; padding:14px 16px; border-radius:8px; width:300px;
    box-shadow:0 3px 16px rgba(0,0,0,0.4);
  }
  #editorScenarioModal h4 { margin:0 0 8px; font-size:13px; }
  #editorScenarioModal input, #editorScenarioModal select {
    width:100%; padding:4px 6px; border:1px solid #d5d9de; border-radius:4px;
    font:12px sans-serif; box-sizing:border-box;
  }
  #editorScenarioModal button {
    font:11.5px sans-serif; padding:4px 9px; margin:0 4px 0 0; border:1px solid #c8cdd3;
    background:#f7f8fa; border-radius:4px; cursor:pointer;
  }
  #editorScenarioModal button.primary { background:#2b6cb0; border-color:#2b6cb0; color:white; }
  #editorScenarioModal .divider { margin:9px 0; border-top:1px solid #e5e8eb; }
  #editorScenarioError { color:#c0392b; min-height:14px; margin:4px 0; font-size:11px; }
  /* Always visible -- in AND out of edit mode (spec) -- so `display` is set
     here, not toggled by enter/exitEditMode. Out of edit mode it reports the
     reserved "current" scenario's own population-weighted level of service. */
  /* max-width reserves 230px on EACH side (the top-right mapLegend's own
     ~220px column, mirrored on the left for the top-left zoom control) --
     not just 220px total. Being horizontally centered (`translateX(-50%)`),
     this bar grows equally in both directions from center, so a
     "100vw - <legend width>" budget only accounted for one side and let the
     bar's right edge run straight into the legend on any window narrower
     than ~1000px (reproduced headless: mapLegend and editSummaryBar
     bounding rects genuinely overlapped at 768px width). Reserving the
     margin on both sides guarantees no overlap at any window width. */
  #editSummaryBar {
    display:block; position:absolute; top:10px; left:50%; transform:translateX(-50%);
    z-index:1200; background:white; padding:5px 12px; border-radius:6px;
    box-shadow:0 1px 6px rgba(0,0,0,0.3); font:12px sans-serif; color:#333; white-space:nowrap;
    /* Item 3 (verbatim user request): "make sure all this text and the
       number fit in the panel and otherwise make the panel wider" --
       widened from `100vw - 460px` (which could clip/ellipsis a custom
       scenario's full line: name + score + cost + impact + impact/$M +
       emoji) to `100vw - 40px`, just enough margin to clear the corner UI
       (basemap switcher etc.) on a normal window. */
    max-width:calc(100vw - 40px); overflow:hidden; text-overflow:ellipsis;
  }
  #editSummaryBar .sep { color:#c3c8ce; margin:0 5px; }
  #editSummaryBar .k { color:#777; }
  #editSummaryBar .v { font-weight:600; }
  #editSummaryBar .emoji { font-size:14px; margin-left:5px; vertical-align:-1px; }
  #editSummaryBar .stale { color:#b8860b; }
  #editSummaryName { cursor:pointer; }
  #editSummaryName:hover { text-decoration:underline; }
</style>

<!-- Top-center summary bar (Phase 4). One compact line, always on screen:
     scenario name, population-weighted level of service, and -- for a custom
     (non-"current") scenario -- total cost, impact, impact-per-$M and the
     impact_score emoji. Labels are deliberately abbreviated (spec: "make the
     labels compact") so the whole line fits at a normal window width. -->
<div id="editSummaryBar"><b id="editSummaryName">-</b><span id="editSummaryRest"></span></div>

<!-- Item 11: grouped next to the statistics (table/plots) button, bottom-left corner. -->
<div id="editModeButton" title="Scenario editor">&#9998;</div>

<div id="editorPanel">
  <h4 id="editorPanelTitle">Scenario editor</h4>
  <div id="editorScenarioLine" style="margin-bottom:4px;"></div>
  <button id="editorCompute" class="primary">&#9889; Compute</button>
  <div id="editorTopButtonsRow">
    <button id="editorSwitchScenario">Switch scenario</button>
    <button id="editorExport">Export</button>
    <button id="editorImport">Import</button>
    <input type="file" id="editorImportFile" accept="application/json" style="display:none;">
  </div>
  <div id="editorComputeStatus" style="color:#666;margin-top:3px;"></div>
  <div class="divider"></div>
  <div id="editorRouteListWrap">
    <label>Routes</label>
    <div id="editorRouteList"></div>
    <button id="editorNewRoute" class="primary">+ New route</button>
  </div>
  <div id="editorRouteEditor" style="display:none;">
    <div class="divider"></div>
    <label>Route name</label>
    <input type="text" id="editRouteName" placeholder="e.g. Green Line X">
    <label>Mode</label>
    <select id="editRouteMode">""" + mode_options + """</select>
    <label>Color</label>
    <input type="color" id="editRouteColor" value="#8b0000">
    <label>Headway (minutes)</label>
    <input type="number" id="editRouteHeadway" min="1" step="1" value="10">
    <div class="divider"></div>
    <label>Geometry</label>
    <div id="editorGeomButtons">
      <button id="editBtnDraw">Draw</button>
      <button id="editBtnFinishDraw" class="primary" style="display:none;">Finish</button>
      <button id="editBtnDeleteLastNode" style="display:none;">Delete last node</button>
      <button id="editBtnMove">Move node</button>
      <button id="editBtnAddNode">Add node</button>
      <button id="editBtnDelNode">Delete node</button>
      <button id="editBtnExtend">Extend route</button>
      <div style="display:flex;">
        <button id="editBtnAddStation" style="flex:1;">Add station</button>
        <button id="editBtnDelStation" style="flex:1;">Delete station</button>
      </div>
      <button id="editBtnGradeSep">Grade separation</button>
    </div>
    <div id="editorHint"></div>
    <div id="editorRouteStats" style="color:#666;margin-top:4px;"></div>
    <!-- Phase 3: per-segment grade-separation assignment. Shown only while
         the "Grade separation" mode is active; the route itself simultaneously
         switches to grade-separation coloring so the mode is self-documenting
         (see `renderActiveRoute`). -->
    <div id="editorGradeSepWrap" style="display:none;">
      <div class="divider"></div>
      <label>Grade separation &mdash; <span id="editorSegLabel">no segment selected</span></label>
      <div id="editorGradeSepButtons"></div>
      <button id="editorGradeSepApplyAll" style="display:none;width:100%;margin:3px 0 0;">Apply to all route</button>
      <div id="editorGradeSepLegend" style="color:#666;margin:2px 0 0;"></div>
      <label>Speed (km/h) <span style="font-weight:400;color:#888;">&mdash; this segment</span></label>
      <div style="display:flex;gap:4px;align-items:center;">
        <input type="number" id="editorSegSpeed" min="1" step="1" style="flex:1;">
        <button id="editorSegSpeedReset" style="flex:none;margin:0;">Reset</button>
      </div>
      <div id="editorSegSpeedNote" style="color:#666;margin-top:2px;"></div>
    </div>
    <div id="editorCostReadout" style="margin-top:5px;color:#222;"></div>
    <div class="divider"></div>
    <button id="editorDoneRoute" class="primary">Done</button>
    <button id="editorDeleteRoute">Delete route</button>
  </div>
</div>

<div id="editorScenarioModal">
  <div class="box">
    <h4 id="editorScenarioModalTitle">Choose a scenario</h4>
    <div id="editorPickExisting" style="display:none;">
      <label style="font-weight:600;">Continue an existing scenario</label>
      <select id="editorScenarioSelect"></select>
      <button id="editorContinueScenario" class="primary" style="margin-top:6px;">Continue</button>
      <div class="divider"></div>
    </div>
    <label style="font-weight:600;">Create a new scenario</label>
    <input type="text" id="editorNewScenarioName" placeholder="Scenario name">
    <div id="editorScenarioError"></div>
    <button id="editorCreateScenario" class="primary">Create</button>
    <button id="editorCancelScenario">Cancel (exit edit mode)</button>
  </div>
</div>
"""


def _editor_js(map_var: Optional[str], params: dict, pop_chunk_cfg: Optional[dict] = None) -> str:
    """All editor behavior.

    Placeholder-substituted (`__MAP_REF__`/`__PARAMS__`) rather than written
    as an f-string: this block is almost entirely JS object/function braces,
    and doubling every one of them would make it unreadable and
    trivially breakable.
    """
    map_ref = f'window["{map_var}"]' if map_var else "null"
    if pop_chunk_cfg is None:
        pop_chunk_cfg = {"dir": POP_CHUNK_DIRNAME, "deg": POP_CHUNK_DEG, "res": 11}
    geoid_chunk_cfg = {"dir": GEOID_CHUNK_DIRNAME, "deg": POP_CHUNK_DEG, "level": CENSUS_LIVE_RECOLOR_LEVEL}
    js = r"""
window.__editParams = __PARAMS__;
// Where the client fetches the H3 res-11 population/baseline-access chunks
// from, and the degree size of one chunk -- both written by
// `transitlos.map.pop_chunks.build_population_chunks`, so the two sides can
// never disagree about the partitioning scheme.
window.__popChunkCfg = __POPCHUNKS__;
// Item 6 (2026-08-16): GEOID<->h3-cell chunks, same grid/partitioning as the
// population chunks above (see `transitlos.map.pop_chunks.build_geoid_chunks`)
// -- lets `computeAccess()` find which census block group each touched h3
// cell belongs to without ever downloading a full metro-wide mapping. A
// study with no census data (or that hasn't run the chunk generator) simply
// has no `geoid_chunks/index.json` to fetch; `loadGeoidChunkIndex()` treats
// that identically to a missing pop-chunks index (degrades to a no-op).
window.__geoidChunkCfg = __GEOIDCHUNKS__;

// ---------------------------------------------------------------------------
// DATA MODEL (in memory only; the JSON export/import below is the persistence)
//
//   window.__editorState = {
//     active:          bool,                 // edit mode on/off
//     scenarios:       [Scenario, ...],      // user-created only
//     activeScenario:  string|null,          // scenario NAME, or the reserved
//                                            // "current" (never editable)
//     activeRouteId:   string|null,
//     mode:            'idle'|'draw'|'move'|'addnode'|'delnode'|
//                      'extend'|'addstation'|'delstation',
//     restore:         {...}                 // pre-edit-mode layer state
//   }
//
//   Scenario = { name: string, routes: [Route, ...] }
//
//   Route = {
//     id:               string,              // 'r<counter>'
//     name:             string,              // GTFS route_short_name-ish
//     mode:             'rail'|'tram'|'brt'|'bus',
//     color:            '#rrggbb',
//     headway_minutes:  number,
//     user_created:     true,                // false => read-only (real network)
//     points: [ { lat, lng, is_station:bool }, ... ],
//     // one entry per SEGMENT, i.e. points.length - 1 entries, kept in sync
//     // by `syncSegments()`:
//     segments: [ { grade_separation: 'mixed'|...|null,
//                   speed_kmh_override?: number }, ... ],   // PHASE 3
//     cost_musd: number|null,        // PHASE 3: null while any segment is
//                                    // still unassigned (i.e. "partial")
//     cost_breakdown: {...}          // PHASE 3: routeCost()'s full result
//   }
//
// PHASE 3: segments[i].grade_separation drawn from
// __editParams.grade_separations[mode], an optional per-segment
// speed_kmh_override on top of the computed speed, and route.cost_musd from
// __editParams.costs (per-km by grade separation + per-station, a station
// charged at the more expensive of its two adjacent segments).
// `mode: 'gradesep'` is the assignment mode; while it is active the route
// renders in grade-separation colors instead of route.color.
//
// PHASE 4 adds no new persisted route fields -- the recompute's output is
// scenario-level and transient, held in the closure's `lastCompute` (readable
// via `window.__editor.lastCompute()`):
//
//   lastCompute = {
//     ok, reason,
//     n_stations, n_stations_scored,        // stations found / actually scored
//     n_chunks, chunk_keys,                 // population chunks fetched
//     n_cells_in_buffer, n_cells_improved,
//     impact,                               // sum((new - old) * population)
//     cost_musd, cost_partial,
//     baseline_access, new_access,          // metro population-weighted means
//     impact_per_musd, total_population,
//     max_gain_cell,                        // the single most-improved cell,
//                                           // with every input of its own
//                                           // score (for hand-checking)
//     signature                             // staleness key, see scenarioSignature
//   }
// ---------------------------------------------------------------------------

(function() {
  var mapRef = __MAP_REF__;
  if (!mapRef || !window.L) return;
  var P = window.__editParams || {};
  var MODES = P.modes || {};
  var RESERVED = ((P.scenarios || {}).reserved_current_name) || 'current';
  // Distance (metres) within which a click while drawing/extending counts as
  // "on" an existing transit stop and triggers the make-it-a-station prompt.
  var STOP_SNAP_M = 60;
  // Phase 3: how a segment with no grade separation assigned yet is drawn --
  // gray and dashed, so "still needs assignment" is unmistakable next to the
  // solid, colored assigned segments.
  var GS_UNSET_COLOR = '#9aa0a6';
  var GS_UNSET_DASH = '6,7';

  var S = {
    active: false,
    scenarios: [],
    activeScenario: null,
    activeRouteId: null,
    mode: 'idle',
    restore: {},
    _routeCounter: 0
  };
  window.__editorState = S;

  // The edited route gets its own pane ABOVE the real transit-line panes
  // (448/449/450, see `_routes_layer_block`) and below `markerPane` (600):
  // in the default overlay pane (400) the real network drew on top of the
  // line the user was actively editing.
  if (!mapRef.getPane('editorPane')) {
    mapRef.createPane('editorPane');
    mapRef.getPane('editorPane').style.zIndex = 460;
  }
  var editLayer = L.layerGroup();
  // Item 6/4: EVERY route the user has drawn (in the current scenario),
  // other than the one actively selected/being edited, is drawn here in a
  // plain, non-highlighted style -- always on the map, in and out of edit
  // mode, so a route never "disappears" after Draw finishes or the panel is
  // closed. Only actual stations get a marker; plain nodes get none (no
  // Geoman-style vertex dots on the always-visible display).
  var otherRoutesLayer = L.layerGroup();
  otherRoutesLayer.addTo(mapRef);
  // Item 7: one shared station-marker builder for every route (active or
  // not) -- always the real mode icon, never a plain dot. Falls back to a
  // black-ringed circle only if the stops layer's helper never published
  // (e.g. a page with no stops data at all).
  // Item 4: once a Compute has run, `p.__stopScore` (set below by
  // `computeAccess()`) holds THIS point's own computed stop_score -- tint
  // the marker with it via the exact same blue scale real GTFS stops use
  // (`window.__blueFor`, normalized against the same `window.__stopScoreDomain`
  // published by `_stops_layer_block`), so a drawn stop reads identically to
  // a real one. Before the first Compute (or if a point has no score yet,
  // e.g. missing headway/grade separation), falls back to the route's own
  // color -- the old fixed behavior -- so an uncomputed route still looks
  // reasonable rather than a uniform "no data" tint.
  function stationTint(p, r) {
    if (p.__stopScore != null && isFinite(p.__stopScore) && window.__blueFor && window.__stopScoreDomain) {
      var domain = window.__stopScoreDomain, sMin = domain[0], sMax = domain[1];
      var t = (p.__stopScore - sMin) / Math.max(sMax - sMin, 1e-9);
      return window.__blueFor(t);
    }
    return r.color;
  }
  // Item 5: a station-marker popup mirroring the real-GTFS-stop popup's
  // table layout (`_stops_layer_block`'s `popupHtml`, above) -- same rows
  // (route/mode/headway/speed/stop score), built from the drawn route's own
  // live data instead of `window.__stopsData`. Built lazily, same reason as
  // the real one: reflects whatever's current (e.g. a just-run Compute's
  // `p.__stopScore`) at open time rather than being stale from marker build.
  function stationPopupHtml(p, r) {
    return function() {
      var tint = stationTint(p, r);
      var rows = '';
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;vertical-align:top;">route</td>' +
        '<td style="padding:2px 0;">' + (window.__routeBadgeHtml ? window.__routeBadgeHtml(r.name || '(unnamed)', r.mode, 10) : (r.name || '(unnamed)')) + '</td></tr>';
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">mode</td><td style="padding:2px 0;">' + r.mode + '</td></tr>';
      if (r.headway_minutes != null) {
        rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">headway</td><td style="padding:2px 0;">' + r.headway_minutes + ' min</td></tr>';
      }
      var spd = routeAvgSpeedKmh(r);
      if (spd != null) {
        rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">speed</td><td style="padding:2px 0;">' + spd.toFixed(1) + ' km/h</td></tr>';
      }
      rows += '<tr><td style="font-weight:700;padding:4px 8px 0 0;border-top:1px solid #eee;white-space:nowrap;">stop score</td>' +
        '<td style="padding:4px 0 0;border-top:1px solid #eee;font-weight:700;white-space:nowrap;color:' + (p.__stopScore != null ? tint : '#333') + ';">' +
        (p.__stopScore != null && isFinite(p.__stopScore) ? p.__stopScore.toFixed(2) : '-') + '</td></tr>';
      return '<div style="font:12px sans-serif;">' +
        '<div style="font-weight:700;margin-bottom:4px;white-space:nowrap;">' + (r.name || 'Station') + ' (' + r.mode + ')</div>' +
        '<table style="border-collapse:collapse;width:100%;">' + rows + '</table></div>';
    };
  }
  function stationMarker(p, r) {
    var m;
    if (window.__modeBadgeHtml) {
      var tint = stationTint(p, r);
      var html = '<div style="text-align:center;">' + window.__modeBadgeHtml(r.mode, tint, 20) +
        (p.__stopScore != null && isFinite(p.__stopScore) && window.__scoreBadgeHtml
          ? window.__scoreBadgeHtml(p.__stopScore.toFixed(2), tint) : '') +
        '</div>';
      var icon = L.divIcon({
        html: html,
        className: 'editor-station-icon', iconSize: [20, 26]
      });
      m = L.marker([p.lat, p.lng], {icon: icon});
    } else {
      m = L.circleMarker([p.lat, p.lng], {radius: 5, color: '#111', weight: 2, fillColor: '#fff', fillOpacity: 1});
    }
    // Item 5: same click-to-open popup interaction real GTFS stops have
    // (`marker.bindPopup(popupHtml, ...)` in `_stops_layer_block`) -- not a
    // stripped-down alternative.
    m.bindPopup(stationPopupHtml(p, r), {minWidth: 210, maxWidth: 260});
    return m;
  }
  function renderOtherRoutes() {
    otherRoutesLayer.clearLayers();
    var sc = activeScenarioObj();
    if (!sc) return;
    sc.routes.forEach(function(r) {
      if (S.active && r.id === S.activeRouteId) return;   // that one gets the highlight style instead
      if (!r.points || r.points.length < 2) return;
      var latlngs = r.points.map(function(p) { return [p.lat, p.lng]; });
      var line = L.polyline(latlngs, {color: r.color, weight: 4, opacity: 0.85, interactive: true, bubblingMouseEvents: false});
      // Item 5: same hover-tooltip interaction real routes have
      // (`line.bindTooltip(r.l, {sticky:true, direction:'top'})` in
      // `_routes_layer_block`) -- a freshly drawn route reads identically.
      line.bindTooltip(r.name || '(unnamed route)', {sticky: true, direction: 'top'});
      line.addTo(otherRoutesLayer);
      r.points.forEach(function(p) {
        if (!p.is_station) return;   // no dot for plain nodes -- stations only
        // Item 7: a station marker ALWAYS shows the route's mode-specific
        // icon (rail/tram/bus), in and out of edit mode, on every route --
        // never a generic dot -- same helper `_stops_layer_block` uses for
        // real GTFS stops (`window.__modeBadgeHtml`, published above).
        stationMarker(p, r).addTo(otherRoutesLayer);
      });
    });
  }
  window.__renderOtherRoutes = renderOtherRoutes;
  var routeLine = null;          // L.polyline for the active route
  var vertexMarkers = [];        // one per point, index kept in marker.__idx
  var extendFrom = null;         // 'start' | 'end' | null (extend mode only)
  var segLayers = [];            // Phase 3: per-segment polylines, index in layer.__seg
  var selectedSeg = -1;          // Phase 3: segment index picked for assignment

  function $(id) { return document.getElementById(id); }
  function setHint(t) { $('editorHint').textContent = t || ''; }

  // --- helpers ------------------------------------------------------------
  function scenarioByName(n) {
    for (var i = 0; i < S.scenarios.length; i++) if (S.scenarios[i].name === n) return S.scenarios[i];
    return null;
  }
  function activeScenarioObj() { return scenarioByName(S.activeScenario); }
  function isReserved(n) { return String(n || '').trim().toLowerCase() === RESERVED.toLowerCase(); }
  function activeRoute() {
    var sc = activeScenarioObj();
    if (!sc || !S.activeRouteId) return null;
    for (var i = 0; i < sc.routes.length; i++) if (sc.routes[i].id === S.activeRouteId) return sc.routes[i];
    return null;
  }
  // One segment record per adjacent point pair -- created empty here,
  // populated by Phase 3's grade-separation UI.
  // Item 9: a brand-new segment starts with its mode's configured default
  // grade separation (`__editParams.default_grade_separation[mode]`) rather
  // than unset, so a freshly-drawn route already costs and scores something
  // sensible before the user has touched the grade-separation panel at all.
  function defaultGradeSep(mode, r) {
    // A route-level live override (item 12's always-interactive "Grade
    // separation" select) wins over the mode's static config default.
    if (r && r._defaultGradeSepOverride && gradeSepOptions(mode).indexOf(r._defaultGradeSepOverride) >= 0) {
      return r._defaultGradeSepOverride;
    }
    var d = (P.default_grade_separation || {})[mode];
    if (!d) return null;
    return (gradeSepOptions(mode).indexOf(d) >= 0) ? d : null;
  }
  function syncSegments(r) {
    var want = Math.max(0, r.points.length - 1);
    while (r.segments.length < want) r.segments.push({grade_separation: defaultGradeSep(r.mode, r)});
    while (r.segments.length > want) r.segments.pop();
  }

  // -------------------------------------------------------------------------
  // PHASE 3: grade separation, speed, cost.
  //
  // Every number below comes from `default_edit_params.json` (`window.__editParams`),
  // never hardcoded here -- same discipline as the Discretization tab's
  // `__jsSpeedScore`/`__jsStopScore` mirrors, which take their constants from
  // `window.__disc` rather than a second hand-copied set.
  // -------------------------------------------------------------------------

  // Which grade separations this route's mode allows (rail: elevated/normal/
  // underground; tram/BRT/bus: elevated/separated/mixed/underground).
  function gradeSepOptions(mode) {
    return ((P.grade_separations || {})[mode]) || [];
  }
  function gradeSepColor(gs) {
    if (!gs) return GS_UNSET_COLOR;
    return ((P.grade_separation_colors || {})[gs]) || GS_UNSET_COLOR;
  }

  // Mirrors `default_edit_params.json`'s `speeds` block, which is fixed by
  // explicit instruction rather than derived: rail runs at a flat 40 km/h
  // whatever its grade separation (only its 45 s/stop dwell applies), while
  // tram/BRT/bus speed is a pure function of grade separation
  // (mixed 10 / separated 15 / elevated|underground 20 km/h) plus a 15 s/stop
  // dwell. Returns null when the segment has no grade separation yet and the
  // mode needs one to answer.
  function computedSegmentSpeedKmh(mode, gradeSep) {
    var sp = (P.speeds || {})[mode];
    if (!sp) return null;
    if (typeof sp.kmh === 'number') return sp.kmh;          // rail: grade-separation-independent
    var by = sp.kmh_by_grade_separation || {};
    if (!gradeSep) return null;
    return (typeof by[gradeSep] === 'number') ? by[gradeSep] : null;
  }
  function dwellSecondsPerStop(mode) {
    var sp = (P.speeds || {})[mode];
    return sp && typeof sp.dwell_seconds_per_stop === 'number' ? sp.dwell_seconds_per_stop : 0;
  }
  // The speed actually in force for a segment: the user's per-segment
  // override when one is set, otherwise the computed value. (Ambiguity call:
  // the spec's "a box with the speed just in case user wants to edit it too"
  // is applied PER SEGMENT, mirroring grade separation itself -- see the
  // report/docstring reasoning.)
  function effectiveSegmentSpeedKmh(r, i) {
    var seg = r.segments[i];
    if (seg && typeof seg.speed_kmh_override === 'number' && isFinite(seg.speed_kmh_override)) {
      return seg.speed_kmh_override;
    }
    return computedSegmentSpeedKmh(r.mode, seg ? seg.grade_separation : null);
  }

  function segmentLengthKm(r, i) {
    if (i < 0 || i + 1 >= r.points.length) return 0;
    var a = L.latLng(r.points[i].lat, r.points[i].lng);
    var b = L.latLng(r.points[i + 1].lat, r.points[i + 1].lng);
    return mapRef.distance(a, b) / 1000.0;
  }

  // A station's cost is charged at "the most expensive grade separation of
  // its two adjacent segments" (per spec and costs.md) -- "most expensive"
  // compared by that mode's own `station_musd` table, not by a fixed
  // ordering, so the rule stays correct if the numbers are re-tuned. An
  // endpoint station has only one adjacent segment, so that one decides.
  function stationGradeSep(r, i) {
    var st = ((P.costs || {})[r.mode] || {}).station_musd || {};
    var cands = [];
    if (i - 1 >= 0 && r.segments[i - 1] && r.segments[i - 1].grade_separation) cands.push(r.segments[i - 1].grade_separation);
    if (i < r.segments.length && r.segments[i] && r.segments[i].grade_separation) cands.push(r.segments[i].grade_separation);
    var best = null, bestC = -Infinity;
    cands.forEach(function(gs) {
      var c = (typeof st[gs] === 'number') ? st[gs] : -Infinity;
      if (c > bestC) { bestC = c; best = gs; }
    });
    return best;
  }

  // Route cost = sum over segments of length_km * cost_per_km_musd[mode][gs]
  //            + sum over stations of station_musd[mode][station's gs].
  // Segments with no grade separation yet contribute nothing and are counted
  // in `n_uncosted`, so the readout can say "partial" instead of quietly
  // reporting a wrong total.
  function routeCost(r) {
    var c = (P.costs || {})[r.mode] || {};
    var perKm = c.cost_per_km_musd || {}, stCost = c.station_musd || {};
    var segTotal = 0, nCosted = 0, nSeg = Math.max(0, r.points.length - 1);
    for (var i = 0; i < nSeg; i++) {
      var gs = r.segments[i] && r.segments[i].grade_separation;
      if (!gs || typeof perKm[gs] !== 'number') continue;
      segTotal += segmentLengthKm(r, i) * perKm[gs];
      nCosted += 1;
    }
    var stTotal = 0, nStations = 0, nStationsCosted = 0;
    for (var j = 0; j < r.points.length; j++) {
      if (!r.points[j].is_station) continue;
      nStations += 1;
      var sgs = stationGradeSep(r, j);
      if (sgs && typeof stCost[sgs] === 'number') { stTotal += stCost[sgs]; nStationsCosted += 1; }
    }
    // Item 7: fleet-size / vehicle cost estimate. One-way travel time = sum of
    // segment run times (length / effective speed) + dwell time at every
    // station; the classic fleet-size formula is
    //   vehicles_needed = ceil(round_trip_minutes / headway_minutes)
    // round_trip_minutes = 2 x one-way minutes (there and back). A route with
    // no headway set, or no costed (speed-bearing) segments, can't estimate a
    // fleet, so it's left null rather than silently reported as zero.
    var oneWayMin = 0, speedKnown = (nSeg > 0);
    var dwellS = dwellSecondsPerStop(r.mode);
    for (var si = 0; si < nSeg; si++) {
      var spKmh = effectiveSegmentSpeedKmh(r, si);
      if (!spKmh || spKmh <= 0) { speedKnown = false; continue; }
      oneWayMin += (segmentLengthKm(r, si) / spKmh) * 60;
    }
    oneWayMin += nStations * (dwellS / 60);
    var vehicleCostPerUnit = (P.vehicle_cost_musd || {})[r.mode];
    var headwayMin = parseFloat(r.headway_minutes);
    var vehiclesNeeded = null, vehicleFleetMusd = null;
    if (speedKnown && oneWayMin > 0 && isFinite(headwayMin) && headwayMin > 0) {
      vehiclesNeeded = Math.max(1, Math.ceil((2 * oneWayMin) / headwayMin));
      if (typeof vehicleCostPerUnit === 'number') vehicleFleetMusd = vehiclesNeeded * vehicleCostPerUnit;
    }
    var infraTotal = segTotal + stTotal;
    return {
      segments_musd: segTotal, stations_musd: stTotal,
      vehicles_needed: vehiclesNeeded, vehicle_fleet_musd: vehicleFleetMusd,
      infrastructure_musd: infraTotal,
      total_musd: infraTotal + (vehicleFleetMusd || 0),
      n_segments: nSeg, n_costed: nCosted, n_uncosted: nSeg - nCosted,
      n_stations: nStations, n_stations_costed: nStationsCosted,
      length_km: (function() { var t = 0; for (var k = 0; k < nSeg; k++) t += segmentLengthKm(r, k); return t; })(),
      round_trip_minutes: (speedKnown && oneWayMin > 0) ? 2 * oneWayMin : null,
      partial: nCosted < nSeg
    };
  }

  // Headway aggregation, per `default_edit_params.json`'s `headway_aggregation`
  // block: frequencies add, so 1/H = 1/h1 + 1/h2 (+ ...). Callers are
  // responsible for only passing headways of the SAME mode -- the params'
  // `combine_across_modes: false` says different-mode routes sharing a stop
  // are scored independently, never merged. Phase 4 consumes this in the
  // live stop-score recompute; Phase 3 only defines it.
  function aggregateHeadway(headways) {
    if (!headways || !headways.length) return null;
    var invSum = 0, n = 0;
    for (var i = 0; i < headways.length; i++) {
      var h = parseFloat(headways[i]);
      if (!isFinite(h) || h <= 0) continue;
      invSum += 1.0 / h;
      n += 1;
    }
    if (!n || invSum <= 0) return null;
    return 1.0 / invSum;
  }
  window.__aggregateHeadway = aggregateHeadway;

  // -------------------------------------------------------------------------
  // PHASE 4: client-side access-score recompute + the top-center summary bar.
  //
  // Everything here runs in the browser off plain static files -- no server
  // logic, by explicit architectural decision. The population/baseline-access
  // grid it needs (H3 res 11, ~1.6M cells for Boston -- far too much to inline
  // into map.html) is pre-partitioned by `transitlos.map.pop_chunks` into small
  // per-lat/lon-square JSON files, and only the handful of squares covering the
  // buffer around the scenario's own stops are ever fetched. See that module's
  // docstring for why the partition key is a degree grid and not an H3 parent
  // (the client would otherwise need an H3 library just to name a chunk).
  //
  // The scoring itself does NOT reimplement anything: it calls the exact same
  // `__jsStopScore` / `__jsWalkDecay` mirrors of `transitlos.scoring` the
  // Discretization tab already uses (published onto `window` at the end of
  // `_stats_panel_js` -- they are function declarations inside that script's
  // own `load` closure and are otherwise unreachable from here).
  // -------------------------------------------------------------------------

  var PC = window.__popChunkCfg || {dir: 'pop_chunks', deg: 0.04, res: 11};
  var AR = P.access_recompute || {};
  // Outer straight-line catchment radius. `walk_decay` is an offset-power
  // curve that never reaches zero, so without a cutoff a single stop would
  // touch the whole metro; 2000 m is the real pipeline's own outer
  // walk-distance step (see the params file's comment).
  var ACCESS_R_M = (typeof AR.max_walk_distance_m === 'number') ? AR.max_walk_distance_m : 2000;
  var HEADWAY_CV = (typeof AR.headway_cv === 'number') ? AR.headway_cv : 0.3;

  var chunkCache = {};        // key -> rows array, once fetched
  var chunkIndex = null;      // {key: 1} of chunks that actually exist
  // Test/inspection hooks: which chunk files this page has actually pulled.
  window.__fetchedChunks = [];
  window.__popChunkCache = chunkCache;

  function popChunkKey(lat, lng) {
    return Math.floor(lat / PC.deg) + '_' + Math.floor(lng / PC.deg);
  }

  function loadChunkIndex() {
    if (chunkIndex) return Promise.resolve(chunkIndex);
    return fetch(PC.dir + '/index.json').then(function(r) {
      if (!r.ok) throw new Error('no index');
      return r.json();
    }).then(function(j) {
      chunkIndex = {};
      (j.keys || []).forEach(function(k) { chunkIndex[k] = 1; });
      window.__popChunkIndexMeta = {n_chunks: j.n_chunks, n_cells: j.n_cells, deg: j.deg, res: j.res};
      return chunkIndex;
    }).catch(function() {
      // No index (population chunks were never generated for this study):
      // the recompute degrades to "unavailable" rather than throwing, and
      // the summary bar says so.
      chunkIndex = null;
      return null;
    });
  }

  function loadChunk(key) {
    if (chunkCache[key]) return Promise.resolve(chunkCache[key]);
    return fetch(PC.dir + '/' + key + '.json').then(function(r) {
      if (!r.ok) throw new Error('missing chunk ' + key);
      return r.json();
    }).then(function(j) {
      chunkCache[key] = j.cells || [];
      window.__fetchedChunks.push(key);
      return chunkCache[key];
    }).catch(function() {
      chunkCache[key] = [];
      return chunkCache[key];
    });
  }

  // Item 6 (2026-08-16): the GEOID<->h3-cell chunk loaders, mirroring
  // loadChunkIndex/loadChunk above exactly but for `window.__geoidChunkCfg`
  // -- same chunk keys, so a chunk fetched here always matches the same
  // geographic square already loaded through `loadChunk()`.
  var GC = window.__geoidChunkCfg || {dir: 'geoid_chunks', deg: 0.04, level: 'blockgroup'};
  var geoidChunkCache = {};
  var geoidChunkIndex = null;

  function loadGeoidChunkIndex() {
    if (geoidChunkIndex) return Promise.resolve(geoidChunkIndex);
    return fetch(GC.dir + '/index.json').then(function(r) {
      if (!r.ok) throw new Error('no geoid index');
      return r.json();
    }).then(function(j) {
      geoidChunkIndex = {};
      (j.keys || []).forEach(function(k) { geoidChunkIndex[k] = 1; });
      return geoidChunkIndex;
    }).catch(function() {
      geoidChunkIndex = null;
      return null;
    });
  }

  function loadGeoidChunk(key) {
    if (geoidChunkCache[key]) return Promise.resolve(geoidChunkCache[key]);
    return fetch(GC.dir + '/' + key + '.json').then(function(r) {
      if (!r.ok) throw new Error('missing geoid chunk ' + key);
      return r.json();
    }).then(function(j) {
      geoidChunkCache[key] = (j && j.map) || {};
      return geoidChunkCache[key];
    }).catch(function() {
      geoidChunkCache[key] = {};
      return geoidChunkCache[key];
    });
  }

  // Great-circle distance in metres. Leaflet's `map.distance` would do the
  // same thing, but it allocates an L.LatLng per call and this runs once per
  // (cell, station) pair over tens of thousands of cells.
  function haversineM(lat1, lng1, lat2, lng2) {
    var R = 6371008.8;   // mean Earth radius, same value Leaflet's spherical CRS uses
    var d2r = Math.PI / 180;
    var dLat = (lat2 - lat1) * d2r, dLng = (lng2 - lng1) * d2r;
    var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * d2r) * Math.cos(lat2 * d2r) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
  }

  // BRT is drawn/labelled as its own mode but SCORES exactly as tram
  // (`modes.brt.scores_as` in the params file, mirroring
  // `transitlos.scoring.mode.MODE_SCORES`' rail/tram/bus tiers).
  function scoringMode(m) { return ((MODES[m] || {}).scores_as) || m; }

  // Commercial (door-to-door) average speed of the whole route: track time
  // from each segment's own effective speed, plus this mode's dwell at every
  // station except the terminus. This is the route-level number
  // `stop_score`'s `speed_score` factor wants -- the real pipeline's
  // `avg_speed_kmh` is likewise a per-route, dwell-inclusive figure, not an
  // instantaneous per-segment one. Returns null while any segment still has
  // no grade separation (and therefore no speed) assigned.
  function routeAvgSpeedKmh(r) {
    var nSeg = Math.max(0, r.points.length - 1);
    if (!nSeg) return null;
    var km = 0, hours = 0;
    for (var i = 0; i < nSeg; i++) {
      var v = effectiveSegmentSpeedKmh(r, i);
      if (!v || !isFinite(v) || v <= 0) return null;
      var l = segmentLengthKm(r, i);
      km += l; hours += l / v;
    }
    var nSt = r.points.filter(function(p) { return p.is_station; }).length;
    hours += Math.max(0, nSt - 1) * dwellSecondsPerStop(r.mode) / 3600.0;
    if (hours <= 0 || km <= 0) return null;
    return km / hours;
  }

  // Every headway serving this station that may be combined with the new
  // route's own, per the params' `headway_aggregation` block: SAME scoring
  // mode only (`combine_across_modes: false`), within `STOP_SNAP_M`. Two
  // sources -- the real current network (`window.__stopsData`, whose
  // `headway_minutes` is already the aggregated figure for that stop) and
  // the scenario's own other routes' stations.
  function headwaysAt(lat, lng, route) {
    var out = [];
    if (route.headway_minutes > 0) out.push(route.headway_minutes);
    var want = scoringMode(route.mode);
    var recs = (window.__stopsData && window.__stopsData.records) || [];
    for (var i = 0; i < recs.length; i++) {
      var rec = recs[i];
      if (String(rec.mode || '').toLowerCase() !== want) continue;
      if (!(rec.headway_minutes > 0)) continue;
      if (haversineM(lat, lng, rec.lat, rec.lon) > STOP_SNAP_M) continue;
      out.push(rec.headway_minutes);
    }
    var sc = activeScenarioObj();
    if (sc) sc.routes.forEach(function(r2) {
      if (r2 === route) return;
      if (scoringMode(r2.mode) !== want || !(r2.headway_minutes > 0)) return;
      for (var j = 0; j < r2.points.length; j++) {
        var p = r2.points[j];
        if (!p.is_station) continue;
        if (haversineM(lat, lng, p.lat, p.lng) <= STOP_SNAP_M) { out.push(r2.headway_minutes); return; }
      }
    });
    return out;
  }

  // The study-area baseline the bar reports: the population-weighted mean
  // level_of_service over the whole metro, straight from the stats panel's own
  // data (`window.__statsData`) rather than recomputed here. Note that a
  // population-weighted mean of population-weighted means over a partition is
  // exactly the population-weighted mean over the underlying res-11 cells, so
  // reading it off the coarser stats resolution is exact, not an
  // approximation -- and it means the recompute's `impact` (sum of
  // gain*population over res-11 cells) converts to a metro-wide access delta
  // by a single division by the same total population.
  var _baseline = undefined;
  function baselineStats() {
    if (_baseline !== undefined) return _baseline;
    _baseline = null;
    var sd = window.__statsData;
    if (sd) {
      var d = sd.metro || sd[Object.keys(sd)[0]];
      if (d && d.level_of_service && d.population) {
        // 2026-09-02 (explicit user request: "top bar should be median not
        // mean") -- was a population-weighted MEAN, inconsistent with the
        // Place-rank tab/combined-map overview, which both show a
        // population-weighted MEDIAN over this exact same data -- a highly
        // skewed access distribution (many cells at exactly 0, a smaller
        // set of well-served dense cells) makes mean and median diverge
        // sharply (real case: Boston, mean 0.24 vs median 0.02), which
        // read as "the numbers don't match" even though both were
        // individually correct. Now genuinely the same statistic
        // everywhere.
        var pairs = [];
        var den = 0;
        for (var i = 0; i < d.level_of_service.length; i++) {
          var a = d.level_of_service[i], p = d.population[i];
          if (a == null || p == null || isNaN(a) || isNaN(p) || p < 0) continue;
          pairs.push([a, p]); den += p;
        }
        if (den > 0) {
          pairs.sort(function(x, y) { return x[0] - y[0]; });
          var cum = 0, half = den / 2, median = pairs[pairs.length - 1][0];
          for (var j = 0; j < pairs.length; j++) {
            cum += pairs[j][1];
            if (cum >= half) { median = pairs[j][0]; break; }
          }
          _baseline = {access: median, population: den};
        }
      }
    }
    return _baseline;
  }

  // Changed-since-last-Compute detection: everything the recompute reads off
  // a route (geometry, station flags, mode, headway, grade separations,
  // speed overrides), and nothing it doesn't (name, color).
  function scenarioSignature(sc) {
    if (!sc) return '';
    return JSON.stringify((sc.routes || []).map(function(r) {
      return [r.mode, r.headway_minutes,
        r.points.map(function(p) { return [p.lat.toFixed(6), p.lng.toFixed(6), p.is_station ? 1 : 0]; }),
        (r.segments || []).map(function(s) { return [s.grade_separation || null, s.speed_kmh_override || null]; })];
    }));
  }

  // impact_per_million_dollar -> emoji, from the params' `impact_score`
  // block. AMBIGUITY RESOLVED: the spec asks for three emoji at 100/200/300
  // AND for linear interpolation between them, which cannot both be literally
  // true of a discrete glyph. Both halves are honored: `score01` is the
  // genuine linear interpolation between the sad and happy thresholds (shown
  // in the bar's tooltip), and the glyph shown is the nearest of the three
  // configured anchors on that same linear scale, with >= happy and <= sad
  // pinned exactly.
  function impactEmoji(x) {
    var cfg = P.impact_score || {}, em = cfg.emoji || {};
    var happy = cfg.happy_threshold, sad = cfg.sad_threshold, med = cfg.medium_value;
    if (x == null || !isFinite(x) || typeof happy !== 'number' || typeof sad !== 'number') {
      return {emoji: '', score01: null};
    }
    var score01 = Math.max(0, Math.min(1, (x - sad) / (happy - sad)));
    if (x >= happy) return {emoji: em.happy || '', score01: score01};
    if (x <= sad) return {emoji: em.sad || '', score01: score01};
    var anchors = [[sad, em.sad], [(typeof med === 'number' ? med : (sad + happy) / 2), em.medium], [happy, em.happy]];
    var best = anchors[1], bestD = Infinity;
    anchors.forEach(function(a) { var d = Math.abs(x - a[0]); if (d < bestD) { bestD = d; best = a; } });
    return {emoji: best[1] || '', score01: score01};
  }
  window.__impactEmoji = impactEmoji;

  var lastCompute = null;
  // Per-scenario history of the last successful Compute, keyed by scenario
  // name -- `lastCompute` alone only ever reflects whichever scenario was
  // active the moment Compute was last pressed, so the stats panel's
  // scenario dropdown (item 1) needs this separate map to know which OTHER
  // scenarios have ever been computed, and to fetch their own results
  // without disturbing the currently active one.
  var computeResultsByScenario = {};

  function scenarioCostMusd(sc) {
    var total = 0, partial = false;
    (sc ? sc.routes : []).forEach(function(r) {
      var c = routeCost(r);
      total += c.total_musd;
      if (c.partial) partial = true;
    });
    return {total_musd: total, partial: partial};
  }

  // The recompute proper. Returns a Promise resolving to the result object
  // (also stashed on `window.__editor.lastCompute`).
  function computeAccess() {
    var sc = activeScenarioObj();
    var res = {
      ok: false, reason: '', n_stations: 0, n_stations_scored: 0,
      n_chunks: 0, n_cells_in_buffer: 0, n_cells_improved: 0,
      impact: 0, cost_musd: 0, cost_partial: false,
      baseline_access: null, new_access: null, impact_per_musd: null,
      signature: scenarioSignature(sc)
    };
    var base = baselineStats();
    res.baseline_access = base ? base.access : null;
    res.new_access = res.baseline_access;
    var cost = scenarioCostMusd(sc);
    res.cost_musd = cost.total_musd; res.cost_partial = cost.partial;

    if (!window.__jsStopScore || !window.__jsWalkDecay || !window.__disc) {
      res.reason = 'scoring mirrors unavailable (no stats panel on this map)';
      lastCompute = res; renderSummary(); return Promise.resolve(res);
    }
    if (!base) { res.reason = 'no baseline stats'; lastCompute = res; renderSummary(); return Promise.resolve(res); }

    // Every station of EVERY route in the scenario (spec: the whole current
    // scenario, not just the selected route).
    var stations = [];
    (sc ? sc.routes : []).forEach(function(r) {
      var spd = routeAvgSpeedKmh(r);
      r.points.forEach(function(p) {
        if (!p.is_station) return;
        var st = {lat: p.lat, lng: p.lng, route: r, point: p, speed_kmh: spd, stop_score: null};
        st.headway_minutes = aggregateHeadway(headwaysAt(p.lat, p.lng, r));
        if (spd && st.headway_minutes) {
          st.stop_score = window.__jsStopScore(scoringMode(r.mode), spd, st.headway_minutes, HEADWAY_CV);
        }
        // Item 4: stash the computed score directly on the route's own point
        // object (not just this transient `stations` array) -- `stationMarker`
        // reads `p.__stopScore` from there so a marker rebuilt at ANY later
        // time (route re-render, re-entering edit mode, etc.) keeps showing
        // its last-computed color without needing `lastCompute` at hand.
        p.__stopScore = st.stop_score;
        stations.push(st);
      });
    });
    res.n_stations = stations.length;
    var usable = stations.filter(function(s) { return s.stop_score != null && isFinite(s.stop_score); });
    res.n_stations_scored = usable.length;
    res.stations = stations;
    if (!usable.length) {
      res.reason = stations.length
        ? 'stations need a headway and a grade separation on every segment'
        : 'no stations yet';
      lastCompute = res; renderSummary(); return Promise.resolve(res);
    }

    return loadChunkIndex().then(function(idx) {
      if (!idx) {
        res.reason = 'population chunks not available (run pop_chunks generator)';
        lastCompute = res; renderSummary(); return res;
      }
      // Which chunk squares the union of the stations' buffers covers.
      var keys = {};
      usable.forEach(function(s) {
        var dLat = ACCESS_R_M / 111320.0;
        var dLng = ACCESS_R_M / (111320.0 * Math.max(0.05, Math.cos(s.lat * Math.PI / 180)));
        var i0 = Math.floor((s.lat - dLat) / PC.deg), i1 = Math.floor((s.lat + dLat) / PC.deg);
        var j0 = Math.floor((s.lng - dLng) / PC.deg), j1 = Math.floor((s.lng + dLng) / PC.deg);
        for (var i = i0; i <= i1; i++) for (var j = j0; j <= j1; j++) {
          var k = i + '_' + j;
          if (idx[k]) keys[k] = 1;
        }
      });
      var keyList = Object.keys(keys);
      res.n_chunks = keyList.length;
      res.chunk_keys = keyList;
      // Item 6: fetch the matching GEOID chunks (same keys, see `loadGeoidChunk`)
      // alongside the population chunks -- bounded by the same "just the
      // drawn area" locality, never the whole metro. `loadGeoidChunkIndex()`
      // degrades to `null` (no fetches) for a study with no census chunks.
      var geoidChunksReady = loadGeoidChunkIndex().then(function(gIdx) {
        if (!gIdx) return [];
        return Promise.all(keyList.filter(function(k) { return gIdx[k]; }).map(loadGeoidChunk));
      });
      return Promise.all([Promise.all(keyList.map(loadChunk)), geoidChunksReady]).then(function(both) {
        var chunks = both[0];
        var speedMps = window.__disc.walkingSpeedMps;
        var impact = 0, nBuf = 0, nImp = 0;
        var sample = null;
        // h3_cell -> recomputed level_of_service for every chunk row this
        // scenario's stations reach, whether or not that station raised the
        // cell's score. Consumed by the stats panel's `getStatsData` (see
        // `_stats_panel_js`) to swap recomputed values into the
        // regression/R^2/ANOVA math for the active scenario's metro/core
        // rows -- joined on `h3_cell`, which `_stats_json_data` includes in
        // `window.__statsData` for exactly this purpose. equity_flag /
        // development percentages are never read through this map, so they
        // stay pinned to the static baseline per spec.
        var h3AccessOverrides = {};
        chunks.forEach(function(rows) {
          for (var ci = 0; ci < rows.length; ci++) {
            var row = rows[ci];                       // [h3, lat, lng, pop, access]
            var lat = row[1], lng = row[2], pop = row[3], oldA = row[4];
            var best = oldA, inBuf = false, bestStop = null, bestD = null;
            for (var si = 0; si < usable.length; si++) {
              var s = usable[si];
              var d = haversineM(lat, lng, s.lat, s.lng);
              if (d > ACCESS_R_M) continue;
              inBuf = true;
              // accessibility_score = stop_score * D(walk_time), exactly
              // accessibility_score.py -- with walk_time derived from the
              // straight-line distance at the region's own walking speed
              // (`parameters.walking_speed_mps`, via window.__disc), NOT a
              // street-network distance. That simplification is the explicit
              // instruction for this what-if tool.
              var a = s.stop_score * window.__jsWalkDecay(d / speedMps / 60.0);
              // "Best nearby stop wins": accessibility_score is a per-cell
              // max over stops, so a new stop only ever raises a cell's score
              // (never lowers it), and the pre-edit value stands in for every
              // stop of the real network already serving that cell.
              if (a > best) { best = a; bestStop = s; bestD = d; }
            }
            if (!inBuf) continue;
            nBuf += 1;
            h3AccessOverrides[row[0]] = {access: best, gainBase: oldA, population: pop};
            if (best > oldA) {
              nImp += 1;
              impact += (best - oldA) * pop;
              if (!sample || (best - oldA) > (sample.gain)) {
                sample = {h3_cell: row[0], lat: lat, lng: lng, population: pop,
                          old_access: oldA, new_access: best, gain: best - oldA,
                          distance_m: bestD, walk_minutes: bestD / speedMps / 60.0,
                          stop_score: bestStop ? bestStop.stop_score : null,
                          stop_headway_minutes: bestStop ? bestStop.headway_minutes : null,
                          stop_speed_kmh: bestStop ? bestStop.speed_kmh : null};
              }
            }
          }
        });
        res.n_cells_in_buffer = nBuf;
        res.n_cells_improved = nImp;
        res.impact = impact;
        res.h3AccessOverrides = h3AccessOverrides;
        res.max_gain_cell = sample;
        res.new_access = base.access + impact / base.population;
        res.total_population = base.population;
        res.impact_per_musd = (res.cost_musd > 0) ? (impact / res.cost_musd) : null;
        res.ok = true;
        lastCompute = res;
        if (S.activeScenario) computeResultsByScenario[S.activeScenario] = res;
        // Item 5: feed the base map's own hexagon/circle fill color the same
        // recomputed values the stats panel already reads (see
        // `_H3_OVERRIDE_LOOKUP_JS`) -- a plain global rather than something
        // threaded through Leaflet, since the style functions are ordinary
        // JS closures with no other channel back to this one. Left in place
        // (not cleared) after a failed Compute or a route edit that hasn't
        // been re-Computed yet, same staleness tolerance `lastCompute`
        // itself already has.
        window.__h3AccessOverrides = h3AccessOverrides;
        redrawH3Shapes();
        // Item 6: derive GEOID-level deltas from the exact same
        // `h3AccessOverrides` just computed -- for every touched h3 cell,
        // if a fetched geoid chunk says which GEOID it belongs to, add its
        // population-weighted (new - old) contribution to that GEOID's
        // running sum. `_census_geoid_override_lookup_js` (build.py) divides
        // this sum by the polygon's OWN total population and adds it to the
        // baked `level_of_service` -- i.e. only the touched cells' share of the
        // population-weighted mean is updated; untouched member cells keep
        // contributing their original baked share implicitly, since they
        // never appear in this sum. See `pop_chunks.build_geoid_chunks_from_h3_and_census`'s
        // docstring for the one simplification this makes (no nearest-match
        // fallback at polygon edges).
        var geoidAccessOverrides = {};
        Object.keys(geoidChunkCache).forEach(function(k) {
          var m = geoidChunkCache[k];
          for (var cellId in m) {
            var ov = h3AccessOverrides[cellId];
            if (!ov || ov.access == null || ov.gainBase == null) continue;
            var gid = m[cellId];
            var pop = (ov.population != null && ov.population >= 0) ? ov.population : 0;
            geoidAccessOverrides[gid] = (geoidAccessOverrides[gid] || 0) + (ov.access - ov.gainBase) * pop;
          }
        });
        window.__geoidAccessOverrides = geoidAccessOverrides;
        redrawCensusShapes();
        // Item 4: `p.__stopScore` was just (re)written on every station point
        // above -- rebuild the markers so the new stop-score tint actually
        // shows. Both the always-visible "other routes" layer and (if one is
        // active) the highlighted edit-mode route need it; `renderActiveRoute`
        // is declared further down this closure but hoisted, so it's callable
        // here.
        renderOtherRoutes();
        if (S.active && S.activeRouteId) renderActiveRoute();
        renderSummary();
        return res;
      });
    });
  }

  // Only the h3_cell-keyed shape layers can possibly change color from
  // `window.__h3AccessOverrides` (see `_H3_OVERRIDE_LOOKUP_JS` -- census is
  // GEOID-keyed and silently no-ops, streets/development have their own
  // unrelated style functions) -- redrawing those two is enough to pick up
  // a fresh Compute's overrides, without the cost of also redrawing every
  // other vector-tile layer on the page.
  function redrawH3Shapes() {
    if (!window.__vgLayers) return;
    Object.keys(window.__vgLayers).forEach(function(k) {
      if (k.indexOf('hexagons:') !== 0 && k.indexOf('circles:') !== 0) return;
      if (window.__vgLayers[k].redraw) window.__vgLayers[k].redraw();
    });
  }

  // Item 6: the census "blockgroup" level's style function is the only one
  // that reads `window.__geoidAccessOverrides` (see `CENSUS_LIVE_RECOLOR_LEVEL`
  // in build.py); other census levels' style function is the old GEOID-agnostic
  // no-op, but redrawing all `census:*` layers is cheap and simplest, and a
  // no-op style change is a harmless repaint for the other levels.
  function redrawCensusShapes() {
    if (!window.__vgLayers) return;
    Object.keys(window.__vgLayers).forEach(function(k) {
      if (k.indexOf('census:') !== 0) return;
      if (window.__vgLayers[k].redraw) window.__vgLayers[k].redraw();
    });
  }

  function fmtNum(v, dp) {
    if (v == null || !isFinite(v)) return '-';
    return v.toLocaleString(undefined, {minimumFractionDigits: dp, maximumFractionDigits: dp});
  }

  // The one-line top-center bar. Order (spec): name, level of service, then --
  // custom scenarios only -- cost, impact, impact/$M and the emoji. Labels
  // abbreviated so it fits on one line.
  function renderSummary() {
    var barName = $('editSummaryName'), rest = $('editSummaryRest');
    if (!barName || !rest) return;
    var shown = S.activeScenario || RESERVED;
    barName.textContent = shown;
    var base = baselineStats();
    var parts = [];
    function push(k, v, cls) {
      // Empty `k` (used for the top-line score, item 3: "instead of access
      // xx I would like to have just the number xxx") omits the label span
      // entirely -- just the separator dot and the bare value.
      parts.push('<span class="sep">&middot;</span>' + (k ? '<span class="k">' + k + '</span> ' : '') +
                 '<span class="v' + (cls ? ' ' + cls : '') + '">' + v + '</span>');
    }
    if (isReserved(shown)) {
      // The real network: level of service only -- there is nothing to cost and
      // no impact to attribute, since it is not editable.
      push('', base ? fmtNum(base.access, 2) : '-');
      rest.innerHTML = parts.join('');
      rest.title = base ? ('Population-weighted median transit level of service over the metro study area (' +
                           fmtNum(base.population, 0) + ' people)') : '';
      return;
    }
    var sc = activeScenarioObj();
    var stale = !lastCompute || lastCompute.signature !== scenarioSignature(sc);
    var c = scenarioCostMusd(sc);
    var acc = (!stale && lastCompute && lastCompute.new_access != null) ? lastCompute.new_access
              : (base ? base.access : null);
    push('', acc == null ? '-' : fmtNum(acc, 2));
    if (!stale && lastCompute && lastCompute.ok && base) {
      var dv = lastCompute.new_access - base.access;
      parts.push(' <span class="k">(' + (dv >= 0 ? '+' : '') + fmtNum(dv, 2) + ')</span>');
    }
    push('cost', fmtMusd(c.total_musd) + (c.partial ? '*' : ''));
    if (stale) {
      parts.push('<span class="sep">&middot;</span><span class="stale">press Compute</span>');
      rest.innerHTML = parts.join('');
      rest.title = 'The scenario changed since the last Compute.';
      return;
    }
    if (!lastCompute.ok) {
      parts.push('<span class="sep">&middot;</span><span class="stale">' + lastCompute.reason + '</span>');
      rest.innerHTML = parts.join('');
      rest.title = lastCompute.reason;
      return;
    }
    // Impact is population-weighted, and the res-11 grid's per-cell
    // population is fractional (dasymetric), so a small scenario's impact can
    // legitimately be < 1 -- rounding it to a whole number there would print a
    // bare "0" for a real, non-zero result.
    push('impact', fmtNum(lastCompute.impact, Math.abs(lastCompute.impact) < 100 ? 1 : 0));
    var ipm = lastCompute.impact_per_musd;
    push('impact/$M', ipm == null ? '-' : fmtNum(ipm, 1));
    var e = impactEmoji(ipm);
    if (e.emoji) parts.push('<span class="emoji">' + e.emoji + '</span>');
    rest.innerHTML = parts.join('');
    rest.title = 'score = metro population-weighted median transit level of service (baseline median plus this scenario\'s population-weighted delta)' +
      ' | impact = sum over affected res-' + PC.res + ' cells of (new - old) access x population' +
      ' | impact/$M = impact per million USD' +
      (e.score01 == null ? '' : ' | impact_score (0-1, linear between the sad/happy thresholds) = ' + e.score01.toFixed(2)) +
      ' | ' + lastCompute.n_cells_improved + ' cells improved, ' + lastCompute.n_chunks + ' population chunks fetched';
  }
  window.__renderEditSummary = renderSummary;

  function runCompute() {
    var st = $('editorComputeStatus');
    if (st) st.textContent = 'Computing...';
    var btn = $('editorCompute');
    if (btn) btn.disabled = true;
    return computeAccess().then(function(r) {
      if (btn) btn.disabled = false;
      if (!st) return r;
      st.textContent = r.ok
        ? (r.n_stations_scored + ' stations, ' + r.n_chunks + ' chunks, ' +
           r.n_cells_improved.toLocaleString() + ' cells improved')
        : ('Not computed: ' + r.reason);
      return r;
    }).catch(function(err) {
      if (btn) btn.disabled = false;
      if (st) st.textContent = 'Compute failed: ' + err;
      return null;
    });
  }

  // --- enter / exit edit mode --------------------------------------------
  // Redraw of the vector-tile shape layers. `_control_panel_js`'s own
  // `__redrawMatching` is closure-local to that script's `load` handler and
  // therefore unreachable from here (see `__fmtLegendNum`'s comment on why
  // cross-script helpers must live on `window`) -- so this block keeps its
  // own two-line copy rather than reaching for it.
  function redrawShapes() {
    if (!window.__vgLayers) return;
    Object.keys(window.__vgLayers).forEach(function(k) {
      if (k.indexOf('streets:') === 0 || k.indexOf('development:') === 0) return;
      if (window.__vgLayers[k].redraw) window.__vgLayers[k].redraw();
    });
  }

  // Every Leaflet.VectorGrid tile canvas writes `pointerEvents = 'auto'` onto
  // ITSELF (`L.Canvas.Tile.initialize`), and whichever canvas is topmost at a
  // pixel receives the DOM click and discards it if it finds no feature --
  // exactly the bug `__makeStreetsClickThrough` exists to work around for the
  // streets layer. In edit mode the same thing happened to the hexagon/
  // circle/census canvas: neither Leaflet's `map click` nor Geoman's
  // `pm:vertexadded` ever fired, so drawing was silently impossible
  // (reproduced headless: `document.elementFromPoint` returned the shape
  // canvas with `pointer-events: auto`). While editing, ALL vector-tile
  // canvases opt out of hit-testing so clicks reach the map/drawing layer;
  // shape popups are not wanted mid-edit anyway. A CSS class on the map
  // container (not per-canvas JS) so lazily-created tiles are covered too,
  // and `!important` to beat VectorGrid's inline value.
  //
  // The same applies to the two other interactive overlays a click can land
  // on: the real transit-line polylines (their own `routeLinesPane_*` panes,
  // SVG paths that DO consume clicks) and the stop markers / cluster bubbles
  // in `markerPane`. Headless probing of `document.elementFromPoint` at the
  // click coordinates showed both stealing points mid-draw, so both go
  // click-through while editing as well. The stop layers are targeted by
  // their OWN classes (`stop-cluster-icon` / `stop-marker-icon` / the two
  // annotation classes, all from `_stops_layer_block`) rather than by
  // "everything in markerPane": Geoman's vertex markers live in that same
  // pane and must stay clickable/draggable for "Move node", and they also
  // carry Leaflet's generic `leaflet-marker-icon` class, so any selector
  // written in terms of that class hits both (an earlier attempt did
  // exactly that and silently disabled nothing). Descendants are listed
  // explicitly because Leaflet.markercluster's inner div sets its own
  // `pointer-events`, which an inherited value from the icon cannot
  // override -- headless probing found a cluster's INNER div receiving the
  // click and firing `zoomToBoundsOnClick`, jumping the map two zoom
  // levels mid-edit.
  var __editClickThroughStyle = document.createElement('style');
  __editClickThroughStyle.textContent = [
    '.transitlos-editing .leaflet-tile-pane canvas.leaflet-tile,',
    '.transitlos-editing .leaflet-pane[class*="routeLines"],',
    // NB the selector is `routeLines`, not `routeLinesPane`: Leaflet's
    // `createPane('routeLinesPane_bus')` STRIPS the literal "Pane" from the
    // name when building the class, yielding
    // `leaflet-routeLines_bus-pane`.
    // leaflet.css gives every interactive SVG path an explicit
    // `pointer-events: auto`, which an inherited `none` from its pane div
    // cannot override -- the real transit lines kept eating draw clicks
    // until the paths themselves were targeted.
    '.transitlos-editing .leaflet-pane[class*="routeLines"] path,',
    '.transitlos-editing .stop-cluster-icon, .transitlos-editing .stop-cluster-icon *,',
    '.transitlos-editing .stop-marker-icon, .transitlos-editing .stop-marker-icon *,',
    '.transitlos-editing .stop-name-anno, .transitlos-editing .stop-routes-anno',
    '  { pointer-events: none !important; }'
  ].join('\n');
  document.head.appendChild(__editClickThroughStyle);

  // Geoman toggles the map's own interaction handlers while a draw is in
  // progress and RE-ENABLES doubleClickZoom when the draw ends, undoing the
  // one-shot disable done on entering edit mode -- so this is re-applied
  // after every draw and on every mode change, not just once.
  function suspendDblClickZoom() {
    if (S.active && mapRef.doubleClickZoom && mapRef.doubleClickZoom.enabled()) {
      mapRef.doubleClickZoom.disable();
    }
  }

  function enterEditMode() {
    if (S.active) return;
    S.active = true;
    $('editModeButton').classList.add('active');
    mapRef.getContainer().classList.add('transitlos-editing');
    // Geoman finishes a line on double-click, and that same double-click
    // otherwise ALSO fires Leaflet's doubleClickZoom -- the map jumped a
    // zoom level the instant a route was finished, dragging every node out
    // from under the cursor (caught headless: nodes ended up at negative
    // container coordinates right after drawing). Suspended for the whole
    // edit session and restored on exit.
    S.restore.dblClickZoom = !!(mapRef.doubleClickZoom && mapRef.doubleClickZoom.enabled());
    suspendDblClickZoom();
    editLayer.addTo(mapRef);

    // (a) streets off, remembering whether they were on. Driven through the
    // checkbox's own change event so the single source of truth for that
    // layer stays `_control_panel_js`'s handler.
    var streetsCb = $('streetsCheckbox');
    S.restore.streetsWasOn = !!(streetsCb && streetsCb.checked);
    if (streetsCb && streetsCb.checked) {
      streetsCb.checked = false;
      streetsCb.dispatchEvent(new Event('change'));
    }
    // (b) halve whatever shape opacity is currently in effect. This is the
    // global multiplier the "Hex/circle/census opacity" slider drives; the
    // data-driven opacity-by-field/contrast system multiplies on top of it
    // inside `opacityFromField`, so a 0.5x here halves the *rendered*
    // opacity of every shape feature whatever the field settings are.
    var shapeSel = $('shapeSelect');
    var shapeVisible = !shapeSel || shapeSel.value !== 'none';
    S.restore.shapeOpacity = (typeof window.__shapeOpacity === 'number') ? window.__shapeOpacity : 1;
    S.restore.shapeHalved = false;
    if (shapeVisible) {
      window.__shapeOpacity = S.restore.shapeOpacity * 0.5;
      S.restore.shapeHalved = true;
      var lbl = $('shapeOpacityValue');
      if (lbl) lbl.textContent = Math.round(window.__shapeOpacity * 100) + '%';
      redrawShapes();
    }

    if (S.activeScenario) { $('editorPanel').style.display = 'block'; refreshPanel(); }
    else openScenarioModal();
  }

  function exitEditMode() {
    if (!S.active) return;
    setMode('idle');
    S.active = false;
    $('editModeButton').classList.remove('active');
    mapRef.getContainer().classList.remove('transitlos-editing');
    if (S.restore.dblClickZoom && mapRef.doubleClickZoom) mapRef.doubleClickZoom.enable();
    $('editorPanel').style.display = 'none';
    $('editorScenarioModal').style.display = 'none';
    // The summary bar is NOT hidden here: the spec wants it visible in AND
    // out of edit mode. Outside edit mode it reports whatever scenario is
    // still selected, or the reserved "current" one.
    renderSummary();
    mapRef.removeLayer(editLayer);
    renderOtherRoutes();   // the just-active route rejoins the plain always-visible layer

    var streetsCb = $('streetsCheckbox');
    if (S.restore.streetsWasOn && streetsCb && !streetsCb.checked) {
      streetsCb.checked = true;
      streetsCb.dispatchEvent(new Event('change'));
    }
    if (S.restore.shapeHalved) {
      window.__shapeOpacity = S.restore.shapeOpacity;
      var lbl = $('shapeOpacityValue');
      if (lbl) lbl.textContent = Math.round(window.__shapeOpacity * 100) + '%';
      redrawShapes();
    }
  }

  // --- scenario modal -----------------------------------------------------
  function openScenarioModal() {
    var sel = $('editorScenarioSelect');
    sel.innerHTML = '';
    // The reserved "current" scenario always exists as a read-only baseline.
    var opt = document.createElement('option');
    opt.value = RESERVED;
    opt.textContent = RESERVED + " (today's network, read-only)";
    sel.appendChild(opt);
    S.scenarios.forEach(function(sc) {
      var o = document.createElement('option');
      o.value = sc.name; o.textContent = sc.name;
      sel.appendChild(o);
    });
    // First time (no user scenarios yet) the picker is hidden entirely: the
    // only ways out are naming a new scenario or cancelling out of edit mode.
    $('editorPickExisting').style.display = S.scenarios.length ? 'block' : 'none';
    $('editorScenarioModalTitle').textContent = S.scenarios.length ? 'Choose a scenario' : 'Name your first scenario';
    $('editorScenarioError').textContent = '';
    $('editorNewScenarioName').value = '';
    $('editorScenarioModal').style.display = 'flex';
    $('editorPanel').style.display = 'none';
  }

  function createScenario() {
    var name = ($('editorNewScenarioName').value || '').trim();
    var err = $('editorScenarioError');
    if (!name) { err.textContent = 'Please enter a scenario name.'; return; }
    if (isReserved(name)) { err.textContent = '"' + RESERVED + '" is reserved for the real network.'; return; }
    if (scenarioByName(name)) { err.textContent = 'A scenario with that name already exists.'; return; }
    S.scenarios.push({name: name, routes: []});
    S.activeScenario = name;
    S.activeRouteId = null;
    $('editorScenarioModal').style.display = 'none';
    // Creating a scenario (unlike merely switching to one, item 9) is
    // inherently an editing action -- entering edit mode here is the
    // intended behavior, not a side effect to avoid.
    if (!S.active) { enterEditMode(); return; }
    $('editorPanel').style.display = 'block';
    refreshPanel();
  }

  function continueScenario() {
    S.activeScenario = $('editorScenarioSelect').value;
    S.activeRouteId = null;
    $('editorScenarioModal').style.display = 'none';
    // Item 9: switching scenarios is a pure view change -- it must not
    // itself flip `S.active`/enter edit mode. If edit mode is already on,
    // behave as before (open the panel on the newly-selected scenario); if
    // not, just re-render the view (summary bar + always-visible routes)
    // for the new scenario and leave edit mode alone.
    if (S.active) {
      $('editorPanel').style.display = 'block';
      refreshPanel();
    } else {
      renderSummary();
      renderOtherRoutes();
    }
  }

  // --- panel rendering ----------------------------------------------------
  function refreshPanel() {
    var readOnly = isReserved(S.activeScenario);
    // Item 10: the scenario name itself is clickable and opens the same
    // switcher as the "Switch scenario" button.
    $('editorScenarioLine').innerHTML = 'Scenario: <b id="editorScenarioNameClick" style="cursor:pointer;text-decoration:underline dotted;" title="Click to switch scenario">' +
      (S.activeScenario || '-') + '</b>' + (readOnly ? ' <span style="color:#888;">(read-only)</span>' : '');
    var nameEl = $('editorScenarioNameClick');
    if (nameEl) nameEl.addEventListener('click', function() { setMode('idle'); openScenarioModal(); });
    renderSummary();
    var list = $('editorRouteList');
    list.innerHTML = '';
    var sc = activeScenarioObj();
    if (sc) {
      sc.routes.forEach(function(r) { list.appendChild(routeRow(r)); });
      if (!sc.routes.length) list.innerHTML = '<div style="color:#888;">No routes yet.</div>';
    }
    // Real-network routes, listed read-only for context from the data the
    // transit-lines overlay already publishes (`window.__routesData`). They
    // carry no Edit/Delete: only user-created routes are ever editable.
    var real = (window.__routesData && window.__routesData.routes) || [];
    if (real.length) {
      var d = document.createElement('div');
      d.className = 'editor-route-row readonly';
      d.id = 'editorRealRouteCount';
      d.style.marginTop = '4px';
      d.textContent = 'Current network: ' + real.length + ' routes (read-only)';
      list.appendChild(d);
    }
    $('editorNewRoute').style.display = readOnly ? 'none' : 'inline-block';
    $('editorRouteEditor').style.display = (S.activeRouteId && !readOnly) ? 'block' : 'none';
    renderActiveRoute();
    refreshGeomButtons();
    renderOtherRoutes();
  }

  function routeRow(r) {
    var row = document.createElement('div');
    row.className = 'editor-route-row';
    var sw = document.createElement('span');
    sw.className = 'swatch'; sw.style.background = r.color;
    var nm = document.createElement('span');
    nm.className = 'name';
    nm.textContent = r.name + ' (' + r.mode + ', ' + r.points.length + ' pts)';
    row.appendChild(sw); row.appendChild(nm);
    if (r.user_created) {
      var e = document.createElement('button');
      e.textContent = 'Edit'; e.setAttribute('data-edit-route', r.id);
      e.addEventListener('click', function() { selectRoute(r.id); });
      var d = document.createElement('button');
      d.textContent = 'Delete'; d.setAttribute('data-delete-route', r.id);
      d.addEventListener('click', function() { deleteRoute(r.id); });
      row.appendChild(e); row.appendChild(d);
    } else {
      row.classList.add('readonly');
    }
    return row;
  }

  function newRoute() {
    if (isReserved(S.activeScenario)) return;
    var sc = activeScenarioObj();
    if (!sc) return;
    S._routeCounter += 1;
    var mode = 'bus';
    var r = {
      id: 'r' + S._routeCounter,
      name: 'Route ' + S._routeCounter,
      mode: mode,
      color: (MODES[mode] || {}).default_color || '#8b0000',
      headway_minutes: 10,
      user_created: true,
      points: [],
      segments: [],
      cost_musd: null,
      _colorTouched: false
    };
    sc.routes.push(r);
    selectRoute(r.id);
  }

  function selectRoute(id) {
    setMode('idle');
    S.activeRouteId = id;
    var r = activeRoute();
    if (r) {
      $('editRouteName').value = r.name;
      $('editRouteMode').value = r.mode;
      $('editRouteColor').value = r.color;
      $('editRouteHeadway').value = r.headway_minutes;
    }
    refreshPanel();
  }

  function deleteRoute(id) {
    var sc = activeScenarioObj();
    if (!sc) return;
    sc.routes = sc.routes.filter(function(r) { return !(r.id === id && r.user_created); });
    if (S.activeRouteId === id) S.activeRouteId = null;
    setMode('idle');
    refreshPanel();
  }

  // --- geometry rendering -------------------------------------------------
  function clearGeom() {
    editLayer.clearLayers();
    routeLine = null;
    vertexMarkers = [];
    segLayers = [];
  }

  function renderActiveRoute() {
    clearGeom();
    var r = activeRoute();
    // Grade-separation panel is independent of whether the route has any
    // geometry yet -- it must auto-show (default highlighted) the instant
    // draw/extend mode starts, even for a brand-new route with zero points,
    // so it can't sit behind the `!r.points.length` early return below.
    renderGradeSepPanel();
    if (!r || !r.points.length) return;
    var latlngs = r.points.map(function(p) { return [p.lat, p.lng]; });
    if (latlngs.length >= 2) {
      // The edited route always reads as "the one being edited": a black
      // casing underneath (weight 9) with a thinner colored line on top,
      // in its own pane above the real network (see `editorPane`, z 460).
      // What the inner line is colored BY is the mode switch: in
      // grade-separation mode each segment takes its grade-separation color
      // (unset = gray dashed), otherwise the whole route is just route.color.
      editLayer.addLayer(L.polyline(latlngs, {
        color: '#000', weight: 9, opacity: 0.9, pane: 'editorPane', interactive: false
      }));
      routeLine = L.polyline(latlngs, {color: r.color, weight: 5, opacity: 0.95, pane: 'editorPane'});
      routeLine.on('click', onLineClick);
      editLayer.addLayer(routeLine);
      if (S.mode === 'gradesep') {
        // One clickable polyline per segment, drawn on top of `routeLine`
        // so a click lands on a specific segment; `routeLine` itself stays
        // in place (Geoman's "Move node" is attached to it) but is visually
        // covered here.
        for (var si = 0; si < r.points.length - 1; si++) {
          var gs = r.segments[si] ? r.segments[si].grade_separation : null;
          var seg = L.polyline([latlngs[si], latlngs[si + 1]], {
            color: gradeSepColor(gs),
            weight: (si === selectedSeg) ? 8 : 5,
            opacity: 1,
            dashArray: gs ? null : GS_UNSET_DASH,
            pane: 'editorPane'
          });
          seg.__seg = si;
          seg.on('click', onSegmentClick);
          editLayer.addLayer(seg);
          segLayers.push(seg);
        }
      }
    }
    r.points.forEach(function(p, i) {
      var m;
      if (p.is_station) {
        // Item 7: the active route's stations get the same real mode icon
        // as everywhere else -- never a generic dot.
        m = stationMarker(p, r);
        m.setZIndexOffset && m.setZIndexOffset(1000);
      } else {
        // Item 6: a plain (non-station) node is a small black dot, and ONLY
        // ever drawn here -- on the currently-active, edit-mode route. No
        // other layer (`renderOtherRoutes`) draws plain-node markers at all.
        m = L.circleMarker([p.lat, p.lng], {radius: 4, color: '#000', weight: 1, fillColor: '#000', fillOpacity: 1, pane: 'editorPane'});
      }
      m.__idx = i;
      m.on('click', onVertexClick);
      editLayer.addLayer(m);
      vertexMarkers.push(m);
    });
    if (S.mode === 'move') startMove();   // re-attach Geoman after a re-render
    var nSt = r.points.filter(function(p) { return p.is_station; }).length;
    $('editorRouteStats').textContent = r.points.length + ' nodes, ' + nSt + ' stations';
    renderGradeSepPanel();
    renderCost();
  }

  // --- grade-separation panel + live cost readout -------------------------
  function fmtMusd(v) {
    if (v >= 1000) return '$' + (v / 1000).toFixed(2) + 'B';
    return '$' + (Math.round(v * 10) / 10) + 'M';
  }

  function renderCost() {
    var r = activeRoute();
    var el = $('editorCostReadout');
    if (!el) return;
    if (!r || r.points.length < 2) { el.innerHTML = ''; return; }
    var c = routeCost(r);
    // Cached onto the route so export/import and Phase 4's impact-per-$
    // read the same number the panel shows. Null while still partial.
    r.cost_musd = c.partial ? null : c.total_musd;
    r.cost_breakdown = c;
    var head = '<b>Route cost: ' + fmtMusd(c.total_musd) + '</b>';
    if (c.partial) {
      head = '<b>Route cost (partial): ' + fmtMusd(c.total_musd) + '</b>' +
        '<span style="color:#c0392b;"> &mdash; ' + c.n_costed + ' of ' + c.n_segments +
        ' segments costed</span>';
    }
    var vehicleLine = (c.vehicles_needed != null)
      ? (c.vehicles_needed + ' vehicle' + (c.vehicles_needed === 1 ? '' : 's') +
         (c.vehicle_fleet_musd != null ? ' (' + fmtMusd(c.vehicle_fleet_musd) + ')' : '') +
         ' &middot; round trip ' + Math.round(c.round_trip_minutes) + ' min')
      : 'vehicles: set headway to estimate';
    el.innerHTML = head +
      '<div style="color:#666;">' + c.length_km.toFixed(2) + ' km &middot; track ' +
      fmtMusd(c.segments_musd) + ' + stations ' + fmtMusd(c.stations_musd) +
      ' (' + c.n_stations_costed + '/' + c.n_stations + ')</div>' +
      '<div style="color:#666;">' + vehicleLine + '</div>';
  }

  // Single grade-separation control, serving two contexts (replaces the
  // old separate `#editDefaultGradeSep` <select>, deleted as a duplicate):
  //
  //  - "default" mode, while DRAWING a new route or EXTENDING an existing
  //    one (`S.mode === 'draw' || 'extend'`): auto-shown, highlights the
  //    route's current default grade separation, and clicking a button
  //    changes that default live (mirrors what the old dropdown did --
  //    newly-added segments pick it up, and any segment still at the OLD
  //    default is retargeted to the new one; explicit per-segment choices
  //    are left alone).
  //  - "per-segment" mode, while in the dedicated "Grade separation" mode
  //    (`S.mode === 'gradesep'`) with a segment selected: unchanged from
  //    before -- clicking a button assigns just that segment.
  function isDefaultGradeSepMode() {
    return S.mode === 'draw' || S.mode === 'extend';
  }
  // Item 4: outside draw/extend, grade separation is assigned with a
  // "paint bucket" flow -- arm a grade-separation button, then either click
  // segments on the map one at a time or hit "Apply to all route". `armedGradeSep`
  // is the currently-armed value (null = nothing armed); it is reset whenever
  // gradesep mode is entered/left (see `setMode`).
  var armedGradeSep = null;
  function renderGradeSepPanel() {
    var wrap = $('editorGradeSepWrap');
    if (!wrap) return;
    var r = activeRoute();
    var defaultMode = !!(r && isDefaultGradeSepMode());
    var segMode = !!(r && S.mode === 'gradesep');
    var on = defaultMode || segMode;
    wrap.style.display = on ? 'block' : 'none';
    wrap.className = defaultMode ? 'default-mode' : (segMode ? 'paint-mode' : '');
    var applyAllBtn = $('editorGradeSepApplyAll');
    if (applyAllBtn) applyAllBtn.style.display = segMode ? 'block' : 'none';
    if (!on) return;
    var opts = gradeSepOptions(r.mode);
    var host = $('editorGradeSepButtons');
    host.innerHTML = '';
    // In default mode, the highlighted button is the route's live default.
    // In paint mode, the highlighted button is whatever is currently ARMED
    // (not any one segment's value -- segments can each hold different
    // values, so there is nothing single-valued to reflect there).
    var cur = defaultMode ? (r._defaultGradeSepOverride || defaultGradeSep(r.mode, r)) : armedGradeSep;
    opts.forEach(function(gs) {
      var b = document.createElement('button');
      var speed = computedSegmentSpeedKmh(r.mode, gs);
      b.innerHTML = gs + (speed != null ? '<span class="gradesep-btn-speed">' + speed + ' km/h</span>' : '');
      b.setAttribute('data-gradesep', gs);
      b.style.borderLeft = '6px solid ' + gradeSepColor(gs);
      if (gs === cur) b.className = 'on';
      b.addEventListener('click', function() { assignGradeSep(gs); });
      host.appendChild(b);
    });
    $('editorSegLabel').textContent = defaultMode
      ? 'default (applies to new nodes -- live while drawing)'
      : (armedGradeSep
        ? ('armed: ' + armedGradeSep + ' -- click segments to paint, or "Apply to all route"')
        : 'pick a grade separation, then click segments on the map');
    $('editorGradeSepLegend').textContent = 'Dashed gray = not assigned yet.';
    // Speed box: defaults to the computed value for this segment, and shows
    // whether the displayed number is computed or a manual override. Not
    // meaningful in "default" mode (no single segment is selected).
    var sp = $('editorSegSpeed'), note = $('editorSegSpeedNote');
    var seg = (!defaultMode && selectedSeg >= 0) ? r.segments[selectedSeg] : null;
    var computed = seg ? computedSegmentSpeedKmh(r.mode, seg.grade_separation) : null;
    var eff = seg ? effectiveSegmentSpeedKmh(r, selectedSeg) : null;
    sp.disabled = !seg;
    sp.value = (eff === null || eff === undefined) ? '' : eff;
    var overridden = !!(seg && typeof seg.speed_kmh_override === 'number');
    if (!seg) note.textContent = '';
    else if (computed === null) note.textContent = 'Assign a grade separation to get a computed speed (+' + dwellSecondsPerStop(r.mode) + ' s dwell/stop).';
    else note.innerHTML = (overridden ? '<b>Manual override</b> (computed: ' + computed + ' km/h)' : 'Computed from grade separation') +
      ' &middot; +' + dwellSecondsPerStop(r.mode) + ' s dwell/stop';
  }

  // Sets the route's live "default" grade separation (draw/extend mode).
  // Item 3: FORWARD ONLY -- this only changes what `syncSegments` hands to
  // segments created AFTER this point (via `defaultGradeSep(r.mode, r)`
  // reading `r._defaultGradeSepOverride`). It must NOT retroactively touch
  // any segment that already exists (even one that happens to already equal
  // the old default): each segment's `grade_separation` is a value recorded
  // at creation time, not something recomputed from "the current default" on
  // every render. (Earlier behavior here also retargeted already-placed
  // segments still at the old default -- that's exactly the retroactive
  // rewrite the user reported as wrong and this fix removes.)
  function setDefaultGradeSep(r, gs) {
    r._defaultGradeSepOverride = gs || null;
    renderActiveRoute();
  }

  // Item 4: outside draw/extend, clicking a grade-separation button ARMS it
  // ("paint bucket") rather than immediately assigning anything -- clicking
  // the same button again disarms it. Actual assignment happens via
  // `onSegmentClick` (click a segment on the map) or "Apply to all route".
  function assignGradeSep(gs) {
    var r = activeRoute();
    if (!r) return;
    if (isDefaultGradeSepMode()) { setDefaultGradeSep(r, gs); return; }
    armedGradeSep = (armedGradeSep === gs) ? null : gs;
    renderGradeSepPanel();
  }

  function applyGradeSepToSegment(r, i, gs) {
    if (!r.segments[i]) return;
    r.segments[i].grade_separation = gs;
    // A speed override belongs to the user, not the grade separation, so it
    // survives a re-assignment; the note in the panel makes clear the shown
    // number is no longer the computed one.
  }

  function onSegmentClick(e) {
    if (S.mode !== 'gradesep') return;
    L.DomEvent.stop(e);
    var r = activeRoute();
    selectedSeg = e.target.__seg;
    if (r && armedGradeSep) applyGradeSepToSegment(r, selectedSeg, armedGradeSep);
    renderActiveRoute();
  }

  function exitGradeSepPaint() {
    armedGradeSep = null;
    if (S.mode === 'gradesep') setMode('idle');
    else renderGradeSepPanel();
  }

  // --- modes --------------------------------------------------------------
  var GEOM_BUTTONS = {
    draw: 'editBtnDraw', move: 'editBtnMove', addnode: 'editBtnAddNode',
    delnode: 'editBtnDelNode', extend: 'editBtnExtend',
    addstation: 'editBtnAddStation', delstation: 'editBtnDelStation',
    gradesep: 'editBtnGradeSep'
  };
  var HINTS = {
    draw: 'Click the map to add points; click "Finish" when done.',
    move: 'Drag a node to move it. Click "Move node" again to stop.',
    addnode: 'Click ON the route line to insert a node there.',
    delnode: 'Click a node to delete it.',
    extend: 'Click the FIRST or LAST node to pick which end to extend from.',
    addstation: 'Click a node (or a point on the line) to make it a station.',
    delstation: 'Click a station to turn it back into a plain node.',
    gradesep: 'Click a SEGMENT (between two nodes), then pick its grade separation.'
  };

  function setMode(m) {
    // leaving the old mode
    if (S.mode === 'move' && routeLine && routeLine.pm) routeLine.pm.disable();
    extendFrom = null;
    var wasGradeSep = (S.mode === 'gradesep');
    var m0 = S.mode;
    S.mode = m || 'idle';
    // Leaving grade-separation paint mode always disarms it -- re-entering
    // starts from a clean, unarmed state rather than remembering the last
    // color (spec: needs "a clear way to exit paint mode").
    if (wasGradeSep && S.mode !== 'gradesep') armedGradeSep = null;
    if (S.mode !== 'gradesep') selectedSeg = -1;
    Object.keys(GEOM_BUTTONS).forEach(function(k) {
      var b = $(GEOM_BUTTONS[k]);
      if (b) b.className = (k === S.mode) ? 'on' : '';
    });
    setHint(HINTS[S.mode] || '');
    suspendDblClickZoom();
    // Item 1: while actively drawing OR extending, the toolbar shows ONLY
    // "Finish" and "Delete last node" -- every other geometry button
    // (Move/Add node/Delete node/Add station/Delete station, plus "Draw"
    // itself) and the outer "Done"/"Delete route" buttons are hidden until
    // drawing ends. The grade-separation panel is handled separately by
    // `renderGradeSepPanel` (auto-shown via `isDefaultGradeSepMode`).
    // `refreshGeomButtons` runs right after this and would otherwise re-show
    // "Draw" (or the has-geometry set) on top of this.
    var drawing = (S.mode === 'draw' || S.mode === 'extend');
    var finishBtn = $('editBtnFinishDraw');
    if (finishBtn) finishBtn.style.display = drawing ? 'inline-block' : 'none';
    var delLastBtn = $('editBtnDeleteLastNode');
    if (delLastBtn) delLastBtn.style.display = drawing ? 'inline-block' : 'none';
    var doneBtn = $('editorDoneRoute'), delRouteBtn = $('editorDeleteRoute');
    if (doneBtn) doneBtn.style.display = drawing ? 'none' : 'inline-block';
    if (delRouteBtn) delRouteBtn.style.display = drawing ? 'none' : 'inline-block';
    if (drawing) {
      Object.keys(GEOM_BUTTONS).forEach(function(k) { var b = $(GEOM_BUTTONS[k]); if (b) b.style.display = 'none'; });
    }
    if (S.mode === 'move') startMove();
    // Entering or leaving grade-separation mode changes how the route is
    // drawn (grade-separation colors vs. plain route.color) and whether the
    // assignment buttons show, so re-render on either transition. Entering
    // or leaving draw/extend also needs a re-render: the grade-separation
    // button panel now auto-shows (highlighting the current default) in
    // those modes too, not just in dedicated "Grade separation" mode.
    var wasDefaultMode = (m0 === 'draw' || m0 === 'extend');
    if (S.mode === 'gradesep' || wasGradeSep || isDefaultGradeSepMode() || wasDefaultMode) renderActiveRoute();
    if (!drawing) refreshGeomButtons();
  }

  function toggleMode(m) { setMode(S.mode === m ? 'idle' : m); }

  // Which buttons exist depends on whether the route has geometry yet: a
  // brand-new route offers ONLY "Draw" (per spec); everything else appears
  // once the route has a drawn line.
  function refreshGeomButtons() {
    var r = activeRoute();
    var has = !!(r && r.points.length >= 2);
    var drawBtn = $('editBtnDraw');
    drawBtn.style.display = has ? 'none' : 'inline-block';
    // Item 2: "Draw" is THE thing to do on a brand-new route -- give it the
    // same visual prominence as "Compute" (see `#editorCompute`/`.draw-prominent`)
    // whenever it's the offered action, i.e. whenever it's shown at all.
    drawBtn.className = has ? '' : 'draw-prominent';
    ['editBtnMove', 'editBtnAddNode', 'editBtnDelNode', 'editBtnExtend',
     'editBtnAddStation', 'editBtnDelStation', 'editBtnGradeSep'].forEach(function(id) {
      $(id).style.display = has ? 'inline-block' : 'none';
    });
  }

  // --- drawing --------------------------------------------------------
  // Item 3/4: a brand-new route is drawn with the exact same click-to-add
  // mechanism as EXTENDING an existing one (see the map `'click'` handler
  // below) rather than Leaflet-Geoman's own draw tool. Geoman only committed
  // points to `r.points`/`r.segments` when the shape finished (`pm:create`),
  // so `r.segments` did not exist yet while actively drawing -- meaning the
  // live grade-separation default control had nothing to apply to. Routing
  // new-route drawing through the same
  // incremental `r.points.push()` + `syncSegments()` path extend uses keeps
  // both flows visually and behaviorally identical, and keeps `r.segments`
  // live from the very first click.

  function startMove() {
    if (!routeLine || !routeLine.pm) { setHint('Draw the route first.'); return; }
    // `hideMiddleMarkers` keeps Geoman's edit mode strictly a MOVE tool --
    // inserting nodes is the separate "Add node" button, so the two
    // operations stay independent (and independently testable).
    routeLine.pm.enable({allowSelfIntersection: true, hideMiddleMarkers: true, removeLayerBelowMinVertexCount: false});
    routeLine.off('pm:markerdragend pm:edit', syncFromLine);
    routeLine.on('pm:markerdragend pm:edit', syncFromLine);
  }

  function syncFromLine() {
    var r = activeRoute();
    if (!r || !routeLine) return;
    var lls = routeLine.getLatLngs();
    // Station flags are preserved positionally (a move never changes the
    // node count or their order).
    r.points = lls.map(function(ll, i) {
      return {lat: ll.lat, lng: ll.lng, is_station: !!(r.points[i] && r.points[i].is_station)};
    });
    syncSegments(r);
    $('editorRouteStats').textContent = r.points.length + ' nodes, ' +
      r.points.filter(function(p) { return p.is_station; }).length + ' stations';
    // Moving a node changes segment lengths and therefore the route cost.
    renderCost();
  }

  // --- click handling -----------------------------------------------------
  function onVertexClick(e) {
    var r = activeRoute();
    if (!r) return;
    L.DomEvent.stop(e);
    var i = e.target.__idx;
    if (S.mode === 'delnode') {
      r.points.splice(i, 1);
      // Deleting node i merges its two adjacent segments (or drops the single
      // adjacent one at an endpoint): the surviving record keeps its own
      // assignment, and the merged-away one is removed IN PLACE so later
      // segments keep theirs (see the mirror-image comment in insertOnSegment).
      if (r.segments.length) r.segments.splice(Math.min(i, r.segments.length - 1), 1);
      syncSegments(r);
      selectedSeg = -1;
      refreshPanel();
    } else if (S.mode === 'extend') {
      // No branching, ever: only the two endpoints are valid extension
      // points, and a mid-route node is explicitly refused here.
      if (i === 0) { extendFrom = 'start'; setHint('Extending from the START. Click the map to add points.'); }
      else if (i === r.points.length - 1) { extendFrom = 'end'; setHint('Extending from the END. Click the map to add points.'); }
      else { extendFrom = null; setHint('Only the first or last node can be extended (no branching).'); }
    } else if (S.mode === 'addstation') {
      r.points[i].is_station = true;
      refreshPanel();
    } else if (S.mode === 'delstation') {
      // Deliberate choice: "Delete station" removes only the STATION status
      // and leaves the node in place -- removing geometry is what "Delete
      // node" is for, and conflating the two makes a misclick destructive.
      if (r.points[i].is_station) { r.points[i].is_station = false; refreshPanel(); }
      else setHint('That node is not a station.');
    }
  }

  function onLineClick(e) {
    var r = activeRoute();
    if (!r) return;
    if (S.mode === 'gradesep') {
      // Fallback for a click that lands on the underlying route line rather
      // than on one of the per-segment overlays: pick the nearest segment.
      L.DomEvent.stop(e);
      selectedSeg = nearestSegment(r, e.latlng);
      renderActiveRoute();
      return;
    }
    if (S.mode !== 'addnode' && S.mode !== 'addstation') return;
    L.DomEvent.stop(e);
    var ins = insertOnSegment(r, e.latlng);
    if (ins < 0) return;
    if (S.mode === 'addstation') r.points[ins].is_station = true;
    refreshPanel();
  }

  // Project `latlng` onto the closest segment of the route and splice a new
  // point in at that position, so "add node" always SPLITS an existing
  // segment rather than appending. Returns the inserted index (or -1).
  function closestOnRoute(r, latlng) {
    if (r.points.length < 2) return {idx: -1, point: null};
    var p = mapRef.latLngToLayerPoint(latlng);
    var best = -1, bestD = Infinity, bestPt = null;
    for (var i = 0; i < r.points.length - 1; i++) {
      var a = mapRef.latLngToLayerPoint(L.latLng(r.points[i].lat, r.points[i].lng));
      var b = mapRef.latLngToLayerPoint(L.latLng(r.points[i + 1].lat, r.points[i + 1].lng));
      var cp = L.LineUtil.closestPointOnSegment(p, a, b);
      var d = p.distanceTo(cp);
      if (d < bestD) { bestD = d; best = i; bestPt = cp; }
    }
    return {idx: best, point: bestPt};
  }
  // Phase 3: which segment a click is on (used by grade-separation mode).
  function nearestSegment(r, latlng) { return closestOnRoute(r, latlng).idx; }

  function insertOnSegment(r, latlng) {
    if (r.points.length < 2) return -1;
    var hit = closestOnRoute(r, latlng);
    var best = hit.idx, bestPt = hit.point;
    if (best < 0) return -1;
    var ll = mapRef.layerPointToLatLng(bestPt);
    r.points.splice(best + 1, 0, {lat: ll.lat, lng: ll.lng, is_station: false});
    // Phase 3: inserting a node SPLITS segment `best` in two, and both halves
    // inherit its grade separation (and speed override) -- without this the
    // extra segment record `syncSegments` appends would land at the END of
    // the array, silently shifting every later segment's assignment.
    var src = r.segments[best] || {grade_separation: null};
    r.segments.splice(best, 0, {
      grade_separation: src.grade_separation,
      speed_kmh_override: src.speed_kmh_override
    });
    syncSegments(r);
    return best + 1;
  }

  // Existing real transit stops (published by `_stops_layer_block` as
  // `window.__stopsData`) plus the stations of the scenario's other routes,
  // for the "you clicked on an existing stop -- stop there?" prompt.
  function nearbyStop(latlng) {
    var best = null, bestD = STOP_SNAP_M;
    var recs = (window.__stopsData && window.__stopsData.records) || [];
    for (var i = 0; i < recs.length; i++) {
      var d = mapRef.distance(latlng, L.latLng(recs[i].lat, recs[i].lon));
      if (d < bestD) { bestD = d; best = {lat: recs[i].lat, lng: recs[i].lon, name: recs[i].stop_name || 'stop'}; }
    }
    var sc = activeScenarioObj();
    if (sc) sc.routes.forEach(function(r2) {
      if (r2.id === S.activeRouteId) return;
      r2.points.forEach(function(p) {
        if (!p.is_station) return;
        var d = mapRef.distance(latlng, L.latLng(p.lat, p.lng));
        if (d < bestD) { bestD = d; best = {lat: p.lat, lng: p.lng, name: r2.name + ' station'}; }
      });
    });
    return best;
  }

  function askStation(point) {
    var near = nearbyStop(L.latLng(point.lat, point.lng));
    if (!near) return false;
    // Deliberately a plain `confirm()` -- the spec allows it, and it keeps
    // the prompt synchronous with the click that triggered it.
    if (window.confirm('There is an existing stop here (' + near.name + '). Stop there?')) {
      point.is_station = true;
      // Snap to the existing stop's EXACT coordinates rather than the raw
      // click point, so the new route genuinely shares that stop (and the
      // access-recompute / cost model see them as the same location).
      point.lat = near.lat;
      point.lng = near.lng;
      return true;
    }
    return false;
  }

  // Geoman owns the click stream while drawing, so for a brand-new route the
  // prompt is applied once, over all drawn points, right after the line is
  // finished. Extending (our own click handler) prompts per click, at click
  // time, as specified.
  function maybePromptStations(r) {
    r.points.forEach(function(p) { askStation(p); });
  }

  mapRef.on('click', function(e) {
    if (!S.active) return;
    // Item 3/4: a fresh draw always appends at the END, same as extending
    // from the end -- no `extendFrom` gate needed since there is only one
    // valid direction to grow a route that doesn't have geometry yet.
    if (S.mode === 'draw' || (S.mode === 'extend' && extendFrom)) {
      var r = activeRoute();
      if (!r) return;
      var pt = {lat: e.latlng.lat, lng: e.latlng.lng, is_station: false};
      askStation(pt);
      if (S.mode === 'extend' && extendFrom === 'start') r.points.unshift(pt);
      else r.points.push(pt);
      syncSegments(r);
      refreshPanel();
    }
  });

  // Item 5/1: undoes the last-placed node during active drawing (fresh draw
  // OR extend), standard "undo last point" behavior. Exposed both as a
  // right-click shortcut (`contextmenu`, unchanged) AND as an explicit
  // visible "Delete last node" toolbar button (item 1) -- same function
  // either way, since the button is just another way to trigger it.
  function deleteLastNode() {
    if (S.mode === 'draw') {
      var r = activeRoute();
      if (!r || !r.points.length) return;
      r.points.pop();
      syncSegments(r);
      refreshPanel();
    } else if (S.mode === 'extend' && extendFrom) {
      var r2 = activeRoute();
      if (!r2 || r2.points.length < 2) return;
      if (extendFrom === 'start') r2.points.shift(); else r2.points.pop();
      syncSegments(r2);
      refreshPanel();
    }
  }
  mapRef.on('contextmenu', function(e) {
    if (!S.active) return;
    if (S.mode === 'draw' || (S.mode === 'extend' && extendFrom)) L.DomEvent.stop(e);
    deleteLastNode();
  });
  var delLastNodeBtn = $('editBtnDeleteLastNode');
  if (delLastNodeBtn) delLastNodeBtn.addEventListener('click', deleteLastNode);

  // Item 4: paint mode's other application path -- apply the currently
  // armed grade separation to EVERY segment of the active route at once.
  var applyAllBtn = $('editorGradeSepApplyAll');
  if (applyAllBtn) applyAllBtn.addEventListener('click', function() {
    var r = activeRoute();
    if (!r || !armedGradeSep) return;
    r.segments.forEach(function(seg) { seg.grade_separation = armedGradeSep; });
    renderActiveRoute();
  });

  // Item 4: Escape exits grade-separation paint mode (a "clear way to exit"
  // besides re-clicking the armed button).
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && S.mode === 'gradesep') exitGradeSepPaint();
  });

  // --- metadata form ------------------------------------------------------
  $('editRouteName').addEventListener('input', function(e) {
    var r = activeRoute(); if (!r) return;
    r.name = e.target.value;
  });
  $('editRouteHeadway').addEventListener('input', function(e) {
    var r = activeRoute(); if (!r) return;
    var v = parseFloat(e.target.value);
    r.headway_minutes = isFinite(v) ? v : null;
  });
  $('editRouteColor').addEventListener('input', function(e) {
    var r = activeRoute(); if (!r) return;
    r.color = e.target.value;
    r._colorTouched = true;   // an explicit pick is never overwritten by mode
    renderActiveRoute();
  });
  $('editRouteMode').addEventListener('change', function(e) {
    var r = activeRoute(); if (!r) return;
    r.mode = e.target.value;
    if (!r._colorTouched) {
      r.color = (MODES[r.mode] || {}).default_color || r.color;
      $('editRouteColor').value = r.color;
      renderActiveRoute();
    }
    // Changing the mode changes the valid grade separations
    // (`__editParams.grade_separations[mode]`) and therefore the cost model:
    // any assignment the new mode does not offer (e.g. rail after switching
    // from bus's "mixed") is reset to the new mode's configured default
    // grade separation (`__editParams.default_grade_separation[mode]`)
    // rather than left as an invalid value (or cleared to null), either of
    // which has no cost/speed table entry and would silently drop the
    // segment's cost to $0.
    var valid = gradeSepOptions(r.mode);
    if (r._defaultGradeSepOverride && valid.indexOf(r._defaultGradeSepOverride) < 0) {
      r._defaultGradeSepOverride = null;
    }
    var fallback = (P.default_grade_separation || {})[r.mode];
    if (!fallback || valid.indexOf(fallback) < 0) fallback = valid[0] || null;
    r.segments.forEach(function(seg) {
      if (seg.grade_separation && valid.indexOf(seg.grade_separation) < 0) {
        seg.grade_separation = defaultGradeSep(r.mode, r) || fallback;
      }
    });
    renderActiveRoute();
  });

  // Per-segment speed override. Applied to the SELECTED SEGMENT only (the
  // spec's "a box with the speed just in case user wants to edit it too" does
  // not say per-segment or route-wide; per-segment is the reading consistent
  // with grade separation itself being per-segment, and it is what the box
  // shows -- it is labelled "this segment"). Blank/invalid clears the
  // override and falls back to the computed value.
  $('editorSegSpeed').addEventListener('input', function(e) {
    var r = activeRoute();
    if (!r || selectedSeg < 0 || !r.segments[selectedSeg]) return;
    var v = parseFloat(e.target.value);
    if (isFinite(v) && v > 0) r.segments[selectedSeg].speed_kmh_override = v;
    else delete r.segments[selectedSeg].speed_kmh_override;
  });
  $('editorSegSpeedReset').addEventListener('click', function() {
    var r = activeRoute();
    if (!r || selectedSeg < 0 || !r.segments[selectedSeg]) return;
    delete r.segments[selectedSeg].speed_kmh_override;
    renderGradeSepPanel();
  });

  // --- wiring -------------------------------------------------------------
  Object.keys(GEOM_BUTTONS).forEach(function(k) {
    $(GEOM_BUTTONS[k]).addEventListener('click', function() { toggleMode(k); });
  });
  $('editorNewRoute').addEventListener('click', newRoute);
  // Item 2/3: an explicit "Finish" control for ending a new-route draw --
  // the route stays on the map exactly as drawn (points already committed
  // live to `r.points`/`r.segments` on every click, same as extend), only
  // draw-mode itself stops.
  var finishDrawBtn = $('editBtnFinishDraw');
  if (finishDrawBtn) finishDrawBtn.addEventListener('click', function() {
    setMode('idle');
    refreshPanel();
  });
  $('editorCompute').addEventListener('click', runCompute);
  $('editorDoneRoute').addEventListener('click', function() {
    var r = activeRoute();
    // Item 5: a route with fewer than 2 stations cannot be scored (no
    // catchment pair to walk between), so saving/finishing it is blocked
    // with a clear inline warning instead of silently letting it through.
    var nSt = r ? r.points.filter(function(p) { return p.is_station; }).length : 0;
    if (r && nSt < 2) {
      setHint('This route needs at least 2 stations before it can be saved (currently ' + nSt + '). Use "Add station" to mark stops.');
      return;
    }
    setMode('idle'); S.activeRouteId = null; refreshPanel();
  });
  $('editorDeleteRoute').addEventListener('click', function() { deleteRoute(S.activeRouteId); });
  $('editorSwitchScenario').addEventListener('click', function() { setMode('idle'); openScenarioModal(); });
  $('editorCreateScenario').addEventListener('click', createScenario);
  $('editorContinueScenario').addEventListener('click', continueScenario);
  $('editorCancelScenario').addEventListener('click', function() {
    $('editorScenarioModal').style.display = 'none';
    exitEditMode();
  });

  // JSON export / import -- the only persistence, by spec (no localStorage).
  $('editorExport').addEventListener('click', function() {
    var blob = new Blob([JSON.stringify({scenarios: S.scenarios}, null, 2)], {type: 'application/json'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'transitlos_scenarios.json';
    a.click();
    URL.revokeObjectURL(a.href);
  });
  $('editorImport').addEventListener('click', function() { $('editorImportFile').click(); });
  $('editorImportFile').addEventListener('change', function(e) {
    var f = e.target.files && e.target.files[0];
    if (!f) return;
    var rd = new FileReader();
    rd.onload = function() {
      try {
        var data = JSON.parse(rd.result);
        if (data && data.scenarios) {
          S.scenarios = data.scenarios.filter(function(sc) { return sc && !isReserved(sc.name); });
          S.scenarios.forEach(function(sc) {
            sc.routes = (sc.routes || []).map(function(r) {
              r.user_created = true; r.segments = r.segments || []; syncSegments(r);
              S._routeCounter = Math.max(S._routeCounter, parseInt(String(r.id).replace(/\D/g, ''), 10) || 0);
              return r;
            });
          });
          S.activeScenario = S.scenarios.length ? S.scenarios[0].name : null;
          S.activeRouteId = null;
          refreshPanel();
        }
      } catch (err) { window.alert('Could not read that scenario file.'); }
    };
    rd.readAsText(f);
  });

  // --- the edit toggle button ----------------------------------------------
  // Item 11: a plain absolute div grouped with #statsButton (bottom-left),
  // not a Leaflet corner control -- matches how #statsButton itself is wired.
  var editBtnEl = document.getElementById('editModeButton');
  if (editBtnEl) {
    L.DomEvent.disableClickPropagation(editBtnEl);
    L.DomEvent.on(editBtnEl, 'click', function() { S.active ? exitEditMode() : enterEditMode(); });
  }
  // Item 10: the top-center summary bar's scenario name is also clickable --
  // enters edit mode (if needed) and opens the scenario switcher.
  var summaryNameEl = document.getElementById('editSummaryName');
  if (summaryNameEl) {
    L.DomEvent.disableClickPropagation(summaryNameEl);
    L.DomEvent.on(summaryNameEl, 'click', function() {
      // Item 9: switching scenarios (which is what this click opens the
      // picker for) must be a pure view change -- it must NOT enter edit
      // mode as a side effect. `openScenarioModal`/`continueScenario` work
      // fine whether or not `S.active` is true (see `continueScenario`).
      if (S.active) setMode('idle');
      openScenarioModal();
    });
  }
  L.DomEvent.disableClickPropagation(document.getElementById('editorPanel'));
  L.DomEvent.disableClickPropagation(document.getElementById('editorScenarioModal'));

  // Test/automation hook -- and the seam Phase 3/4 code should use rather
  // than reaching into this closure.
  window.__editor = {
    state: S, enter: enterEditMode, exit: exitEditMode, setMode: setMode,
    newRoute: newRoute, selectRoute: selectRoute, refresh: refreshPanel,
    activeRoute: activeRoute, insertOnSegment: insertOnSegment,
    // Phase 3 seam (Phase 4 consumes the speed/cost/headway helpers).
    selectSegment: function(i) { selectedSeg = i; renderActiveRoute(); },
    selectedSegment: function() { return selectedSeg; },
    assignGradeSep: assignGradeSep,
    gradeSepOptions: gradeSepOptions,
    segmentLengthKm: segmentLengthKm,
    computedSegmentSpeedKmh: computedSegmentSpeedKmh,
    effectiveSegmentSpeedKmh: effectiveSegmentSpeedKmh,
    stationGradeSep: stationGradeSep,
    routeCost: routeCost,
    aggregateHeadway: aggregateHeadway,
    // Phase 4 seam.
    compute: runCompute,
    computeAccess: computeAccess,
    lastCompute: function() { return lastCompute; },
    computeResultForScenario: function(name) { return computeResultsByScenario[name] || null; },
    renderSummary: renderSummary,
    baselineStats: baselineStats,
    routeAvgSpeedKmh: routeAvgSpeedKmh,
    headwaysAt: headwaysAt,
    scoringMode: scoringMode,
    scenarioCostMusd: scenarioCostMusd,
    impactEmoji: impactEmoji,
    popChunkKey: popChunkKey,
    haversineM: haversineM,
    accessRadiusM: ACCESS_R_M,
    headwayCv: HEADWAY_CV,
    segmentStyles: function() {
      return segLayers.map(function(s) {
        return {idx: s.__seg, color: s.options.color, weight: s.options.weight, dash: s.options.dashArray || null};
      });
    }
  };

  // Initial paint: the bar is on screen from page load, before edit mode has
  // ever been entered, showing the "current" (real network) level of service.
  renderSummary();
})();
"""
    return (
        js.replace("__MAP_REF__", map_ref)
        .replace("__PARAMS__", json.dumps(params))
        .replace("__POPCHUNKS__", json.dumps(pop_chunk_cfg))
        .replace("__GEOIDCHUNKS__", json.dumps(geoid_chunk_cfg))
    )


def _editor_block(
    map_var: Optional[str],
    params: Optional[dict] = None,
    pop_chunk_cfg: Optional[dict] = None,
) -> str:
    """CDN tags + HTML + `load`-deferred JS for the scenario/route editor."""
    params = params if params is not None else load_edit_params()
    return (
        f'<link rel="stylesheet" href="{GEOMAN_CSS}">\n'
        f'<script src="{GEOMAN_JS}"></script>\n'
        + _editor_html(params)
        + "<script>\nwindow.addEventListener('load', function() {\n"
        + _editor_js(map_var, params, pop_chunk_cfg)
        + "\n});\n</script>\n"
    )


def _maplibre_compute_access_js(
    params: dict,
    pop_chunk_cfg: dict,
    stats_data_json: str,
    region: str,
) -> str:
    """MapLibre port of `computeAccess()` (Phase 4 of `_editor_js`) -- item 2 of the session's priority list.

    A deliberately SCOPED port, not a 1:1 port of the Folium version:

    - Single current route only (no named-scenario system, no per-segment
      grade separation) -- `window.__routeState()` (the existing MapLibre
      draw tool) supplies the geometry; every drawn point is treated as a
      station (Folium lets the user pick which points are stations one at a
      time -- here they all are, a simplification given the port's time
      budget); mode/headway/grade-separation are one control each for the
      WHOLE route rather than per-segment.
    - Live recolor targets the `hexagons:h3_{POP_CHUNK_RES}` level only
      (verified to exist for boston_city, matching the population chunks'
      own resolution exactly -- so no h3-parent rollup is needed, unlike
      Folium's per-feature `_h3_override_lookup_js`). Coarser hex
      resolutions and the census/circle levels are NOT live-recolored by
      this port.
    - Cost/fleet-size estimate (`routeCost` in the Folium version) is not
      ported -- only the accessibility recompute itself.

    Everything it DOES compute reuses the exact same formulas as the Folium
    version (`__jsStopScore`, `__jsWalkDecay`, walk-buffer/distance math),
    copied verbatim from `_stats_panel_js`/`_editor_js` in this file, so the
    numbers this produces are not a second, drifting implementation.
    """
    consts_js = _discretization_params_js(region)
    return f"""
window.__editParams = {json.dumps(params)};
window.__popChunkCfg = {json.dumps(pop_chunk_cfg)};
window.__geoidChunkCfg = {json.dumps({"dir": GEOID_CHUNK_DIRNAME, "deg": pop_chunk_cfg.get("deg", POP_CHUNK_DEG), "level": CENSUS_LIVE_RECOLOR_LEVEL})};
window.__statsData = {stats_data_json};
{consts_js}

(function() {{
  var P = window.__editParams || {{}};
  var MODES = P.modes || {{}};
  var PC = window.__popChunkCfg || {{dir: 'pop_chunks', deg: 0.04, res: 11}};
  var HEX_LEVEL = 'hexagons:h3_' + PC.res;
  var AR = P.access_recompute || {{}};
  var ACCESS_R_M = (typeof AR.max_walk_distance_m === 'number') ? AR.max_walk_distance_m : 2000;
  var HEADWAY_CV = (typeof AR.headway_cv === 'number') ? AR.headway_cv : 0.3;

  // --- formula mirrors, copied verbatim from `_stats_panel_js` (kept in
  // sync via `window.__disc`, baked from the same Python dataclass) ---
  function __jsSpeedScore(avgSpeedKmh) {{
    var b = window.__disc.modeSpeedBreakpointsKmh;
    var b0 = b[0], b1 = b[1], b2 = b[2];
    if (avgSpeedKmh <= 0) return 0.0;
    if (avgSpeedKmh < b0) return 0.5 * (avgSpeedKmh / b0);
    if (avgSpeedKmh < b1) return 0.5 + 0.35 * (avgSpeedKmh - b0) / (b1 - b0);
    if (avgSpeedKmh < b2) return 0.85 + 0.15 * (avgSpeedKmh - b1) / (b2 - b1);
    return 1.0;
  }}
  function __jsFrequencyScore(headwayMinutes) {{
    var sat = window.__disc.headwaySaturationMinutes, scale = window.__disc.headwayDecayScaleMinutes;
    if (headwayMinutes <= sat) return 1.0;
    return Math.exp(-(headwayMinutes - sat) / scale);
  }}
  function __jsReliabilityScore(headwayCv) {{
    return Math.exp(-headwayCv / window.__disc.reliabilityCvScale);
  }}
  var __MODE_SCORES = {{ rail: 1.0, tram: Math.pow(1.0 * (1.0 / 1.15), 0.5), bus: 1.0 / 1.15 }};
  function scoringMode(m) {{ return ((MODES[m] || {{}}).scores_as) || m; }}
  function __jsStopScore(mode, speedKmh, headwayMinutes, headwayCv) {{
    var w = window.__disc.weights;
    var m = __MODE_SCORES[scoringMode(mode)];
    var s = __jsSpeedScore(speedKmh);
    var f = __jsFrequencyScore(headwayMinutes);
    var r = __jsReliabilityScore(headwayCv == null ? 0.3 : headwayCv);
    return Math.pow(m, w[0]) * Math.pow(s, w[1]) * Math.pow(f, w[2]) * Math.pow(r, w[3]);
  }}
  function __jsWalkDecay(walkTimeMinutes) {{
    var d = window.__disc;
    var tSat = d.walkDistanceSaturationM / d.walkingSpeedMps / 60.0;
    if (walkTimeMinutes <= tSat) return 1.0;
    return Math.pow(1.0 + (walkTimeMinutes - tSat) / d.walkT0BaseMinutes, -d.walkDecayShapeP);
  }}

  function haversineM(lat1, lng1, lat2, lng2) {{
    var R = 6371008.8, d2r = Math.PI / 180;
    var dLat = (lat2 - lat1) * d2r, dLng = (lng2 - lng1) * d2r;
    var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * d2r) * Math.cos(lat2 * d2r) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
  }}

  // --- chunk loading, copied verbatim from `_editor_js` ---
  var chunkCache = {{}}, chunkIndex = null;
  window.__fetchedChunks = [];
  function loadChunkIndex() {{
    if (chunkIndex) return Promise.resolve(chunkIndex);
    return fetch(PC.dir + '/index.json').then(function(r) {{
      if (!r.ok) throw new Error('no index'); return r.json();
    }}).then(function(j) {{
      chunkIndex = {{}};
      (j.keys || []).forEach(function(k) {{ chunkIndex[k] = 1; }});
      return chunkIndex;
    }}).catch(function() {{ chunkIndex = null; return null; }});
  }}
  function loadChunk(key) {{
    if (chunkCache[key]) return Promise.resolve(chunkCache[key]);
    return fetch(PC.dir + '/' + key + '.json').then(function(r) {{
      if (!r.ok) throw new Error('missing chunk ' + key); return r.json();
    }}).then(function(j) {{
      chunkCache[key] = j.cells || [];
      window.__fetchedChunks.push(key);
      return chunkCache[key];
    }}).catch(function() {{ chunkCache[key] = []; return chunkCache[key]; }});
  }}

  function speedKmhFor(mode, gradeSep) {{
    var sp = (P.speeds || {{}})[mode];
    if (!sp) return null;
    if (typeof sp.kmh === 'number') return sp.kmh;
    var by = sp.kmh_by_grade_separation || {{}};
    return (gradeSep && typeof by[gradeSep] === 'number') ? by[gradeSep] : null;
  }}

  var _baseline = undefined;
  function baselineStats() {{
    if (_baseline !== undefined) return _baseline;
    _baseline = null;
    var sd = window.__statsData;
    if (sd) {{
      var d = sd.metro || sd[Object.keys(sd)[0]];
      if (d && d.level_of_service && d.population) {{
        // 2026-09-02 (explicit user request: "top bar should be median not
        // mean") -- was a population-weighted MEAN, inconsistent with the
        // Place-rank tab/combined-map overview, which both show a
        // population-weighted MEDIAN over this exact same data -- a highly
        // skewed access distribution (many cells at exactly 0, a smaller
        // set of well-served dense cells) makes mean and median diverge
        // sharply (real case: Boston, mean 0.24 vs median 0.02), which
        // read as "the numbers don't match" even though both were
        // individually correct. Now genuinely the same statistic
        // everywhere.
        var pairs = [];
        var den = 0;
        for (var i = 0; i < d.level_of_service.length; i++) {{
          var a = d.level_of_service[i], p = d.population[i];
          if (a == null || p == null || isNaN(a) || isNaN(p) || p < 0) continue;
          pairs.push([a, p]); den += p;
        }}
        if (den > 0) {{
          pairs.sort(function(x, y) {{ return x[0] - y[0]; }});
          var cum = 0, half = den / 2, median = pairs[pairs.length - 1][0];
          for (var j = 0; j < pairs.length; j++) {{
            cum += pairs[j][1];
            if (cum >= half) {{ median = pairs[j][0]; break; }}
          }}
          _baseline = {{access: median, population: den}};
        }}
      }}
    }}
    return _baseline;
  }}

  // --- UI: extend the existing route-draw toolbar (`_route_draw_html_js`) ---
  var toolbar = document.getElementById('route-toolbar');
  var panel = document.createElement('div');
  panel.className = 'rt-group';
  var modeOptions = Object.keys(MODES).map(function(m) {{
    return '<option value="' + m + '">' + (MODES[m].label || m) + '</option>';
  }}).join('');
  panel.innerHTML =
    '<label style="display:block;margin-bottom:4px;">Mode <select id="caMode" style="margin-left:4px;">' + modeOptions + '</select></label>' +
    '<label style="display:block;margin-bottom:4px;">Headway (min) <input id="caHeadway" type="number" value="10" min="0.5" step="0.5" style="width:50px;margin-left:4px;"></label>' +
    '<label style="display:block;margin-bottom:6px;"><input id="caGradeSep" type="checkbox"> Grade-separated</label>' +
    '<div id="caCostReadout" style="margin-bottom:6px;color:#333;"></div>' +
    '<button id="caComputeBtn" type="button" class="rt-primary">Compute access</button>' +
    '<div id="caStatus" style="margin-top:4px;color:#555;max-width:220px;"></div>' +
    '<div class="rt-group">' +
      '<button id="caSaveScenarioBtn" type="button" class="rt-primary">Save as scenario</button>' +
    '</div>';
  // Round 17 (item 5): this config panel (mode/headway/grade-separation/
  // cost readout/compute/save) now lives INSIDE the "Create new route"
  // flow's config section (`#routeConfigSection`, revealed alongside the
  // draw toolbox), not appended loose onto the toolbar -- matches Folium's
  // `_editor_html` layout (mode/headway/color/geometry/cost all under one
  // route-editor block) and the user's explicit ask for "config for the
  // cost and grade separation" to appear when creating a new route.
  var configSection = document.getElementById('routeConfigSection');
  if (configSection) configSection.appendChild(panel);
  else if (toolbar) toolbar.appendChild(panel);
  // Panel starts hidden along with the rest of the draw toolbar's tool
  // buttons -- editing (including this compute/scenario panel) is gated
  // behind the scenario panel's "+ Create new route" button, matching the
  // draw tool's own lock in `_route_draw_html_js`. The default/baseline
  // network AND every already-saved scenario are read-only (round 17: this
  // now extends round 12's "baseline is read-only" rule to ALL saved
  // scenarios, not just baseline -- see `saveAsScenario` below, which
  // re-locks after a successful save).
  panel.style.display = '';

  // --- Item A (round 6): a real multi-scenario system, ported from Folium's
  // `_editor_js` scenario save/list/switch model (S.scenarios/activeScenario)
  // but scoped to this port's single-route-at-a-time editor. Each saved
  // scenario snapshots the drawn route's points + mode/headway/grade-sep AND
  // (once Computed) its own `h3AccessOverrides` result, so the stats panel's
  // `getStatsData(area, scenario)` / `__populateSecondaryScenarioSelect` can
  // pull a real second series for ANY saved scenario, not just the one
  // currently showing on the map -- exactly the same `window.__editor.state`
  // + `computeResultForScenario(name)` shape the stats-panel JS already
  // expects (see `getStatsData` and `__populateSecondaryScenarioSelect` in
  // `_stats_panel_js`, unchanged by this port).
  var RESERVED = ((P.scenarios || {{}}).reserved_current_name) || 'current';
  var S = {{ scenarios: [], activeScenario: null }};
  var computeResultsByScenario = {{}};

  function scenarioByName(n) {{
    for (var i = 0; i < S.scenarios.length; i++) if (S.scenarios[i].name === n) return S.scenarios[i];
    return null;
  }}

  // Round 17 (item 5): the scenario picker is now a real LIST
  // (`#routeScenarioList`, in the panel `_route_draw_html_js` builds),
  // rendered on open by the edit-mode button, not a `<select>`. Every saved
  // scenario is view-only -- clicking "View" loads its route onto the map
  // WITHOUT unlocking the draw tool, extending round 12's "baseline is
  // read-only" rule to every saved scenario (per the user's explicit
  // "Existing routes cannot be edited"). Only "+ Create new route" unlocks
  // editing, for a brand-new, not-yet-saved route.
  function refreshScenarioList() {{
    var listEl = document.getElementById('routeScenarioList');
    var emptyEl = document.getElementById('routeScenarioEmpty');
    if (!listEl) return;
    if (!S.scenarios.length) {{
      listEl.innerHTML = '';
      if (emptyEl) emptyEl.style.display = '';
    }} else {{
      if (emptyEl) emptyEl.style.display = 'none';
      listEl.innerHTML = S.scenarios.map(function(sc) {{
        var active = (S.activeScenario === sc.name);
        return '<div class="rt-scenario-row"' + (active ? ' style="border-color:#2563eb;"' : '') + '>' +
          '<span class="rt-scenario-name">' + sc.name + '<br><span class="rt-scenario-lock">&#128274; read-only</span></span>' +
          '<button type="button" data-scenario-view="' + sc.name + '">View</button>' +
        '</div>';
      }}).join('');
      Array.prototype.forEach.call(listEl.querySelectorAll('[data-scenario-view]'), function(btn) {{
        btn.addEventListener('click', function() {{ viewScenario(btn.getAttribute('data-scenario-view')); }});
      }});
    }}
    window.__populateAllSecondaryScenarioSelects && window.__populateAllSecondaryScenarioSelects();
  }}

  function saveAsScenario() {{
    var status = document.getElementById('caStatus');
    var suggested = S.activeScenario || '';
    var name = window.prompt('Save current route as scenario named:', suggested);
    if (!name) return;
    name = String(name).trim();
    if (!name) return;
    if (name === RESERVED) {{ window.alert('"' + RESERVED + '" is reserved for the baseline network.'); return; }}
    var route = currentRoute();
    if (!route.points.length) {{ window.alert('Draw a route before saving it as a scenario.'); return; }}
    var entry = {{
      name: name, route: route,
      gradeSepChecked: document.getElementById('caGradeSep').checked,
    }};
    var existing = scenarioByName(name);
    if (existing) {{ existing.route = entry.route; existing.gradeSepChecked = entry.gradeSepChecked; }}
    else {{ S.scenarios.push(entry); }}
    S.activeScenario = name;
    // Carry the most recent Compute forward as this scenario's own result if
    // it was computed against the same route that's being saved right now.
    if (lastCompute && lastCompute.ok) computeResultsByScenario[name] = lastCompute;
    refreshScenarioList();
    status.textContent = 'Saved scenario "' + name + '". It is now read-only -- use "+ Create new route" to start another.';
    // Round 17: a scenario becomes IMMUTABLE the moment it's saved -- re-lock
    // the draw tool/config panel and drop back to the scenario-list view,
    // exactly like `loadScenario`/`viewScenario` already do for a picked
    // existing scenario.
    if (window.__setRouteEditLocked) window.__setRouteEditLocked(true);
    var scenarioPanel = document.getElementById('routeScenarioPanel');
    var toolButtons = document.getElementById('routeToolButtons');
    if (scenarioPanel) scenarioPanel.style.display = 'block';
    if (toolButtons) toolButtons.style.display = 'none';
  }}

  // Loads a saved scenario's route for VIEWING only -- does not unlock
  // editing (round 17: existing routes cannot be edited).
  function viewScenario(name) {{
    var status = document.getElementById('caStatus');
    var sc = scenarioByName(name);
    if (!sc) return;
    S.activeScenario = name;
    var pts = sc.route.points.map(function(p) {{ return [p.lng, p.lat]; }});
    var stFlags = sc.route.points.map(function(p) {{ return !!p.is_station; }});
    if (window.__setRoutePoints) window.__setRoutePoints(pts, stFlags);
    document.getElementById('caMode').value = sc.route.mode;
    document.getElementById('caHeadway').value = sc.route.headway_minutes;
    document.getElementById('caGradeSep').checked = !!sc.gradeSepChecked;
    lastCompute = computeResultsByScenario[name] || null;
    if (status) status.textContent = 'Viewing scenario "' + name + '" (read-only)' + (lastCompute ? ' -- last computed.' : ' -- not computed yet.');
    // Viewing never unlocks the draw tool -- show the config/draw panel in
    // its locked state so mode/headway/cost are visible but not editable.
    if (window.__setRouteEditLocked) window.__setRouteEditLocked(true);
    var scenarioPanel = document.getElementById('routeScenarioPanel');
    var toolButtons = document.getElementById('routeToolButtons');
    if (scenarioPanel) scenarioPanel.style.display = 'none';
    if (toolButtons) toolButtons.style.display = '';
    refreshScenarioList();
  }}
  // `loadScenario` kept as an alias -- `window.__editor.loadScenario` (the
  // top-center bar's own scenario select, round 13) expects this name.
  var loadScenario = viewScenario;

  document.getElementById('caSaveScenarioBtn').addEventListener('click', function() {{ saveAsScenario(); }});
  refreshScenarioList();

  // Round 17 (item 4): the edit-mode emoji button (`rtEditModeBtn`, in
  // `_route_draw_html_js`) is the SOLE entry point into the scenario system
  // -- clicking it opens the scenario-list panel (replacing round 12's
  // standalone "+ New Scenario" button). "+ Create new route" inside that
  // panel is what actually unlocks editing for a brand-new route.
  // Item 3 (verbatim user request): the whole route-editor box is hidden by
  // default and only shown once this standalone toggle button (rendered
  // OUTSIDE `#route-toolbar` so it stays clickable while the box itself is
  // `display:none`) is clicked -- see the `.rt-open` CSS class added in
  // `_route_draw_html_js`.
  var editModeBtn = document.getElementById('routeEditToggleBtn');
  var routeToolbarBox = document.getElementById('route-toolbar');
  var scenarioPanelEl = document.getElementById('routeScenarioPanel');
  if (editModeBtn) editModeBtn.addEventListener('click', function() {{
    if (!scenarioPanelEl || !routeToolbarBox) return;
    var opening = !routeToolbarBox.classList.contains('rt-open');
    routeToolbarBox.classList.toggle('rt-open', opening);
    scenarioPanelEl.style.display = opening ? 'block' : 'none';
    editModeBtn.className = opening ? 'rt-active' : '';
    if (opening) {{
      // Opening the panel always re-shows the (read-only) list first --
      // matches the user's explicit "first a list of the user created
      // routes" flow, even if a "Create new route" draw was in progress.
      document.getElementById('routeToolButtons').style.display = 'none';
      refreshScenarioList();
    }}
  }});

  function startNewScenario() {{
    var name = window.prompt('New scenario name:', '');
    if (!name) return;
    name = String(name).trim();
    if (!name) return;
    if (name === RESERVED) {{ window.alert('"' + name + '" is reserved for the baseline network.'); return; }}
    if (window.__setRoutePoints) window.__setRoutePoints([], []);
    S.activeScenario = null; // unsaved-until-"Save as scenario"
    lastCompute = null;
    var status = document.getElementById('caStatus');
    if (status) status.textContent = 'New scenario "' + name + '" started -- draw a route, then Save as scenario.';
    var savedNameField = document.getElementById('caSaveScenarioBtn');
    if (savedNameField) savedNameField.setAttribute('data-suggested-name', name);
    if (window.__setRouteEditLocked) window.__setRouteEditLocked(false);
    if (scenarioPanelEl) scenarioPanelEl.style.display = 'none';
    document.getElementById('routeToolButtons').style.display = '';
    if (editModeBtn) editModeBtn.className = '';
  }}
  var createNewBtn = document.getElementById('routeCreateNewBtn');
  if (createNewBtn) createNewBtn.addEventListener('click', startNewScenario);
  // `saveAsScenario`'s own prompt pre-fills from `S.activeScenario` -- also
  // honor the name just typed into "+ Create new route" as the suggestion,
  // so the user doesn't have to retype it.
  var _origSaveAsScenario = saveAsScenario;
  saveAsScenario = function() {{
    var btn = document.getElementById('caSaveScenarioBtn');
    var suggested = btn ? btn.getAttribute('data-suggested-name') : null;
    if (suggested && !S.activeScenario) {{
      var name = window.prompt('Save current route as scenario named:', suggested);
      if (!name) return;
      name = String(name).trim();
      if (!name) return;
      if (name === RESERVED) {{ window.alert('"' + RESERVED + '" is reserved for the baseline network.'); return; }}
      var route = currentRoute();
      if (!route.points.length) {{ window.alert('Draw a route before saving it as a scenario.'); return; }}
      var entry = {{ name: name, route: route, gradeSepChecked: document.getElementById('caGradeSep').checked }};
      var existing = scenarioByName(name);
      if (existing) {{ existing.route = entry.route; existing.gradeSepChecked = entry.gradeSepChecked; }}
      else {{ S.scenarios.push(entry); }}
      S.activeScenario = name;
      if (lastCompute && lastCompute.ok) computeResultsByScenario[name] = lastCompute;
      btn.removeAttribute('data-suggested-name');
      refreshScenarioList();
      document.getElementById('caStatus').textContent = 'Saved scenario "' + name + '". It is now read-only.';
      if (window.__setRouteEditLocked) window.__setRouteEditLocked(true);
      var toolButtons2 = document.getElementById('routeToolButtons');
      if (scenarioPanelEl) scenarioPanelEl.style.display = 'block';
      if (toolButtons2) toolButtons2.style.display = 'none';
      if (editModeBtn) editModeBtn.className = '';
      return;
    }}
    _origSaveAsScenario();
  }};

  // Route cost estimate (round 17, item 5's "config for the cost"): ported
  // from Folium's `_editor_js` route-cost formula (see that module's
  // `// Route cost = sum over segments ... + sum over stations ...`), reusing
  // the SAME real cost model Folium already has -- `P.costs[mode]` from
  // `default_edit_params.json` (`cost_per_km_musd`/`station_musd` per
  // grade-separation tier), not a fabricated/placeholder number. If a build's
  // params have no cost entry for a mode, the readout says so explicitly.
  function haversineKm(a, b) {{
    var R = 6371.0088, d2r = Math.PI / 180;
    var dLat = (b[1] - a[1]) * d2r, dLng = (b[0] - a[0]) * d2r;
    var s = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(a[1] * d2r) * Math.cos(b[1] * d2r) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
  }}
  function routeCostEstimate(route) {{
    var cc = (P.costs || {{}})[route.mode];
    if (!cc) return null;
    var kmRate = (cc.cost_per_km_musd || {{}})[route.grade_separation];
    var stRate = (cc.station_musd || {{}})[route.grade_separation];
    if (kmRate == null && stRate == null) return null;
    var lengthKm = 0;
    for (var i = 1; i < route.points.length; i++) {{
      lengthKm += haversineKm([route.points[i - 1].lng, route.points[i - 1].lat], [route.points[i].lng, route.points[i].lat]);
    }}
    var nStations = route.points.filter(function(p) {{ return p.is_station; }}).length;
    var segMusd = kmRate != null ? lengthKm * kmRate : 0;
    var stMusd = stRate != null ? nStations * stRate : 0;
    return {{ length_km: lengthKm, n_stations: nStations, segments_musd: segMusd, stations_musd: stMusd, total_musd: segMusd + stMusd }};
  }}
  function refreshCostReadout() {{
    var el = document.getElementById('caCostReadout');
    if (!el) return;
    var route = currentRoute();
    var c = routeCostEstimate(route);
    if (!c) {{ el.textContent = 'Cost: no cost config for this mode.'; return; }}
    el.innerHTML = 'Est. cost: <b>$' + c.total_musd.toFixed(1) + 'M</b>' +
      ' (track $' + c.segments_musd.toFixed(1) + 'M + ' + c.n_stations + ' stop(s) $' + c.stations_musd.toFixed(1) + 'M)';
  }}
  ['caMode', 'caHeadway', 'caGradeSep'].forEach(function(id) {{
    var el = document.getElementById(id);
    if (el) el.addEventListener('change', refreshCostReadout);
  }});
  // The route's geometry/stops change via map clicks (draw/add-node/add-stop
  // etc.), not DOM `change` events, so those alone can't trigger a cost
  // refresh -- a cheap poll (matches the round-13 top bar's own
  // `setInterval` pattern for the same "many different mutation sites"
  // reason) keeps the readout live while a route is unlocked/being edited.
  setInterval(function() {{ if (!window.__routeEditLocked) refreshCostReadout(); }}, 500);

  function currentRoute() {{
    var rs = window.__routeState ? window.__routeState() : {{points: [], stations: []}};
    var mode = document.getElementById('caMode').value;
    var headway = parseFloat(document.getElementById('caHeadway').value);
    var gradeSepChoices = (P.grade_separations || {{}})[mode] || [];
    var separated = document.getElementById('caGradeSep').checked;
    // Simplification (see this function's docstring): one grade-separation
    // choice for the whole route -- the most-separated option when checked,
    // the mode's own `default_grade_separation` (its normal at-grade tier)
    // when not.
    var gradeSep = separated
      ? (gradeSepChoices[0] || (P.default_grade_separation || {{}})[mode])
      : ((P.default_grade_separation || {{}})[mode]);
    var stFlags = rs.stations || [];
    return {{
      mode: mode, headway_minutes: headway, grade_separation: gradeSep,
      points: rs.points.map(function(p, i) {{ return {{lat: p[1], lng: p[0], is_station: !!stFlags[i]}}; }}),
    }};
  }}

  // Cleared individually via `__clearAccessOverride` rather than
  // `window.__clearAllAccessOverrides(HEX_LEVEL)` -- that clear-all call
  // needs a feature actually loaded/rendered for the removal to resolve on
  // this MapLibre version, and throws ("A feature id is required to remove
  // its specific state property") when called with no id at all, which it
  // is here. Tracking exactly the ids this function itself set sidesteps
  // that entirely.
  var lastOverrideIds = [];
  // Round 18 (item 6 of the user's feedback list): the same override
  // mechanism extended to circles (identical h3 ids to hexagons, since
  // `circ_map` is built over the exact same `resolutions_levels` as
  // `hex_map` -- see `build_city_map`), census polygons (GEOID-keyed, via
  // a population-weighted delta-sum port of Folium's own
  // `_census_geoid_override_lookup_js`/`_editor_js` item-6 logic), and
  // streets (no numeric access ramp of their own -- highlighted with a
  // fixed color when within the buffer, not recolored along a ramp).
  var lastCircleOverrideIds = [];
  var lastCensusOverrideIds = [];
  var lastStreetOverrideIds = [];
  var CIRCLES_LEVEL = 'circles:h3_' + PC.res;
  var CENSUS_LEVEL = 'census:census_' + {json.dumps(CENSUS_LIVE_RECOLOR_LEVEL)};
  var STREETS_LEVEL = 'streets:edges';
  var STREET_AFFECTED_COLOR = '#ff6b00';
  var GC = window.__geoidChunkCfg || {{dir: 'geoid_chunks', deg: PC.deg, level: {json.dumps(CENSUS_LIVE_RECOLOR_LEVEL)}}};
  var geoidChunkCache = {{}}, geoidChunkIndex = null;
  function loadGeoidChunkIndex() {{
    if (geoidChunkIndex) return Promise.resolve(geoidChunkIndex);
    return fetch(GC.dir + '/index.json').then(function(r) {{
      if (!r.ok) throw new Error('no geoid index'); return r.json();
    }}).then(function(j) {{
      geoidChunkIndex = {{}};
      (j.keys || []).forEach(function(k) {{ geoidChunkIndex[k] = 1; }});
      return geoidChunkIndex;
    }}).catch(function() {{ geoidChunkIndex = null; return null; }});
  }}
  function loadGeoidChunk(key) {{
    if (geoidChunkCache[key]) return Promise.resolve(geoidChunkCache[key]);
    return fetch(GC.dir + '/' + key + '.json').then(function(r) {{
      if (!r.ok) throw new Error('missing geoid chunk ' + key); return r.json();
    }}).then(function(j) {{
      geoidChunkCache[key] = (j && j.map) || {{}};
      return geoidChunkCache[key];
    }}).catch(function() {{
      geoidChunkCache[key] = {{}};
      return geoidChunkCache[key];
    }});
  }}
  // One-time cache of every currently-loaded census feature's baked
  // level_of_service/population, keyed by its promoted id (same GEOID the
  // geoid chunks use) -- used to turn a population-weighted delta SUM
  // (see below) back into an absolute new score. Rebuilt on every Compute
  // since which tiles/features are loaded can change as the user pans.
  function censusFeatureLookup() {{
    var out = {{}};
    if (!window.__hasMapLevel(CENSUS_LEVEL)) return out;
    var feats = window.__mapLevelQueryFeatures(CENSUS_LEVEL, {{sourceLayer: 'census_' + {json.dumps(CENSUS_LIVE_RECOLOR_LEVEL)}}});
    feats.forEach(function(f) {{
      if (f.id == null) return;
      var p = f.properties || {{}};
      // Round 18: property naming varies by pipeline vintage -- pre-redesign
      // `population`/`acs_population`, post-`pyCensus`-redesign (round 16)
      // `acs5_population`/`dhc_population` (see that round's notes) -- try
      // them in order, first positive value wins.
      var pop = p.population || p.acs_population || p.acs5_population || p.dhc_population || 0;
      out[String(f.id)] = {{access: p.level_of_service || 0, population: pop}};
    }});
    return out;
  }}
  var lastCompute = null;
  function computeAccess() {{
    var status = document.getElementById('caStatus');
    var route = currentRoute();
    var base = baselineStats();
    if (!window.__hasMapLevel(HEX_LEVEL)) {{
      status.textContent = 'no ' + HEX_LEVEL + ' layer on this map';
      return Promise.resolve(null);
    }}
    if (!route.points.length) {{ status.textContent = 'draw a route first'; return Promise.resolve(null); }}
    var speedKmh = speedKmhFor(route.mode, route.grade_separation);
    if (!speedKmh || !route.headway_minutes) {{
      status.textContent = 'set a valid mode/headway (and grade separation, for bus/tram/brt)';
      return Promise.resolve(null);
    }}
    var stopScore = __jsStopScore(route.mode, speedKmh, route.headway_minutes, HEADWAY_CV);
    // Round 17 (item 5): only points flagged as stops (`is_station`, from
    // the "Add stops" checkbox / Add Stop button in `_route_draw_html_js`)
    // count as stations for the access-recompute buffer -- a plain route
    // node contributes geometry only. Falls back to treating every point as
    // a station if none are flagged, so a route drawn before this round's
    // stop-flagging exists (or with stops never marked) still computes.
    var stations = route.points.filter(function(p) {{ return p.is_station; }});
    if (!stations.length) stations = route.points;

    return loadChunkIndex().then(function(idx) {{
      if (!idx) {{ status.textContent = 'population chunks not available'; return null; }}
      var keys = {{}};
      stations.forEach(function(s) {{
        var dLat = ACCESS_R_M / 111320.0;
        var dLng = ACCESS_R_M / (111320.0 * Math.max(0.05, Math.cos(s.lat * Math.PI / 180)));
        var i0 = Math.floor((s.lat - dLat) / PC.deg), i1 = Math.floor((s.lat + dLat) / PC.deg);
        var j0 = Math.floor((s.lng - dLng) / PC.deg), j1 = Math.floor((s.lng + dLng) / PC.deg);
        for (var i = i0; i <= i1; i++) for (var j = j0; j <= j1; j++) {{
          var k = i + '_' + j;
          if (idx[k]) keys[k] = 1;
        }}
      }});
      var keyList = Object.keys(keys);
      // Round 18: fetch the matching GEOID chunks alongside the population
      // chunks (same key/locality scheme, see `loadGeoidChunk` above) --
      // ported from Folium's `_editor_js` item-6 comment. Degrades to `[]`
      // (no census overrides) when this study has no geoid_chunks at all.
      var geoidChunksReady = loadGeoidChunkIndex().then(function(gIdx) {{
        if (!gIdx) return [];
        return Promise.all(keyList.filter(function(k) {{ return gIdx[k]; }}).map(loadGeoidChunk));
      }});
      return Promise.all([Promise.all(keyList.map(loadChunk)), geoidChunksReady]).then(function(both) {{
        var chunks = both[0];
        var speedMps = (window.__disc && window.__disc.walkingSpeedMps) || 1.4;
        var nBuf = 0, nImp = 0, impact = 0;
        // childH3 -> {{access, gainBase, population, lat, lng}} -- same shape
        // the stats panel's `getStatsData(area, scenario)` rollup
        // (`_stats_panel_js`) already expects from
        // `window.__editor.lastCompute().h3AccessOverrides` /
        // `computeResultForScenario(name).h3AccessOverrides`, so a saved
        // scenario's "Compare with scenario" series is real recomputed data,
        // not a stub. `lat`/`lng` (round 18) are extra fields only the new
        // circles/census/stops overrides below read.
        var h3AccessOverrides = {{}};
        lastOverrideIds.forEach(function(id) {{ window.__clearAccessOverride(HEX_LEVEL, id); }});
        lastOverrideIds = [];
        var haveCircles = window.__hasMapLevel(CIRCLES_LEVEL);
        lastCircleOverrideIds.forEach(function(id) {{ window.__clearAccessOverride(CIRCLES_LEVEL, id); }});
        lastCircleOverrideIds = [];
        chunks.forEach(function(rows) {{
          for (var ci = 0; ci < rows.length; ci++) {{
            var row = rows[ci];
            var lat = row[1], lng = row[2], pop = row[3], oldA = row[4];
            var best = oldA, inBuf = false;
            for (var si = 0; si < stations.length; si++) {{
              var s = stations[si];
              var d = haversineM(lat, lng, s.lat, s.lng);
              if (d > ACCESS_R_M) continue;
              inBuf = true;
              var a = stopScore * __jsWalkDecay(d / speedMps / 60.0);
              if (a > best) best = a;
            }}
            if (!inBuf) continue;
            nBuf += 1;
            h3AccessOverrides[row[0]] = {{access: best, gainBase: oldA, population: pop, lat: lat, lng: lng}};
            if (best > oldA) {{
              nImp += 1;
              impact += (best - oldA) * pop;
              var color = window.__colorForValue(HEX_LEVEL, best);
              if (color) {{
                window.__setAccessOverride(HEX_LEVEL, row[0], color);
                lastOverrideIds.push(row[0]);
              }}
              // Circles: an identical h3 grid to hexagons (same
              // `resolutions_levels` -- see `build_city_map`), so the exact
              // same cell id + color applies directly, no re-derivation.
              if (haveCircles) {{
                var circleColor = window.__colorForValue(CIRCLES_LEVEL, best) || color;
                if (circleColor && window.__setAccessOverride(CIRCLES_LEVEL, row[0], circleColor)) {{
                  lastCircleOverrideIds.push(row[0]);
                }}
              }}
            }}
          }}
        }});
        // Round 18: census polygons -- GEOID-keyed, no per-feature h3 id, so
        // roll each touched h3 cell's (new - old) access up to its parent
        // GEOID via the geoid chunk map, exactly mirroring Folium's own
        // `_editor_js` item-6 logic (population-weighted delta SUM, then
        // divided by the polygon's own total population and added to its
        // baked level_of_service -- see `_census_geoid_override_lookup_js`'s
        // docstring for why this reproduces the original build-time mean).
        lastCensusOverrideIds.forEach(function(id) {{ window.__clearAccessOverride(CENSUS_LEVEL, id); }});
        lastCensusOverrideIds = [];
        var geoidAccessOverrides = {{}};
        var haveCensus = window.__hasMapLevel(CENSUS_LEVEL);
        if (haveCensus) {{
          both[1].forEach(function(m) {{
            for (var cellId in m) {{
              var ov = h3AccessOverrides[cellId];
              if (!ov || ov.access == null || ov.gainBase == null) continue;
              var gid = m[cellId];
              var cpop = (ov.population != null && ov.population >= 0) ? ov.population : 0;
              geoidAccessOverrides[gid] = (geoidAccessOverrides[gid] || 0) + (ov.access - ov.gainBase) * cpop;
            }}
          }});
          var censusLookup = censusFeatureLookup();
          Object.keys(geoidAccessOverrides).forEach(function(gid) {{
            var delta = geoidAccessOverrides[gid];
            var cf = censusLookup[String(gid)];
            if (!cf || !(cf.population > 0) || delta === 0) return;
            var newScore = Math.max(0, Math.min(1, cf.access + delta / cf.population));
            var color = window.__colorForValue(CENSUS_LEVEL, newScore);
            if (color && window.__setAccessOverride(CENSUS_LEVEL, gid, color)) {{
              lastCensusOverrideIds.push(gid);
            }}
          }});
        }}
        // Round 18: streets. `street_overlay` layers ARE colored by a real
        // numeric score (`_score_maplibre_paint("line", score_col, ...)` in
        // `build_city_map`, the street's own baked `level_of_service`), and
        // `_add_group` (geohierarchy's maplibre render.py) now captures
        // `line-color` ramps into `window.__scoreInterpolators` the same as
        // fill/circle (round 18 render.py fix) -- so an affected street's
        // override color is genuinely value-derived via
        // `window.__colorForValue`, not a flat highlight. "Affected" = any
        // loaded edge feature passing within ACCESS_R_M of at least one
        // station (checked against the edge's own vertex coordinates, a
        // close approximation of point-to-segment distance, cheap in JS);
        // its new value is the max h3AccessOverrides access among cells near
        // the triggering vertex, falling back to `STREET_AFFECTED_COLOR`
        // only if no interpolator/nearby cell exists (e.g. this study has no
        // streets score ramp at all).
        lastStreetOverrideIds.forEach(function(id) {{ window.__clearAccessOverride(STREETS_LEVEL, id); }});
        lastStreetOverrideIds = [];
        var haveStreets = window.__hasMapLevel(STREETS_LEVEL);
        if (haveStreets && window.__hasMapLevel(STREETS_LEVEL)) {{
          var streetH3Cells = Object.keys(h3AccessOverrides).map(function(k) {{ return h3AccessOverrides[k]; }});
          var edgeFeats = window.__mapLevelQueryFeatures(STREETS_LEVEL, {{sourceLayer: 'edges'}});
          edgeFeats.forEach(function(f) {{
            if (f.id == null || !f.geometry) return;
            var coords = f.geometry.type === 'LineString' ? [f.geometry.coordinates]
              : f.geometry.type === 'MultiLineString' ? f.geometry.coordinates : [];
            var affected = false, bestNearby = null;
            for (var li = 0; li < coords.length; li++) {{
              for (var vi = 0; vi < coords[li].length; vi++) {{
                var vlng = coords[li][vi][0], vlat = coords[li][vi][1];
                for (var si2 = 0; si2 < stations.length; si2++) {{
                  if (haversineM(vlat, vlng, stations[si2].lat, stations[si2].lng) <= ACCESS_R_M) {{
                    affected = true;
                    break;
                  }}
                }}
                for (var hi2 = 0; hi2 < streetH3Cells.length; hi2++) {{
                  var hc2 = streetH3Cells[hi2];
                  if (haversineM(vlat, vlng, hc2.lat, hc2.lng) <= ACCESS_R_M) {{
                    if (bestNearby == null || hc2.access > bestNearby) bestNearby = hc2.access;
                  }}
                }}
              }}
            }}
            if (!affected) return;
            var streetColor = (bestNearby != null && window.__colorForValue(STREETS_LEVEL, bestNearby)) || STREET_AFFECTED_COLOR;
            if (window.__setAccessOverride(STREETS_LEVEL, f.id, streetColor)) {{
              lastStreetOverrideIds.push(f.id);
            }}
          }});
        }}
        // Round 18: recompute the STOPS' own displayed access -- distinct
        // from the population-side h3AccessOverrides above. For every real
        // GTFS stop within ACCESS_R_M of the drawn route's stations, look up
        // the h3 grid cells (from the SAME `h3AccessOverrides` this Compute
        // just built) within a small local buffer of the stop's own
        // coordinates via straight-line (haversine) distance, and take the
        // max of their NEW access values as that stop's new local-access
        // number -- the max-over-nearby-cells mirrors the same "best nearby
        // wins" convention `accessibility_score` itself uses. Written to a
        // new `__new_access` property (not `stop_score`, which is the
        // stop's own service-quality metric, a different number) so the
        // score label can show it without redefining stop_score's meaning.
        var nStopsAffected = 0;
        if (window.__stopsFC && window.__stopsFC.features && window.__stopsFC.features.length) {{
          var STOP_LOCAL_BUFFER_M = Math.min(ACCESS_R_M, 600);
          var h3Cells = Object.keys(h3AccessOverrides).map(function(k) {{ return h3AccessOverrides[k]; }});
          window.__stopsFC.features.forEach(function(sf) {{
            var slng = sf.geometry.coordinates[0], slat = sf.geometry.coordinates[1];
            var nearStation = stations.some(function(s) {{ return haversineM(slat, slng, s.lat, s.lng) <= ACCESS_R_M; }});
            if (!nearStation) return;
            var bestNear = null;
            for (var hi = 0; hi < h3Cells.length; hi++) {{
              var hc = h3Cells[hi];
              if (haversineM(slat, slng, hc.lat, hc.lng) <= STOP_LOCAL_BUFFER_M) {{
                if (bestNear == null || hc.access > bestNear) bestNear = hc.access;
              }}
            }}
            if (bestNear != null) {{
              sf.properties.__new_access = bestNear;
              nStopsAffected += 1;
            }}
          }});
          if (nStopsAffected > 0 && window.__refreshStopsData) window.__refreshStopsData();
        }}
        var newAccess = base ? (base.access + impact / base.population) : null;
        status.textContent = nBuf + ' cells in buffer, ' + nImp + ' improved, ' +
          lastCircleOverrideIds.length + ' circles, ' + lastCensusOverrideIds.length + ' census polys, ' +
          lastStreetOverrideIds.length + ' streets, ' + nStopsAffected + ' stops recomputed' +
          (newAccess != null ? ', metro access ' + base.access.toFixed(2) + ' -> ' + newAccess.toFixed(2) : '');
        var result = {{
          ok: true, n_cells_in_buffer: nBuf, n_cells_improved: nImp, impact: impact,
          stop_score: stopScore, speed_kmh: speedKmh, baseline_access: base ? base.access : null,
          new_access: newAccess, mode: route.mode, grade_separation: route.grade_separation,
          h3AccessOverrides: h3AccessOverrides,
          n_circles_overridden: lastCircleOverrideIds.length,
          n_census_overridden: lastCensusOverrideIds.length,
          n_streets_affected: lastStreetOverrideIds.length,
          n_stops_recomputed: nStopsAffected,
        }};
        window.__lastComputeAccess = result;
        lastCompute = result;
        // Cache this Compute under whichever scenario is currently active
        // (if any) so `computeResultForScenario(name)` can return it later
        // even after the user switches to/computes a DIFFERENT scenario.
        if (S.activeScenario) computeResultsByScenario[S.activeScenario] = result;
        refreshScenarioList();
        return result;
      }});
    }});
  }}
  document.getElementById('caComputeBtn').addEventListener('click', function() {{
    document.getElementById('caStatus').textContent = 'Computing...';
    refreshCostReadout();
    computeAccess();
  }});
  refreshCostReadout();
  // Test/automation hook.
  window.__computeAccess = computeAccess;

  // Item A (round 6): the seam the stats panel's `getStatsData`/
  // `__populateSecondaryScenarioSelect` already read via try/catch fallback
  // (see `_stats_panel_js`) -- same shape as Folium's `window.__editor`
  // (`state.scenarios`/`state.activeScenario`, `lastCompute()`,
  // `computeResultForScenario(name)`), scoped to what this port's
  // single-route-at-a-time editor actually needs.
  window.__editor = {{
    state: S,
    lastCompute: function() {{ return lastCompute; }},
    computeResultForScenario: function(name) {{ return computeResultsByScenario[name] || null; }},
    saveAsScenario: saveAsScenario,
    loadScenario: loadScenario,
    // Round-13 top bar (`_inject_maplibre_topbar_into_saved_html`) reads
    // these to show the baseline level of service for the reserved/"current"
    // scenario, since `lastCompute()` is null until a real edited scenario
    // has actually been computed.
    reservedName: RESERVED,
    baselineStats: function() {{ return baselineStats(); }},
  }};
}})();
"""


def _maplibre_stops_routes_js(
    stops_gdf: Optional[gpd.GeoDataFrame],
    routes_gdf: Optional[gpd.GeoDataFrame],
    score_domain: Tuple[float, float],
) -> str:
    """MapLibre-native stops + routes + labels + popups overlay (item 5 of the round-4 port).

    Folium's version (`_stops_layer_block`/`_routes_layer_block`) is built on
    real Leaflet markers (`L.marker`/`L.divIcon`) inside a
    `Leaflet.markercluster` group plus `L.polyline`/`L.layerGroup` -- none of
    which exist on the MapLibre page. This is a from-scratch MapLibre-native
    equivalent using GeoJSON sources + MapLibre's own built-in clustering
    (`cluster: true` on a GeoJSON source, simpler here than Leaflet's plugin
    approach), covering the same four pieces of content the user flagged as
    missing:
      - stops (as a clustered circle layer, colored by `stop_score` on the
        same blue scale Folium uses, with a count layer for clusters)
      - routes (as one line layer per mode: bus/tram/rail, zoom-gated by
        `ROUTE_MODE_MIN_ZOOM`, matching Folium's per-mode weight/z-order)
      - labels (stop-name symbol layer, visible only at high zoom so a real
        city's thousands of stops don't spam the map)
      - popups (click handlers on both stops and routes showing the same
        core fields Folium's popup table does: name/mode/route/headway/
        speed/stop_score for stops, route label for routes)

    Deliberately simplified vs. Folium's version -- not a byte-for-byte
    port: no custom mode-emoji cluster icon that shows the cluster's BEST
    member (MapLibre cluster count layers show a plain number), no paged
    on-map "routes" annotation badges, no `all_routes`-driven per-route
    mode-visibility checkbox wiring. Those are cosmetic/interaction
    refinements on top of a now-present base layer, not missing content.
    """
    stops_json = _stops_json_data(stops_gdf, score_domain) if stops_gdf is not None and not stops_gdf.empty else None
    routes_json = _routes_geojson_data(routes_gdf) if routes_gdf is not None and not routes_gdf.empty else None
    weights_json = json.dumps(_ROUTE_MODE_WEIGHT)
    mode_fallback_json = json.dumps(MODE_FALLBACK_COLOR)

    parts = ["(function() {"]
    parts.append("  if (!map) return;")
    parts.append("  function whenReady(fn) { if (map.isStyleLoaded()) fn(); else map.once('load', fn); }")

    if routes_json is not None:
        parts.append(f"""
  var routesFC = {routes_json};
  var routeWeights = {weights_json};
  var modeFallback = {mode_fallback_json};
  window.__routeModeVisible = {{bus: true, tram: true, rail: true}};
  // Round 15 (item 1 of the route-badge port): published so the stops
  // overlay's route badge chips can look up each route's real GTFS
  // `route_color`/`route_text_color` by name, exactly mirroring Folium's
  // `window.__routeStyles` (`_routes_json_data`) -- see `_route_styles_dict`.
  window.__routeStyles = routesFC.styles || {{}};
  whenReady(function() {{
    map.addSource('__routes_overlay', {{type: 'geojson', data: routesFC}});
    // Issue 2 (verbatim user report): route lines read as visually
    // indistinguishable from street lines. Fixed with the standard "cased
    // line" cartographic technique -- a slightly WIDER plain-black line
    // drawn first (the casing), then the route's own colored line drawn
    // narrower on top of it, so every route reads as a two-tone stroke
    // (colored fill + black outline) regardless of what color it shares
    // with nearby streets. Two real `line` layers rather than
    // `line-gap-width`/`line-border-color` -- those paint properties are
    // real but only became available in fairly recent maplibre-gl-js
    // releases (line-border-* landed in v5), and this project has no
    // pinned maplibre-gl CDN/version constant anywhere in this file (the
    // MapLibre HTML shell + its script tag are written by the
    // `geohierarchy` package's `save_maplibre`, not here) to check safely
    // against -- two plain `line` layers are supported by every
    // maplibre-gl-js version and need no such check.
    ['bus', 'tram', 'rail'].forEach(function(m) {{
      var baseWidth = routeWeights[m] || 2;
      map.addLayer({{
        id: '__routes_casing_' + m,
        type: 'line',
        source: '__routes_overlay',
        filter: ['==', ['get', 'mode'], m],
        minzoom: (routesFC.min_zoom && routesFC.min_zoom[m]) || 0,
        layout: {{'line-cap': 'round', 'line-join': 'round'}},
        paint: {{
          'line-color': '#000000',
          'line-width': baseWidth + 2,
          'line-opacity': 0.85,
        }},
      }});
      map.addLayer({{
        id: '__routes_line_' + m,
        type: 'line',
        source: '__routes_overlay',
        filter: ['==', ['get', 'mode'], m],
        minzoom: (routesFC.min_zoom && routesFC.min_zoom[m]) || 0,
        layout: {{'line-cap': 'round', 'line-join': 'round'}},
        paint: {{
          'line-color': ['coalesce', ['get', 'color'], modeFallback[m] || '#8b0000'],
          'line-width': baseWidth,
          'line-opacity': 0.95,
        }},
      }});
    }});
    // Round-14: "route short names" always render on route lines (no
    // checkbox any more -- user asked for these to appear unconditionally).
    // Item 5 (round 14, verbatim user request): only at high zoom -- a
    // route's on-line label at low zoom just spams a whole zoomed-out city
    // with text. A route can ask for a finer (higher) threshold via its
    // GeoJSON `route_label_min_zoom` property (falls back to the layer's own
    // base minzoom when absent) -- MapLibre layer `minzoom` is a single
    // scalar though, so the per-feature override is additionally enforced
    // with a `['step', ['zoom'], ...]` opacity guard evaluated per-feature;
    // the layer's own `minzoom` covers the common case cheaply (skips the
    // layer entirely below it) and the paint-level opacity expression
    // handles any route that asked for something higher.
    //
    // Follow-up (verbatim user request): a single label per line read as a
    // vestigial "route name once" stamp, not an always-identifiable route --
    // wanted REPEATED labels along the whole shape at a fixed interval, at a
    // RANGE of zooms (not only right at the old zoom>=16 cutoff), denser
    // when zoomed in and sparser when zoomed out so repeats don't overlap
    // each other along a curve at low zoom. `symbol-placement: 'line'`
    // (already set) is what makes MapLibre repeat a symbol layer's label
    // along a LineString to begin with; `symbol-spacing` (px between
    // repeats) is the interval knob, made a zoom-interpolated expression
    // here instead of the old fixed `200`. Issue 2 (verbatim user report):
    // route labels should activate at the SAME zoom level streets
    // themselves become visible -- this file's Folium/Leaflet streets
    // overlay (`_edges_style_js`/`set_resolution("edges", ...)`, see the
    // `street_min_zoom` parameter threaded through
    // `_inject_controls_into_saved_html`'s caller) defaults to zoom 14, so
    // the base `minzoom` here is matched to that same value (was 11, itself
    // a prior lowering from 16) rather than an independently-chosen number.
    map.addLayer({{
      id: '__routes_label',
      type: 'symbol',
      source: '__routes_overlay',
      minzoom: 14,
      layout: {{
        'symbol-placement': 'line',
        'text-field': ['get', 'route_label'],
        'text-size': 11,
        'symbol-spacing': [
          'interpolate', ['linear'], ['zoom'],
          14, 320,
          16, 220,
          18, 140,
          20, 90,
        ],
        'visibility': 'visible',
      }},
      paint: {{
        'text-color': '#666666', 'text-halo-color': '#ffffff', 'text-halo-width': 1.4,
        // `step`'s stop values must be number LITERALS per the MapLibre
        // style spec -- a `['coalesce', ['get', ...], 11]` expression there
        // is invalid and makes `createExpression` return an error, which
        // makes this whole `map.addLayer(...)` throw (uncaught, since
        // nothing here wraps it in try/catch), aborting the rest of this
        // `map.on('load', ...)` callback -- including `__updateRouteLayers`
        // never getting defined/called. That's why the labels never
        // rendered on ANY map. `case` has no such literal-stop restriction,
        // so it can compare against the per-feature threshold directly.
        'text-opacity': ['case',
          ['>=', ['zoom'], ['coalesce', ['get', 'route_label_min_zoom'], 14]], 1, 0],
      }},
    }});
    // Item 6 (user request, this round): route linestrings must carry NO
    // click-popup or hover-tooltip at all -- the route's name is instead
    // shown as a permanent MapLibre-native on-line text label (the
    // `__routes_label` symbol layer just above, `minzoom: 16`). This used
    // to bind both a click `routePopup` (route_label/mode table) and a
    // mousemove `routeHoverPopup` (Folium-tooltip-style follow-cursor
    // label) on every `__routes_line_*` layer; both are removed rather than
    // reworked since the on-line label now covers exactly the same
    // information (route name), natively, without a click or hover at all.
    // Wired to the same bus/tram/rail checkboxes as the Folium editor would
    // use, if this page's control panel provides them; otherwise all modes
    // stay visible (default true above).
    window.__updateRouteLayers = function() {{
      var visibleModes = [];
      ['bus', 'tram', 'rail'].forEach(function(m) {{
        var visible = window.__routeModeVisible[m] !== false;
        map.setLayoutProperty('__routes_line_' + m, 'visibility', visible ? 'visible' : 'none');
        // Issue 2's cased-line casing layer must hide/show in lockstep with
        // its colored line, or toggling a mode off left its black outline
        // showing on its own.
        map.setLayoutProperty('__routes_casing_' + m, 'visibility', visible ? 'visible' : 'none');
        if (visible) visibleModes.push(m);
      }});
      // Route on-line labels (`__routes_label`) aren't split per-mode into
      // separate layers like the lines are, so hiding a mode's line alone
      // left its label floating with no visible route under it. Filtering
      // the shared label layer to only the currently-visible modes makes it
      // disappear along with its route, on top of the existing zoom-16 gate
      // (`minzoom`/`text-opacity` above) -- both conditions now apply.
      map.setFilter('__routes_label', ['in', ['get', 'mode'], ['literal', visibleModes]]);
    }};
    window.__updateRouteLayers();
  }});
""")

    if stops_json is not None:
        parts.append(f"""
  var stopsData = {stops_json};
  var sMin = stopsData.score_domain[0], sMax = stopsData.score_domain[1];
  var blueScale = stopsData.blue_scale;
  function blueFor(t) {{
    t = Math.max(0, Math.min(1, isFinite(t) ? t : 0));
    var scaled = t * (blueScale.length - 1);
    var i = Math.floor(scaled), frac = scaled - i;
    if (i >= blueScale.length - 1) return blueScale[blueScale.length - 1];
    function hex2rgb(h) {{ h = h.replace('#', ''); return [parseInt(h.substr(0,2),16), parseInt(h.substr(2,2),16), parseInt(h.substr(4,2),16)]; }}
    var a = hex2rgb(blueScale[i]), b = hex2rgb(blueScale[i + 1]);
    var rgb = [0, 1, 2].map(function(k) {{ return Math.round(a[k] + (b[k] - a[k]) * frac); }});
    return 'rgb(' + rgb.join(',') + ')';
  }}
  var stopsFC = {{
    type: 'FeatureCollection',
    features: stopsData.records.map(function(r, i) {{
      return {{
        type: 'Feature',
        id: i,
        geometry: {{type: 'Point', coordinates: [r.lon, r.lat]}},
        properties: r,
      }};
    }}),
  }};
  // Round 18: published so the route-editor's `computeAccess()`
  // (`_maplibre_compute_access_js`) can find every stop's coordinates/
  // properties and, after recomputing an affected stop's `__new_access`
  // property in place, push the mutated FeatureCollection back through the
  // GeoJSON source's `setData` -- the same "just re-set the source data"
  // pattern MapLibre GeoJSON sources use in place of Leaflet's per-marker
  // `setIcon`/text update.
  window.__stopsFC = stopsFC;
  window.__refreshStopsData = function() {{
    var src = map.getSource('__stops_overlay');
    if (src) src.setData(stopsFC);
  }};
  whenReady(function() {{
    // Bug fix (user report): a stop-marker click must never also open the
    // hex/circle/census shape popup underneath it. `geohierarchy`'s
    // MapLibre renderer (`transitlos.map.build`'s shape layers) checks
    // `window.__mapClickPriorityLayers` at click time (see its own click
    // handler's comment) before opening its own popup -- publish the stop
    // layers here so it can suppress itself whenever a click also lands on
    // a rendered stop marker/cluster, regardless of which module's JS
    // happened to register its click listener first.
    window.__mapClickPriorityLayers = (window.__mapClickPriorityLayers || []).concat(['__stops_mode_icon', '__stops_cluster']);
    // Item (verbatim user request): "the stop group circle to be colored
    // with the stop score colors by the best stops of the group" --
    // `clusterProperties` asks Supercluster (MapLibre's clustering engine)
    // to compute `max(stop_score)` per cluster as it builds each zoom
    // level's clusters, giving a real per-cluster aggregate property
    // (`max_stop_score`) the paint expression below can read -- the native
    // MapLibre equivalent of the Leaflet path's `iconCreateFunction`
    // picking the best child marker by hand.
    var clusterColorExpr = ['interpolate', ['linear'], ['get', 'max_stop_score']];
    var scaleN = Math.max(blueScale.length - 1, 1);
    blueScale.forEach(function(c, i) {{ clusterColorExpr.push(i / scaleN, c); }});
    map.addSource('__stops_overlay', {{
      type: 'geojson', data: stopsFC,
      cluster: true, clusterRadius: 18, clusterMaxZoom: 16,
      clusterProperties: {{'max_stop_score': ['max', ['coalesce', ['get', 'stop_score'], 0]]}},
    }});
    map.addLayer({{
      id: '__stops_cluster',
      type: 'circle',
      source: '__stops_overlay',
      filter: ['has', 'point_count'],
      paint: {{
        'circle-color': clusterColorExpr,
        // Item (verbatim user request): "make the stops circles a bit
        // smaller especially the maximum circles for very large groups" --
        // shrunk from 14/18/24 to 10/13/16.
        'circle-radius': ['step', ['get', 'point_count'], 10, 25, 13, 100, 16],
        'circle-stroke-width': 1.5,
        'circle-stroke-color': '#ffffff',
        'circle-opacity': 0.85,
      }},
    }});
    map.addLayer({{
      id: '__stops_cluster_count',
      type: 'symbol',
      source: '__stops_overlay',
      filter: ['has', 'point_count'],
      layout: {{'text-field': ['get', 'point_count_abbreviated'], 'text-size': 12}},
      // Round 15 (item 4): black text + a real white halo so the count
      // reads clearly over any cluster-circle color, matching the user's
      // explicit "black text with white border" ask.
      paint: {{'text-color': '#000000', 'text-halo-color': '#ffffff', 'text-halo-width': 1.8, 'text-halo-blur': 0.3}},
    }});
    // Stop icons -- NO circle background/symbol at all (nothing drawn
    // under the icon). Per user request, the mode glyph is now a real Font
    // Awesome Free Solid icon (fa-train / fa-train-tram / fa-bus-simple --
    // SAME glyph choices already used by the Leaflet/Folium path's marker
    // icons, see the `MODE_META`/free-glyph mapping near line 1029) instead
    // of a color emoji. MapLibre's own text-field/glyph pipeline can't
    // render arbitrary webfont icons directly (its SDF glyphs come from a
    // vector-tile font stack, not an installed webfont), so the glyph is
    // rasterized once per mode onto an offscreen <canvas> using the FA
    // webfont (loaded via the CDN `<link>` added in
    // `_inject_maplibre_stops_routes_into_saved_html`), its RGB flattened to
    // solid white while KEEPING its real alpha (the glyph's own
    // silhouette), producing a proper SDF-style icon that
    // `map.addImage(..., {{sdf: true}})` tints per-feature via `icon-color`
    // -- the visible shape is the FA glyph's silhouette, its color is the
    // access-score tint (SAME `blueFor` scale every other stop-score
    // visualization on this map uses), and there is no background shape.
    var FA_GLYPH = {{rail: '\\uf238', tram: '\\ue5b4', bus: '\\uf55e'}}; // fa-train / fa-train-tram / fa-bus-simple
    function drawModeIcon(mode) {{
      var size = 48;
      var c = document.createElement('canvas');
      c.width = size; c.height = size;
      var ctx = c.getContext('2d');
      ctx.clearRect(0, 0, size, size);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '900 ' + Math.round(size * 0.72) + 'px "Font Awesome 6 Free"';
      ctx.fillText(FA_GLYPH[mode], size / 2, size / 2 + 1);
      var img = ctx.getImageData(0, 0, size, size);
      var d = img.data;
      for (var i = 0; i < d.length; i += 4) {{ d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; }}
      return img;
    }}
    // Rasterizing before the FA webfont has actually loaded silently draws
    // the fallback (sans-serif, blank at these codepoints) -- wait for
    // `document.fonts` to settle first so the icons don't come out empty on
    // a slow network. `document.fonts.load` targets the exact `900 ...`
    // face used above; `document.fonts.ready` alone isn't enough because a
    // face is only "ready" once something has actually requested it.
    function addModeIcons() {{
      ['rail', 'tram', 'bus'].forEach(function(m) {{
        map.addImage('__mode_icon_' + m, drawModeIcon(m), {{sdf: true}});
      }});
    }}
    if (document.fonts && document.fonts.load) {{
      document.fonts.load('900 32px "Font Awesome 6 Free"').then(addModeIcons).catch(addModeIcons);
    }} else {{
      addModeIcons();
    }}
    map.addLayer({{
      id: '__stops_mode_icon',
      type: 'symbol',
      source: '__stops_overlay',
      filter: ['!', ['has', 'point_count']],
      layout: {{
        'icon-image': ['match', ['get', 'mode'], 'rail', '__mode_icon_rail', 'tram', '__mode_icon_tram', '__mode_icon_bus'],
        // Bumped from 0.42 to 0.55 (user request: "a bit larger"), then
        // trimmed slightly (user request: "a bit smaller") alongside
        // tighter clustering (`clusterRadius`) so more individual stops
        // stay ungrouped without crowding each other.
        'icon-size': 0.48,
        'icon-allow-overlap': true,
        'icon-ignore-placement': true,
      }},
      paint: {{
        'icon-color': [
          'interpolate', ['linear'], ['coalesce', ['get', 'stop_score'], sMin],
          sMin, blueFor(0), sMax, blueFor(1),
        ],
        'icon-halo-color': '#ffffff',
        'icon-halo-width': 1,
      }},
    }});
    // Round 15 (item 3): small stop-score text under each icon, using
    // MapLibre's built-in `number-format` expression so it always shows
    // exactly 2 decimals like Folium's `fmt2`, without precomputing a
    // string property server-side.
    map.addLayer({{
      id: '__stops_score_label',
      type: 'symbol',
      source: '__stops_overlay',
      filter: ['!', ['has', 'point_count']],
      layout: {{
        // Round 18: once `computeAccess()` has recomputed this stop's
        // local access from the (now-updated) h3 grid within its own
        // buffer, `__new_access` wins over the static `stop_score` --
        // `coalesce` picks the first non-null, so unaffected stops still
        // just show their original stop_score, unchanged.
        'text-field': ['case', ['any', ['has', '__new_access'], ['has', 'stop_score']],
          ['number-format', ['coalesce', ['get', '__new_access'], ['get', 'stop_score']],
            {{'min-fraction-digits': 2, 'max-fraction-digits': 2}}], ''],
        'text-size': 9,
        'text-offset': [0, 1.15],
        'text-anchor': 'top',
        'text-allow-overlap': true,
        'text-ignore-placement': true,
      }},
      paint: {{'text-color': '#000000', 'text-halo-color': '#ffffff', 'text-halo-width': 1.4}},
    }});
    // Stop-name labels: high zoom only, so a real city's thousands of stops
    // don't spam the map at low zoom (matches Folium's ANNO_MIN_ZOOM gate).
    map.addLayer({{
      id: '__stops_label',
      type: 'symbol',
      source: '__stops_overlay',
      filter: ['!', ['has', 'point_count']],
      minzoom: 16,
      layout: {{
        'text-field': ['get', 'stop_name'],
        'text-size': 10,
        'text-offset': [0, 2.1],
        'text-anchor': 'top',
        'visibility': 'none',
      }},
      paint: {{'text-color': '#1a1a1a', 'text-halo-color': '#ffffff', 'text-halo-width': 1.2}},
    }});

    // --- Route badge chips (item 1 of the round-15 port) --------------------
    // Ported from Folium's `_stops_layer_block` (`routeStyleFor`/
    // `splitRoutes`/`routeBadgeHtml`/`badgesLabelHtml`/`attachPager`), which
    // the user pointed to as the reference implementation: a small pill per
    // route filled with that route's real GTFS color (text in its GTFS text
    // color), paginated with prev/next arrows at a fixed count per page so a
    // stop served by many routes doesn't spam the map.
    var modeFallbackColor = {mode_fallback_json};
    function normRouteName(s) {{ return String(s).replace(/[_\\s]+/g, ' ').trim().toLowerCase(); }}
    function routeStyleFor(name, fallbackMode) {{
      var styles = window.__routeStyles || {{}};
      var hit = styles[normRouteName(name)];
      if (hit) return {{color: hit[0], text: hit[1], mode: hit[2]}};
      var mode = fallbackMode || 'bus';
      return {{color: modeFallbackColor[mode] || modeFallbackColor.bus, text: '#FFFFFF', mode: mode}};
    }}
    function splitRoutes(routeStr) {{
      if (!routeStr) return [];
      return String(routeStr).split(',').map(function(s) {{ return s.trim(); }}).filter(function(s) {{ return s.length > 0; }});
    }}
    function routeBadgeHtml(name, fallbackMode, fontPx) {{
      var st = routeStyleFor(name, fallbackMode);
      var display = String(name).replace(/_/g, ' ');
      return '<span style="display:inline-block;background:' + st.color + ';color:' + st.text + ';' +
        'font:700 ' + (fontPx || 10) + 'px sans-serif;line-height:1.35;padding:1px 5px;border-radius:7px;' +
        'margin:1px 3px 1px 0;white-space:nowrap;box-shadow:0 0 0 1px rgba(0,0,0,0.18);">' + display + '</span>';
    }}
    window.__routeBadgeHtml = routeBadgeHtml;
    function modeVisible(mode) {{
      var vis = window.__routeModeVisible;
      return !vis || vis[mode] !== false;
    }}
    // Item 7 (this round, verbatim user request): route badges are scoped
    // per `stop_id` (the individual platform), NOT aggregated to the parent
    // station -- `r.route` is `_stops_json_data`'s `route_col`, which prefers
    // the STATION-WIDE `all_routes` field when present (see that function's
    // docstring), so badges built from it were showing the wrong (station,
    // not platform) route set -- and for a station with many routes spread
    // across several platforms, most of those routes don't actually serve
    // THIS stop_id, which is very likely why they looked "broken/missing"
    // for busy stops. `stop_routes` is the real per-stop_id field (falls
    // back to `r.route` only for a record from a build that predates it).
    function visibleRouteNames(r) {{
      return splitRoutes(r.stop_routes || r.route).filter(function(n) {{ return modeVisible(routeStyleFor(n, r.mode).mode); }});
    }}
    function prettyName(s) {{ return s ? String(s).replace(/_/g, ' ') : s; }}

    var ROUTES_PER_PAGE = 4, MAX_ANNOTATED = 400;
    var routePage = {{}};
    var badgeMarkers = {{}};
    window.__stopRouteBadgesOn = false;

    function badgeEl(r, idx) {{
      var names = visibleRouteNames(r);
      if (!names.length) return null;
      var pages = Math.ceil(names.length / ROUTES_PER_PAGE);
      var page = Math.min(routePage[idx] || 0, pages - 1);
      var slice = names.slice(page * ROUTES_PER_PAGE, (page + 1) * ROUTES_PER_PAGE);
      var inner = slice.map(function(n) {{ return routeBadgeHtml(n, r.mode, 9); }}).join('');
      var pager = '';
      if (pages > 1) {{
        pager = '<span style="display:inline-block;white-space:nowrap;font:600 9px sans-serif;color:#333;">' +
          '<span class="stop-route-pg" data-dir="-1" style="cursor:pointer;padding:0 3px;">&#9664;</span>' +
          (page + 1) + '/' + pages +
          '<span class="stop-route-pg" data-dir="1" style="cursor:pointer;padding:0 3px;">&#9654;</span></span>';
      }}
      var el = document.createElement('div');
      el.style.cssText = 'white-space:nowrap;background:rgba(255,255,255,0.9);border-radius:4px;padding:1px 3px;pointer-events:auto;';
      el.innerHTML = inner + pager;
      var buttons = el.querySelectorAll('.stop-route-pg');
      for (var i = 0; i < buttons.length; i++) {{
        (function(btn) {{
          btn.addEventListener('click', function(ev) {{
            ev.stopPropagation();
            var names2 = visibleRouteNames(r);
            var pages2 = Math.ceil(names2.length / ROUTES_PER_PAGE);
            var dir = parseInt(btn.getAttribute('data-dir'), 10);
            routePage[idx] = ((routePage[idx] || 0) + dir + pages2) % pages2;
            window.__refreshStopBadges();
          }});
        }})(buttons[i]);
      }}
      return el;
    }}

    // Badges must exist ONLY on stops that are ACTUALLY rendered as
    // individual markers right now -- i.e. exactly the features MapLibre's
    // own clustering has resolved to the `__stops_mode_icon` layer (not
    // folded into a `__stops_cluster` bubble). Rather than re-guessing the
    // clustering state with a duplicated zoom threshold (drifts out of sync
    // with the source's real `clusterRadius`/`clusterMaxZoom`, and ignores
    // that clustering can still leave points grouped exactly AT that zoom),
    // ask the map layer itself via `queryRenderedFeatures` -- the same
    // source of truth the icon layer uses to decide what's a dot vs a
    // cluster this frame. Each rendered feature's `id` is the `stopsData`
    // index (set 1:1 when `stopsFC` was built above), so no extra lookup.
    var __lastBadgeIdxKey = null;
    window.__updateStopRouteBadges = function() {{
      if (!window.__stopRouteBadgesOn) {{
        Object.keys(badgeMarkers).forEach(function(k) {{ badgeMarkers[k].remove(); delete badgeMarkers[k]; }});
        __lastBadgeIdxKey = null;
        return;
      }}
      var individualFeatures;
      try {{
        individualFeatures = map.queryRenderedFeatures({{layers: ['__stops_mode_icon']}});
      }} catch (err) {{
        // Layer not added/style not loaded yet -- nothing to badge this frame.
        return;
      }}
      // Dedupe across overlapping tiles, cap, then skip the (relatively
      // expensive) marker teardown/rebuild entirely on frames where the set
      // of individually-rendered stops hasn't actually changed (e.g. a pure
      // pan `render` tick, or route-page-button repaints).
      var seen = {{}}, idxs = [];
      for (var i = 0; i < individualFeatures.length && idxs.length < MAX_ANNOTATED; i++) {{
        var idx = individualFeatures[i].id;
        if (idx == null || seen[idx]) continue;
        seen[idx] = true;
        idxs.push(idx);
      }}
      idxs.sort(function(a, b) {{ return a - b; }});
      var key = idxs.join(',');
      if (key === __lastBadgeIdxKey) return;
      __lastBadgeIdxKey = key;
      Object.keys(badgeMarkers).forEach(function(k) {{ badgeMarkers[k].remove(); delete badgeMarkers[k]; }});
      idxs.forEach(function(idx) {{
        var r = stopsData.records[idx];
        if (!r) return;
        var el = badgeEl(r, idx);
        if (!el) return;
        var m = new maplibregl.Marker({{element: el, anchor: 'left', offset: [10, -6]}})
          .setLngLat([r.lon, r.lat]).addTo(map);
        badgeMarkers[idx] = m;
      }});
    }};
    window.__refreshStopBadges = window.__updateStopRouteBadges;
    // `render` fires every frame the map is animating/interacting (zoom,
    // pan, cluster re-resolution), which is what actually changes which
    // features are individual vs clustered -- `zoomend`/`moveend` alone left
    // a visible lag (and, at the moment a stop got swallowed into a
    // cluster, a stale badge lingering) since clustering can change even
    // mid-drag/zoom, not just once motion has fully settled.
    // NOTE: MapLibre's `Evented.on(type, listener)` -- unlike
    // jQuery/Leaflet -- does NOT support a single space-separated multi-
    // event string; `'zoomend moveend render'` silently registered a
    // listener for a nonexistent event that never fired (real bug: badges
    // only ever updated once, on checkbox toggle, and never deactivated on
    // zoom-out since the `render`/`zoomend` re-check never ran). Three
    // separate `.on()` calls instead.
    map.on('zoomend', window.__updateStopRouteBadges);
    map.on('moveend', window.__updateStopRouteBadges);
    map.on('render', window.__updateStopRouteBadges);

    // `maxWidth` explicit: MapLibre's own default (240px, via its
    // `.maplibregl-popup-content` CSS) would otherwise clip the wider
    // 340px inner `<div>` below regardless of that div's own inline style.
    var stopPopup = new maplibregl.Popup({{closeButton: true, closeOnClick: true, maxWidth: '360px'}});
    function fmt2(v) {{ return (v == null) ? null : Number(v).toFixed(2); }}

    // Round 19 (item 5 of round-15's original popup redesign, previously
    // deferred): real `parent_station` (round-16's data-pipeline fix means
    // it's now genuinely populated -- real GTFS linkage where the agency
    // provides it, else a 150m proximity-grouping fallback, never null) and
    // a real per-stop_id `stop_routes` field (distinct from station-wide
    // `all_routes`) let this build the sections round-15/17/18 deferred:
    // routes at this exact stop_id, "connections" (other routes reachable
    // via the same parent station but not stopping here), and a data panel
    // split into per-platform (mode -- genuinely computed per stop_id, see
    // `stops.py`'s `own_mode_df`) vs. per-station (headway/speed/stop
    // score/reliability -- genuinely computed at the `at="parent_station"`
    // grouping level, so honestly the SAME number for every platform
    // sharing a station, not fabricated per-platform figures).
    function stationSiblings(parentStation) {{
      if (!parentStation) return [];
      return stopsData.records.filter(function(r) {{ return r.parent_station === parentStation; }});
    }}
    // Item 7 (this round, verbatim user request): "do NOT show a
    // 'reliability: N/A (not measured)' line if reliability was not
    // measured for ANY stop in the dataset -- omit the whole row entirely
    // when it's globally unmeasured, not just show N/A per stop." Computed
    // once from the whole dataset (matches `_stops_json_data`'s docstring:
    // `reliability` is only ever non-null when the pipeline genuinely
    // measured headway coefficient-of-variation -- never true on this
    // codebase's real feeds today, so this is `false` for every build seen
    // so far, and the two reliability rows below are dropped entirely).
    var __reliabilityMeasuredAnywhere = stopsData.records.some(function(r) {{ return r.reliability != null; }});
    function fmtReliability(p) {{
      return (p.reliability != null) ? fmt2(p.reliability) : 'N/A (not measured)';
    }}
    map.on('click', '__stops_mode_icon', function(e) {{
      // Bug fix (user report): clicking a stop marker must NEVER also open
      // the hex/circle/census shape popup underneath it. This map embeds
      // MapLibre GL (this `map`) as an overlay inside the outer Leaflet map
      // that owns the hex/circle/census `L.geoJson` layers and their
      // `bindPopup`/`GeoJsonPopup` click handlers (see `_shape_popup_js`).
      // MapLibre's own per-layer `map.on('click', layerId, ...)` firing does
      // NOT stop the underlying browser click event from bubbling up
      // through the DOM to Leaflet's map-container click listener, which
      // does its own independent hit-test against its Canvas/SVG-rendered
      // shape layers and would open a second, wrong popup for whatever
      // shape happens to sit under this stop. `stopPropagation` on the
      // real DOM event (`e.originalEvent`, MapLibre's own `MapMouseEvent`
      // wrapper) is what actually prevents that bubbling; MapLibre's own
      // `e.preventDefault()` only suppresses MapLibre's default handling,
      // not the DOM event Leaflet listens for.
      if (e.originalEvent && e.originalEvent.stopPropagation) {{ e.originalEvent.stopPropagation(); }}
      var p = e.features[0].properties;
      var t = (p.stop_score != null) ? (p.stop_score - sMin) / Math.max(sMax - sMin, 1e-9) : 0;

      // "Routes at this stop": prefer the real per-stop_id `stop_routes`
      // field; fall back to the older station-level `route`/`all_routes`
      // field for any record that predates it (e.g. a cached build).
      var ownRouteStr = p.stop_routes || p.route || '';
      var ownRouteNames = splitRoutes(ownRouteStr).filter(function(n) {{ return modeVisible(routeStyleFor(n, p.mode).mode); }});
      var routeCell = ownRouteNames.length
        ? ownRouteNames.map(function(n) {{ return routeBadgeHtml(n, p.mode, 10); }}).join('')
        : '-';

      // "Connections via parent station": every route in the station-wide
      // `all_routes` set that ISN'T already shown as a route at this exact
      // stop_id -- only meaningful (and only rendered) when this stop's
      // `parent_station` genuinely groups more than this one stop_id.
      var siblings = stationSiblings(p.parent_station);
      var stationAllRoutesStr = p.all_routes || ownRouteStr;
      var stationRouteNames = splitRoutes(stationAllRoutesStr);
      var ownSet = {{}};
      ownRouteNames.forEach(function(n) {{ ownSet[normRouteName(n)] = true; }});
      var connectionNames = stationRouteNames.filter(function(n) {{ return !ownSet[normRouteName(n)]; }});
      var hasRealStation = p.parent_station && (siblings.length > 1 || connectionNames.length > 0);

      // Station label for the "connecting routes via ..." header: NOT
      // `prettyName(p.parent_station)` (real bug -- when no GTFS agency
      // supplies a real parent-station id, `parent_station` falls back to a
      // synthetic id built by joining the grouped stop_ids together, e.g.
      // "26738_25025_22532_5426_28440"; `prettyName` only swaps underscores
      // for spaces, so that rendered as a run of unrelated-looking numbers
      // instead of a place name). Use this stop's own real `stop_name`
      // instead -- always human-readable, and every stop already carries
      // one (the popup's title falls back to 'Stop' only if it's blank; if
      // that fallback ever triggers here too, the label just reads "near
      // Stop", not wrong, only generic).
      var stationLabel = prettyName(p.stop_name || 'this stop');
      // Real parent-station name, when the GTFS agency's `parent_station`
      // points at a genuine `stops.txt` row of its own (see `stops.py`'s
      // `parent_station_name` -- null on the `stop_group_distance`
      // proximity-fallback path, which has no such row). Falls back to
      // `stationLabel` (this platform's own name) so the "parent station
      // data" section header always reads sensibly even without a real
      // station-level name.
      var parentStationLabel = prettyName(p.parent_station_name || stationLabel);

      // Bug fix (user report, "El Arenal" station, 2026-08-29): the "stop
      // group"/"parent station data" section used to just repeat `p.mode`
      // (THIS platform's own mode) under a "mode" label that implied it was
      // the STATION's aggregated mode -- misleading whenever a station mixes
      // modes with very different service levels (e.g. rail + connecting
      // buses), since the headway/speed/score numbers in that same section
      // are genuinely computed at the parent-station level while the mode
      // label silently claimed to be this one platform's mode instead of
      // reflecting what that station-level data is actually FOR (e.g. a
      // station shown as "mode: rail" whose headway is really bus-dominated).
      // `p.station_modes` (see `stops.py`) is the real, cheaply-computed set
      // of mode classes actually present at this station (e.g. "bus + rail"
      // for a mixed station, just "rail" for a real single-mode one) --
      // showing that instead of a single (possibly wrong) mode name is
      // honest about what the combined headway/speed number below actually
      // covers, without requiring a separate, expensive mode-restricted
      // headway recomputation on the backend (attempted and reverted --
      // OOM'd a real GTFS feed even at 22 GB, see `stops.py`'s comment).
      var stationMode = (hasRealStation && p.station_modes) ? p.station_modes : p.mode;
      // Item (verbatim user request): "connecting routes via [station] here
      // I would like each single stop id with its routes of the parent
      // station. So stop id name, routes, stop id name, routes etc" --
      // replaces the old single flattened/deduplicated route-badge list
      // (which lost which specific platform each route actually serves)
      // with one row per sibling stop_id sharing this parent station, each
      // showing its own name and its own routes -- the same per-platform
      // breakdown a previous, since-dropped "connecting stops" section
      // used to show, now merged into this section instead of living
      // alongside a separate flattened one.
      var connectionsHtml = '';
      if (hasRealStation && siblings.length) {{
        var siblingRowsHtml = siblings.map(function(s) {{
          var sRouteNames = splitRoutes(s.stop_routes || s.route || '').filter(function(n) {{
            return modeVisible(routeStyleFor(n, s.mode).mode);
          }});
          var sBadges = sRouteNames.length
            ? sRouteNames.map(function(n) {{ return routeBadgeHtml(n, s.mode, 10); }}).join('')
            : '<span style="color:#888;">no routes</span>';
          return '<div style="margin-top:4px;"><div style="font-size:11px;">' + prettyName(s.stop_name || 'Stop') +
            '</div><div style="margin-top:2px;">' + sBadges + '</div></div>';
        }}).join('');
        connectionsHtml += '<div style="margin-top:6px;"><div style="font-weight:600;color:#555;font-size:11px;">stops at ' +
          stationLabel + '</div>' + siblingRowsHtml + '</div>';
      }}
      // No `hasRealStation`: this stop_id has no real grouping (its own
      // `parent_station` fallback IS its own `stop_id`, single-member
      // "station") -- gracefully omit the section entirely rather than show
      // an empty/fake one.

      var rows = '';
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;vertical-align:top;">routes here</td><td style="padding:2px 0;">' + routeCell + '</td></tr>';
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">mode (this stop)</td><td>' + p.mode + '</td></tr>';
      // Item 7: the "reliability" row is omitted ENTIRELY (not shown as
      // "N/A") when reliability was not measured for ANY stop in this
      // dataset -- see `__reliabilityMeasuredAnywhere` above.
      if (__reliabilityMeasuredAnywhere) {{
        rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">reliability (this stop)</td><td>' + fmtReliability(p) + '</td></tr>';
      }}
      // Renamed from "stop group data" to "parent station data" (verbatim
      // user request -- "stop group" was confusing users) and now also
      // shows the real parent station NAME (previously missing entirely)
      // and its genuine dominant-mode-based `mode` (`stationMode`, see
      // above -- previously this row just repeated `p.mode`, this
      // platform's own mode, which is the exact bug the user reported).
      rows += '<tr><td colspan="2" style="padding-top:4px;border-top:1px solid #eee;font-weight:600;color:#555;font-size:11px;">' +
        (hasRealStation ? ('parent station data (' + parentStationLabel + ')') : 'parent station data') + '</td></tr>';
      rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">mode</td><td>' + stationMode +
        (p.stop_score != null ? ' <span style="color:#888;">(score ' + fmt2(p.stop_score) + ')</span>' : '') + '</td></tr>';
      if (p.headway_minutes != null) {{
        rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">headway</td><td>' + fmt2(p.headway_minutes) + ' min' +
          (p.frequency_score != null ? ' <span style="color:#888;">(score ' + fmt2(p.frequency_score) + ')</span>' : '') + '</td></tr>';
      }}
      if (p.avg_speed_kmh != null) {{
        rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">speed</td><td>' + fmt2(p.avg_speed_kmh) + ' km/h' +
          (p.speed_score != null ? ' <span style="color:#888;">(score ' + fmt2(p.speed_score) + ')</span>' : '') + '</td></tr>';
      }}
      if (__reliabilityMeasuredAnywhere) {{
        // No separate "(score X)" here -- `reliability` IS already a 0-1
        // score (see `_stops_json_data`'s docstring), unlike headway/speed
        // which are raw physical units alongside their own score field.
        rows += '<tr><td style="font-weight:600;color:#555;padding:2px 8px 2px 0;">reliability</td><td>' + fmtReliability(p) + '</td></tr>';
      }}
      rows += '<tr><td style="font-weight:700;padding:4px 8px 0 0;border-top:1px solid #eee;">stop score</td><td style="font-weight:700;border-top:1px solid #eee;color:' + blueFor(t) + ';">' + (p.stop_score != null ? fmt2(p.stop_score) : '-') + '</td></tr>';
      // Widened from 260px -> 340px (user report: too narrow for the
      // parent-station-data section's new station-name header + longer
      // route-badge rows).
      var html = '<div style="font:12px sans-serif;max-width:340px;"><div style="font-weight:700;margin-bottom:4px;">' +
        prettyName(p.stop_name || 'Stop') + ' (' + p.mode + ')</div><table style="border-collapse:collapse;width:100%;">' + rows + '</table>' +
        connectionsHtml + '</div>';
      stopPopup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    }});
    map.on('click', '__stops_cluster', function(e) {{
      // Same fix as `__stops_mode_icon` above: a cluster click must not
      // also open the shape layer underneath it.
      if (e.originalEvent && e.originalEvent.stopPropagation) {{ e.originalEvent.stopPropagation(); }}
      var f = map.queryRenderedFeatures(e.point, {{layers: ['__stops_cluster']}})[0];
      var clusterId = f.properties.cluster_id;
      map.getSource('__stops_overlay').getClusterExpansionZoom(clusterId, function(err, zoom) {{
        if (err) return;
        map.easeTo({{center: f.geometry.coordinates, zoom: zoom}});
      }});
    }});
    ['__stops_mode_icon', '__stops_cluster'].forEach(function(id) {{
      map.on('mouseenter', id, function() {{ map.getCanvas().style.cursor = 'pointer'; }});
      map.on('mouseleave', id, function() {{ map.getCanvas().style.cursor = ''; }});
    }});
    window.__stopsOverlayLoaded = true;
  }});
""")

    # Real reported gap (user testing feedback, round-8-followup): the
    # stops/routes/labels overlay above was fully implemented and rendering
    # but had NO on-page toggle -- Folium's `_control_panel_html` gives the
    # user "Stops"/"Show routes"+bus/tram/rail/"stop names"/"Route labels"
    # checkboxes (see that function), MapLibre had none, even though
    # `window.__routeModeVisible`/`window.__updateRouteLayers` already
    # existed as callable-but-unexposed JS. Wire real checkboxes (injected
    # as HTML by `_inject_maplibre_stops_routes_into_saved_html`) to the
    # already-working plumbing instead of adding a button that only calls
    # a console function.
    parts.append("""
  whenReady(function() {
    function onChg(id, fn) { var el = document.getElementById(id); if (el) el.addEventListener('change', fn); }
    function setVis(layerId, visible) {
      if (map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
    }
    onChg('ovStopsCheckbox', function(e) {
      ['__stops_mode_icon', '__stops_score_label', '__stops_cluster', '__stops_cluster_count'].forEach(function(id) { setVis(id, e.target.checked); });
      if (!e.target.checked) window.__stopRouteBadgesOn = false;
      if (window.__updateStopRouteBadges) window.__updateStopRouteBadges();
    });
    onChg('ovStopNamesCheckbox', function(e) { setVis('__stops_label', e.target.checked); });
    // Round 15 (item 1): "route names checkbox should be on every stop" --
    // per-stop paginated route-badge chips, distinct from the "Show routes"
    // route-LINE toggle above and from the always-on `__routes_label`
    // on-line labels (round 14).
    onChg('ovStopRouteBadgesCheckbox', function(e) {
      window.__stopRouteBadgesOn = e.target.checked;
      if (window.__updateStopRouteBadges) window.__updateStopRouteBadges();
    });
    onChg('ovShowRoutesCheckbox', function(e) {
      window.__routesMasterVisible = e.target.checked;
      ['bus', 'tram', 'rail'].forEach(function(m) {
        var v = e.target.checked && window.__routeModeVisible[m] !== false;
        setVis('__routes_line_' + m, v);
        // Issue 2's cased-line casing layer must follow the colored line.
        setVis('__routes_casing_' + m, v);
      });
    });
    ['bus', 'tram', 'rail'].forEach(function(m) {
      onChg('ovRoute_' + m + '_Checkbox', function(e) {
        window.__routeModeVisible[m] = e.target.checked;
        var master = document.getElementById('ovShowRoutesCheckbox');
        var v = (!master || master.checked) && e.target.checked;
        setVis('__routes_line_' + m, v);
        setVis('__routes_casing_' + m, v);
      });
    });
  });
""")

    parts.append("})();")
    return "\n".join(parts)


def _routes_geojson_data(routes_gdf: gpd.GeoDataFrame) -> str:
    """GeoJSON FeatureCollection version of `_routes_json_data`, for MapLibre's GeoJSON source.

    Same rounding/fallback-color/mode conventions as `_routes_json_data`
    (which emits Leaflet's `[lat, lon]` polyline order instead) -- this
    emits standard GeoJSON `[lon, lat]` coordinate order.
    """
    gdf = routes_gdf.to_crs(4326) if routes_gdf.crs is not None else routes_gdf
    features = []
    min_zoom = dict(ROUTE_MODE_MIN_ZOOM)
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        mode = str(row.get("mode") or "bus")
        color = str(row.get("color") or MODE_FALLBACK_COLOR.get(mode, "#8b0000"))
        label = str(row.get("route_label") or row.get("route_id") or "")
        geometry = json.loads(shapely.to_geojson(geom))
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {"route_label": label, "mode": mode, "color": color},
        })
    styles = _route_styles_dict(gdf)
    return json.dumps({
        "type": "FeatureCollection", "features": features, "min_zoom": min_zoom, "styles": styles,
    })


def _inject_maplibre_stops_routes_into_saved_html(path: str, stops_gdf, routes_gdf, score_domain: Tuple[float, float]) -> str:
    """Splice `_maplibre_stops_routes_js`'s stops/routes/labels/popups overlay into a MapLibre `map.html`.

    Same "share the page's own `<script>` scope" pattern as
    `_inject_maplibre_editor_into_saved_html` -- appended just before the
    trailing `</script>\\n</body>` marker so `map` (a `const` declared in
    that same script tag) is already in scope.
    """
    if (stops_gdf is None or stops_gdf.empty) and (routes_gdf is None or routes_gdf.empty):
        return path
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    # MapLibre never gets the Font Awesome CDN `<link>` that
    # `_inject_controls_into_saved_html` adds for the Folium branch (that
    # function is Folium-only, see its call site). The stop-icon rasterizer
    # below (`drawModeIcon`) needs the FA webfont loaded to render bus/tram/
    # rail glyphs onto its offscreen canvas, so add it here instead, in
    # `<head>`, so the browser starts fetching it as early as possible.
    fa_link = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">\n'
    if "font-awesome" not in html and "</head>" in html:
        html = html.replace("</head>", fa_link + "</head>", 1)
    has_stops = stops_gdf is not None and not stops_gdf.empty
    has_routes = routes_gdf is not None and not routes_gdf.empty
    # Real on-page checkboxes for the stops/routes overlay -- see the
    # docstring on the checkbox-wiring block appended in
    # `_maplibre_stops_routes_js` for why this exists (functionality was
    # already implemented but had no clickable UI). Placed in the same
    # top-left "Route editor" panel's visual neighborhood via its own small
    # box, inserted right before the route-toolbar div.
    rows = []
    if has_stops:
        rows.append('<label><input type="checkbox" id="ovStopsCheckbox" checked> Stops</label><br>')
    if has_routes:
        rows.append('<label><input type="checkbox" id="ovShowRoutesCheckbox" checked> Show routes</label><br>')
        rows.append(
            '<label><input type="checkbox" id="ovRoute_bus_Checkbox" checked> bus</label> '
            '<label><input type="checkbox" id="ovRoute_tram_Checkbox" checked> tram</label> '
            '<label><input type="checkbox" id="ovRoute_rail_Checkbox" checked> rail</label><br>'
        )
    if has_stops:
        rows.append('<label><input type="checkbox" id="ovStopNamesCheckbox"> stop names</label><br>')
        rows.append('<label><input type="checkbox" id="ovStopRouteBadgesCheckbox"> route badges (per stop)</label><br>')
    # Round-14: route short names on route lines now always render (no
    # checkbox) -- see `_maplibre_stops_routes_js`'s `__routes_label` layer.
    # Round-14: user feedback -- "overlays should be part of the
    # layercontrol" -- fold these checkboxes into the consolidated
    # `#layer-switcher-body` box (geohierarchy's `render.py`) instead of a
    # separate floating "Overlays" box, so there's one legend+layer-control
    # panel instead of two. Inserted right after the box's own "Layers"
    # heading, before the named/overlay group radios/checkboxes it already
    # has, with an <hr> separator.
    overlay_controls_html = (
        '<div id="overlay-controls">\n<strong>Stops &amp; routes</strong><br>\n'
        + "\n".join(rows) + "\n</div>\n<hr>\n"
    )
    div_marker = '<strong>Layers</strong><br>'
    if div_marker in html:
        html = html.replace(div_marker, div_marker + "\n" + overlay_controls_html, 1)
    else:
        # Fallback for older/other templates without the consolidated box.
        alt_marker = '<div id="route-toolbar"'
        if alt_marker in html:
            html = html.replace(alt_marker, overlay_controls_html + alt_marker, 1)
    js = _maplibre_stops_routes_js(stops_gdf, routes_gdf, score_domain)
    marker = "</script>\n</body>"
    if marker not in html:
        raise ValueError(f"{path}: expected trailing '{marker}' from save_multi_maplibre, not found")
    html = html.replace(marker, js + "\n" + marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _inject_maplibre_topbar_into_saved_html(path: str, enable_place_comparison: bool = False) -> str:
    """Splice a top-center place/scenario/results bar into a MapLibre `map.html` (item C, round 13).

    User feedback (verbatim): "top center I want selector for place (if
    multiple places), scenario, and the results (level of service, costs if
    edits, impact etc)." Built as one cohesive pill/bar, reusing plumbing
    that already exists rather than re-deriving it:
      - place selector: only rendered `if enable_place_comparison` (same
        flag `_stats_panel_js` already gates its own place-rank tab and
        "Compare with place" selects on) -- given the `.comparePlaceSelect`
        class so the stats panel's own `__populatePlaceSelects()` (see
        `_stats_panel_js`) fills it from `CS_PLACE_MANIFEST` for free, no
        separate roster/JS needed. Correctly absent for boston_city, which
        was built with `enable_place_comparison=False`.
      - scenario selector: drives `window.__editor.state.scenarios` /
        `window.__editor.loadScenario(name)` (`_maplibre_compute_access_js`,
        exposed on `window.__editor` above) -- the SAME model the
        left-panel "Route editor"'s own scenario-list panel drives, so
        switching either one keeps the other in sync (both just mutate
        `S.activeScenario` via the same `loadScenario`).
      - results summary: reads `window.__editor.lastCompute()` for the
        active scenario's `new_access`/`impact`/`cost_musd` (null until
        that scenario has actually been Computed at least once -- shown as
        "-"), and `window.__editor.baselineStats()` for the
        reserved/baseline network's plain level of service (no cost/impact --
        it has no edits to attribute them to).

    Polls every 800ms (`setInterval`) rather than hooking every mutation
    site (`loadScenario`/`saveAsScenario`/the compute button) individually --
    cheap given how small the DOM diff is, and robust to the several
    different places `S.activeScenario`/`lastCompute` can change from
    (this bar's own select, the left panel's scenario-list panel, "+ Create
    new route", "Save as scenario", "Compute").
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    if "window.__editor" not in html and "'__editor'" not in html:
        # No stats_by_area / no editor wired in for this build -- nothing
        # for the bar to show (mirrors the other injectors' silent no-ops
        # when their prerequisite data is absent).
        pass
    place_select_html = (
        '<select id="topBarPlaceSelect" class="comparePlaceSelect" '
        'style="border:none;background:transparent;font:600 12px sans-serif;color:#1a1a1a;padding:2px 4px;">'
        "</select>"
        '<span style="width:1px;background:#ddd;align-self:stretch;margin:0 6px;"></span>'
        if enable_place_comparison else ""
    )
    bar_html = f"""
<div id="top-center-bar" style="position:absolute; top:10px; left:50%; transform:translateX(-50%);
  z-index:6; background:white; border-radius:20px; box-shadow:0 1px 6px rgba(0,0,0,0.25);
  font:12px sans-serif; padding:6px 14px; display:flex; align-items:center; gap:0;
  max-width:92vw; white-space:nowrap;">
  {place_select_html}
  <select id="topBarScenarioSelect" style="border:none;background:transparent;font:600 12px sans-serif;
    color:#1a1a1a;padding:2px 4px; cursor:pointer;"></select>
  <span style="width:1px;background:#ddd;align-self:stretch;margin:0 10px;"></span>
  <span id="topBarResults" style="color:#333; overflow-x:auto;"></span>
</div>
"""
    marker = "<div id=\"map\"></div>"
    if marker not in html:
        raise ValueError(f"{path}: expected '{marker}' from save_multi_maplibre, not found")
    html = html.replace(marker, marker + bar_html, 1)

    js = """
(function() {
  function fmtNum(v, d) {
    if (v == null || !isFinite(v)) return '-';
    d = (d == null) ? 2 : d;
    return Number(v).toFixed(d);
  }
  function refreshTopBar() {
    var ed = window.__editor;
    var sel = document.getElementById('topBarScenarioSelect');
    var out = document.getElementById('topBarResults');
    if (!ed || !sel || !out) return;
    var reserved = ed.reservedName || 'current';
    var options = [{value: '', label: reserved + ' (baseline)'}].concat(
      (ed.state.scenarios || []).map(function(sc) { return {value: sc.name, label: sc.name}; })
    );
    var want = (ed.state.activeScenario != null) ? ed.state.activeScenario : '';
    var optionsHtml = options.map(function(o) {
      return '<option value="' + o.value + '"' + (o.value === want ? ' selected' : '') + '>' + o.label + '</option>';
    }).join('');
    if (sel.getAttribute('data-html') !== optionsHtml) {
      sel.innerHTML = optionsHtml;
      sel.setAttribute('data-html', optionsHtml);
    }
    sel.value = want;

    // Item 2 (verbatim user request): fields with a null value (e.g. cost/
    // impact on the read-only baseline, which has no edits to attribute
    // them to) are OMITTED entirely, not shown as "-" -- build a list of
    // only the fields that actually have a value, then join with the same
    // separator.
    var isBaseline = !ed.state.activeScenario;
    var fields = [];
    if (isBaseline) {
      var base = ed.baselineStats ? ed.baselineStats() : null;
      // Item 3 (verbatim user request): "instead of access xx I would like
      // to have just the number xxx" -- bare value, no "access" label.
      if (base && base.access != null) fields.push('<b>' + fmtNum(base.access, 2) + '</b>');
      // cost/impact are never applicable to the baseline (no edits) -- omitted, not "-".
    } else {
      var lc = ed.lastCompute ? ed.lastCompute() : null;
      if (lc && lc.ok) {
        if (lc.new_access != null) fields.push('<b>' + fmtNum(lc.new_access, 2) + '</b>');
        if (lc.cost_musd != null) fields.push('cost <b>$' + fmtNum(lc.cost_musd, 1) + 'M</b>');
        if (lc.impact != null) fields.push('impact <b>' + fmtNum(lc.impact, Math.abs(lc.impact) < 100 ? 1 : 0) + '</b>');
      }
    }
    var html = fields.length ? fields.join(' &nbsp;|&nbsp; ') : '<span style="color:#888;">not computed yet</span>';
    out.innerHTML = html;
  }
  document.addEventListener('change', function(e) {
    if (e.target && e.target.id === 'topBarScenarioSelect') {
      var ed = window.__editor;
      if (ed && ed.loadScenario) ed.loadScenario(e.target.value);
      refreshTopBar();
    }
  });
  setInterval(refreshTopBar, 800);
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    refreshTopBar();
  } else {
    document.addEventListener('DOMContentLoaded', refreshTopBar);
  }
  window.__refreshTopBar = refreshTopBar;
})();
"""
    marker2 = "</script>\n</body>"
    if marker2 not in html:
        raise ValueError(f"{path}: expected trailing '{marker2}' from save_multi_maplibre, not found")
    html = html.replace(marker2, js + "\n" + marker2, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _inject_maplibre_legend_into_saved_html(path: str, score_domain: Tuple[float, float], show_development: bool) -> str:
    """Splice `_legend_html`'s access-score/stop-score/circle/opacity legend into a MapLibre `map.html`.

    Round-8-followup fix for a real reported gap: the MapLibre page had NO
    legend at all (Folium's `_legend_html` was only ever wired into the
    Folium/`_inject_controls_into_saved_html` branch). `_legend_html` is
    plain, renderer-agnostic HTML (a handful of absolutely-unpositioned
    divs, no Leaflet control dependency) -- Folium promotes it into an
    `L.control`; here it's simpler to just pin it top-right with inline
    absolute positioning, matching `#layer-switcher`'s own placement
    pattern in `save_multi_maplibre`. Circle-size/opacity/development
    sub-legends start `display:none` (same as Folium's default) and are now
    toggled/populated live by `geohierarchy.maps.maplibre.render`'s own
    `style_controls_js`/`switcher_js` (`__updateCircleLegend`/
    `__updateOpacityLegend`/the `__overlay_group` "development" case) --
    MapLibre gained real "Circle size by"/"Opacity by" controls in an
    earlier round, and this legend wiring was the one piece of that port
    left undone until now.
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    legend_div = _legend_html(score_domain, show_development)
    placeholder = "<!--LEGEND_PLACEHOLDER-->"
    if placeholder in html:
        # Round-13: merged into the single #layer-switcher box (`save_multi_maplibre`
        # now reserves a `#legend-slot` there) instead of a second floating panel --
        # the legend is always visible, the "Layers" button below it expands/collapses
        # the rest of the layer controls.
        html = html.replace(placeholder, legend_div, 1)
    else:
        # Fallback for HTML saved before round 13's marker existed.
        wrapped = (
            '<div id="mapLegendWrap" style="position:absolute; top:10px; right:180px; z-index:5;">\n'
            + legend_div
            + "\n</div>\n"
        )
        marker = "</body>"
        if marker not in html:
            raise ValueError(f"{path}: expected trailing '{marker}', not found")
        html = html.replace(marker, wrapped + marker, 1)
    # Bug fix (user report: "popup box ... not implemented" on MapLibre
    # hexagon/circle/census click): `save_multi_maplibre`'s per-shape
    # `popupFns` (see `_shape_popup_field_js`-generated `window.__shapePopupHtml
    # ? window.__shapePopupHtml(properties) : ''` fallback around build.py:2570)
    # calls `window.__shapePopupHtml`, but `_shape_popup_js()` -- the function
    # that actually DEFINES `__shapePopupHtml` on `window` -- was only ever
    # spliced into the legacy Folium/Leaflet branch via
    # `_inject_controls_into_saved_html` (build.py:9388). The MapLibre branch
    # never called it, so every hexagon/circle/census click hit the `: ''`
    # fallback and opened an empty popup bubble. Runs LAST here (this
    # injector already runs last in the MapLibre chain, see its call site's
    # comment) so it can use the same `</script>\n</body>` splice point every
    # earlier MapLibre injector also depends on, without disturbing them.
    popup_marker = "</script>\n</body>"
    if popup_marker in html:
        html = html.replace(
            popup_marker, f"{_shape_popup_js()}\n</script>\n</body>", 1
        )
    else:
        # Fallback: no `</script>\n</body>` splice point found (unexpected
        # page shape) -- inject a standalone `<script>` block right before
        # `</body>` instead of silently dropping the popup implementation.
        marker = "</body>"
        if marker not in html:
            raise ValueError(f"{path}: expected trailing '{marker}', not found")
        html = html.replace(
            marker, f"<script>\n{_shape_popup_js()}\n</script>\n" + marker, 1
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def build_download_panel_block(manifest_json: str, base_path: str = "results", place_options: Optional[List[dict]] = None) -> str:
    """Shared HTML+JS for the "Download data" panel.

    Used by both `_inject_maplibre_download_panel_into_saved_html` (one
    city's own map, `place_options=None`) and `combined_map.py` (which
    passes `place_options` -- one `{value, label, base_path}` per city --
    to add the place dropdown described in the task; `base_path` there is
    each city's own relative path from the combined map's HTML location,
    e.g. `"../beerseba/results"`, since the combined map has no downloads
    of its own to point at).

    `manifest_json` is a JSON object (baked in, same pattern as
    `_stats_json_data`) shaped:
        {"scopes": ["metro","core"], "h3_resolutions": [5,7,9,11],
         "census_levels": ["county","tract",...], "has_stops": true,
         "has_routes": true}
    For the combined map, `manifest_json` is instead a JSON *mapping* of
    place value -> that same per-place object (`window.__downloadManifests`),
    since different cities can have different resolutions/levels/census
    availability.
    """
    place_select_html = ""
    if place_options:
        opts = "".join(
            f'<option value="{p["value"]}">{p["label"]}</option>' for p in place_options
        )
        place_select_html = f"""
  <label style="display:block;font-weight:600;margin-bottom:4px;">Place</label>
  <select id="dlPlaceSelect" style="width:100%;margin-bottom:10px;font:12px sans-serif;padding:3px;">{opts}</select>
"""
    base_paths_json = "null"
    if place_options:
        import json as _json
        base_paths_json = _json.dumps({p["value"]: p["base_path"] for p in place_options})
    return f"""
<!-- Client-side zip assembly for the multi-resolution/multi-level download
     (user report: "download a zip file with all geoparquets" when more than
     one resolution/level is selected -- this page is static/offline-
     generated with no server, so the zip has to be built in-browser from the
     fetched geoparquet Blobs). JSZip is pulled from cdnjs, the same CDN +
     exact-version-pin convention already used for font-awesome elsewhere in
     this module (see `fa_link` above) -- picked over fflate (initially
     tried) because fflate is not actually published on cdnjs (a real 404
     live-verified against the allowed CDN, not a hypothetical), while
     JSZip's 3.10.1 UMD build is. -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<style>
  #downloadButton {{ transition: transform 0.12s, box-shadow 0.12s; }}
  #downloadButton:hover {{ transform: scale(1.08); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }}
</style>
<!-- Bottom-left corner, immediately RIGHT of #statsButton (same `bottom`,
     `left` offset by that button's own 34px width + a 10px gap) rather than
     stacked directly above it -- stacking put this button's own circle
     flush against/overlapping #statsButton's top edge (both are absolutely
     positioned with no shared layout flow to keep them apart), a real
     visual bug reported live. Side-by-side avoids that regardless of
     which one is clicked first. -->
<div id="downloadButton" title="Download data" style="position:absolute;bottom:56px;left:54px;z-index:1000;
  background:white;width:34px;height:34px;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,0.3);
  display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer;">⬇️</div>
<div id="downloadPanel" style="display:none;position:absolute;bottom:56px;left:98px;z-index:1000;background:white;
  width:280px;border-radius:6px;box-shadow:0 1px 6px rgba(0,0,0,0.35);font:12px sans-serif;padding:12px;">
  <div style="font-weight:700;margin-bottom:8px;">Download data</div>
  {place_select_html}
  <label style="display:block;font-weight:600;margin-bottom:4px;">Scope</label>
  <select id="dlScopeSelect" style="width:100%;margin-bottom:10px;font:12px sans-serif;padding:3px;">
    <option value="metro">Metro</option>
    <option value="core">City core</option>
  </select>
  <div style="margin-bottom:6px;">
    <label style="display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="dlH3Check" checked> H3 grid
    </label>
    <!-- Multi-select (user report: "should allow user to decide downloading
         multiple resolutions ... so multiple should be selectable and not
         only one") -- a checkbox list, matching this panel's own existing
         Stops/Routes checkbox style, replaces what used to be a single
         `<select>` picking exactly one resolution. Populated in JS
         (`populateSelects`) from the manifest, same as the old `<select>`
         was. -->
    <div id="dlH3ResList" style="margin-top:3px;max-height:90px;overflow-y:auto;border:1px solid #ddd;border-radius:3px;padding:3px 6px;"></div>
  </div>
  <div id="dlCensusRow" style="margin-bottom:6px;">
    <label style="display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="dlCensusCheck"> Census
    </label>
    <div id="dlCensusLevelList" style="margin-top:3px;max-height:90px;overflow-y:auto;border:1px solid #ddd;border-radius:3px;padding:3px 6px;"></div>
  </div>
  <div style="margin-bottom:6px;">
    <label id="dlStopsRow" style="display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="dlStopsCheck"> Stops
    </label>
  </div>
  <div style="margin-bottom:10px;">
    <label id="dlRoutesRow" style="display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="dlRoutesCheck"> Route shapes
    </label>
  </div>
  <button id="dlGoButton" style="width:100%;padding:6px;background:#3a6ea8;color:white;border:none;
    border-radius:5px;cursor:pointer;font:600 12px sans-serif;">Download</button>
</div>
<script>
(function() {{
  window.__downloadManifests = {manifest_json};
  window.__downloadBasePaths = {base_paths_json};
  var basePath = {base_path!r};

  function currentManifest() {{
    var m = window.__downloadManifests;
    var placeSel = document.getElementById('dlPlaceSelect');
    if (placeSel) return m[placeSel.value];
    return m;
  }}
  function currentBasePath() {{
    var placeSel = document.getElementById('dlPlaceSelect');
    if (placeSel && placeSel.value && window.__downloadBasePaths) {{
      return window.__downloadBasePaths[placeSel.value];
    }}
    return basePath;
  }}

  // Builds one checkbox-per-value inside `container` (replacing whatever
  // was a single `<select>` before) -- shared by the H3-resolution and
  // census-level pickers below, both now multi-select. First value defaults
  // checked (matches the old `<select>`'s "always has *a* value selected"
  // behavior) so a user who never touches these still gets the same single
  // download they used to.
  function populateCheckboxList(container, values, labelFn) {{
    container.innerHTML = values.map(function(v, i) {{
      var id = container.id + '_' + i;
      return '<label style="display:flex;align-items:center;gap:6px;padding:1px 0;">' +
        '<input type="checkbox" class="' + container.id + 'Item" value="' + v + '"' +
        (i === 0 ? ' checked' : '') + '> ' + labelFn(v) + '</label>';
    }}).join('');
  }}

  function checkedValues(container) {{
    return Array.prototype.slice.call(container.querySelectorAll('input[type=checkbox]:checked'))
      .map(function(cb) {{ return cb.value; }});
  }}

  function populateSelects() {{
    var man = currentManifest();
    if (!man) return;
    populateCheckboxList(
      document.getElementById('dlH3ResList'), man.h3_resolutions || [],
      function(r) {{ return 'Resolution ' + r; }}
    );
    var censusRow = document.getElementById('dlCensusRow');
    var levels = man.census_levels || [];
    if (levels.length) {{
      censusRow.style.display = '';
      populateCheckboxList(
        document.getElementById('dlCensusLevelList'), levels, function(l) {{ return l; }}
      );
    }} else {{
      censusRow.style.display = 'none';
      document.getElementById('dlCensusCheck').checked = false;
    }}
    document.getElementById('dlStopsRow').style.display = man.has_stops ? '' : 'none';
    if (!man.has_stops) document.getElementById('dlStopsCheck').checked = false;
    document.getElementById('dlRoutesRow').style.display = man.has_routes ? '' : 'none';
    if (!man.has_routes) document.getElementById('dlRoutesCheck').checked = false;
  }}

  function triggerDownload(url) {{
    var a = document.createElement('a');
    a.href = url;
    a.download = url.split('/').pop();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }}

  function triggerBlobDownload(blob, filename) {{
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() {{ URL.revokeObjectURL(url); }}, 1000);
  }}

  // Fetches every selected geoparquet file and, when more than one is
  // selected, bundles them into a single .zip built entirely client-side
  // (this is a static page with no server to zip on) via JSZip -- see the
  // `<script src=".../jszip...">` tag above. A single selected file still
  // downloads directly as before (no zip overhead/extra click for the
  // common one-file case).
  function downloadSelected(urls) {{
    if (!urls.length) return;
    if (urls.length === 1) {{
      triggerDownload(urls[0]);
      return;
    }}
    Promise.all(urls.map(function(url) {{
      return fetch(url).then(function(resp) {{
        if (!resp.ok) throw new Error('fetch failed: ' + url + ' (' + resp.status + ')');
        return resp.arrayBuffer();
      }}).then(function(buf) {{
        return {{name: url.split('/').pop(), data: buf}};
      }});
    }})).then(function(files) {{
      var zip = new JSZip();
      // De-dupe filenames (e.g. same census level name across scopes should
      // never collide here since scope is fixed per click, but stay safe).
      var seen = {{}};
      files.forEach(function(f) {{
        var name = f.name;
        if (seen[name] != null) {{
          seen[name] += 1;
          var dot = name.lastIndexOf('.');
          name = dot === -1 ? (name + '_' + seen[name]) : (name.slice(0, dot) + '_' + seen[name] + name.slice(dot));
        }} else {{
          seen[name] = 0;
        }}
        zip.file(name, f.data);
      }});
      return zip.generateAsync({{type: 'blob', compression: 'DEFLATE', compressionOptions: {{level: 6}}}});
    }}).then(function(blob) {{
      triggerBlobDownload(blob, 'transitlos_data.zip');
    }}).catch(function(err) {{
      console.error('[download panel] zip build failed', err);
      alert('Download failed: ' + err.message);
    }});
  }}

  document.getElementById('downloadButton').addEventListener('click', function() {{
    var panel = document.getElementById('downloadPanel');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  }});

  var placeSel = document.getElementById('dlPlaceSelect');
  if (placeSel) placeSel.addEventListener('change', populateSelects);

  document.getElementById('dlGoButton').addEventListener('click', function() {{
    var scope = document.getElementById('dlScopeSelect').value;
    var base = currentBasePath();
    var urls = [];
    if (document.getElementById('dlH3Check').checked) {{
      checkedValues(document.getElementById('dlH3ResList')).forEach(function(res) {{
        urls.push(base + '/' + scope + '/downloads/h3_res' + res + '.parquet');
      }});
    }}
    if (document.getElementById('dlCensusCheck').checked) {{
      checkedValues(document.getElementById('dlCensusLevelList')).forEach(function(level) {{
        urls.push(base + '/' + scope + '/downloads/census_' + level + '.parquet');
      }});
    }}
    if (document.getElementById('dlStopsCheck').checked) {{
      urls.push(base + '/downloads/stops.parquet');
    }}
    if (document.getElementById('dlRoutesCheck').checked) {{
      urls.push(base + '/downloads/routes.parquet');
    }}
    // Multiple resolutions/levels (or any combination of >1 file total) now
    // bundle into one .zip -- see `downloadSelected`. A single selected file
    // still falls straight through to a plain `<a download>`, same as
    // before.
    downloadSelected(urls);
  }});

  populateSelects();
}})();
</script>
"""


def _inject_maplibre_download_panel_into_saved_html(path: str, downloads_manifest: dict) -> str:
    """Splice the "Download data" button + panel into a MapLibre `map.html`.

    User-requested feature: a real button/panel (matching the existing
    `#statsButton`/`#statsPanel` bottom-left stack style) offering
    checkboxes/pickers for H3 resolution, census level, stops, and route
    shapes, scoped by metro/city-core, driven entirely by
    `downloads_manifest` (computed at build time in `pipeline.py` from
    what this specific city's pipeline run actually wrote to
    `results/{metro,core}/downloads/...` -- see that call site). Clicking
    "Download" triggers real browser downloads via synthesized
    `<a download>` clicks, one per checked item, staggered to dodge the
    browser's multi-download popup-spam guard.

    No place dropdown here -- this is the single-city map; see
    `combined_map.py`'s own call into `build_download_panel_block` (shared
    helper, this function's `place_options=None` case) for the multi-city
    version.
    """
    import json as _json

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    block = build_download_panel_block(_json.dumps(downloads_manifest), base_path="results")
    marker = "</body>"
    if marker not in html:
        raise ValueError(f"{path}: expected trailing '{marker}' from save_multi_maplibre, not found")
    html = html.replace(marker, block + marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _inject_maplibre_editor_into_saved_html(
    path: str,
    params: dict,
    pop_chunk_cfg: dict,
    stats_data_json: str,
    region: str = "global",
) -> str:
    """Splice `_maplibre_compute_access_js`'s output into a MapLibre `map.html` (item 2 of the session's port).

    `save_multi_maplibre` (geohierarchy) already wrote the route-draw tool
    and live-recolor plumbing this hooks into; this function runs AFTER that
    save, appending the computeAccess port's JS just before the page's
    single closing `</script>` tag so it shares that script's scope (`map`,
    `window.__routeState`, `window.__setAccessOverride`, etc. are all
    already defined there by the time this code runs, since script tags
    execute top-to-bottom).
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    js = _maplibre_compute_access_js(params, pop_chunk_cfg, stats_data_json, region)
    marker = "</script>\n</body>"
    if marker not in html:
        raise ValueError(f"{path}: expected trailing '{marker}' from save_multi_maplibre, not found")
    html = html.replace(marker, js + "\n" + marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _inject_maplibre_stats_panel_into_saved_html(
    path: str,
    stats_by_area: Dict[str, gpd.GeoDataFrame],
    is_us: bool,
    region: str = "global",
    enable_place_comparison: bool = True,
    share_source_map: Optional[Dict[str, str]] = None,
) -> str:
    """Splice the Distribution/Regression/ANOVA/R2/City-rank stats panel into a MapLibre `map.html`.

    Reuses `_stats_panel_block` verbatim (same helper Folium's
    `_inject_controls_into_saved_html` uses) -- its markup is plain
    absolutely-positioned `<div>`s with no Leaflet dependency, and its JS
    reads only `window.__statsData`/`window.__editParams`/the optional
    `window.__editor` (see the call site's comment for how the "Compare
    with scenario" degrades gracefully without it). Appended before the
    same trailing `</script>\\n</body>` marker
    `_inject_maplibre_editor_into_saved_html` uses, so it runs after that
    script (order doesn't actually matter here since the two don't share
    state at load time, but keeping them adjacent mirrors Folium's ordering).
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    block = _stats_panel_helper_js() + _stats_panel_block(
        stats_by_area, is_us, region, enable_place_comparison, share_source_map=share_source_map
    )
    marker = "</body>"
    if marker not in html:
        raise ValueError(f"{path}: expected trailing '{marker}' from save_multi_maplibre, not found")
    html = html.replace(marker, block + marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _inject_controls_into_saved_html(
    path: str,
    circle_fields: List[str],
    opacity_fields: List[str],
    show_development: bool,
    score_domain: Tuple[float, float],
    radius_field_domains: Dict[str, tuple],
    opacity_field_domains: Dict[str, tuple],
    stats_by_area: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    is_us: bool = False,
    stops_gdf: Optional[gpd.GeoDataFrame] = None,
    routes_gdf: Optional[gpd.GeoDataFrame] = None,
    region: str = "global",
    default_shape: str = "hexagons",
    enable_place_comparison: bool = True,
    share_source_map: Optional[Dict[str, str]] = None,
    radius_field_domains_by_res: Optional[Dict[int, Dict[str, tuple]]] = None,
    circle_zoom_bands: Optional[Dict[int, tuple]] = None,
) -> None:
    """Post-process an already-`.save()`d map HTML file to add controls + legend + stats panel.

    `mapControls` and `mapLegend` are added as real Leaflet controls
    (`L.control({{position: 'topright'}})`) rather than absolutely
    positioned page-level `<div>`s -- Leaflet stacks every control sharing
    a corner vertically on its own, so this is what makes them land below
    the native layer control (which is also `topright`) without any manual
    pixel math.
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    map_var = _extract_map_var(html)
    # "Circle size by" defaults to pop_density when it's an available field
    # (the single most broadly-useful default for this dropdown), falling
    # back to the first available field otherwise.
    default_circle_field = _preferred_density_field(circle_fields) or (circle_fields[0] if circle_fields else None)

    base_layer_vars = _extract_base_layer_vars(html)

    controls_and_legend_js = ""
    if map_var:
        controls_and_legend_js = f"""
(function() {{
  var legendCtl = L.control({{position: 'topright'}});
  legendCtl.onAdd = function() {{
    var div = document.getElementById('mapLegend');
    L.DomEvent.disableClickPropagation(div);
    return div;
  }};
  legendCtl.addTo(window["{map_var}"]);
  L.DomEvent.disableClickPropagation(document.getElementById('mapControls'));
}})();
"""

    # `mapControls` + its `mapSettingsButton` toggle are their own
    # absolutely-positioned bottom-right elements (mirroring
    # `statsButton`/`statsPanel`'s bottom-left pattern) -- unlike
    # `mapLegend`, which stays a real Leaflet `L.control` docked top-right --
    # so only the legend needs the "hidden until promoted" wrapper trick.
    injected = (
        _control_panel_html(circle_fields, opacity_fields, default_circle_field)
        + f'<div style="display:none;">{_legend_html(score_domain, show_development)}</div>\n'
        + f"<script>\n{_vectorgrid_click_polyfill_js()}\n{_color_interp_js()}\n{_field_value_js()}\n"
        + f"{_opacity_helper_js()}\n{_shape_popup_js()}\n</script>\n"
        + "<script>\nwindow.addEventListener('load', function() {\n"
        + "document.getElementById('mapLegend').parentElement.style.display = '';\n"
        + controls_and_legend_js
        + _control_panel_js(
            map_var, default_circle_field, radius_field_domains, opacity_field_domains, base_layer_vars,
            default_shape=default_shape,
            radius_field_domains_by_res=radius_field_domains_by_res,
            circle_zoom_bands=circle_zoom_bands,
        )
        + "\n});\n</script>\n"
    )

    if stats_by_area:
        injected += _stats_panel_block(
            stats_by_area, is_us, region, enable_place_comparison, share_source_map=share_source_map
        )

    if stops_gdf is not None and not stops_gdf.empty:
        layer_control_var = _extract_layer_control_var(html)
        if layer_control_var:
            # Bridge the `let`-scoped folium variable onto `window` by bare
            # identifier reference (see `_extract_layer_control_var`'s
            # docstring). Deferred to the `load` event, NOT run as a bare
            # top-level script -- folium emits some of its own script
            # content (including this `let layer_control_XXXX = ...` line)
            # *after* the literal `</body>` tag this gets inserted before
            # (browsers still parse/run it, HTML parsing recovers past a
            # premature `</body>`, but that means DOCUMENT ORDER puts our
            # script first if run eagerly -- a real bug found 2026-08-12:
            # referencing the identifier before its `let` had executed threw
            # a ReferenceError, silently caught, leaving
            # `window.__layerControlRef` null forever). `window`'s `load`
            # event only fires after every synchronous script on the page --
            # including folium's trailing ones -- has already run, so the
            # `let` binding is guaranteed to exist by then.
            injected += (
                "<script>\nwindow.addEventListener('load', function() {\n"
                f"try {{ window.__layerControlRef = {layer_control_var}; }} "
                f"catch (e) {{ window.__layerControlRef = null; }}\n"
                "});\n</script>\n"
            )
        injected += _stops_layer_block(stops_gdf, score_domain, map_var, layer_control_var)

    if routes_gdf is not None and not routes_gdf.empty:
        injected += _routes_layer_block(routes_gdf, map_var)

    # Scenario/route editor ("edit mode"). Last, so its `load` handler runs
    # after the control panel's and the routes/stops blocks' -- it reads
    # `#streetsCheckbox`, `window.__shapeOpacity`, `window.__routesData` and
    # `window.__stopsData`, all published by those.
    injected += _editor_block(map_var)

    if "</body>" in html:
        html = html.replace("</body>", injected + "</body>")
    else:
        html += injected

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _level_zoom_bands(levels: List[Hashable], total_zoom_levels: int = 26) -> Dict[Hashable, tuple[int, int]]:
    """Split `[0, total_zoom_levels - 1]` across `levels`, using fixed real-world zoom anchors.

    Every `HierarchyMap`'s levels must jointly partition the *entire*
    `[0, 25]` zoom range with no gaps (`resolution.validate_resolutions`
    enforces this even for a single-level map) -- so each independently
    built `HierarchyMap` (shape hierarchies, the streets overlay, the
    development overlay) needs its own full-range partition, not a shared
    slice of it.

    2026-09-01 (user reference points, replacing the earlier
    proportional-weighting scheme; shifted one zoom level lower the same
    day per follow-up feedback). 2026-09-02 follow-up (explicit user
    request: "make the zoom levels less strict... require one zoom level
    less to change to the next... careful in places like Hamburg the
    maximum census resolution is municipality so it should be active much
    earlier"): shifted again, AND the coarse/middle boundary was pulled in
    further specifically so an absolutely-coarse tier (state/province/
    municipality-equivalent) doesn't need to wait for a deep zoom just
    because a given country happens to have several such levels between
    it and the true finest one. Block-sized geometries (the finest level --
    a census block, or H3's finest resolution) now activate at zoom 13 or
    deeper; the single coarsest level (nation/state) shares zoom 0-6; any
    levels strictly BETWEEN coarsest and finest (county/municipality/
    district and similar) now split the WIDER zoom 7-12 band (6 zoom
    levels, up from 5) -- e.g. Hamburg's own 5-level hierarchy (nation,
    state, municipality, district, grid) now shows municipality from
    around zoom 8 instead of zoom 10-11. Applies identically to census
    admin levels and H3 resolutions -- both call sites already share this
    one function.

    Concretely (`n = len(levels)`, coarsest-to-finest order):
      - `n == 1`: that single level covers the whole range.
      - `n == 2`: the coarser level covers 0-12 (there's no real
        "county/municipal" tier distinct from "coarsest" here, so it just
        persists right up to the finest level's zoom-13 floor); the finest
        covers 13 to the end.
      - `n >= 3`: the single coarsest level gets 0-6; every level strictly
        between coarsest and finest (the "county/municipal and anything
        finer-but-not-finest" tier) splits 7-12 evenly, in coarse-to-fine
        order; the finest level gets 13 to the end.

    Falls back to an even split (still gapless, every level >=1 zoom
    level) if `total_zoom_levels` is too small to fit these fixed anchors
    (`< 14`, i.e. no room left for the finest level's own zoom-13-to-end
    span) -- not a realistic case at the real default of 26, but keeps the
    function well-defined for a hypothetically smaller total.

    Args:
        levels: Ordered level keys, coarsest to finest (H3 resolutions
            ascending, or census level names coarse-to-fine) -- order
            determines which anchor each level gets, not just remainder
            assignment.
        total_zoom_levels: Number of zoom levels to partition, starting at
            0 (default 26, i.e. the full `[0, 25]`).

    Returns:
        Mapping of level -> `(min_zoom, max_zoom)`, jointly and exactly
        partitioning `[0, total_zoom_levels - 1]`.
    """
    n = len(levels)
    total = total_zoom_levels
    if n == 1:
        return {levels[0]: (0, total - 1)}

    # 2026-09-01/02: shifted twice lower than the original user reference
    # points (block >=15, municipal ~10) -- the strict anchors made finer
    # geometry (and, worse, absolutely-coarse admin tiers like
    # municipality) activate later than felt natural in practice.
    FINEST_MIN_ZOOM = 13
    COARSEST_MAX_ZOOM = 6

    if total < FINEST_MIN_ZOOM + 1:
        # Not enough room for the fixed anchors -- even split fallback,
        # same shape as the old degenerate-case handling.
        base = total // n
        spans = [base] * n
        for i in range(total - base * n):
            spans[-(i + 1)] += 1
        spans = [max(1, s) for s in spans]
        drift = total - sum(spans)
        i = n - 1
        while drift != 0:
            step = 1 if drift > 0 else -1
            if spans[i] + step >= 1:
                spans[i] += step
                drift -= step
            i = (i - 1) % n
        bands: Dict[Hashable, tuple[int, int]] = {}
        z = 0
        for level, span in zip(levels, spans):
            bands[level] = (z, z + span - 1)
            z += span
        return bands

    if n == 2:
        return {levels[0]: (0, FINEST_MIN_ZOOM - 1), levels[1]: (FINEST_MIN_ZOOM, total - 1)}

    # n >= 3: coarsest = [0, 6], finest = [13, total-1], every level in
    # between splits [7, 12] as evenly as possible (6 zoom levels), in
    # coarse-to-fine order -- an earlier middle level never starts later
    # than a finer one.
    bands = {levels[0]: (0, COARSEST_MAX_ZOOM), levels[-1]: (FINEST_MIN_ZOOM, total - 1)}
    middle_levels = levels[1:-1]
    n_middle = len(middle_levels)
    middle_total = FINEST_MIN_ZOOM - (COARSEST_MAX_ZOOM + 1)  # 6, for the real 13/7/6 anchors
    base = middle_total // n_middle
    spans = [base] * n_middle
    for i in range(middle_total - base * n_middle):
        spans[i] += 1  # earlier (coarser) middle levels absorb any remainder first
    spans = [max(1, s) for s in spans]
    drift = middle_total - sum(spans)
    i = n_middle - 1
    while drift != 0:
        step = 1 if drift > 0 else -1
        if spans[i] + step >= 1:
            spans[i] += step
            drift -= step
        i = (i - 1) % n_middle
    z = COARSEST_MAX_ZOOM + 1
    for level, span in zip(middle_levels, spans):
        bands[level] = (z, z + span - 1)
        z += span
    return bands


def _finite_domain(series) -> tuple[float, float]:
    """Compute a `(min, max)` domain from a pandas Series, ignoring NaN/inf."""
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    return (float(values.min()), float(values.max()))
