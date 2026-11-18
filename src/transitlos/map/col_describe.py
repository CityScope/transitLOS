"""Plain-English descriptions for map columns that have no real definition in a census schema.

`describe_column(col, label, source)` builds one sentence from the column's name pattern (election results, shares, rates, means,
counts, ...) so every "i" button on the map has a text that says what the number is. `is_weak(description, label)` flags a stored
description that only repeats the label.
"""
from __future__ import annotations

import re
from typing import Optional

_KNOWN = {
    "owner_rate": "Percentage of households that own their home.",
    "renter_rate": "Percentage of households that rent their home.",
    "employment_rate": "Percentage of the working-age population that is employed.",
    "unemployment_rate": "Percentage of the labour force that is unemployed and looking for work.",
    "labor_force_rate": "Percentage of the working-age population that is in the labour force (employed or looking for work).",
    "labor_force_participation_rate": "Percentage of the working-age population that is in the labour force (employed or looking for work).",
    "poverty_rate": "Percentage of residents living below the poverty line.",
    "car_free_rate": "Percentage of households without a car.",
    "no_vehicle_rate": "Percentage of households without a vehicle.",
    "car_ownership_rate": "Percentage of households that own at least one car.",
    "rent_burden_rate": "Percentage of renting households that spend 30% or more of their income on rent.",
    "bachelors_rate": "Percentage of adults (25 and over) with a bachelor's degree or higher.",
    "high_school_rate": "Percentage of adults (25 and over) whose highest education is high school.",
    "less_than_hs_rate": "Percentage of adults (25 and over) who did not finish high school.",
    "some_college_rate": "Percentage of adults (25 and over) with some college education but no bachelor's degree.",
    "academic_degree_rate": "Percentage of adults with an academic (university) degree.",
    "self_employed_rate": "Percentage of employed residents who are self-employed.",
    "foreign_resident_rate": "Percentage of residents who are foreign nationals.",
    "institutionalized_resident_rate": "Percentage of residents living in institutions (care homes, hospitals, prisons, dormitories).",
    "transit_share": "Percentage of commuters who travel to work by public transport.",
    "walk_commute_share": "Percentage of commuters who walk to work.",
    "bike_commute_share": "Percentage of commuters who cycle to work.",
    "car_commute_share": "Percentage of commuters who travel to work by car.",
    "urban_share": "Percentage of the population living in urban areas.",
    "local_worker_share": "Percentage of resident workers who work in the same municipality where they live.",
    "commuter_in_share": "Percentage of the jobs located in the area that are held by people who live elsewhere.",
    "worldpop_male_share": "Percentage of the WorldPop-estimated population that is male.",
    "worldpop_female_share": "Percentage of the WorldPop-estimated population that is female.",
    "worldpop_children_share": "Percentage of the WorldPop-estimated population under 18.",
    "worldpop_elderly_share": "Percentage of the WorldPop-estimated population aged 65 and over.",
    "worldpop_adult_share": "Percentage of the WorldPop-estimated population aged 18 to 64.",
    "worldpop_urban_share": "Percentage of the WorldPop-estimated population living in urban land.",
    "education_university_share": "Percentage of adults with a university degree.",
    "education_secondary_share": "Percentage of adults whose highest education is secondary school.",
    "education_primary_or_less_share": "Percentage of adults with primary education or less.",
    "education_other_tertiary_share": "Percentage of adults with non-university tertiary education (college, vocational or trade).",
    "cbs_population_share_0_19": "Percentage of residents aged 0 to 19.",
    "cbs_population_share_20_64": "Percentage of residents aged 20 to 64.",
    "cbs_population_share_65_plus": "Percentage of residents aged 65 and over.",
    "population_share_0_19": "Percentage of residents aged 0 to 19.",
    "population_share_20_64": "Percentage of residents aged 20 to 64.",
    "population_share_65_plus": "Percentage of residents aged 65 and over.",
    "ageMedian": "Median age of residents.",
    "sexRatio": "Number of men per 100 women.",
}
_ELEC = re.compile(r"^(?:(?P<scope>national|state|municipal)_)?(?P<year>\d{4})(?P<kind>[A-Za-z]*)_(?P<rest>.+)$")
_UNIT_WORDS = {"share": "percentage", "rate": "percentage"}


def _words(name: str):
    return [w.lower() for w in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name.replace("_", " ")) if w]


def _election(logical: str) -> Optional[str]:
    m = re.match(r"^elec_(national|state|municipal)_(\d{4})([A-Za-z]*)_(.+)$", logical)
    if m:
        scope, year, kind, rest = m.groups()
        what = f"the {year} {scope} election" + (f" ({kind})" if kind else "")
    else:
        m = re.match(r"^(\d{4})_(president|senate)_(.+)$", logical)
        if not m:
            return None
        year, office, rest = m.groups()
        what = f"the {year} US {office} election"
    if rest.startswith("party_"):
        party = re.sub(r"_(share|votes)$", "", rest[6:]).replace("_", " ").strip().title()
        return f"{'Votes' if rest.endswith('_votes') else 'Share of the valid votes'} won by {party} in {what}."
    return {
        "total_votes": f"Total valid votes cast in {what}.",
        "turnout_rate": f"Turnout: percentage of registered voters who voted in {what}.",
        "left_share": f"Share of the valid votes for left-wing parties in {what} (parties classified as left bloc by this study).",
        "right_share": f"Share of the valid votes for right-wing parties in {what} (parties classified as right bloc by this study).",
        "other_share": f"Share of the valid votes for all other parties in {what}.",
        "democrat_share": f"Share of the valid votes for the Democratic candidate in {what}.",
        "republican_share": f"Share of the valid votes for the Republican candidate in {what}.",
    }.get(rest)


# ---- pipeline-derived families (2026-11-13: every "i" says what the number is, where it comes from and how it is computed) ----
_DERIVED = {
    "losAtHomesWorkersMean": "Transit service at workers' homes: for the people who work in this area, the average level of service (0-100) at the places "
                             "where they live, weighted by the number of commuters on each home-work flow (origin-destination commuting data). "
                             "High values mean the area's workforce lives where transit is good.",
    "losAtWorkplacesResidentsMean": "Transit service at workplaces: for the people who live in this area, the average level of service (0-100) at the places "
                                    "where they work, weighted by the number of commuters on each home-work flow (origin-destination commuting data). "
                                    "High values mean residents' jobs are where transit is good.",
    "commuteTimeResidentsMean": "Average one-way commute time (minutes) of the people who live in the area, from the origin-destination commuting data.",
    "commuteTimeWorkersMean": "Average one-way commute time (minutes) of the people who work in the area, from the origin-destination commuting data.",
    "commuteDistanceResidentsMean": "Average straight-line commute distance (km) from the area's homes to its residents' workplaces (origin-destination data).",
    "commuteDistanceWorkersMean": "Average straight-line commute distance (km) from the homes of the area's workers to the area (origin-destination data).",
    "commuteSpeedResidentsMean": "Average commute speed (km/h) of the area's residents: commute distance divided by commute time.",
    "commuteSpeedWorkersMean": "Average commute speed (km/h) of the area's workers: commute distance divided by commute time.",
    "gba_builtSurfacePopulation": "Residents estimated from the built floor area of the GlobalBuildingAtlas (footprint x height / 3 m per floor): in each census unit, "
                                  "census residents per built m2, times the built area of each active H3 cell (cells near streets). Scaled with one factor so the "
                                  "city total equals the census total (factor in results/rescale_factors.json).",
    "gba_population_overestimation_vs_census": "How many more residents the built-area estimate puts in the cell than the census: max(0, estimate - census), "
                                               "both scaled to the same city total. Shows where buildings suggest more people than the census counts.",
    "gba_population_underestimation_vs_census": "How many fewer residents the built-area estimate puts in the cell than the census: max(0, census - estimate), "
                                                "both scaled to the same city total.",
    "gba_overestimation_share": "Built-area estimate overestimation divided by the built-area estimate (percent): the part of the estimate the census does not confirm.",
    "gba_underestimation_share": "Built-area estimate underestimation divided by the census residents (percent): the part of the census population the buildings miss.",
    "population_overestimation_vs_census": "How many more residents WorldPop puts in the cell than the census: max(0, WorldPop - census), WorldPop scaled to the "
                                           "census total with one factor.",
    "population_underestimation_vs_census": "How many fewer residents WorldPop puts in the cell than the census: max(0, census - WorldPop), WorldPop scaled to the "
                                            "census total with one factor.",
    "worldpop_overestimation_share": "WorldPop overestimation divided by the WorldPop estimate (percent).",
    "worldpop_underestimation_share": "WorldPop underestimation divided by the census residents (percent).",
    "gba_buildingCount": "Number of buildings (GlobalBuildingAtlas footprints) whose centre lies in the cell. Building stock: never rescaled.",
    "gba_buildingFootprintArea": "Total ground footprint (m2) of the GlobalBuildingAtlas buildings in the cell. Never rescaled.",
    "gba_builtFloorArea": "Total built floor area (m2) of the GlobalBuildingAtlas buildings in the cell: footprint x height / 3 m per floor. Never rescaled.",
    "gba_buildingHeightMax": "Height (m) of the tallest GlobalBuildingAtlas building in the cell.",
    "gba_buildingHeightMedian": "Median height (m) of the GlobalBuildingAtlas buildings in the cell.",
    "gba_buildingFootprintMax": "Largest building footprint (m2) in the cell (GlobalBuildingAtlas).",
    "gba_buildingFootprintMedian": "Median building footprint (m2) in the cell (GlobalBuildingAtlas).",
    "gba_builtFloorAreaMax": "Largest building floor area (m2) in the cell (GlobalBuildingAtlas, footprint x height / 3).",
    "gba_builtFloorAreaMedian": "Median building floor area (m2) in the cell (GlobalBuildingAtlas, footprint x height / 3).",
    "pop_density_1km": "Residents per km2 within 1 km of the cell: the population of all H3 cells whose centres lie within 1 km of this cell's centre, "
                       "divided by their area. A smoother density that describes the neighbourhood rather than the single cell.",
    "pop_density_500m": "Residents per km2 within 500 m of the cell: the population of all H3 cells whose centres lie within 500 m of this cell's centre, "
                        "divided by their area. On census areas: the residents-weighted mean of their cells' values.",
    "pop_density_2km": "Residents per km2 within 2 km of the cell: the population of all H3 cells whose centres lie within 2 km of this cell's centre, "
                       "divided by their area. On census areas: the residents-weighted mean of their cells' values.",
    "pop_density_3km": "Residents per km2 within 3 km of the cell: the population of all H3 cells whose centres lie within 3 km of this cell's centre, "
                       "divided by their area. On census areas: the residents-weighted mean of their cells' values.",
    "pop_density_4km": "Residents per km2 within 4 km of the cell: the population of all H3 cells whose centres lie within 4 km of this cell's centre, "
                       "divided by their area. On census areas: the residents-weighted mean of their cells' values.",
    "pop_density_5km": "Residents per km2 within 5 km of the cell: the population of all H3 cells whose centres lie within 5 km of this cell's centre, "
                       "divided by their area. On census areas: the residents-weighted mean of their cells' values.",
    "pop_density": "Residents per km2 of the cell or area.",
    "population_density": "Census residents per km2 of the cell or area.",
    "pop_jobs_total": "Residents plus workplaces (jobs located in the area).",
    "pop_jobs_density": "Residents plus workplaces per km2.",
    "jobs_share": "Workplaces as a percentage of residents plus workplaces: near 0 in purely residential areas, near 100 in job centres.",
    "jobs_per_resident_ratio": "Jobs located in the area per resident.",
    "census_residents": "Residents counted by the census, split from census units to H3 cells by built floor area.",
    "worldpop_residents": "Residents estimated by WorldPop (100 m grid), scaled to the census total with one factor (factor in rescale_factors.json).",
}
_SHARE_OF = {   # global-schema share columns: (what, of what)
    "under18": ("residents under 18", "all residents"), "over65": ("residents aged 65 and over", "all residents"),
    "adult": ("adult residents", "all residents"), "children": ("children", "all residents"), "elderly": ("residents aged 65 and over", "all residents"),
    "female": ("female residents", "all residents"), "male": ("male residents", "all residents"),
    "foreign_born": ("residents born abroad", "all residents"), "university": ("adults with a university degree", "adults"),
    "renter_households": ("renting households", "all households"), "owner_households": ("owner-occupied households", "all households"),
    "minority_ethnicity": ("residents of a minority ethnic group", "all residents"), "poverty": ("residents below the poverty line", "all residents"),
    "religious": ("religious residents", "all residents"), "minority_religion": ("residents of a minority religion", "all residents"),
    "zero_vehicle_households": ("households without a car", "all households"), "transit_commute": ("commuters who use public transport", "all commuters"),
    "car_commute": ("commuters who drive or ride in a car", "all commuters"), "walk_commute": ("commuters who walk", "all commuters"),
    "bike_commute": ("commuters who cycle", "all commuters"), "jewish": ("Jewish residents", "all residents"), "muslim": ("Muslim residents", "all residents"),
    "christian": ("Christian residents", "all residents"), "druze": ("Druze residents", "all residents"), "haredim": ("Haredi (ultra-Orthodox) residents", "all residents"), "education_basic_or_less": ("adults with basic education or less", "adults"),
}


def _global_definitions():
    global _GLOBAL_DEFS
    try:
        return _GLOBAL_DEFS
    except NameError:
        pass
    _GLOBAL_DEFS = {}
    try:
        import json
        from pathlib import Path

        import pycensus

        p = Path(pycensus.__file__).resolve().parent / "schemas" / "global_schema.json"
        _GLOBAL_DEFS = {f["feature_name"]: f.get("definition", "") for f in json.loads(p.read_text(encoding="utf-8"))}
    except Exception:  # noqa: BLE001
        pass
    return _GLOBAL_DEFS


def _derived_or_global(col: str, logical: str, source: str) -> Optional[str]:
    for key in (col, logical):
        if key in _DERIVED:
            return _DERIVED[key]
    defs = _global_definitions()
    src = f" Source: {source}." if source and "Derived" not in source else ""
    for key in (logical, col):
        if defs.get(key):
            return defs[key] + src
    m = re.match(r"^([a-z0-9]+)_(.+)$", col)   # a source prefix in front of a global name: worldpop_malePopulation, eustat_femalePopulation ...
    if m and defs.get(m.group(2)):
        what = "WorldPop estimate" if m.group(1) == "worldpop" else f"{source or m.group(1)}"
        return f"{defs[m.group(2)]} ({what}.)" if m.group(1) == "worldpop" else defs[m.group(2)] + src
    m = re.match(r"^(.+)_(share|rate)$", logical)
    if m and m.group(1) in _SHARE_OF:
        what, of = _SHARE_OF[m.group(1)]
        return (f"{what[:1].upper() + what[1:]} as a percentage of {of}: recomputed from the two counts at every map level (so coarser cells are weighted by their "
                f"population, not averaged).{src}")
    return None


def describe_column(col: str, label: str, source: str = "", logical: Optional[str] = None) -> str:
    """One sentence saying what `col` (shown as `label`, from `source`) holds.

    Args:
        col: Column name.
        label: Display label.
        source: Dataset name.
        logical: Column name without its source prefix.

    Returns:
        The sentence.
    """
    logical = logical or col
    for key in (logical, col):
        if key in _KNOWN:
            return _KNOWN[key]
    el = _election(logical)
    if el:
        return el
    dg = _derived_or_global(col, logical, source)
    if dg:
        return dg
    base = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    low = base[:1].lower() + base[1:] if base else logical
    src = f" Source: {source}." if source else ""
    last = _words(logical)[-1] if _words(logical) else ""
    if last in ("share", "rate") or low.endswith((" share", " rate")):
        core = re.sub(r"\s+(share|rate)$", "", low).strip()
        return (f"Percentage of the relevant total (residents, households, dwellings or workers, as the source counts them) in the category “{core}”: "
                f"the category's count divided by that total, recomputed at every map level from the two counts.{src}")
    if last in ("mean", "median", "average", "avg") or low.startswith(("mean ", "median ", "average ")):
        return f"{base}, as published by the source for the area.{src}"
    if last in ("ratio", "index"):
        return f"{base}: ratio/index value for the area as published by the source.{src}"
    if last == "density":
        return f"{base}: amount per square kilometre of the area.{src}"
    return f"Number of {low} counted in the cell or area (a count: coarser map levels add up the finer cells).{src}"


def is_weak(description: str, label: str) -> bool:
    """True when `description` is empty, tiny or just the label again.

    Args:
        description: A description.
        label: The column's label.
    """
    d = re.sub(r"[\s.]+$", "", (description or "").strip().lower())
    l = re.sub(r"\s*\([^)]*\)\s*$", "", (label or "").strip().lower())
    if "derived by this study from its source columns" in d or "a ratio derived by this study from the census" in d or "derived by this study from the counts" in d:
        return True
    return len(d) < 30 or d == l or d == (label or "").strip().lower() or d.replace("_", " ") == l
