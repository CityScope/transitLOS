"""Simplified column list + relevance ordering + election-party rules for the stats panel / dropdowns of a MapLibre page.

Build time (`column_select_script`): from the finest grid (`level_of_service` + `population` weights) compute, per candidate column,
  * `r2`    weighted R^2 against level_of_service (best of raw / log, like the Regression tab) -> dropdown order of regression / opacity-by,
  * `anova` |difference in mean level_of_service between the high and low half| (like the Census tab) -> order of the other selectors,
  * `simple` the simplified column set: every derived / global-schema column, plus the single most relevant (highest R^2) column of each
    local topic (elections, housing price, rent, race ...; two for race); election parties: only the 4 most voted parties per election
    when they hold > 5% of the votes.
Run time (`COLUMN_SELECT_JS`): `window.__colVisible(col)` (simplified unless the "Extended census" checkbox is on, plus the scope rule: at
nation scope no state / municipal election parties and, outside the US, only the left / right proxies), used by the Census and Regression
tabs and by the custom dropdowns (which are also ordered by relevance). Pass `simplified_columns` to override the heuristic.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

_TOPICS = [  # (topic, regex, how many to keep)
    ("election", r"elec|vote|party|left_|right_|democrat|republican", 2),
    ("rent", r"rent", 1),
    ("housing price", r"price|sale|sold|buy|home_?value|house_?value|median_?value", 1),
    ("race", r"race|white|black|ethnic|hispanic|latino|asian|foreign|immigr", 2),
    ("income", r"income|wage|salary", 1),
    ("education", r"educat|degree|school", 1),
    ("age", r"age_|_age|elderly|young|children", 1),
]
_PARTY = re.compile(r"^(?P<prefix>.*elec_(?:national|state|municipal)_\d{4})_party_(?P<party>.+?)_(?P<kind>votes|share)$")
_ALWAYS = {"population", "level_of_service", "pop_density", "pop_density_500m", "pop_density_1km", "pop_density_2km", "pop_density_3km", "pop_density_4km", "pop_density_5km", "pop_jobs_density", "jobs_share"}


def _wcorr2(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 8:
        return 0.0
    x, y, w = x[m], y[m], w[m]
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    cov = (w * (x - mx) * (y - my)).sum() / sw
    vx, vy = (w * (x - mx) ** 2).sum() / sw, (w * (y - my) ** 2).sum() / sw
    return float(cov * cov / (vx * vy)) if vx > 0 and vy > 0 else 0.0


def _anova_diff(x, y, w) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 8:
        return 0.0
    x, y, w = x[m], y[m], w[m]
    med = float(np.median(x))
    lo, hi = x < med, x >= med
    if not lo.any() or not hi.any():
        lo, hi = x <= med, x > med
        if not lo.any() or not hi.any():
            return 0.0
    return float(abs((w[hi] * y[hi]).sum() / w[hi].sum() - (w[lo] * y[lo]).sum() / w[lo].sum()))


def _num(v) -> float:
    """`v` as a sortable number (NaN / None sort last)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -1.0
    return v if v == v else -1.0


def column_stats(gdf, fields: Iterable[str], score_col: str = "level_of_service") -> Dict[str, Dict[str, float]]:
    """`{field: {r2, anova}}` for every numeric field present on `gdf`.

    Args:
        gdf: Finest map level.
        fields: Columns to rate.
        score_col: Score column.
    """
    y = np.asarray(gdf[score_col], dtype=float) if score_col in gdf.columns else None
    w = np.asarray(gdf["population"], dtype=float) if "population" in gdf.columns else np.ones(len(gdf))
    out: Dict[str, Dict[str, float]] = {}
    if y is None:
        return out
    pop = np.asarray(gdf["population"], dtype=float) if "population" in gdf.columns else None
    from .build import is_relative_field as _is_relative
    for f in fields:
        if f not in gdf.columns:
            continue
        try:
            x = np.asarray(gdf[f], dtype=float)
        except (TypeError, ValueError):
            continue
        with np.errstate(all="ignore"):
            r2 = _wcorr2(x, y, w)
            pos = np.where(x > 0, x, np.nan)
            r2 = max(r2, _wcorr2(np.log(pos), y, w))
            # Census-tab relevance (|difference of mean level_of_service between the high and the low half|). A COUNT column is ranked by its SHARE equivalent (count per
            # resident): the Census tab compares relative columns, and a raw count mostly measures how big the cell is.
            xa = x
            if pop is not None and not _is_relative(f):
                xa = np.where(pop > 0, x / np.where(pop > 0, pop, np.nan), np.nan)
            out[f] = {"r2": r2, "anova": _anova_diff(xa, y, w)}
    return out


def top_parties(gdf, max_parties: int = 4, min_share: float = 0.05) -> set:
    """Party columns to keep: per election (prefix) the `max_parties` most voted parties whose vote share is above `min_share`.

    Args:
        gdf: Finest map level.
        max_parties: Parties kept per election.
        min_share: Smallest vote share kept.
    """
    w = np.asarray(gdf["population"], dtype=float) if "population" in gdf.columns else None
    groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c in gdf.columns:
        m = _PARTY.match(str(c))
        if not m:
            continue
        v = np.asarray(gdf[c], dtype=float)
        if m.group("kind") == "votes":
            tot = float(np.nansum(v))
        else:   # share column: weight by population
            tot = float(np.nansum(v * w)) if w is not None else float(np.nansum(v))
        groups.setdefault(m.group("prefix"), {}).setdefault(m.group("party"), {})[m.group("kind")] = tot
    keep = set()
    for prefix, parties in groups.items():
        score = {p: (d.get("votes", d.get("share", 0.0))) for p, d in parties.items()}
        total = sum(score.values()) or 1.0
        ranked = sorted(score, key=score.get, reverse=True)
        chosen = [p for p in ranked[:max_parties] if score[p] / total > min_share]
        for p in chosen:
            keep.update(f"{prefix}_party_{p}_{k}" for k in ("votes", "share"))
    all_party_cols = {c for c in gdf.columns if _PARTY.match(str(c))}
    return keep, all_party_cols


_LEVELS = ("national", "state", "municipal")


def _elec(level: str, kind: str, us: bool) -> List[str]:
    """Candidates for one election concept: the global name, the US party names, then the year-labelled columns of the city maps
    (`re:` = regular expression; the latest year wins)."""
    g = {"left": "left_share", "right": "right_share", "turnout": "turnout_rate"}[kind]
    out = []
    if us:
        k = {"left": "democrat_share", "right": "republican_share", "turnout": "turnout"}[kind]
        out += [f"elec_{level}_{k}"] + ([f"2024_president_{k}"] if level == "national" else []) + ([f"2024_senate_{k}"] if level == "state" else [])
    out += [f"elec_{level}_{g}", f"elec_{level}_turnout"] if kind == "turnout" else [f"elec_{level}_{g}"]
    out += [f"re:^elec_{level}_(\\d{{4}})_{g}$", f"re:^elec_{level}_(\\d{{4}})[a-z]+_{g}$"]
    return out


def template(us: bool) -> List[tuple]:
    """The fixed simplified template: `(concept, [candidate columns, canonical name first], ...)`; the first candidate present wins.

    Args:
        us: US map (US column names).
    """
    T = [
        ("Population", ["population"]),
        ("Jobs", ["jobs", "jobs_total", "workplaceJobs", "lodes_wac_jobsTotal", "statcan_jobsTotal", "sii_jobsTotal", "ba_jobsTotal"]),
        ("Population + jobs", ["pop_jobs_total", "populationAndJobs"]),
        ("Population density", ["population_density", "pop_density", "worldpop_population_density"]),
        ("Population density within 500 m", ["pop_density_500m"]),
        ("Population density within 1 km", ["pop_density_1km"]),
        ("Population density within 2 km", ["pop_density_2km"]),
        ("Population density within 3 km", ["pop_density_3km"]),
        ("Population density within 4 km", ["pop_density_4km"]),
        ("Population density within 5 km", ["pop_density_5km"]),
        ("Jobs density", ["jobs_density"]),
        ("Population + jobs density", ["pop_jobs_density", "population_and_jobs_density"]),
        # counts (2026-11-13: count-only selectors such as Opacity by / Circle size by need them; share selectors ignore them)
        ("Male population", ["malePopulation", "ine_malePopulation", "worldpop_malePopulation"]),
        ("Female population", ["femalePopulation", "ine_femalePopulation", "worldpop_femalePopulation"]),
        ("Population under 18", ["under18Population", "ine_under18Population", "worldpop_under18Population"]),
        ("Population 65 and over", ["over65Population", "ine_over65Population", "worldpop_over65Population"]),
        ("University-educated population", ["educationUniversityPopulation", "ine_educationUniversityPopulation"]),
        ("Foreign-born population", ["foreignBornPopulation", "ine_foreignBornPopulation"]),
        ("Population in poverty", ["povertyPopulation", "ine_povertyPopulation"]),
        ("Employed residents", ["employedResidents", "ine_employedResidents"]),
        ("Households", ["households", "ine_households"]),
        ("Owner households", ["ownerHouseholds"]),
        ("Renter households", ["renterHouseholds"]),
        ("Single-person households", ["singlePersonHouseholds"]),
        ("Dwellings", ["dwellings", "ine_dwellings"]),
        ("Buildings", ["buildingCount", "ine_catastro_buildings", "gba_buildingCount"]),
        ("Built floor area", ["builtFloorArea", "ine_catastro_floor_area_m2_total", "gba_builtFloorArea"]),
        ("Transit commuters", ["publicTransportCommuters"]),
        ("Car commuters", ["carCommuters"]),
        ("Age: under 18 share", ["under18_share", "under18Population_share", "worldpop_children_share", "ine_under18Population_share"]),
        ("Age: over 65 share", ["over65_share", "over65Population_share", "worldpop_elderly_share", "ine_over65Population_share"]),
        ("Gender: male share", ["male_share", "worldpop_male_share"]),
        ("Gender: female share", ["female_share", "worldpop_female_share"]),
        ("Education: university share", ["university_share", "educationUniversity_share", "educationUniversityPopulation_share", "bachelors_or_higher_share", "ine_educationUniversityPopulation_share"]),
        ("Education: basic or less share", ["education_basic_or_less_share", "educationPrimaryOrLess_share", "educationPrimaryOrLessPopulation_share", "less_than_high_school_share", "ine_educationPrimaryOrLessPopulation_share"]),
        ("Household income", ["median_household_income", "meanHouseholdIncome", "ine_meanHouseholdIncome", "ine_incomeGrossHouseholdMean", "eustat_incomeFamilyMean"]),
        ("Poverty share", ["poverty_share"]),
        ("Unemployment rate", ["unemployment_rate"]),
        ("Labor force participation", ["labor_force_participation_rate"]),
        ("Ethnicity: minority share", ["minority_ethnicity_share"]),
        ("Religion: religious share", ["religious_share"]),
        ("Religion: minority religion share", ["minority_religion_share"]),
        ("Owner households share", ["owner_households_share"]),
        ("Renter households share", ["renter_households_share"]),
        ("Gini index", ["income_gini_index", "giniIndex", "ine_giniIndex"]),
        ("WorldPop overestimation", ["population_overestimation_vs_census"]),
        ("WorldPop underestimation", ["population_underestimation_vs_census"]),
        ("WorldPop overestimation (share)", ["worldpop_overestimation_share"]),
        ("WorldPop underestimation (share)", ["worldpop_underestimation_share"]),
        ("Built-area population estimate", ["gba_builtSurfacePopulation"]),
        ("Built-area estimate overestimation", ["gba_population_overestimation_vs_census"]),
        ("Built-area estimate underestimation", ["gba_population_underestimation_vs_census"]),
        ("Built-area estimate overestimation (share)", ["gba_overestimation_share"]),
        ("Built-area estimate underestimation (share)", ["gba_underestimation_share"]),
    ]
    if us:
        T += [("Ethnicity: white share", ["white_share"]), ("Ethnicity: black share", ["black_share"]),
              ("Owner share", ["owner_share"]), ("Renter share", ["renter_share"]), ("Rent burden", ["rent_burden_median"]), ("Rent per room", ["rent_per_room_mean"])]
    else:
        T += [("Ethnicity: foreign-born share", ["foreign_born_share", "foreignBornPopulation_share", "foreignBorn_share", "ine_foreignBornPopulation_share", "eustat_foreignBornPopulation_share"])]
    T += [
        ("Mean household size", ["mean_household_size", "householdSizeMean", "ine_householdSizeMean", "eustat_householdSizeMean"]),
        # rent and house price on every map (user 2026-11-16): the whole-home value first, per m2 where that is all there is
        ("Rent price", ["rent_per_home_mean", "median_gross_rent", "rent_median", "ushs_rent_median", "median_contract_rent", "rent_per_m2_mean"]),
        ("House price", ["price_per_home_mean", "median_home_value", "home_value_median", "ushs_home_value_median", "price_per_m2_mean", "listing_price_per_m2"]),
        ("Price per m2", ["listing_price_per_m2"] if us else ["price_per_m2_mean"]),
        ("Rent per m2", [] if us else ["rent_per_m2_mean"]),
        ("Mean home size m2", ["listing_median_area_m2"] if us else ["home_size_m2_mean"]),
        ("Built before 1945", ["dwellings_built_before_1945_share"]),
        ("Built 1945-1970", ["dwellings_built_1945_1970_share"]),
        ("Built 1970-2000", ["dwellings_built_1970_2000_share"]),
        ("Built after 2000", ["dwellings_built_after_2000_share"]),
    ]
    T += [
        ("Mobility: cars per household", ["cars_per_household_ratio", "cars_per_household", "vehicles_per_household", "mean_vehicles_per_household"]),
        ("Mobility: households without a car share", ["zero_vehicle_households_share", "no_car_households_share", "households_without_car_share"]),
        ("Mobility: car commute share", ["car_commute_share"]),
        ("Mobility: walk commute share", ["walk_commute_share"]),
        ("Mobility: bike commute share", ["bike_commute_share"]),
        ("Mobility: transit commute share", ["transit_commute_share", "public_transport_commute_share", "commute_transit_share"]),
        ("Mobility: regular trips on up to 3 days a week (share)", ["regular_flows_up_to_3_days_week_share_est", "2024_regular_flows_up_to_3_days_week_share_est"]),
        ("Commute time of residents", ["commuteTimeResidentsMean", "commuteTimeMean", "mean_commute_time_minutes"]),
        ("Commute time of workers", ["commuteTimeWorkersMean"]),
        ("Commute distance of residents", ["commuteDistanceResidentsMean", "commuteDistanceMean"]),
        ("Commute distance of workers", ["commuteDistanceWorkersMean"]),
        ("Commute speed of residents", ["commuteSpeedResidentsMean", "commuteSpeedMean"]),
        ("Commute speed of workers", ["commuteSpeedWorkersMean"]),
        ("Transit service at workplaces", ["losAtWorkplacesResidentsMean"]),   # the other end of the commute (LODES flows), usa_study/add_od_los_columns.py
        ("Transit service at workers' homes", ["losAtHomesWorkersMean"]),
    ]
    for lv in _LEVELS:
        for kind, label in (("left", "Democrat share" if us else "left share"), ("right", "Republican share" if us else "right share"), ("turnout", "turnout")):
            T.append((f"Election {lv}: {label}", _elec(lv, kind, us)))
    if not us:
        T.append(("Tourism: tourist dwellings share", ["tourist_dwellings_share", "ine_tourist_dwellings_share_2026M05"]))
    return T


_RELIGION = re.compile(r"(catholic|christian|muslim|jewish|protestant|evangelic|hindu|buddhist|baptist|latter_day|orthodox|sikh|no_religion|nonreligious|druze)", re.I)


def main_religions(finest, fields: List[str], top: int = 4, min_share: float = 0.03) -> List[str]:
    """The main religions of the place: the religion share columns with the highest population-weighted mean (up to `top`, each >= 3%).

    Args:
        finest: Finest map level.
        fields: Available columns.
        top: Most religions kept.
        min_share: Smallest mean share kept.
    """
    cand = [f for f in fields if _RELIGION.search(f) and re.search(r"(share|pct)$", f) and "congregation" not in f]
    if finest is None or not cand:
        return []
    try:
        w = np.asarray(finest["population"], dtype=float) if "population" in finest else None
    except Exception:  # noqa: BLE001
        w = None
    scores = []
    for c in cand:
        try:
            v = np.asarray(finest[c], dtype=float)
        except Exception:  # noqa: BLE001
            continue
        m = np.isfinite(v) & (np.isfinite(w) & (w > 0) if w is not None else True)
        if not np.any(m):
            continue
        mean = float(np.average(v[m], weights=w[m] if w is not None else None))
        scale = 100.0 if np.nanmax(v[m]) > 1.5 else 1.0
        scores.append((mean / scale, c))
    return [c for sh, c in sorted(scores, reverse=True)[:top] if sh >= min_share]


def template_columns(fields: List[str], us: bool):
    """Resolve the template against the available fields -> (columns in template order, {concept: column or None}).

    Args:
        fields: Available columns.
        us: US map.
    """
    have = set(fields)
    resolved: Dict[str, Optional[str]] = {}
    for concept, cands in template(us):
        resolved[concept] = None
        for c in cands:
            if c.startswith("re:"):   # e.g. elec_national_2024_left_share: the latest year
                rx = re.compile(c[3:])
                hits = sorted((f for f in fields if rx.match(f)), key=lambda f: rx.match(f).group(1), reverse=True)
                if hits:
                    resolved[concept] = hits[0]
                    break
            elif c in have:
                resolved[concept] = c
                break
        if resolved[concept] is None:   # a source-prefixed twin of a canonical name (ine_xxx / eustat_xxx / worldpop_xxx)
            resolved[concept] = next((f for c in cands if not c.startswith("re:") for f in fields if f.endswith("_" + c) and f.split("_")[0] in ("ine", "eustat", "worldpop", "ba", "sii", "statcan", "lodes")), None)
    cols = [c for c in resolved.values() if c]
    return list(dict.fromkeys(cols)), resolved


COLUMN_SELECT_JS = r"""
(function() {
  var S = window.__colSelect || {};
  window.__extendedCensus = false;
  function isPartyCol(c) { return /elec_(national|state|municipal)_\d{4}_party_/.test(c); }
  function scopeNation() { try { return window.__scopeLevel ? window.__scopeLevel() === 'nation' : false; } catch (e) { return false; } }
  var isUS = !!S.us;
  // Minor parties are judged on the active scope: the nation's most voted parties when the scope is the nation (or zoomed out), the
  // state's / county's / place's own when the scope is that area. Same rule as pycensus `minor_party_columns` (top 4 above 5%).
  var PARTY_RE = /^(.*elec_(?:national|state|municipal)_\d{4}[A-Za-z]*)_party_(.+?)_(votes|share)$/, majorCache = new WeakMap();
  function majorParties(d) {
    if (majorCache.has(d)) return majorCache.get(d);
    var groups = {}, w = d.population;
    Object.keys(d).forEach(function(c) {
      var m = PARTY_RE.exec(c); if (!m || !Array.isArray(d[c])) return;
      var a = d[c], t = 0;
      for (var i = 0; i < a.length; i++) { var v = a[i]; if (v == null || isNaN(v)) continue; t += (m[3] === 'votes') ? v : v * (w && w[i] != null ? w[i] : 1); }
      var g = groups[m[1]] = groups[m[1]] || {}; (g[m[2]] = g[m[2]] || {})[m[3]] = t;
    });
    var keep = {};
    Object.keys(groups).forEach(function(pre) {
      var sc = {}, tot = 0;
      Object.keys(groups[pre]).forEach(function(p) { var o = groups[pre][p]; sc[p] = o.votes != null ? o.votes : (o.share || 0); tot += sc[p]; });
      tot = tot || 1;
      Object.keys(sc).sort(function(a, b) { return sc[b] - sc[a]; }).slice(0, 4).forEach(function(p) { if (sc[p] / tot > 0.05) keep[pre + '|' + p] = true; });
    });
    majorCache.set(d, keep);
    return keep;
  }
  function majorInScope(c) {
    var m = PARTY_RE.exec(c);
    if (!m || !window.__scopeData) return true;
    try { var d = window.__scopeData(); if (!d || !d.population) return true; return !!majorParties(d)[m[1] + '|' + m[2]]; } catch (e) { return true; }
  }
  window.__colVisible = function(c) {
    if (c === 'level_of_service' || c === 'population') return true;
    if (!majorInScope(c)) return false;
    // country-wide scope, also in the extended census: no party of a regional / municipal election (a regional party such as EH Bildu shows only
    // where that election is held, i.e. when the scope is its region or smaller)
    if (scopeNation() && isPartyCol(c) && /(^|_)elec_(state|municipal)_/.test(c)) return false;
    if (!window.__extendedCensus && S.simple && S.simple.indexOf(c) < 0) return false;
    if (scopeNation() && !window.__extendedCensus) {
      // country-wide scope: no state / municipal parties; left / right proxies instead of national parties (except in the US)
      if (/(^|_)elec_(state|municipal)_|_(senate|governor)_/.test(c)) return false;
      if (!isUS && isPartyCol(c)) return false;
    }
    return true;
  };
  var rank = function(sel) {
    if (sel === 'regFieldSelect' || sel === 'placeRankWeightSelect') return window.__r2Order || S.r2 || [];   // live R2 order of the active scope once the R2 bars were drawn
    return S.anova || [];   // opacity-by, circle-size-by and the other selectors: Census-tab relevance (absolute value; a count by its share equivalent)
  };
  window.__colOrderFor = function(selId, values) {   // values (array) ordered by relevance; unknown ones keep their place at the end
    if (/^(regFieldSelect|placeRankWeightSelect)$/.test(selId) && window.__refreshR2Order) window.__refreshR2Order();
    var r = rank(selId), pos = {}; r.forEach(function(c, i) { pos[c] = i; });
    if (selId === 'regFieldSelect') {   // the regression always starts with population and jobs, else population, else WorldPop population
      var pref = ['pop_jobs_density', 'jobs_and_population_density', 'pop_jobs_total', 'population_density', 'pop_density', 'population', 'worldpop_population_density', 'worldpop_residents'];
      for (var pi = 0; pi < pref.length; pi++) if (values.indexOf(pref[pi]) >= 0) { pos[pref[pi]] = -1; break; }
    }
    return values.map(function(v, i) { return [v, i]; }).sort(function(a, b) {
      var pa = pos[a[0]] == null ? 1e6 + a[1] : pos[a[0]], pb = pos[b[0]] == null ? 1e6 + b[1] : pos[b[0]];
      return pa - pb;
    }).map(function(x) { return x[0]; });
  };
  function addToggle() {
    var host = document.getElementById('statsPanel');
    if (!host || document.getElementById('extCensusCheck')) return;
    var tabs = host.querySelector('.statsTabBtn'); tabs = tabs && tabs.parentNode;
    if (!tabs || !S.simple) return;
    var lab = document.createElement('label');
    lab.id = 'extCensusRow'; lab.style.cssText = 'display:block;margin:4px 0 2px;font:11px sans-serif;color:#555;cursor:pointer;';
    lab.innerHTML = '<input type="checkbox" id="extCensusCheck" style="vertical-align:middle;"> Extended census (all columns)';
    tabs.parentNode.insertBefore(lab, tabs.nextSibling);
    document.getElementById('extCensusCheck').addEventListener('change', function(e) {
      window.__extendedCensus = e.target.checked;
      document.querySelectorAll('.extCensusSync').forEach(function(x) { x.checked = e.target.checked; });
      if (window.__renderActiveTab) window.__renderActiveTab();
    });
  }
  function addLegendToggle() {
    var row = document.getElementById('opacitySelect'); row = row && row.closest('div');
    if (!row || document.getElementById('extCensusLegendCheck') || !S.simple) return;
    var lab = document.createElement('label');
    lab.style.cssText = 'display:block;font:11px sans-serif;color:#555;margin:2px 0;cursor:pointer;';
    lab.innerHTML = '<input type="checkbox" id="extCensusLegendCheck" class="extCensusSync" style="vertical-align:middle;"> Extended census';
    row.parentNode.insertBefore(lab, row.nextSibling);
    lab.querySelector('input').addEventListener('change', function(e) {
      window.__extendedCensus = e.target.checked;
      var c = document.getElementById('extCensusCheck'); if (c) c.checked = e.target.checked;
      if (window.__renderActiveTab) window.__renderActiveTab();
    });
  }
  var n = 0, t = setInterval(function() { addToggle(); addLegendToggle(); if (++n > 40 || (document.getElementById('extCensusCheck') && document.getElementById('extCensusLegendCheck'))) clearInterval(t); }, 500);
  window.__setExtendedCensus = function(on) { window.__extendedCensus = !!on; };
})();
"""


def column_select_script(finest, fields: List[str], is_us: bool, global_logical=None, strip=None,
                         simplified: Optional[List[str]] = None, score_col: str = "level_of_service") -> str:
    """`simplified` overrides the fixed template; relevance (R^2 / ANOVA) only orders the dropdowns.

    Args:
        finest: Finest map level.
        fields: Available columns.
        is_us: US map.
        global_logical: Global logical name per column.
        strip: Function removing source prefixes.
        simplified: Columns of the simplified list (overrides the template).
        score_col: Score column.

    Returns:
        ``<script>`` block of the column selector.
    """
    stats = column_stats(finest, fields, score_col)
    concepts = {}
    if simplified is not None:
        simple = list(simplified)
    else:
        simple, concepts = template_columns(list(fields), is_us)
        rel = main_religions(finest, fields)
        simple = list(dict.fromkeys(simple + rel))
        concepts.update({f"Religion: {c}": c for c in rel})
    r2 = sorted(stats, key=lambda c: -_num(stats[c]["r2"]))
    anova = sorted(stats, key=lambda c: -_num(stats[c]["anova"]))
    payload = {"simple": simple, "r2": r2, "anova": anova, "us": bool(is_us), "concepts": concepts}
    return "<script>\nwindow.__colSelect = " + json.dumps(payload) + ";\n</script>\n<script>" + COLUMN_SELECT_JS + "</script>\n"
