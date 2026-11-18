"""Ranking tab of the US / Spain stats panel (nation-wide maps whose stats come from per-state `stats_chunks/`).

`build_ranking_table(chunk_dir)` streams every `<state>.json.gz` once and writes `ranking.json`: for each geometry level that the chunks
carry (state / county / place), the population-weighted mean and median `level_of_service` of every unit, for each weight column that exists
(population, jobs-like count columns). Levels whose geometry is not in the chunks (metropolitan areas, tracts, electoral districts,
school districts ...) are listed in the tab as "not available" rather than faked.
`RANKING_JS` is the tab: level picker, weight column, mean / median, restricted to the scope chosen in the scope bar.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .pop_range import POP_RANGE_JS

LABELS = {
    "us": {"state": "State", "cbsa": "Metro area (Census CBSA)", "county": "County", "place": "Place", "tract": "Census tract", "congressional": "Congressional district (national)", "legislative": "State legislative district (upper + lower)", "school": "School district", "voting": "Voting district"},
    "spain": {"state": "Autonomous community", "metro": "Metro area", "county": "Province", "place": "Municipality", "district": "Census district", "section": "Census section"},
}
UNAVAILABLE = {"us": [], "spain": []}
_WEIGHT_HINTS = ("jobs", "workplace")


def _wstats(v, w, idx, n):
    """Per-group weighted mean and weighted median of v (weights w) for group index idx in [0, n)."""
    m = (idx >= 0) & np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w, idx = v[m], w[m], idx[m]
    mean = np.full(n, np.nan)
    med = np.full(n, np.nan)
    tot = np.bincount(idx, weights=w, minlength=n)
    with np.errstate(all="ignore"):
        mean = np.bincount(idx, weights=w * v, minlength=n) / tot
    order = np.lexsort((v, idx))
    v, w, idx = v[order], w[order], idx[order]
    cw = np.cumsum(w)
    starts = np.searchsorted(idx, np.arange(n), side="left")
    ends = np.searchsorted(idx, np.arange(n), side="right")
    for g in range(n):
        a, b = starts[g], ends[g]
        if a == b:
            continue
        base = cw[a - 1] if a else 0.0
        half = base + (cw[b - 1] - base) / 2
        med[g] = v[min(a + int(np.searchsorted(cw[a:b], half)), b - 1)]
    return mean, med, tot


CBSA_FILE = "list1.xlsx"   # Census Bureau delineation file (county -> CBSA), cached in TransitLOSStudies/cache/census_cbsa
CBSA_URL = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"


def _cbsa_rows(county_rows, weights) -> List[dict]:
    """Official Census CBSAs (metro + micro areas): county rows aggregated by CBSA (mean exact, median = weighted median of county medians)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cache = None
    for up in range(3, 8):   # .../PTLOS/transitLOS/src/transitlos/map -> PTLOS
        cand = os.path.join(here, *([".."] * up), "TransitLOSStudies", "cache", "census_cbsa")
        if os.path.isdir(os.path.dirname(cand)):
            cache = cand
            break
    if cache is None:
        return []
    path = os.path.join(cache, CBSA_FILE)
    try:
        import pandas as pd

        if not os.path.isfile(path):
            import urllib.request

            os.makedirs(cache, exist_ok=True)
            urllib.request.urlretrieve(CBSA_URL, path)
        d = pd.read_excel(path, header=2, dtype=str).dropna(subset=["CBSA Code", "FIPS State Code", "FIPS County Code"])
    except Exception as exc:  # noqa: BLE001
        print(f"[ranking] CBSA table unavailable: {exc!r}")
        return []
    d["fips"] = d["FIPS State Code"] + d["FIPS County Code"]
    by_fips = dict(zip(d["fips"], zip(d["CBSA Code"], d["CBSA Title"], d["Metropolitan/Micropolitan Statistical Area"])))
    groups: Dict[str, List[dict]] = {}
    for r in county_rows:
        hit = by_fips.get(r.get("id"))
        if hit:
            groups.setdefault(hit[0], []).append((r, hit))
    rows = []
    for code, items in groups.items():
        hit = items[0][1]
        row = {"name": hit[1] + (" (micro)" if hit[2].startswith("Micro") else ""), "state": items[0][0]["state"], "states": sorted({r["state"] for r, _ in items}), "id": code}
        for wcol in weights:
            ws = [(r.get("w_" + wcol), r.get("mean_" + wcol), r.get("median_" + wcol)) for r, _ in items if r.get("w_" + wcol)]
            if not ws:
                continue
            tot = sum(w for w, _, _ in ws)
            row["w_" + wcol] = round(tot, 1)
            row["mean_" + wcol] = round(sum(w * m for w, m, _ in ws) / tot, 2)
            acc, half = 0.0, tot / 2
            for w, _, md in sorted(ws, key=lambda x: x[2]):
                acc += w
                if acc >= half:
                    row["median_" + wcol] = md
                    break
        _set_share(row, _transit_of([r for r, _ in items]))   # CBSA = sum of ALL its counties
        if "mean_population" in row:
            rows.append(row)
    return rows


def _metro_rows(place_rows, weights, map_path) -> List[dict]:
    """Metropolitan areas from a study-provided {"areas": {code: name}, "muni": {municipality id: code}} file: municipality rows aggregated
    (mean exact, median = weighted median of municipality medians)."""
    if not os.path.isfile(map_path):
        return []
    m = json.load(open(map_path))
    groups: Dict[str, List[dict]] = {}
    for r in place_rows:
        code = m["muni"].get(r.get("id"))
        if code:
            groups.setdefault(code, []).append(r)
    rows = []
    for code, items in groups.items():
        row = {"name": m["areas"][code], "state": items[0]["state"], "states": sorted({r["state"] for r in items}), "id": code}
        for wcol in weights:
            ws = [(r.get("w_" + wcol), r.get("mean_" + wcol), r.get("median_" + wcol)) for r in items if r.get("w_" + wcol) and r.get("mean_" + wcol) is not None]
            if not ws:
                continue
            tot = sum(w for w, _, _ in ws)
            row["w_" + wcol] = round(tot, 1)
            row["mean_" + wcol] = round(sum(w * mn for w, mn, _ in ws) / tot, 2)
            acc = 0.0
            for w, _, md in sorted(ws, key=lambda x: x[2]):
                acc += w
                if acc >= tot / 2:
                    row["median_" + wcol] = md
                    break
        _set_share(row, _transit_of(items))   # Spain metro area: only when EVERY municipality has survey data
        if "mean_population" in row:
            rows.append(row)
    return rows


def _unit_rows(df, key, namer, state, extra=None):
    """Rows (population-weighted mean / median level of service) of the units in `df` (cols GEOID/key, population, level_of_service)."""
    df = df[np.isfinite(df["level_of_service"]) & (df["population"] > 0)]
    if df.empty:
        return []
    codes, uniq = np.unique(df[key].astype(str).to_numpy(), return_inverse=True)[1], None
    uniq = np.unique(df[key].astype(str).to_numpy())
    mean, med, tot = _wstats(df["level_of_service"].to_numpy(float), df["population"].to_numpy(float), codes, len(uniq))
    rows = []
    for i, u in enumerate(uniq):
        if np.isfinite(mean[i]):
            r = {"name": namer(u), "state": state, "id": u, "w_population": round(float(tot[i]), 1),
                 "mean_population": round(float(mean[i]), 2), "median_population": round(float(med[i]), 2)}
            if extra:
                r.update(extra(u))
            rows.append(r)
    return rows


def _extra_levels(chunk_dir, states, out):
    """Levels read from the per-state / per-community geometry tables of the study (`data/<State>/...geoparquet`): US tract, school district,
    congressional and state legislative districts; Spain census district and section (section GEOID = province 2 + municipality 3 + district 2 + section 3)."""
    import pandas as pd

    data_root = os.path.dirname(os.path.dirname(os.path.abspath(chunk_dir)))
    us = any(st["name"] == "Alabama" for st in states)
    res: Dict[str, List[dict]] = {}
    muni_names = {p["id"]: p["name"] for st in states for p in st["places"]}
    cols = ["GEOID", "population", "level_of_service"]
    if us:
        county_names = {c["id"]: c["name"] for st in states for c in st["counties"]}
        spec = {"congressional": ("congressional_district.geoparquet", "Congressional district"),
                "school": ("school_district.geoparquet", "School district"),
                "legislative": ("state_legislative_district.geoparquet", "State legislative district"),
                "voting": ("voting_district.geoparquet", "Voting district")}
        for lvl, (fn, label) in spec.items():
            rows: List[dict] = []
            for st in states:
                f = os.path.join(data_root, st["key"], fn)
                if os.path.isfile(f):
                    rows += _unit_rows(pd.read_parquet(f, columns=cols), "GEOID", lambda g, st=st, label=label: f"{st['name']} - {label} {g[2:]}", st["key"])
            if rows:
                res[lvl] = rows
        tracts: List[dict] = []
        for st in states:
            base = os.path.join(data_root, st["key"], "counties")
            for cid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
                f = os.path.join(base, cid, "tract.geoparquet")
                try:
                    if os.path.isfile(f):
                        tracts += _unit_rows(pd.read_parquet(f, columns=cols), "GEOID", lambda g, st=st: f"{county_names.get(g[:5], g[:5])}, {st['name']} - tract {g[5:]}", st["key"])
                except Exception:  # a county file being rewritten by the data pipeline: skipped
                    continue
        if tracts:
            res["tract"] = tracts
        return res
    sections: List[dict] = []
    for st in states:
        f = os.path.join(data_root, st["key"], "section.geoparquet")
        if os.path.isfile(f):
            sections += _unit_rows(pd.read_parquet(f, columns=cols), "GEOID", lambda g: f"{muni_names.get(g[:5], g[:5])} - district {g[5:7]} section {g[7:]}", st["key"],
                                   extra=lambda g: {"prov": g[:2], "muni": g[:5]})
    if sections:
        res["section"] = sections
        sd = pd.DataFrame(sections)
        # districts: aggregate sections by the 7-digit prefix (weighted mean exact; median = weighted median of section medians)
        grp: Dict[str, List[dict]] = {}
        for r in sections:
            grp.setdefault(r["id"][:7], []).append(r)
        drows = []
        for g, items in grp.items():
            tot = sum(r["w_population"] for r in items)
            acc, half, md = 0.0, tot / 2, items[0]["median_population"]
            for r in sorted(items, key=lambda x: x["median_population"]):
                acc += r["w_population"]
                if acc >= half:
                    md = r["median_population"]
                    break
            drows.append({"name": f"{muni_names.get(g[:5], g[:5])} - district {g[5:7]}", "state": items[0]["state"], "id": g, "prov": g[:2], "muni": g[:5],
                          "w_population": round(tot, 1), "mean_population": round(sum(r["w_population"] * r["mean_population"] for r in items) / tot, 2), "median_population": md})
        res["district"] = drows
    return res


def _transit_of(items, min_cover: float = 1.0):
    """Transit share of a larger area from its parts' COUNTS (sum of transit commuters / sum of commuters): only upscaling, never a share
    spread down. None unless the parts with data hold at least `min_cover` of the parts' population (1.0: every part must have data)."""
    tot = sum(r.get("w_population") or 0.0 for r in items)
    have = [r for r in items if r.get("transit_den")]
    cov = sum(r.get("w_population") or 0.0 for r in have)
    if not have or (tot > 0 and cov / tot < min_cover - 1e-9):
        return None
    num, den = sum(r["transit_num"] for r in have), sum(r["transit_den"] for r in have)
    return (round(num / den, 4), num, den) if den > 0 else None


def _set_share(row, agg):
    if agg:
        row["transit_share"], row["transit_num"], row["transit_den"] = agg


def _attach_transit_shares(out: Dict[str, List[dict]], chunk_dir: str, states) -> None:
    """`transit_share` (0-1) on the ranking rows, ONLY where the source publishes the geography or it is an exact sum of published parts
    (never a share spread down from a municipal or larger unit; values resampled from a sub-municipal source such as block groups are fine):
      US     ACS 5-year commuters by public transport / workers 16+ of tracts, counties, places and the congressional / legislative / school /
             voting districts (the last ones resampled from block groups); states from their own ACS totals, CBSAs summed from their counties.
      Spain  INE ECEPOV-2021 persons 16+ by main means of transport to work or study: municipalities > 50,000 / provincial capitals and
             provinces as published (data/modal_split_ecepov2021.json); communities summed from their provinces, metro areas from their
             municipalities, only when every part has data."""
    import pandas as pd

    data_root = os.path.dirname(os.path.dirname(os.path.abspath(chunk_dir)))
    us = any(st["name"] == "Alabama" for st in states)
    counts: Dict[str, Dict[str, tuple]] = {}
    if us:
        def read(f, lvl):
            try:
                d = pd.read_parquet(f, columns=["GEOID", "publicTransportCommuters", "acs5_workers"])
            except Exception:  # noqa: BLE001
                return
            w = pd.to_numeric(d["acs5_workers"], errors="coerce"); t = pd.to_numeric(d["publicTransportCommuters"], errors="coerce")
            ok = (w > 0) & t.notna()
            counts.setdefault(lvl, {}).update({g: (float(a), float(b)) for g, a, b in zip(d["GEOID"][ok].astype(str), t[ok], w[ok])})
        for st in states:
            base = os.path.join(data_root, st["key"])
            for lvl, fn in (("county", "county.geoparquet"), ("place", "place.geoparquet"), ("congressional", "congressional_district.geoparquet"),
                            ("school", "school_district.geoparquet"), ("legislative", "state_legislative_district.geoparquet"), ("voting", "voting_district.geoparquet")):
                if os.path.isfile(os.path.join(base, fn)):   # places / districts: resampled from block groups (a sub-municipal source: allowed)
                    read(os.path.join(base, fn), lvl)
            cd = os.path.join(base, "counties")
            for cid in sorted(os.listdir(cd)) if os.path.isdir(cd) else []:
                f = os.path.join(cd, cid, "tract.geoparquet")
                if os.path.isfile(f):
                    read(f, "tract")
    else:
        f = os.path.join(data_root, "modal_split_ecepov2021.json")
        if not os.path.isfile(f):
            return
        ms = json.load(open(f, encoding="utf-8"))
        counts["place"] = {k: (v["public"], v["total"]) for k, v in ms.get("muni", {}).items() if v.get("total")}
        counts["county"] = {k: (v["public"], v["total"]) for k, v in ms.get("prov", {}).items() if v.get("total")}
    for lvl, m in counts.items():
        for r in out.get(lvl, []):
            c = m.get(str(r.get("id")))
            if c and c[1] > 0:
                r["transit_num"], r["transit_den"], r["transit_share"] = c[0], c[1], round(c[0] / c[1], 4)
    for r in out.get("state", []):   # US: the state's own ACS totals; Spain: autonomous communities summed from ALL their provinces
        if us:
            f = os.path.join(data_root, r.get("state", ""), "state.geoparquet")
            try:
                d = pd.read_parquet(f, columns=["publicTransportCommuters", "acs5_workers"])
                a, b = float(pd.to_numeric(d["publicTransportCommuters"]).sum()), float(pd.to_numeric(d["acs5_workers"]).sum())
                if b > 0:
                    _set_share(r, (round(a / b, 4), a, b))
            except Exception:  # noqa: BLE001
                pass
        else:
            _set_share(r, _transit_of([c for c in out.get("county", []) if c.get("state") == r.get("state")]))


def build_ranking_table(chunk_dir: str) -> str:
    """``ranking.json``: score and weights of every state, county and place from the stats chunks (and
    ``ranking_top.json``).

    Args:
        chunk_dir: Stats chunk folder (with ``index.json``).

    Returns:
        Path of ``ranking.json``.
    """
    index = json.load(open(os.path.join(chunk_dir, "index.json")))
    states = index["states"]
    out: Dict[str, List[dict]] = {"state": [], "county": [], "place": []}
    weights_seen: List[str] = ["population"]
    for st in states:
        path = os.path.join(chunk_dir, st["key"] + ".json.gz")
        if not os.path.isfile(path):
            continue
        d = json.load(gzip.open(path))
        v = np.asarray(d["level_of_service"], dtype=float)
        wcols = ["population"] + [c for c in d if any(h in c.lower() for h in _WEIGHT_HINTS) and c.lower().replace("_", "") in ("jobs", "jobstotal", "workplacejobs", "popjobstotal", "ldwacjobstotal", "lodeswacjobstotal", "statcanjobstotal", "siijobstotal", "bajobstotal")]
        for c in wcols:
            if c not in weights_seen:
                weights_seen.append(c)
        cols = {"state": (np.zeros(len(v), dtype=int), 1, [st["name"]]),
                "county": (np.asarray(d.get("county", [-1] * len(v)), dtype=int), len(st["counties"]), [c["name"] for c in st["counties"]]),
                "place": (np.asarray(d.get("place", [-1] * len(v)), dtype=int), len(st["places"]), [p["name"] for p in st["places"]])}
        if "unweighted" not in weights_seen:
            weights_seen.append("unweighted")
        for level, (idx, n, names) in cols.items():
            rows = [{"name": nm, "state": st["key"], "i": i} for i, nm in enumerate(names)]
            for c in wcols + ["unweighted"]:
                w = np.ones(len(v)) if c == "unweighted" else np.asarray(d[c], dtype=float)   # unweighted: every cell counts the same
                mean, med, tot = _wstats(v, w, idx, n)
                for i, r in enumerate(rows):
                    if np.isfinite(mean[i]):
                        r["w_" + c] = round(float(tot[i]), 1)
                        r["mean_" + c] = round(float(mean[i]), 2)
                        r["median_" + c] = round(float(med[i]), 2)
            if level in ("county", "place"):
                for r in rows:
                    r["id"] = (st["counties"] if level == "county" else st["places"])[r["i"]]["id"]
            out[level].extend(r for r in rows if "mean_population" in r)
        # population + jobs composite weight, when a jobs column exists in the chunk
        jc = next((c for c in wcols if c != "population"), None)
        if jc:
            wpj = np.asarray(d["population"], dtype=float) + np.nan_to_num(np.asarray(d[jc], dtype=float))
            if "population+jobs" not in weights_seen:
                weights_seen.append("population+jobs")
            for level, (idx, n, names) in cols.items():
                mean, med, tot = _wstats(v, wpj, idx, n)
                rows = [r for r in out[level] if r["state"] == st["key"]]
                for r in rows:
                    i = r["i"]
                    if i < n and np.isfinite(mean[i]):
                        r["w_population+jobs"], r["mean_population+jobs"], r["median_population+jobs"] = round(float(tot[i]), 1), round(float(mean[i]), 2), round(float(med[i]), 2)
    try:
        out.update(_extra_levels(chunk_dir, states, out))
    except Exception as exc:  # noqa: BLE001
        print(f"[ranking] extra levels skipped: {exc!r}")
    try:
        _attach_transit_shares(out, chunk_dir, states)
    except Exception as exc:  # noqa: BLE001
        print(f"[ranking] transit shares skipped: {exc!r}")
    cbsa = _cbsa_rows(out["county"], weights_seen)
    if cbsa:
        out["cbsa"] = cbsa
    metro = _metro_rows(out.get("place", []), weights_seen, os.path.join(chunk_dir, "metro_map.json"))
    if metro:
        out["metro"] = metro
    path = os.path.join(chunk_dir, "ranking.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"levels": out, "weights": weights_seen}, f, separators=(",", ":"))
    write_ranking_top(chunk_dir, out)
    return path


_TOP_KEYS = ("name", "state", "states", "id", "mean_population", "w_population")


def write_ranking_top(chunk_dir: str, levels: Optional[dict] = None) -> str:
    """`ranking_top.json`: the state / county / place rows with only what the top bar's score needs, so the page does not download the
    whole ranking table (tens of MB) at start; `ranking.json` is fetched when the stats panel is opened.

    Args:
        chunk_dir: Stats chunk folder.
        levels: Ranking levels; ``None`` = read ``ranking.json``.

    Returns:
        Path of ``ranking_top.json``.
    """
    if levels is None:
        levels = json.load(open(os.path.join(chunk_dir, "ranking.json")))["levels"]
    top = {lv: [{k: r[k] for k in _TOP_KEYS if k in r} for r in levels.get(lv, [])] for lv in ("state", "county", "place")}
    path = os.path.join(chunk_dir, "ranking_top.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"levels": top}, f, separators=(",", ":"))
    return path


# ---- "Rank by" any share / count column (2026-11-14, user) -------------------------------------------------------------------------------
# The study's stats-chunk step writes `rank_cells/<state>.parquet` (`write_rank_cells`): finer cells than the res-8 chunks (US res 9, Spain
# res 10) with the same map columns and, per ranking level, the id of the unit each cell centre falls in. `build_rank_columns` aggregates them
# for every level (count columns summed; share / rate / density / median ... columns population-weighted means, the rule of
# `build.is_relative_field`) and writes one small file per (level, column) aligned with that level's rows in `ranking.json`, fetched by the
# tab only when the column is picked.
RANK_SKIP = {"h3", "h3_cell", "county", "place", "level_of_service"}
RANK_CELLS = "rank_cells"
RANK_COLS = "ranking_cols"


def write_rank_cells(cells, state_key: str, units: dict, out_dir: str) -> str:
    """`cells` (column `h3`, `population` and the map's numeric columns) + per level in `units` ({level: GeoDataFrame GEOID/geometry,
    or a callable on the joined ids}) the unit id of each cell centre -> `<out_dir>/rank_cells/<state_key>.parquet`.

    Args:
        cells: Cells with ``h3``, ``population`` and numeric columns.
        state_key: State key.
        units: ``{level: GeoDataFrame or function}``.
        out_dir: Stats chunk folder.

    Returns:
        Path of the parquet.
    """
    import geopandas as gpd
    import h3
    import pandas as pd

    df = cells.reset_index(drop=True)
    ll = np.asarray([h3.cell_to_latlng(c) for c in df["h3"].astype(str)], dtype=float).reshape(-1, 2)
    pts = gpd.GeoDataFrame({"_i": np.arange(len(df))}, geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326)
    out = {"population": pd.to_numeric(df.get("population"), errors="coerce").astype("float32")}
    for c in df.columns:
        if c not in RANK_SKIP and c != "population" and pd.api.types.is_numeric_dtype(df[c]):
            out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    out["u_state"] = np.full(len(df), state_key, dtype=object)
    derived = {}
    for level, g in units.items():
        if callable(g):
            derived[level] = g
            continue
        if g is None or not len(g):
            continue
        g = g[["GEOID", "geometry"]].to_crs(4326)
        j = gpd.sjoin(pts, g, how="left", predicate="within").drop_duplicates("_i").sort_values("_i")
        out["u_" + level] = j["GEOID"].astype(object).where(j["GEOID"].notna(), None).to_numpy()
    for level, fn in derived.items():   # e.g. Spain census district = the section id's first 7 characters
        out["u_" + level] = fn(out)
    path = os.path.join(out_dir, RANK_CELLS, state_key + ".parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_parquet(path)
    return path


def _sig(v: float):
    if v is None or not np.isfinite(v):
        return None
    if v == 0:
        return 0
    return float(f"{v:.4g}")


def build_rank_columns(chunk_dir: str) -> int:
    """`ranking_cols/columns.json` + `ranking_cols/<level>/<i>.json` (see the block comment). Returns the number of files written.

    Args:
        chunk_dir: Stats chunk folder.
    """
    import pandas as pd

    from .build import is_relative_field

    cdir = os.path.join(chunk_dir, RANK_CELLS)
    if not os.path.isdir(cdir):
        return 0
    table = json.load(open(os.path.join(chunk_dir, "ranking.json")))
    rows_by_level = table["levels"]
    files = sorted(f for f in os.listdir(cdir) if f.endswith(".parquet"))
    # extra groupings: US CBSA from the county ids, Spain metro areas from the municipality ids
    maps = {}
    cb = _cbsa_map()
    if cb:
        maps["cbsa"] = ("county", cb)
    mp = os.path.join(chunk_dir, "metro_map.json")
    if os.path.isfile(mp):
        maps["metro"] = ("place", json.load(open(mp))["muni"])
    cols, acc = None, {}   # acc[level][col] = (sum v*w or v, sum w) per unit id
    for f in files:
        d = pd.read_parquet(os.path.join(cdir, f))
        if cols is None:
            cols = []
        for c in d.columns:
            if not c.startswith("u_") and c != "population" and c not in cols:
                cols.append(c)
        w = d["population"].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        levels = {c[2:]: d[c].to_numpy() for c in d.columns if c.startswith("u_")}
        for lvl, (base, m) in maps.items():
            if base in levels:
                levels[lvl] = np.array([m.get(x) if isinstance(x, str) else None for x in levels[base]], dtype=object)
        allc = [c for c in d.columns if not c.startswith("u_")]
        for lvl, ids in levels.items():
            ok = np.array([isinstance(x, str) and x != "" for x in ids])
            if not ok.any():
                continue
            u, inv = np.unique(ids[ok].astype(str), return_inverse=True)
            A = acc.setdefault(lvl, {})
            for c in allc:
                v = d[c].to_numpy(dtype=float)[ok]
                fin = np.isfinite(v)
                rel = is_relative_field(c)
                ww = w[ok] if rel else np.ones(fin.size)
                num = np.bincount(inv, weights=np.where(fin, v * ww if rel else v, 0.0), minlength=len(u))
                den = np.bincount(inv, weights=np.where(fin, ww, 0.0), minlength=len(u))
                prev = A.setdefault(c, {})
                for k, uid in enumerate(u):
                    if den[k] > 0:
                        a, b = prev.get(uid, (0.0, 0.0))
                        prev[uid] = (a + num[k], b + den[k])
    if not cols:
        return 0
    cols = ["population"] + [c for c in cols if c != "population"]
    rel = [bool(is_relative_field(c)) for c in cols]
    out_dir = os.path.join(chunk_dir, RANK_COLS)
    import shutil

    shutil.rmtree(out_dir, ignore_errors=True)
    import re

    pct_re = re.compile(r"(share|rate|ratio|pct|percent)$", re.I)
    vmax = {}
    for A in acc.values():   # percent-like columns stored as 0-1 (Spain) or 0-100 (US ACS): the tab needs the scale to show a %
        for i, c in enumerate(cols):
            if rel[i] and pct_re.search(c) and c in A:
                vmax[c] = max(vmax.get(c, 0.0), max((abs(t[0] / t[1]) for t in A[c].values() if t[1] > 0), default=0.0))
    pct = [0 if not (rel[i] and pct_re.search(c)) else (100 if vmax.get(c, 0.0) > 1.5 else 1) for i, c in enumerate(cols)]
    n, avail = 0, {}
    for lvl, rows in rows_by_level.items():
        A = acc.get(lvl)
        if not A:
            continue
        ids = [str(r.get("state")) if lvl == "state" else (str(r["id"]) if r.get("id") is not None else None) for r in rows]
        os.makedirs(os.path.join(out_dir, lvl), exist_ok=True)
        have = []
        for i, c in enumerate(cols):
            m = A.get(c)
            if not m:
                continue
            vals = []
            for uid in ids:
                t = m.get(uid) if uid is not None else None
                vals.append(None if t is None or t[1] <= 0 else _sig(t[0] / t[1] if rel[i] else t[0]))
            if sum(v is not None for v in vals) < 2:   # a column needs at least two units to rank
                continue
            with gzip.open(os.path.join(out_dir, lvl, f"{i}.json.gz"), "wt", encoding="utf-8", compresslevel=6) as fh:
                json.dump(vals, fh, separators=(",", ":"))
            have.append(i)
            n += 1
        avail[lvl] = have
    with open(os.path.join(out_dir, "columns.json"), "w") as fh:
        json.dump({"columns": cols, "relative": rel, "pct": pct, "levels": avail}, fh, separators=(",", ":"))
    return n


def _cbsa_map() -> dict:
    """{county fips: CBSA code} from the Census delineation file (cached; see `_cbsa_rows`)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(3, 8):
        cand = os.path.join(here, *([".."] * up), "TransitLOSStudies", "cache", "census_cbsa", CBSA_FILE)
        if os.path.isfile(cand):
            try:
                import pandas as pd

                d = pd.read_excel(cand, header=2, dtype=str).dropna(subset=["CBSA Code", "FIPS State Code", "FIPS County Code"])
                return dict(zip(d["FIPS State Code"] + d["FIPS County Code"], d["CBSA Code"]))
            except Exception:  # noqa: BLE001
                return {}
    return {}


RANKING_JS = POP_RANGE_JS + r"""
(function() {
  var CFG = __RCFG__, DATA = null, level = 'state', method = 'mean', weight = 'population', minPop = 0, maxPop = null;
  // "Rank by": 'score' (the transit score, weight + mean / median), 'transit' (published transit share) or 'col:<i>' (any share / count column
  // of ranking_cols/columns.json; counts summed over the area, the rest population-weighted means of its cells), fetched per level on demand
  var rankBy = 'score', COLS = null, colCache = {};
  function gunzipJson(url) {
    return fetch(url).then(function(r) { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }).then(function(buf) {
      var u8 = new Uint8Array(buf);
      if (u8[0] === 0x1f && u8[1] === 0x8b && typeof DecompressionStream !== 'undefined') return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
      return JSON.parse(new TextDecoder().decode(u8));
    });
  }
  function loadCols() {
    if (COLS !== null) return;
    COLS = false;
    fetch(CFG.dir + '/ranking_cols/columns.json').then(function(r) { return r.ok ? r.json() : null; }).then(function(m) { COLS = m || false; if (m) render(); }).catch(function() {});
  }
  function colValues(lvl, i) {   // the column's values aligned with DATA.levels[lvl]; null while loading / unavailable
    var k = lvl + '/' + i;
    if (colCache[k] === undefined) {
      colCache[k] = null;
      gunzipJson(CFG.dir + '/ranking_cols/' + lvl + '/' + i + '.json.gz').then(function(a) { colCache[k] = a; render(); }).catch(function() { colCache[k] = false; render(); });
    }
    return colCache[k];
  }
  function esc(s) { return String(s).replace(/[&<>]/g, function(c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]; }); }
  function addTab() {
    var tabs = document.querySelector('#statsPanel .statsTabBtn'); tabs = tabs && tabs.parentNode;
    if (!tabs || document.getElementById('statsTab_ranking')) return !!document.getElementById('statsTab_ranking');
    var btn = document.createElement('button'); btn.className = 'statsTabBtn'; btn.setAttribute('data-tab', 'ranking'); btn.textContent = 'Ranking';
    tabs.insertBefore(btn, tabs.firstChild);
    var panel = document.createElement('div'); panel.id = 'statsTab_ranking'; panel.className = 'statsTabPanel'; panel.style.display = 'none';
    var last = document.querySelector('#statsPanel .statsTabPanel'); last.parentNode.appendChild(panel);
    btn.addEventListener('click', function() {
      window.__statsTab = 'ranking';
      document.querySelectorAll('.statsTabBtn').forEach(function(b) { b.classList.remove('active'); }); btn.classList.add('active');
      document.querySelectorAll('.statsTabPanel').forEach(function(p) { p.style.display = 'none'; }); panel.style.display = 'block';
      render();
    });
    var orig = window.__renderActiveTab;
    window.__renderActiveTab = function() { if (window.__statsTab === 'ranking') { render(); return; } if (orig) orig(); };
    btn.click();
    return true;
  }
  var SN = {}, TOP = null, topTried = false;
  function statsOpen() { var p = document.getElementById('statsPanel'); return !!p && p.style.display !== 'none'; }
  function loadTop() {
    if (topTried) return; topTried = true;
    fetch(CFG.dir + '/index.json').then(function(r) { return r.json(); }).then(function(ix) { (ix.states || []).forEach(function(x) { SN[x.key] = x.name; }); }).catch(function() {});
    fetch(CFG.dir + '/ranking_top.json').then(function(r) { return r.ok ? r.json() : null; }).then(function(d) { if (d) TOP = d; }).catch(function() {});
  }
  loadTop();
  function load(cb) {
    if (DATA) return cb();
    var ixp = fetch(CFG.dir + '/index.json').then(function(r) { return r.json(); }).then(function(ix) { (ix.states || []).forEach(function(x) { SN[x.key] = x.name; }); }).catch(function() {});
    ixp.then(function() { return fetch(CFG.dir + '/ranking.json'); }).then(function(r) { return r.json(); }).then(function(d) { DATA = d; if (!window.__rkWeightSet) { window.__rkWeightSet = 1; if (d.weights.indexOf('population+jobs') >= 0) weight = 'population+jobs'; } cb(); })
      .catch(function() { document.getElementById('statsTab_ranking').innerHTML = '<div style="color:#a33;font:12px sans-serif;"></div>'; });
  }
  function scopeSel() {   // {state, county, place} of the primary scope bar
    var lv = window.__scopeLevel ? window.__scopeLevel() : 'nation';
    var lab = window.__scopeLabel ? window.__scopeLabel('metro') : '';
    return {lv: lv, label: lab};
  }
  // Top-centre score of a nation-wide map: the editor's baseline needs the whole map's cells in the page, which these maps do not
  // carry, so the bar shows the population-weighted mean score of the scope picked in the scope bar (from the ranking table).
  // The score follows the zoom and ALWAYS the centre of the map (not the majority of the view): nation, then the state / autonomous
  // community under the centre, then its county / province, then its place / municipality; with no place at the centre it stays at
  // the county. Rows come from the ranking table (population-weighted means).
  var TOP_ZOOMS = {nationMax: 6.5, state: CFG.min_zoom || 5, county: 8, place: 10};
  // The features under the map centre, from the visible fill layers whose source covers it only (a nationwide page has thousands of chunk
  // layers: querying all of them took ~2 s and ran on a timer), cached per view and never recomputed while the map moves.
  var centreCache = {sig: null, fs: []};
  window.__centreFeatures = function(m) {
    if (m.isMoving && m.isMoving()) return centreCache.fs;
    var c = m.getCenter(), sig = c.lng.toFixed(6) + ',' + c.lat.toFixed(6) + ',' + m.getZoom().toFixed(3), fs = [];
    if (sig === centreCache.view && Date.now() - centreCache.t < 3000) return centreCache.fs;   // same view: skip even the layer scan
    var view = sig;
    var all = m.getLayersOrder ? m.getLayersOrder() : m.getStyle().layers.map(function(l) { return l.id; });
    var fills = all.filter(function(id) { var L = m.getLayer(id); return L && L.type === 'fill' && !/^__bm_/.test(id); });
    var ls = window.__liveLayers ? window.__liveLayers(m, fills, c) : fills;
    sig += '|' + ls.join(',');
    if (sig === centreCache.sig) { centreCache.view = view; centreCache.t = Date.now(); return centreCache.fs; }
    try { var pt = m.project(c); if (ls.length) fs = m.queryRenderedFeatures([[pt.x - 2, pt.y - 2], [pt.x + 2, pt.y + 2]], {layers: ls}); } catch (e) {}
    centreCache = {sig: sig, view: view, t: Date.now(), fs: fs};
    return fs;
  };
  function centreIds(m) {
    var ids = [];
    window.__centreFeatures(m).forEach(function(f) {
      var pr = f.properties || {};
      for (var k in pr) if (/^(geoid|id)$/i.test(k) && (typeof pr[k] === 'string' || typeof pr[k] === 'number') && String(pr[k]).length >= 2) ids.push(String(pr[k]));
    });
    return ids;
  }
  function rowByIds(rows, ids, state) {
    var best = null, bl = -1;
    (rows || []).forEach(function(r) {
      if (state && r.state !== state && (r.states || []).indexOf(state) < 0) return;
      var rid = r.id != null ? String(r.id) : null;
      if (rid && rid.length > bl && ids.some(function(v) { return v === rid || (v.length > rid.length && v.indexOf(rid) === 0); })) { best = r; bl = rid.length; }
    });
    return best;
  }
  function natMean(L) {
    var num = 0, den = 0;
    (L.state || []).forEach(function(q) { if (q.mean_population != null && q.w_population > 0) { num += q.mean_population * q.w_population; den += q.w_population; } });
    return den > 0 ? num / den : null;
  }
  window.__scopeScore = function() {
    var src = DATA || TOP;
    if (!src) return null;
    var L = src.levels, m = window.__rkMap;
    window.__scopeScoreName = CFG.country === 'spain' ? 'Spain' : 'United States';
    // 2026-11-13 fix: the score follows the scope bar -- at the nation level it is the national value, never the state / community that
    // happens to sit under the map centre at the default view (the default zoom is already past the state threshold)
    if (!m || !m.getZoom || !window.__stateAt) return natMean(L);
    if (m.getZoom() < TOP_ZOOMS.nationMax) return natMean(L);   // low zoom (the whole country in view): the national score
    var z = m.getZoom(), c = m.getCenter();
    if (z < TOP_ZOOMS.state) return natMean(L);
    var st = window.__stateAt(c.lng, c.lat);
    if (!st) return natMean(L);
    var row = null;
    (L.state || []).forEach(function(r) { if (r.state === st) row = r; });
    if (z >= TOP_ZOOMS.county) {
      var ids = centreIds(m), cty = rowByIds(L.county, ids, st);
      if (cty) { row = cty; if (z >= TOP_ZOOMS.place) { var pl = rowByIds(L.place, ids, st); if (pl) row = pl; } }
    }
    if (row && row.mean_population != null) { window.__scopeScoreName = row.name + (row.state && SN[row.state] && row.name !== SN[row.state] ? ', ' + SN[row.state] : ''); return row.mean_population; }
    return natMean(L);
  };
  function fillTopBar() {   // the editor's own top-bar refresh needs the scenario selector, which these maps do not have: write the score here
    var out = document.getElementById('topBarResults'), sel = document.getElementById('topBarScenarioSelect');
    if (!out || (sel && sel.value !== '')) return;
    var v = window.__scopeScore ? window.__scopeScore() : null;
    if (v != null) { var h = '<b>' + v.toFixed(2) + '</b>'; if (out.innerHTML !== h) out.innerHTML = h;
      out.title = 'Transit score of ' + (window.__scopeScoreName || '') + ' (population-weighted mean of its cells)'; }
  }
  setInterval(fillTopBar, 700);
  var lastScopeSig = null;
  setInterval(function() {   // the list follows the scope bar (nation <-> state <-> county / municipality) as the map is zoomed or the user picks one
    if (!DATA) return;
    var sig = '';
    try { sig = scopeSel().lv + '|' + (window.__scopeStateKey && window.__scopeStateKey()) + '|' + (window.__scopeCountyId && window.__scopeCountyId()) + '|' + (window.__scopePlaceId && window.__scopePlaceId()); } catch (e) {}
    if (sig === lastScopeSig) return;
    var first = lastScopeSig === null; lastScopeSig = sig;
    if (!first) render();
  }, 700);
  function render() {
    var el = document.getElementById('statsTab_ranking'); if (!el) return;
    if (!statsOpen()) return;   // the ranking table loads when the stats panel is opened (the panel's button re-renders the active tab)
    load(function() {
      loadCols();
      var L = CFG.labels, s = scopeSel(), h = '';
      // Zoomed into a state / autonomous community the ranking lists its counties / provinces / municipalities (the "state" level only has
      // meaning for the whole nation): a stale 'state' choice moves to the county level, which then only lists the active state.
      var inState = s.lv !== 'nation' && s.lv !== 'none';
      if (inState && level === 'state') level = (DATA.levels.county || []).length ? 'county' : 'place';
      if (method === 'transit') { method = 'mean'; rankBy = 'transit'; }
      var hasTransit = (DATA.levels[level] || []).some(function(r) { return r.transit_share != null; });
      var colAvail = (COLS && COLS.levels && COLS.levels[level]) || [];
      if (rankBy === 'transit' && !hasTransit) rankBy = 'score';
      if (rankBy.indexOf('col:') === 0 && colAvail.indexOf(+rankBy.slice(4)) < 0) rankBy = 'score';
      var colI = rankBy.indexOf('col:') === 0 ? +rankBy.slice(4) : -1, colName = colI >= 0 ? COLS.columns[colI] : null;
      var colLabels = (COLS && window.__fieldLabelsFor) ? window.__fieldLabelsFor(colAvail.map(function(i) { return COLS.columns[i]; })) : {};
      var colLabel = function(i) { var c = COLS.columns[i]; return colLabels[c] || c; };
      var byOpts = '<option value="score"' + (rankBy === 'score' ? ' selected' : '') + '>Transit score</option>' +
        (hasTransit ? '<option value="transit"' + (rankBy === 'transit' ? ' selected' : '') + '>Transit share</option>' : '') +
        (colAvail.length ? '<optgroup label="Census and other columns">' + colAvail.slice().sort(function(a, b) { return colLabel(a).localeCompare(colLabel(b)); }).map(function(i) {
          return '<option value="col:' + i + '"' + (rankBy === 'col:' + i ? ' selected' : '') + '>' + esc(colLabel(i)) + '</option>'; }).join('') + '</optgroup>' : '');
      var lvlOpts = ['state', 'cbsa', 'metro', 'congressional', 'legislative', 'county', 'place', 'school', 'voting', 'district', 'tract', 'section'].filter(function(k) { return L[k] && (DATA.levels[k] || []).length && !(inState && k === 'state'); }).map(function(k) { return '<option value="' + k + '"' + (k === level ? ' selected' : '') + '>' + L[k] + '</option>'; }).join('');
      var wOpts = DATA.weights.map(function(c) { return '<option value="' + c + '"' + (c === weight ? ' selected' : '') + '>' + (c === 'population' ? 'Population' : c === 'population+jobs' ? 'Population + jobs' : c === 'unweighted' ? 'Unweighted (every cell equal)' : esc(c)) + '</option>'; }).join('');
      h += '<div style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;align-items:flex-end;">' +
        '<label style="flex:2;font:11px sans-serif;">Rank by level<select id="rkLevel" style="width:100%;">' + lvlOpts + '</select></label>' +
        '<label style="flex:1.5;font:11px sans-serif;">Weight<select id="rkWeight" style="width:100%;">' + wOpts + '</select></label>' +
        '<label style="flex:1;font:11px sans-serif;">Method<select id="rkMethod" style="width:100%;"><option value="mean"' + (method === 'mean' ? ' selected' : '') + '>Mean</option><option value="median"' + (method === 'median' ? ' selected' : '') + '>Median</option></select></label></div>' +
        '<label style="display:block;font:11px sans-serif;margin-bottom:6px;">Rank by<select id="rkBy" style="width:100%;">' + byOpts + '</select></label>' +
        '<div id="rkPopRange" style="margin-bottom:6px;"></div>';
      var rows = (DATA.levels[level] || []).slice();
      var colArr = colI >= 0 ? colValues(level, colI) : null;
      if (colI >= 0) rows.forEach(function(r, i) { r.__colv = colArr ? colArr[i] : null; });
      var inScope = rows;
      var selState = null;
      try { selState = (window.__scopeStateKey && window.__scopeStateKey()) || null; } catch (e) {}
      if (s.lv !== 'nation' && s.lv !== 'none' && selState) inScope = rows.filter(function(r) { return (r.states || [r.state]).indexOf(selState) >= 0; });
      var mt = s.lv === 'metro' && window.__scopeMetro ? window.__scopeMetro() : null;
      if (mt) {   // a metropolitan area: its municipalities / counties and everything inside them
        var mem = mt.members || [];
        inScope = rows.filter(function(r) {
          var id = r.id != null ? String(r.id) : '';
          return id && (mem.indexOf(id) >= 0 || (id.length > 5 && mem.indexOf(id.slice(0, 5)) >= 0) || ('metro_' + id) === mt.key);
        });
      }
      var cid = window.__scopeCountyId && window.__scopeCountyId(), pid = window.__scopePlaceId && window.__scopePlaceId();
      if (s.lv === 'county' && cid) inScope = inScope.filter(function(r) { return r.prov ? r.prov === cid : (r.id || '').indexOf(cid) === 0 || r.id === cid; });
      if (s.lv === 'place' && pid) inScope = inScope.filter(function(r) { return r.muni ? r.muni === pid : r.id === pid; });
      var isT = rankBy === 'transit', isC = colI >= 0;
      var key = isT ? 'transit_share' : isC ? '__colv' : method + '_' + weight, wNote = '';
      if (!isT && !isC && !rows.some(function(r) { return r[key] != null; }) && rows.some(function(r) { return r[method + '_population'] != null; })) {
        key = method + '_population';
      }
      var pct = isC ? COLS.pct[colI] : 0, cRel = isC ? COLS.relative[colI] : true;
      var fmtCol = function(v) {
        if (pct === 100) return v.toFixed(1) + '%';
        if (pct === 1) return (100 * v).toFixed(1) + '%';
        var a = Math.abs(v);
        return a >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : a >= 1e4 ? (v / 1e3).toFixed(0) + 'k' : a >= 100 || !cRel ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
      };
      var sc = function(r) { return isT ? (100 * r[key]).toFixed(1) + '%' : isC ? fmtCol(r[key]) : r[key].toFixed(1); };
      var tsCol = function(r) { return isT ? '' : '<span style="flex:none;min-width:40px;text-align:right;color:#2b6cb0;" title="Transit commute share">' + (r.transit_share != null ? (100 * r.transit_share).toFixed(1) + '%' : '\u2013') + '</span>'; };
      var byScore = function(a, b) { return (b[key] - a[key]) || String(a.name).localeCompare(String(b.name)); };
      var rankIt = function(list, f) { var prev = null, pr = 0; list.forEach(function(r, i) { var v = sc(r); if (v !== prev) { pr = i + 1; prev = v; } r[f] = pr; }); };
      rankIt(rows.filter(function(r) { return r[key] != null; }).sort(byScore), '__rk0');
      rankIt(rows.filter(function(r) { return r[key] != null && (r.w_population || 0) >= minPop && (maxPop == null || (r.w_population || 0) <= maxPop); }).sort(byScore), '__rk');
      inScope = inScope.filter(function(r) { return r[key] != null && (r.w_population || 0) >= minPop && (maxPop == null || (r.w_population || 0) <= maxPop); }).sort(byScore);
      rankIt(inScope, '__rkf');   // position among the rows shown (all active filters: scope, population range); __rk0 stays the rank without any filter
      var nAll = rows.filter(function(r) { return r[key] != null; }).length;
      var RH = 30, hl = -1;
      h += '<div style="position:relative;display:flex;gap:4px;align-items:center;margin-bottom:6px;"><input id="rkSearch" type="search" autocomplete="off" placeholder="Search..." style="flex:1;min-width:0;box-sizing:border-box;padding:4px 6px;font:12px sans-serif;">' +
        '<button id="rkTop" title="Top" style="width:22px;height:22px;flex:none;padding:0;font:10px sans-serif;line-height:1;border:1px solid #ccc;background:#f5f6f8;color:#666;border-radius:3px;cursor:pointer;">&#9650;</button><button id="rkBot" title="Bottom" style="width:22px;height:22px;flex:none;padding:0;font:10px sans-serif;line-height:1;border:1px solid #ccc;background:#f5f6f8;color:#666;border-radius:3px;cursor:pointer;">&#9660;</button>' +
        '<div id="rkSug" style="display:none;position:absolute;left:0;right:50px;top:100%;z-index:5;background:#fff;border:1px solid #bbb;max-height:200px;overflow-y:auto;font:12px sans-serif;"></div></div>';
      h += '<div id="rkList" style="position:relative;overflow-y:auto;flex:1 1 auto;min-height:160px;border:1px solid #e3e6ea;"><div id="rkSpacer" style="position:relative;height:' + (inScope.length * RH) + 'px;"></div></div>';
      var max = inScope.length ? Math.max(Math.abs(inScope[0][key]), 1e-9) : 1;
      var lbl = function(r) { return r.name + (level !== 'state' && level !== 'cbsa' && level !== 'metro' && r.state ? ', ' + (SN[r.state] || r.state) : ''); };
      var stLbl = function(r) { return (level !== 'state' && level !== 'cbsa' && level !== 'metro' && r.state) ? (SN[r.state] || r.state) : ''; };
      el.innerHTML = h;
      var list = document.getElementById('rkList'), spacer = document.getElementById('rkSpacer'), sInp = document.getElementById('rkSearch'), sug = document.getElementById('rkSug');
      function draw() {
        var first = Math.max(0, Math.floor(list.scrollTop / RH) - 4), last = Math.min(inScope.length, first + Math.ceil(list.clientHeight / RH) + 9), o = '';
        for (var i = first; i < last; i++) {
          var r = inScope[i];
          o += '<div title="' + esc(lbl(r)) + '" style="position:absolute;left:0;right:0;top:' + (i * RH) + 'px;height:' + RH + 'px;box-sizing:border-box;padding:2px 6px;' + (i === hl ? 'background:#fff3c4;' : '') + '"><div style="display:flex;gap:6px;align-items:baseline;font:11px sans-serif;color:#444;white-space:nowrap;"><span style="display:flex;flex:1 1 auto;min-width:0;"><span style="flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"><b title="Rank with the active filters">' + r.__rkf + '.</b> ' + esc(r.name) + '</span>' + (stLbl(r) ? '<span style="flex:none;color:#666;white-space:pre;">, ' + esc(stLbl(r)) + '</span>' : '') + '</span><b style="flex:none;">(' + sc(r) + ')</b>' + tsCol(r) + '<span style="flex:none;min-width:30px;text-align:right;color:#888;" title="Rank without filters (of ' + nAll + ')">#' + r.__rk0 + '</span></div>' +
            '<div style="height:8px;background:#eef1f4;border-radius:3px;overflow:hidden;"><div style="width:' + (100 * r[key] / max) + '%;height:100%;background:#3a6ea8;"></div></div></div>';
        }
        spacer.innerHTML = o;
      }
      function goTo(i) { hl = i; list.scrollTop = Math.max(0, i * RH - list.clientHeight / 2 + RH / 2); draw(); }
      list.addEventListener('scroll', draw); draw();
      if (window.ResizeObserver) { if (window.__rkRO) window.__rkRO.disconnect(); window.__rkRO = new ResizeObserver(function() { draw(); }); window.__rkRO.observe(list); }
      document.getElementById('rkTop').addEventListener('click', function() { hl = -1; list.scrollTop = 0; draw(); });
      document.getElementById('rkBot').addEventListener('click', function() { hl = -1; list.scrollTop = list.scrollHeight; draw(); });
      window.__rkGoCenter = function() {
        if (window.__statsTab !== 'ranking' || !window.__rkMap || document.activeElement === sInp) return;
        var m = window.__rkMap; if (m.getZoom() < 10) return;
        var ids = [];
        try { window.__centreFeatures(m).forEach(function(f) { var pr = f.properties || {}; for (var k in pr) { var v = pr[k]; if ((typeof v === 'string' || typeof v === 'number') && /^[0-9A-Za-z]{2,14}$/.test(String(v))) ids.push(String(v)); } }); } catch (e) {}
        var cid = window.__scopeCountyId && window.__scopeCountyId(), pid = window.__scopePlaceId && window.__scopePlaceId(), sk = window.__scopeStateKey && window.__scopeStateKey();
        var best = -1, bl = -1;
        inScope.forEach(function(r, i) {
          var rid = r.id != null ? String(r.id) : null;
          if (rid && ids.some(function(v) { return v === rid || (v.length > rid.length && v.indexOf(rid) === 0 && rid.length >= 2); }) && rid.length > bl) { best = i; bl = rid.length; }
        });
        if (best < 0) { var want = level === 'county' ? cid : level === 'place' ? pid : level === 'state' ? sk : null; if (want) best = inScope.findIndex(function(r) { return r.id === want || r.state === want; }); }
        if (best >= 0 && best !== hl) goTo(best);
      };
      function suggest() {
        var fold = function(t) { return String(t).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase(); }; var q = fold(sInp.value.trim());
        if (!q) { sug.style.display = 'none'; return; }
        var hits = [];
        for (var i = 0; i < inScope.length && hits.length < 40; i++) { var t = fold(lbl(inScope[i])); var p = t.indexOf(q); if (p >= 0) hits.push([p, i]); }
        hits.sort(function(a, b) { return a[0] - b[0] || a[1] - b[1]; });
        sug.innerHTML = hits.slice(0, 10).map(function(h2) { return '<div data-i="' + h2[1] + '" style="padding:3px 6px;cursor:pointer;border-bottom:1px solid #f0f0f0;">' + inScope[h2[1]].__rk + '. ' + esc(lbl(inScope[h2[1]])) + '</div>'; }).join('') || '<div style="padding:3px 6px;color:#888;">No match</div>';
        sug.style.display = 'block';
      }
      sInp.addEventListener('input', suggest);
      sInp.addEventListener('keydown', function(e) { if (e.key === 'Enter') { var f = sug.querySelector('[data-i]'); if (f) { goTo(+f.getAttribute('data-i')); sug.style.display = 'none'; } } });
      sug.addEventListener('mousedown', function(e) { var t = e.target.closest('[data-i]'); if (t) { e.preventDefault(); goTo(+t.getAttribute('data-i')); sug.style.display = 'none'; sInp.blur(); } });
      if (window.__rkGoCenter) setTimeout(window.__rkGoCenter, 0);
      var prHost = document.getElementById('rkPopRange');
      if (prHost && window.__popRange) {
        var pmax = 1; rows.forEach(function(r) { if ((r.w_population || 0) > pmax) pmax = r.w_population; });
        window.__popRange(prHost, { max: pmax, lo: minPop, hi: maxPop, onChange: function(a, b) { minPop = a; maxPop = b; render(); } });
      }
      ['rkLevel', 'rkWeight', 'rkMethod', 'rkBy'].forEach(function(id) {
        document.getElementById(id).addEventListener('change', function(e) {
          if (id === 'rkLevel') level = e.target.value; else if (id === 'rkWeight') weight = e.target.value; else if (id === 'rkBy') rankBy = e.target.value; else method = e.target.value;
          render();
        });
      });
    });
  }
  var n = 0, t = setInterval(function() { if (addTab() || ++n > 60) clearInterval(t); }, 500);
  var n2 = 0, t2 = setInterval(function() {
    var m = [window.__map, window.map, window.__mapRef && window.__mapRef()].filter(function(c) { return c && c.on && c.getCenter; })[0];   // `window.map` can be the #map element
    if (m && m.on && m.getCenter) { clearInterval(t2); window.__rkMap = m; var tm = null; m.on('moveend', function() { clearTimeout(tm); tm = setTimeout(function() { if (window.__rkGoCenter) window.__rkGoCenter(); }, 350); }); }
    else if (++n2 > 120) clearInterval(t2);
  }, 500);
})();
"""


def ranking_script(chunk_dir: str, rel_dir: str, country: str, rebuild: bool = False) -> str:
    """The ranking tab of a nation-wide map, (re)building its tables when they are missing or older than the data.

    Args:
        chunk_dir: Stats chunk folder.
        rel_dir: Path of that folder relative to the page.
        country: Country code of the map.
        rebuild: Rebuild every table.

    Returns:
        ``<script>`` block.
    """
    table = os.path.join(chunk_dir, "ranking.json")
    if rebuild or not os.path.isfile(table):
        build_ranking_table(chunk_dir)
    elif not os.path.isfile(os.path.join(chunk_dir, "ranking_top.json")) or os.path.getmtime(os.path.join(chunk_dir, "ranking_top.json")) < os.path.getmtime(table):
        write_ranking_top(chunk_dir)
    if os.path.isdir(os.path.join(chunk_dir, RANK_CELLS)) and (rebuild or not os.path.isfile(os.path.join(chunk_dir, RANK_COLS, "columns.json"))
                                                                or os.path.getmtime(os.path.join(chunk_dir, RANK_COLS, "columns.json")) < os.path.getmtime(table)):
        try:
            print(f"[ranking] {build_rank_columns(chunk_dir)} rank-by column files")
        except Exception as exc:  # noqa: BLE001
            print(f"[ranking] rank-by columns skipped: {exc!r}")
    cfg = {"dir": rel_dir, "labels": LABELS[country], "unavailable": UNAVAILABLE[country], "rows": 14, "country": country}
    return "<script>\n" + RANKING_JS.replace("__RCFG__", json.dumps(cfg)) + "\n</script>\n"
