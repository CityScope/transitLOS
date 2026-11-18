"""Res-9 data of the click / address popup (user 2026-11-16: "the stats popups stats resolution should be h3 res 9").

The pages' stats data stay at res 8 (a res-9 copy would be ~7 times larger: tens to hundreds of MB), so the popup reads its own small files,
fetched only for the point clicked:

    popup9/index.json                     {"res": 9, "chunk_res": 5, "cols": [...], "scopes": [kind, ...]}
    popup9/c/<res-5 parent>.json.gz       {"h3": [...], "v": [[value per col], ...], "<kind>": [scope id per cell], ...}
    popup9/q/<kind>_<id>.json             {col: quantiles}   (101 points for the wider scopes, 21 for the many small ones)

`write_popup_cells(cells, out_dir, scopes)` writes them from one frame of res-9 cells (column `h3`, the numeric columns of the map and one
id column per scope kind). The browser side is `POPUP9_JS` (window.__popup9): row lookup, nearest populated neighbour, percentile tables.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

POPUP_RES = 9
CHUNK_RES = 5
DIRNAME = "popup9"
_SKIP = {"h3", "h3_cell", "geometry", "_i", "GEOID", "geoid"}


def _num(v: float):
    if v is None or not np.isfinite(v):
        return None
    if v == 0:
        return 0
    return float(f"{v:.5g}")


def _safe(s) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def _quantiles(a: np.ndarray, n: int) -> Optional[List]:
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return None
    return [_num(x) for x in np.percentile(a, np.linspace(0, 100, n))]


def write_popup_cells(cells, out_dir: str, scopes: Dict[str, Iterable], wide: Iterable[str] = (), cols: Optional[List[str]] = None,
                      chunk_res: int = CHUNK_RES, extra_quantiles: Optional[Dict[str, np.ndarray]] = None) -> str:
    """`cells`: pandas frame of populated res-9 cells (`h3` + numeric columns). `scopes`: {kind: array of scope ids, one per cell (None = none)}.
    `wide`: the scope kinds with few, large members (state, nation, metro ...) -> 101-point tables; others get 21 points.
    `extra_quantiles`: {name: boolean mask} for whole-frame tables such as the nation (written as q/<name>.json).

    Args:
        cells: Populated res-9 cells.
        out_dir: Output folder.
        scopes: ``{kind: scope id per cell}``.
        wide: Scope kinds with 101-point tables.
        cols: Columns; ``None`` = every numeric one.
        chunk_res: H3 resolution of the cell chunks.
        extra_quantiles: ``{name: boolean mask}`` whole-frame tables.

    Returns:
        The ``popup9`` folder.
    """
    import h3
    import pandas as pd

    df = cells.reset_index(drop=True)
    cell_col = "h3" if "h3" in df.columns else "h3_cell"
    if cols is None:
        cols = [c for c in df.columns if c not in _SKIP and c not in scopes and not str(c).startswith("u_") and pd.api.types.is_numeric_dtype(df[c])]
    base = os.path.join(out_dir, DIRNAME)
    os.makedirs(os.path.join(base, "c"), exist_ok=True)
    os.makedirs(os.path.join(base, "q"), exist_ok=True)
    V = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64") for c in cols]) if cols else np.zeros((len(df), 0))
    ids = df[cell_col].astype(str).to_numpy()
    parents = np.array([h3.cell_to_parent(c, chunk_res) for c in ids])
    sc = {k: np.asarray(list(v), dtype=object) for k, v in scopes.items()}
    order = np.argsort(parents, kind="stable")
    bounds = np.flatnonzero(np.r_[True, parents[order][1:] != parents[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        out = {"h3": ids[idx].tolist(), "v": [[_num(x) for x in row] for row in V[idx]]}
        for k, arr in sc.items():
            out[k] = [None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else str(x) for x in arr[idx]]
        with gzip.open(os.path.join(base, "c", parents[order][a] + ".json.gz"), "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(out, f, separators=(",", ":"))
    wide = set(wide)
    for k, arr in sc.items():
        n = 101 if k in wide else 21
        keys = pd.Series(arr)
        for sid, idx in keys.groupby(keys).groups.items():
            if sid is None:
                continue
            tab = {c: q for j, c in enumerate(cols) if (q := _quantiles(V[np.asarray(idx), j], n)) is not None}
            with open(os.path.join(base, "q", f"{k}_{_safe(sid)}.json"), "w", encoding="utf-8") as f:
                json.dump(tab, f, separators=(",", ":"))
    for name, mask in (extra_quantiles or {}).items():
        m = np.asarray(mask, dtype=bool)
        tab = {c: q for j, c in enumerate(cols) if (q := _quantiles(V[m, j], 101)) is not None}
        with open(os.path.join(base, "q", f"{_safe(name)}.json"), "w", encoding="utf-8") as f:
            json.dump(tab, f, separators=(",", ":"))
    index = {"res": POPUP_RES, "chunk_res": chunk_res, "cols": list(cols), "scopes": list(sc)}
    with open(os.path.join(base, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    return base


def world_quantiles(frames: Iterable, out_path: str, cols: Optional[List[str]] = None) -> str:
    """101-point table over the res-9 cells of several frames (all the cities of the combined map) -> `out_path`.

    Args:
        frames: Cell tables of the cities.
        out_path: Output JSON.
        cols: Columns; ``None`` = every numeric one.

    Returns:
        ``out_path``.
    """
    import pandas as pd

    parts: Dict[str, list] = {}
    for df in frames:
        for c in df.columns:
            if c in _SKIP or (cols is not None and c not in cols) or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            parts.setdefault(c, []).append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64"))
    tab = {c: q for c, v in parts.items() if (q := _quantiles(np.concatenate(v), 101)) is not None}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tab, f, separators=(",", ":"))
    return out_path


def write_popup_source(cells, key: str, out_dir: str, county=None) -> str:
    """Per state / community, from the stats-chunk step: `<out_dir>/popup_src/<key>.parquet` = the populated res-9 cells (`h3` + the map's
    numeric columns as float32) with `u_county` (the county / province GeoDataFrame `county`, GEOID by cell centre) and `u_state` = `key`.

    Args:
        cells: Populated res-9 cells.
        key: State / community key.
        out_dir: Stats chunk folder.
        county: County / province polygons (GEOID).

    Returns:
        Path of the parquet.
    """
    import geopandas as gpd
    import h3
    import pandas as pd

    df = cells.reset_index(drop=True)
    cell_col = "h3" if "h3" in df.columns else "h3_cell"
    out = {"h3": df[cell_col].astype(str).to_numpy()}
    for c in df.columns:
        if c not in _SKIP and not str(c).startswith("u_") and pd.api.types.is_numeric_dtype(df[c]):
            out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    out["u_state"] = np.full(len(df), key, dtype=object)
    if county is not None and len(county):
        ll = np.asarray([h3.cell_to_latlng(c) for c in out["h3"]], dtype=float).reshape(-1, 2)
        pts = gpd.GeoDataFrame({"_i": np.arange(len(df))}, geometry=gpd.points_from_xy(ll[:, 1], ll[:, 0]), crs=4326)
        j = gpd.sjoin(pts, county[["GEOID", "geometry"]].to_crs(4326), how="left", predicate="within").drop_duplicates("_i").sort_values("_i")
        out["u_county"] = j["GEOID"].astype(object).where(j["GEOID"].notna(), None).to_numpy()
    path = os.path.join(out_dir, "popup_src", key + ".parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_parquet(path)
    return path


def build_popup_from_sources(chunk_dir: str) -> str:
    """`popup9/` of a nation-wide map from every `popup_src/<state>.parquet`: county and state tables, and the nation's (q/nation.json).

    Args:
        chunk_dir: Stats chunk folder with ``popup_src/``.

    Returns:
        The ``popup9`` folder.
    """
    import pandas as pd

    src = os.path.join(chunk_dir, "popup_src")
    files = sorted(f for f in os.listdir(src) if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(os.path.join(src, f)) for f in files], ignore_index=True)
    import shutil

    shutil.rmtree(os.path.join(chunk_dir, DIRNAME), ignore_errors=True)
    scopes = {k: df.pop(c).to_numpy(dtype=object) for k, c in (("county", "u_county"), ("state", "u_state")) if c in df.columns}
    return write_popup_cells(df, chunk_dir, scopes, wide=("state",), extra_quantiles={"nation": np.ones(len(df), dtype=bool)})


# Browser side: window.__popup9(base) -> {row(lng, lat) -> Promise<null | {cell, row, ids}>, q(name) -> Promise<table | null>}
POPUP9_JS = r"""
(function() {
  if (window.__popup9) return;
  var inst = {};
  function getJSON(url, gz) {
    return fetch(url).then(function(r) { if (!r.ok) return null; return gz ? r.arrayBuffer() : r.json(); }).then(function(x) {
      if (!gz || x == null) return x;
      var u8 = new Uint8Array(x);
      if (u8[0] === 0x1f && u8[1] === 0x8b && typeof DecompressionStream !== 'undefined')
        return new Response(new Blob([x]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
      return JSON.parse(new TextDecoder().decode(u8));
    }).catch(function() { return null; });
  }
  window.__popup9 = function(base) {
    if (inst[base]) return inst[base];
    var idx = null, idxP = null, chunks = {}, order = [], qs = {};
    function index() { if (!idxP) idxP = getJSON(base + 'index.json').then(function(j) { idx = j; return j; }); return idxP; }
    function chunk(p) {
      if (!chunks[p]) {
        chunks[p] = getJSON(base + 'c/' + p + '.json.gz', true).then(function(d) {
          if (d) { d.__ix = {}; d.h3.forEach(function(c, i) { d.__ix[c] = i; }); }
          return d;
        });
        order.push(p);
        if (order.length > 40) delete chunks[order.shift()];
      }
      return chunks[p];
    }
    function rowOf(d, i) {
      var r = {}; idx.cols.forEach(function(c, j) { r[c] = d.v[i][j]; });
      var ids = {}; (idx.scopes || []).forEach(function(k) { if (d[k]) ids[k] = d[k][i]; });
      return {row: r, ids: ids};
    }
    var api = {
      row: function(lng, lat) {
        return index().then(function(ix) {
          if (!ix || typeof h3 === 'undefined') return null;
          var cell = h3.latLngToCell(lat, lng, ix.res);
          var near = h3.gridDisk(cell, 2), parents = {};
          near.forEach(function(c) { parents[h3.cellToParent(c, ix.chunk_res)] = 1; });
          var ps = Object.keys(parents);
          return Promise.all(ps.map(chunk)).then(function(ds) {
            var by = {}; ps.forEach(function(p, i) { by[p] = ds[i]; });
            var own = by[h3.cellToParent(cell, ix.chunk_res)];
            if (own && own.__ix[cell] !== undefined) { var o = rowOf(own, own.__ix[cell]); o.cell = cell; return o; }
            // a click on an unpopulated cell: the nearest populated cell within two rings
            var best = null, bd = 1e18, kc = Math.cos(lat * Math.PI / 180);
            near.forEach(function(c) {
              var d = by[h3.cellToParent(c, ix.chunk_res)]; if (!d || d.__ix[c] === undefined) return;
              var ll = h3.cellToLatLng(c), dd = (ll[0] - lat) * (ll[0] - lat) + (ll[1] - lng) * (ll[1] - lng) * kc * kc;
              if (dd < bd) { bd = dd; best = [c, d]; }
            });
            if (!best) return null;
            var o2 = rowOf(best[1], best[1].__ix[best[0]]); o2.cell = best[0]; return o2;
          });
        });
      },
      q: function(name) {
        if (!(name in qs)) qs[name] = getJSON(base + 'q/' + String(name).replace(/[^A-Za-z0-9_.-]/g, '_') + '.json');
        return qs[name];
      }
    };
    inst[base] = api;
    return api;
  };
})();
"""
