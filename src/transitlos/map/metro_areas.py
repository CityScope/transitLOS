"""Metropolitan areas on the nation-wide maps: stats chunks (one per area), the stats index entries and the dotted boundary layer.

`build_metro_chunks` cuts a `<key>.json.gz` per metropolitan area out of the per-state stats chunks (the cells of its member municipalities /
counties, same column layout as a state chunk without the `county` / `place` indexes) and appends `index["metros"]`
(`{key, name, bbox, pop, n, members}`) to `stats_chunks/index.json`; the scope bar then offers each area as a level of its own.
`metro_lines_geojson` turns the area polygons into boundary lines and `metro_lines_script` is the map layer: a thick black dotted line with a white
halo (readable on any fill), visible from `MIN_ZOOM`, with its own checkbox in the layer panel.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, Iterable, List

MIN_ZOOM = 6.5


def _cut(d: dict, rows: List[int]) -> dict:
    return {c: [v[i] for i in rows] for c, v in d.items() if c not in ("county", "place") and isinstance(v, list)}


def build_metro_chunks(chunk_dir: str, areas: Dict[str, str], members: Dict[str, str], member_level: str = "place") -> List[dict]:
    """Write `metro_<code>.json.gz` for every area and extend `index.json`; `members` is {municipality / county id: area code}.

    `member_level` is the stats-chunk index the ids belong to: "place" (Spain municipalities) or "county" (US counties).

    Args:
        chunk_dir: Stats chunk folder (with ``index.json``).
        areas: ``{area code: name}``.
        members: ``{member id: area code}``.
        member_level: ``"place"`` or ``"county"``.

    Returns:
        The area entries added to the index.
    """
    ip = os.path.join(chunk_dir, "index.json")
    index = json.load(open(ip, encoding="utf-8"))
    plural = "places" if member_level == "place" else "counties"
    rows_by_area: Dict[str, Dict[str, List[int]]] = {}
    keys_of_area: Dict[str, List[str]] = {}
    for st in index["states"]:
        ids = [e["id"] for e in st.get(plural, [])]
        code_of = [members.get(i) for i in ids]
        if not any(code_of):
            continue
        d = json.load(gzip.open(os.path.join(chunk_dir, st["key"] + ".json.gz")))
        col = d.get(member_level)
        if col is None:
            continue
        take: Dict[str, List[int]] = {}
        for r, idx in enumerate(col):
            if idx is not None and idx >= 0 and code_of[idx]:
                take.setdefault(code_of[idx], []).append(r)
        for code, rows in take.items():
            part = _cut(d, rows)
            acc = rows_by_area.setdefault(code, {})
            for c, v in part.items():
                acc.setdefault(c, []).extend(v)
        del d
    metros = []
    for code, cols in rows_by_area.items():
        n = len(next(iter(cols.values())))
        if n == 0:
            continue
        key = "metro_" + code
        with gzip.open(os.path.join(chunk_dir, key + ".json.gz"), "wt", encoding="utf-8") as f:
            json.dump(cols, f, separators=(",", ":"))
        ids = sorted(i for i, c in members.items() if c == code)
        pop = float(sum(v for v in cols.get("population", []) if v))
        metros.append({"key": key, "name": areas[code], "pop": round(pop), "n": n, "members": ids})
    metros.sort(key=lambda m: -m["pop"])
    index["metros"] = metros
    json.dump(index, open(ip, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    return metros


def metro_lines_geojson(polygons, name_col: str = "name", tolerance_deg: float = 0.0015) -> dict:
    """Boundary lines (outer and inner rings) of the area polygons (a GeoDataFrame in EPSG:4326), lightly simplified.

    Args:
        polygons: Area polygons (EPSG:4326).
        name_col: Name column.
        tolerance_deg: Simplification tolerance (degrees).
    """
    feats = []
    for _, r in polygons.iterrows():
        g = r.geometry.simplify(tolerance_deg, preserve_topology=True).boundary
        if g.is_empty:
            continue
        feats.append({"type": "Feature", "properties": {"name": str(r[name_col])}, "geometry": json.loads(__import__("shapely").to_geojson(__import__("shapely").set_precision(g, 1e-4)))})
    return {"type": "FeatureCollection", "features": feats}


METRO_JS = r"""
(function() {
  var URL_ = __URL__, LABEL = __LABEL__, MINZ = __MINZ__, P = __PREFIX__, W = __WIDTH__;
  function setup(map) {
    if (map.getSource('__' + P + '_lines')) return;
    map.addSource('__' + P + '_lines', {type: 'geojson', data: URL_ + '?v=' + ((document.querySelector('meta[name="build-id"]') || {}).content || '')});   // loaded by MapLibre itself, not through fetch: versioned like the other data
    var vis = (document.getElementById('ov' + P + 'Checkbox') || {checked: true}).checked ? 'visible' : 'none';
    var before = map.getLayer('__stops_cluster') ? '__stops_cluster' : undefined;
    var w = ['interpolate', ['linear'], ['zoom'], 6, 1.4 * W, 9, 2.6 * W, 12, 3.8 * W, 16, 5.5 * W];
    var wh = ['interpolate', ['linear'], ['zoom'], 6, 2.2 + 1.4 * W, 9, 2.2 + 2.6 * W, 12, 2.2 + 3.8 * W, 16, 2.2 + 5.5 * W];   // dots + 2.2 px of halo
    map.addLayer({id: '__' + P + '_halo', type: 'line', source: '__' + P + '_lines', minzoom: MINZ, layout: {'line-cap': 'round', 'line-join': 'round', visibility: vis},
      paint: {'line-color': '#ffffff', 'line-opacity': 0.8, 'line-width': wh}}, before);
    map.addLayer({id: '__' + P + '_dots', type: 'line', source: '__' + P + '_lines', minzoom: MINZ, layout: {'line-cap': 'round', 'line-join': 'round', visibility: vis},
      paint: {'line-color': '#000000', 'line-width': w, 'line-dasharray': [0.01, 1.9]}}, before);
  }
  function box() {
    if (document.getElementById('ov' + P + 'Checkbox')) return true;
    var host = document.getElementById('overlay-controls') || document.getElementById('layer-switcher-body');
    if (!host) return false;
    var row = document.createElement('div'); row.id = P + 'LinesRow';
    row.innerHTML = '<label><input type="checkbox" id="ov' + P + 'Checkbox" checked> ' + LABEL + '</label>';
    host.appendChild(row);
    row.querySelector('input').addEventListener('change', function(e) {
      var map = window.__map; if (!map) return;
      ['__' + P + '_halo', '__' + P + '_dots'].forEach(function(id) { if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', e.target.checked ? 'visible' : 'none'); });
    });
    return true;
  }
  var n = 0, t = setInterval(function() {
    var map = window.__map;
    if (map && map.getStyle && map.isStyleLoaded && map.isStyleLoaded()) { try { setup(map); } catch (e) {} box(); if (map.getLayer('__' + P + '_dots') && document.getElementById('ov' + P + 'Checkbox')) { clearInterval(t); map.on('styledata', function() { try { setup(map); } catch (e) {} }); } }
    if (++n > 240) clearInterval(t);
  }, 500);
})();
"""


def lines_script(geojson_url: str, label: str, prefix: str, min_zoom: float, width: float = 1.0) -> str:
    """Dotted black boundary layer (white halo) with its own checkbox; `prefix` keeps several such layers apart (metro, core).

    Args:
        geojson_url: URL of the lines GeoJSON.
        label: Checkbox label.
        prefix: Id prefix of the layer.
        min_zoom: First zoom shown.
        width: Line width.
    """
    return "<script>\n" + (METRO_JS.replace("__URL__", json.dumps(geojson_url)).replace("__LABEL__", json.dumps(label)).replace("__MINZ__", str(min_zoom))
                            .replace("__PREFIX__", json.dumps(prefix)).replace("__WIDTH__", str(width))) + "\n</script>\n"


def metro_lines_script(geojson_url: str, label: str = "Metro areas") -> str:
    """The metro-area boundary layer (see :func:`lines_script`).

    Args:
        geojson_url: URL of the lines GeoJSON.
        label: Checkbox label.

    Returns:
        ``<script>`` block.
    """
    return lines_script(geojson_url, label, "metro", MIN_ZOOM)
