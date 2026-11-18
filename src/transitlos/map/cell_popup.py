"""The click / address-search popup of a MapLibre page: always the H3 stats-resolution (H3 res 8) cell under the point.

One script (`CELL_POPUP_JS`) serves the city maps and the US map. It owns the popup rendering (header with level of
service + population, a collapsible table of every statistic of the cell, and for each statistic its percentile among
all the cells of a few scopes) and delegates data access to a page-provided provider:

    window.__cellPopupProvider(lng, lat) -> Promise<null | {cell, row, scopes}>
        row    {column: value} of the res-8 cell,
        scopes [{name, pct(column, value) -> 0..100 | null}]   (e.g. County, State, Nation  /  Urban area, City core, All places)

`window.__openCellPopup(lng, lat, {expanded, title})` opens it (the geohierarchy click handler calls it before falling
back to its own per-shape popup, the address search calls it with `expanded: true` so all the stats show at once).
`percentile_helpers_js()` is the shared client-side percentile machinery (exact over an array, or interpolated in a
101-point quantile table).
"""
from __future__ import annotations

import json
from typing import Dict

H3_JS_CDN = "https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"

# Absolute column -> the relative column built from it (shown in parentheses next to the count).
SKIP_KEYS = ("h3_cell", "geometry", "county", "place", "GEOID", "geoid", "geo_id", "_i")

CELL_POPUP_JS = r"""
(function() {
  if (window.__openCellPopup) return;
  var ABS_TO_REL = __ABS_TO_REL__;
  var SKIP = {h3_cell: 1, geometry: 1, county: 1, place: 1, GEOID: 1, geoid: 1, geo_id: 1, _i: 1, level_of_service: 1, population: 1};
  var popup = null, sortedCache = (typeof WeakMap !== 'undefined') ? new WeakMap() : null;

  // ---- percentiles ---------------------------------------------------------------------------------------------------
  // Percentile rank of v among the finite values of an array (ties count half), 0..100.
  function sortedOf(arr) {
    var s = sortedCache && sortedCache.get(arr);
    if (s) return s;
    s = Float64Array.from(arr.filter(function(x) { return x != null && isFinite(x); })).sort();
    if (sortedCache) sortedCache.set(arr, s);
    return s;
  }
  function lowerBound(s, v) { var lo = 0, hi = s.length; while (lo < hi) { var m = (lo + hi) >> 1; if (s[m] < v) lo = m + 1; else hi = m; } return lo; }
  function upperBound(s, v) { var lo = 0, hi = s.length; while (lo < hi) { var m = (lo + hi) >> 1; if (s[m] <= v) lo = m + 1; else hi = m; } return lo; }
  function pctInArray(arr, v) {
    if (!arr || v == null || !isFinite(v)) return null;
    var s = sortedOf(arr);
    if (s.length < 2) return null;
    return 100 * (lowerBound(s, v) + 0.5 * (upperBound(s, v) - lowerBound(s, v))) / s.length;
  }
  // Same, from an n-point quantile table (q[0] = min ... q[n-1] = max; 101 or 21 points).
  function pctInQuantiles(q, v) {
    var p = pctInQ(q, v);
    return (p == null || !q) ? p : p * 100 / (q.length - 1);
  }
  function pctInQ(q, v) {
    if (!q || q.length < 2 || v == null || !isFinite(v)) return null;
    if (v <= q[0]) return v < q[0] ? 0 : 0;
    if (v >= q[q.length - 1]) return q.length - 1;
    var lo = 0, hi = q.length - 1;
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (q[m] <= v) lo = m; else hi = m; }
    var span = q[hi] - q[lo];
    var firstEq = lo; while (firstEq > 0 && q[firstEq - 1] === q[lo]) firstEq--;
    var lastEq = hi; while (lastEq < q.length - 1 && q[lastEq + 1] === q[hi]) lastEq++;
    if (span === 0) return (firstEq + lastEq) / 2;
    return lo + (v - q[lo]) / span;
  }
  window.__pctInArray = pctInArray; window.__pctInQuantiles = pctInQuantiles;

  // ---- formatting ----------------------------------------------------------------------------------------------------
  function esc(t) { return String(t).replace(/[&<>"]/g, function(c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  function fmt(v, key) {
    if (v == null || (typeof v === 'number' && !isFinite(v))) return '&ndash;';
    if (typeof window.__shapePopupNum === 'function') return window.__shapePopupNum(v, key);
    return Number(v).toLocaleString('en-US', {maximumFractionDigits: 2});
  }
  function labelsFor(keys) {
    if (typeof window.__fieldLabelsFor === 'function') return window.__fieldLabelsFor(keys);
    var o = {}; keys.forEach(function(k) { o[k] = k.replace(/_/g, ' '); }); return o;
  }
  var GROUPS = [
    ['Access & population', /density|^pop_|residents|area_m2/i],
    ['Jobs & economy', /job|worker|labor|unemploy|income|poverty|rent|business|employ|commut|vehicle|building|wage/i],
    ['Households & housing', /household|housing|owner|renter|tenure/i],
    ['Education', /educat|school|degree|bachelor/i],
    ['Demographics', /male|female|age|under18|over65|adult|white|black|asian|native|hispanic|race|born|urban|children|elderly|population/i]
  ];
  // Commuting is matched before the generic jobs pattern so "Car commuters" lands in its own group.
  GROUPS.splice(1, 0, ['Commuting', /commut|transit|walk|bike|car_|drive|travel/i]);
  function groupOf(key) {
    for (var i = 0; i < GROUPS.length; i++) if (GROUPS[i][1].test(key)) return GROUPS[i][0];
    return 'Other';
  }

  function sn(n) { n = String(n).replace(/ County$/, ''); return n.length > 13 ? n.slice(0, 12) + '\u2026' : n; }
  function pctChip(p) {
    if (p == null) return '<span class="cpNone">&ndash;</span>';
    return '<span class="cpPct" title="Higher than ' + Math.round(p) + '% of the cells in this area">' + Math.round(p) + '</span>';
  }

  var STYLE = '.maplibregl-popup.cellPopup{z-index:10010;}.cellPopup .maplibregl-popup-content{padding:0;border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.25);overflow:hidden;}' +
    '.cellPopup .maplibregl-popup-close-button{font-size:16px;color:#fff;right:4px;top:2px;}' +
    '.cp{font:12px/1.3 -apple-system,Segoe UI,Roboto,sans-serif;color:#222;width:min(340px,84vw);}' +
    '.cpHead{background:#2b5f9e;color:#fff;padding:7px 12px 6px;}' +
    '.cpTitle{font-weight:600;font-size:12.5px;margin-right:18px;} .cpSub{font-size:10.5px;opacity:.8;margin-top:1px;}' +
    '.cpKpis{display:flex;gap:8px;padding:8px 10px 2px;} .cpKpi{flex:1;border:1px solid #e6eaef;border-radius:8px;padding:5px 8px;background:#fafbfc;}' +
    '.cpKpi small{display:block;color:#778;font-size:10px;text-transform:uppercase;letter-spacing:.03em;}' +
    '.cpKpi b{font-size:16px;} .cpKpi .cpPcts{margin-top:2px;display:flex;flex-wrap:wrap;gap:0 8px;font-size:10px;color:#99a;}' +
    '.cpBtn{margin:6px 10px 6px;border:1px solid #d6dce3;border-radius:12px;background:#fff;cursor:pointer;padding:2px 10px;font:11px sans-serif;color:#456;}' +
    '.cpBtn:hover{background:#f2f5f9;} .cpTableWrap{max-height:260px;overflow-y:auto;border-top:1px solid #eef0f3;}' +
    '.cp table{border-collapse:collapse;width:100%;} .cp th{position:sticky;top:0;background:#f6f7f9;font-size:9.5px;text-transform:uppercase;color:#99a;font-weight:500;padding:3px 5px;text-align:right;border-bottom:1px solid #e8ebef;white-space:nowrap;z-index:1;}' +
    '.cp th:first-child{text-align:left;padding-left:10px;} .cp td{padding:2px 5px;border-bottom:1px solid #f3f4f6;text-align:right;white-space:nowrap;}' +
    '.cp td:first-child{text-align:left;padding-left:10px;white-space:normal;color:#333;} .cp td.cpVal{font-weight:600;}' +
    '.cp tr.cpGroup td{background:#fafbfc;color:#5b7ca6;font-weight:600;font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;padding:3px 10px;text-align:left;}' +
    '.cpPct{color:#99a;font-size:10.5px;} .cpNone{color:#ccc;} .cpRel{color:#889;font-weight:400;}';

  function ensureStyle() {
    if (document.getElementById('cellPopupStyle')) return;
    var st = document.createElement('style'); st.id = 'cellPopupStyle'; st.textContent = STYLE; document.head.appendChild(st);
  }

  function render(ctx, opts) {
    var row = ctx.row, scopes = ctx.scopes || [];
    var keys = Object.keys(row).filter(function(k) { return !SKIP[k] && k.charAt(0) !== '_' && row[k] != null && isFinite(row[k]); });
    var consumed = {};
    Object.keys(ABS_TO_REL).forEach(function(a) { var r = ABS_TO_REL[a]; if (row[a] != null && row[r] != null) consumed[r] = 1; });
    keys = keys.filter(function(k) { return !consumed[k]; });
    var labels = labelsFor(keys);
    keys.sort(function(a, b) { return (groupOf(a) === groupOf(b)) ? (labels[a] < labels[b] ? -1 : 1) : (GROUPS.map(function(g) { return g[0]; }).concat(['Other']).indexOf(groupOf(a)) - GROUPS.map(function(g) { return g[0]; }).concat(['Other']).indexOf(groupOf(b))); });
    var los = row.level_of_service, pop = row.population;
    var pcts = function(key, v) { return scopes.map(function(s) { var p = null; try { p = s.pct(key, v); } catch (e) {} return p; }); };
    var kpiPcts = function(key, v) {
      var ps = pcts(key, v);
      return '<div class="cpPcts">' + scopes.map(function(s, i) { return ps[i] == null ? '' : '<span title="' + esc(s.name) + '">' + esc(sn(s.name)) + ' ' + Math.round(ps[i]) + '</span>'; }).join('') + '</div>';
    };
    var title = ctx.title || opts.title || ('Hexagon (H3 res. ' + (typeof h3 !== 'undefined' && ctx.cell ? h3.getResolution(ctx.cell) : 9) + ')');
    var html = '<div class="cp"><div class="cpHead"><div class="cpTitle">' + esc(title) + '</div><div class="cpSub">' + esc(ctx.subtitle || ('H3 cell ' + ctx.cell)) + ' &middot; ' + keys.length + ' statistics</div></div>' +
      '<div class="cpKpis"><div class="cpKpi"><small>Level of service</small><b>' + fmt(los, 'level_of_service') + '</b>' + kpiPcts('level_of_service', los) + '</div>' +
      '<div class="cpKpi"><small>Population</small><b>' + fmt(pop, 'population') + '</b>' + kpiPcts('population', pop) + '</div></div>';
    var uid = 'cp' + Math.random().toString(36).slice(2);
    var open = !!opts.expanded;
    html += '<button type="button" class="cpBtn" data-n="' + keys.length + '" data-cp="' + uid + '">' + (open ? '&#9650; Hide' : '&#128202; Show') + ' all ' + keys.length + ' statistics</button>';
    html += '<div class="cpTableWrap" id="' + uid + '" style="display:' + (open ? 'block' : 'none') + '"><table><thead><tr><th>Statistic</th><th>Value</th>' +
      scopes.map(function(s) { return '<th title="Percentile of the value among all cells of ' + esc(s.name) + '">' + esc(sn(s.name)) + ' pct</th>'; }).join('') + '</tr></thead><tbody>';
    var lastGroup = null, cols = 2 + scopes.length;
    keys.forEach(function(k) {
      var g = groupOf(k);
      if (g !== lastGroup) { html += '<tr class="cpGroup"><td colspan="' + cols + '">' + g + '</td></tr>'; lastGroup = g; }
      var val = fmt(row[k], k), rel = ABS_TO_REL[k];
      if (rel && row[rel] != null) val += ' <span class="cpRel">(' + fmt(row[rel], rel) + ')</span>';
      var ps = pcts(k, row[k]);
      html += '<tr><td>' + esc(labels[k]) + '</td><td class="cpVal">' + val + '</td>' + ps.map(function(p) { return '<td>' + pctChip(p) + '</td>'; }).join('') + '</tr>';
    });
    return html + '</tbody></table></div></div>';
  }

  // The res-8 hexagon the popup's numbers come from is outlined (and lightly filled) on the map while the popup is open.
  function highlight(cell, geometry) {
    var map = window.__mainMap; if (!map) return;
    var data = {type: 'FeatureCollection', features: []};
    if (geometry) {
      data.features.push({type: 'Feature', geometry: geometry, properties: {}});
    } else if (cell) {
      var ring = h3.cellToBoundary(cell, true);
      ring.push(ring[0]);
      data.features.push({type: 'Feature', geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}});
    }
    if (!map.getSource('__cell_highlight')) {
      if (!cell && !geometry) return;
      map.addSource('__cell_highlight', {type: 'geojson', data: data});
      map.addLayer({id: '__cell_highlight_line', type: 'line', source: '__cell_highlight',
        paint: {'line-color': '#1f4e8c', 'line-width': 3, 'line-opacity': 0.95}, layout: {'line-join': 'round'}});
    } else {
      map.getSource('__cell_highlight').setData(data);
    }
  }

  // ---- the clicked layer decides the popup: the census polygon (US blocks: their block group) or the H3 cell at the drawn resolution (res 8 at most) ----
  var LEVEL_NAMES = {ccaa: 'Autonomous community', province: 'Province', municipality: 'Municipality', district: 'Census district', section: 'Census section',
    state: 'State', county: 'County', place: 'Place', tract: 'Census tract', blockgroup: 'Block group', block: 'Block', nation: 'Nation'};
  function bgProbe(map) {   // near-invisible block-group layers keep the block-group tiles loaded wherever blocks are drawn, so a click on a block can read its block group
    if (map.__bgProbed) return; map.__bgProbed = true;
    var layers = map.getStyle().layers, minz = 99;
    layers.forEach(function(l) { if (l['source-layer'] === 'census_block' && l.minzoom != null) minz = Math.min(minz, l.minzoom); });
    if (minz === 99) return;   // no block geometry on this map
    layers.forEach(function(l) {
      if (l['source-layer'] === 'census_blockgroup' && l.type === 'fill' && !map.getLayer('__bgp_' + l.id)) {
        map.addLayer({id: '__bgp_' + l.id, type: 'fill', source: l.source, 'source-layer': 'census_blockgroup', minzoom: Math.max(minz - 0.5, 0),
          paint: {'fill-opacity': 0.003, 'fill-color': '#000'}});
      }
    });
  }
  (function() {
    var mm = window.__mainMap;
    if (!mm || !mm.getStyle) return;
    var go = function() { try { bgProbe(mm); } catch (e) {} };
    if (mm.isStyleLoaded && mm.isStyleLoaded()) go(); else mm.once('load', go);
  })();
  function featureCtx(map, f, lng, lat) {
    var sl = f.sourceLayer || (f.layer && f.layer['source-layer']) || '', props = f.properties || {}, m = /^h3_res(\d+)$/.exec(sl);
    if (m) {
      var r = +m[1];
      if (r >= 9) return null;   // res 9 or finer: the res-9 cell popup (with its percentiles)
      var cell = h3.latLngToCell(lat, lng, r);
      return {cell: cell, row: props, scopes: [], title: 'Hexagon (H3 res. ' + r + ')'};
    }
    var c = /^census_(.+)$/.exec(sl);
    if (!c) return null;
    var level = c[1], geom = f.geometry;
    if (level === 'block') {   // US blocks: always the block group
      bgProbe(map);
      var ids = map.getStyle().layers.filter(function(l) { return /^__bgp_/.test(l.id); }).map(function(l) { return l.id; });
      var pt = map.project([lng, lat]), hit = ids.length ? map.queryRenderedFeatures(pt, {layers: ids}) : [];
      if (!hit.length) return null;
      f = hit[0]; level = 'blockgroup'; props = f.properties || {}; geom = f.geometry;
    }
    var gid = props.GEOID || props['_census_' + level + '_GEOID'] || '', label = LEVEL_NAMES[level] || level;
    var nm = props.name || props.NAME || props.NAMELSAD || (window.__geoNameOf && gid ? window.__geoNameOf(level, gid) : '') || gid || '';
    return {cell: null, row: props, scopes: [], geometry: geom, title: label + (nm ? ': ' + nm : ''), subtitle: gid ? 'ID ' + gid : label};
  }

  window.__openCellPopup = function(lng, lat, opts) {
    opts = opts || {};
    var map = window.__mainMap;
    if (opts.feature && map && typeof h3 !== 'undefined') {
      var fc = featureCtx(map, opts.feature, lng, lat);
      if (fc) {
        ensureStyle();
        window.__cellPopupSeq = (window.__cellPopupSeq || 0) + 1;
        if (popup) popup.remove();
        popup = new maplibregl.Popup({closeButton: true, closeOnClick: true, maxWidth: '640px', className: 'cellPopup'});
        popup.setLngLat(opts.at || [lng, lat]).setHTML(render(fc, opts)).addTo(map);
        highlight(fc.cell, fc.geometry);
        popup.on('close', function() { highlight(null); });
        popup.getElement().querySelectorAll('.cpBtn').forEach(function(b) {
          b.addEventListener('click', function() {
            var w = document.getElementById(b.getAttribute('data-cp')); if (!w) return;
            var shown = w.style.display !== 'none'; w.style.display = shown ? 'none' : 'block';
            b.innerHTML = (shown ? '&#128202; Show' : '&#9650; Hide') + ' all ' + b.getAttribute('data-n') + ' statistics';
          });
        });
        return true;
      }
    }
    if (!map || typeof window.__cellPopupProvider !== 'function' || typeof h3 === 'undefined') return false;
    var provider = window.__cellPopupProvider;
    var p = provider(lng, lat);
    if (!p || !p.then) return false;
    ensureStyle();
    var mine = (window.__cellPopupSeq = (window.__cellPopupSeq || 0) + 1);
    p.then(function(ctx) {
      if (mine !== window.__cellPopupSeq) return;
      if (!ctx) { if (popup) popup.remove(); return; }   // a cell with no data (outside the study area): no popup
      if (popup) popup.remove();
      popup = new maplibregl.Popup({closeButton: true, closeOnClick: true, maxWidth: '640px', className: 'cellPopup'});
      popup.setLngLat(opts.at || [lng, lat]).setHTML(render(ctx, opts)).addTo(map);
      highlight(ctx.cell);
      popup.on('close', function() { highlight(null); });
      var el = popup.getElement();
      el.querySelectorAll('.cpBtn').forEach(function(b) {
        b.addEventListener('click', function() {
          var w = document.getElementById(b.getAttribute('data-cp')); if (!w) return;
          var shown = w.style.display !== 'none'; w.style.display = shown ? 'none' : 'block';
          var n = b.getAttribute('data-n');
          b.innerHTML = (shown ? '&#128202; Show' : '&#9650; Hide') + ' all ' + n + ' statistics';
        });
      });
    }).catch(function(e) { console.warn('cell popup failed', e); });
    return true;
  };
})();
"""


def cell_popup_script(abs_to_rel: Dict[str, str]) -> str:
    """Script of the cell popup (percentiles of the clicked res-9 cell against its areas).

    Args:
        abs_to_rel: ``{absolute column: relative column}`` pairs shown together.

    Returns:
        JavaScript text.
    """
    from .popup_cells import POPUP9_JS

    return POPUP9_JS + CELL_POPUP_JS.replace("__ABS_TO_REL__", json.dumps(abs_to_rel))


# City maps: the res-8 cells of `window.__getStatsData('metro' | 'core')` (so a scenario's edits show in the level of service), percentiles
# among the urban area's cells, the city core's cells and every cell of every city (`../world_quantiles.json`, built with the combined map).
CITY_PROVIDER_JS = r"""
(function() {
  var worldQ = null, worldTried = false, worldPromise = null, indexCache = (typeof WeakMap !== 'undefined') ? new WeakMap() : null;
  function loadWorld() {
    if (worldTried) return worldPromise; worldTried = true;
    worldPromise = fetch(window.__worldQuantilesUrl || '../world_quantiles.json').then(function(r) { return r.ok ? r.json() : null; })
      .then(function(j) { worldQ = j; }).catch(function() {});
    return worldPromise;
  }
  function indexOf(d, cell) {
    var m = indexCache && indexCache.get(d.h3_cell);
    if (!m) { m = {}; d.h3_cell.forEach(function(c, i) { m[c] = i; }); if (indexCache) indexCache.set(d.h3_cell, m); }
    return m[cell];
  }
  loadWorld();
  var world9 = null, world9P = null;
  function res9(lng, lat) {
    // the city's res-9 popup files (transitlos.map.popup_cells); scenario edits only exist in the res-8 stats data, so an edited scenario uses those
    var sc = document.getElementById('topBarScenarioSelect'), edited = sc && sc.value !== '' && sc.value !== 'original' && sc.value !== 'baseline';
    if (edited || !window.__popup9 || typeof h3 === 'undefined') return Promise.resolve(null);
    var P = window.__popup9(window.__popup9Base || 'results/popup9/');
    if (!world9P) world9P = fetch(window.__world9Url || '../world_quantiles9.json').then(function(r) { return r.ok ? r.json() : null; }).then(function(j) { world9 = j; }).catch(function() {});
    return P.row(lng, lat).then(function(o) {
      if (!o) return null;
      var NAMES9 = {metro: 'Urban area', official: 'Metro area', municipality: 'Municipality', core: 'City core'}, kinds = [], scopes = [];
      ['metro', 'official', 'municipality', 'core'].forEach(function(k) { if (o.ids[k]) kinds.push(k); });
      return Promise.all(kinds.map(function(k) { return P.q(k + '_' + o.ids[k]); }).concat([world9P])).then(function(tabs) {
        kinds.forEach(function(k, i) { var t = tabs[i]; if (t) scopes.push({name: NAMES9[k], pct: function(c, v) { return t[c] ? window.__pctInQuantiles(t[c], v) : null; }}); });
        scopes.push({name: 'All places', pct: function(c, v) { return world9 && world9[c] ? window.__pctInQuantiles(world9[c], v) : null; }});
        return {cell: o.cell, row: o.row, scopes: scopes};
      });
    }).catch(function() { return null; });
  }
  window.__cellPopupProvider = function(lng, lat) {
    return res9(lng, lat).then(function(r) { return r || res8(lng, lat); });
  };
  function res8(lng, lat) {
    var wp = loadWorld();
    if (typeof window.__getStatsData !== 'function' || typeof h3 === 'undefined') return Promise.resolve(null);
    // every analysis area of the map (the active one first): a cell inside the official or municipal area need not be inside the UN metro area
    var NAMES = {metro: 'Urban area', official: 'Metro area', municipality: 'Municipality', core: 'City core'}, order = [], srcs = [];
    [window.__currentArea, 'metro', 'official', 'municipality', 'core'].forEach(function(k) { if (k && NAMES[k] && order.indexOf(k) < 0) order.push(k); });
    order.forEach(function(k) { var d = null; try { d = window.__getStatsData(k); } catch (e) {} if (d && d.h3_cell && d.h3_cell.length) srcs.push({k: k, d: d}); });
    if (!srcs.length) return Promise.resolve(null);
    var res = h3.getResolution(srcs[0].d.h3_cell[0]), cell = h3.latLngToCell(lat, lng, res), src = null, i;
    for (var a = 0; a < srcs.length && !src; a++) { var ix = indexOf(srcs[a].d, cell); if (ix !== undefined) { src = srcs[a]; i = ix; } }
    if (!src) {   // a click on a coarser circle can land on an unpopulated cell: use the nearest populated neighbour (<= 2 rings)
      var best = null, bd = 1e9, bs = null, kc = Math.cos(lat * Math.PI / 180);
      h3.gridDisk(cell, 2).forEach(function(c) {
        for (var b = 0; b < srcs.length; b++) {
          if (indexOf(srcs[b].d, c) === undefined) continue;
          var ll = h3.cellToLatLng(c), dd = (ll[0] - lat) * (ll[0] - lat) + (ll[1] - lng) * (ll[1] - lng) * kc * kc;
          if (dd < bd) { bd = dd; best = c; bs = srcs[b]; }
          break;
        }
      });
      if (best) { cell = best; src = bs; i = indexOf(bs.d, best); }
    }
    if (!src) return Promise.resolve(null);
    var row = {};
    Object.keys(src.d).forEach(function(k) { var a = src.d[k]; if (Array.isArray(a) && k !== 'h3_cell') row[k] = a[i]; });
    var scopes = [{name: NAMES[src.k], pct: function(k, v) { return src.d[k] ? window.__pctInArray(src.d[k], v) : null; }}];
    srcs.forEach(function(o) { if (o.k === 'core' && src.k !== 'core') scopes.push({name: NAMES.core, pct: function(k, v) { return o.d[k] ? window.__pctInArray(o.d[k], v) : null; }}); });
    scopes.push({name: 'All places', pct: function(k, v) { return worldQ && worldQ[k] ? window.__pctInQuantiles(worldQ[k], v) : null; }});
    return Promise.resolve(wp).then(function() { return {cell: cell, row: row, scopes: scopes}; });
  }
})();
"""
