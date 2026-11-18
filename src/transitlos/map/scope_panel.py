"""Scope selector of the stats panels: which data every tab shows (nation / state / county / place) and what it is
compared with (none / nation / state / county / place), plus the Original / With-edits scenario.

Injected into a MapLibre `map.html` that already has the stats panel (`build._inject_maplibre_stats_panel_into_saved_html`).
Data comes from `stats_chunks/` (written by the study, e.g. `usa_study/code/stats_chunks.py`):
  * `index.json`: `{"states": [{key, name, bbox, rings, counties: [{id, name}], places: [{id, name}], n}]}`;
  * `<key>.json.gz`: column-oriented cells of one state (`h3_cell`, `level_of_service`, ..., `county`, `place` = index or -1).
The nation is `window.__statsData.metro` (the coarse grid embedded in the page).
Rules (user spec): Nation switches everything else off; State needs a state picked (clicking it again deactivates it);
County and Place are mutually exclusive and only offered after a state is picked, choosing from that state's list (type to
filter). Zooming out until the view covers several states resets to the nation and no comparison; zooming into a single state
selects that state automatically.
"""
import json

SCOPE_BAR_HTML = """
<style>#statsAreaBtnRow { display: none !important; }</style>
<div id="scopeBar" style="margin-bottom:8px;">
  <div style="display:flex;gap:6px;">
    <div class="scopeSide" data-side="primary" style="flex:1;position:relative;min-width:0;">
      <button type="button" class="scopeMainBtn" style="width:100%;text-align:left;padding:5px 8px;border:1px solid #d5d9de;border-radius:5px;background:#f5f6f8;cursor:pointer;font:12px sans-serif;color:#333;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></button>
      <div class="scopePop" style="display:none;position:absolute;top:100%;left:0;z-index:1002;background:white;border:1px solid #d5d9de;border-radius:5px;box-shadow:0 2px 8px rgba(0,0,0,.2);padding:8px;margin-top:2px;width:250px;"></div>
    </div>
    <div class="scopeSide" data-side="compare" style="flex:1;position:relative;min-width:0;">
      <button type="button" class="scopeMainBtn" style="width:100%;text-align:left;padding:5px 8px;border:1px solid #d5d9de;border-radius:5px;background:#f5f6f8;cursor:pointer;font:12px sans-serif;color:#333;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></button>
      <div class="scopePop" style="display:none;position:absolute;top:100%;right:0;z-index:1002;background:white;border:1px solid #d5d9de;border-radius:5px;box-shadow:0 2px 8px rgba(0,0,0,.2);padding:8px;margin-top:2px;width:250px;"></div>
    </div>
  </div>
</div>
"""

SCOPE_JS = r"""
(function() {
  var CFG = __CFG__;
  if (CFG.address_zoom) { window.__addressZoom = CFG.address_zoom; window.__addressCountry = "us"; }   // the top bar's address search: US results, zoom 6
  var map = window.__mainMap;
  var INDEX = null, byKey = {}, byMetro = {}, stateData = {}, pending = {}, subsetCache = {}, lru = [];
  var MAX_STATES = 4;   // state tables kept in memory (the nation table is always kept)
  var scope = {
    primary: {on: true, state: null, county: null, place: null, metro: null},
    compare: {on: false, state: null, county: null, place: null, metro: null},
    manual: false,
  };
  var pickerOpen = {primary: null, compare: null};
  window.__scopeActive = true;

  function blank(on) { return {on: on, state: null, county: null, place: null, metro: null}; }
  function levelOf(s) {
    if (!s.on) return 'none';
    if (s.metro) return 'metro';
    if (s.place != null) return 'place';
    if (s.county != null) return 'county';
    if (s.state) return 'state';
    return 'nation';
  }
  function labelOf(s) {
    var lv = levelOf(s), st = s.state && byKey[s.state];
    if (lv === 'none') return 'None';
    if (lv === 'nation') return CFG.nation_name || 'Nation';
    if (lv === 'metro') return byMetro[s.metro].name;
    if (lv === 'state') return st.name;
    if (lv === 'county') return st.counties[s.county].name + ', ' + st.name;
    return st.places[s.place].name + ', ' + st.name;
  }
  window.__scopeLabel = function(area) { return labelOf(area === 'core' ? scope.compare : scope.primary); };
  window.__scopeCompareOn = function() { return scope.compare.on; };
  // Nation level has no edit mode: edits never change nation-scope data (and the editor itself is hidden, see setEditAllowed).
  window.__scopeNoEdits = function(area) { return levelOf(area === 'core' ? scope.compare : scope.primary) === 'nation'; };

  // ---- data ----------------------------------------------------------------------------------------------------
  function fetchState(key) {
    if (stateData[key] || pending[key]) return;
    // the nation sample (tens of MB) loads when the stats panel is opened, not at page start (the panel's button re-renders the tab)
    if (key === 'Nation') { var sp = document.getElementById('statsPanel'); if (!sp || sp.style.display === 'none') return; }
    pending[key] = true;
    fetch(CFG.dir + '/' + key + '.json.gz').then(function(r) { return r.arrayBuffer(); }).then(function(buf) {
      var u8 = new Uint8Array(buf);
      if (u8[0] === 0x1f && u8[1] === 0x8b && typeof DecompressionStream !== 'undefined') {
        var ds = new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')));
        return ds.json();
      }
      return JSON.parse(new TextDecoder().decode(u8));
    }).then(function(d) {
      if (key === 'Nation') { window.__nationFactor = d.factor; delete d.factor; }
      stateData[key] = d; delete pending[key];
      if (key !== 'Nation') { touch(key); evict(); }
      rerender();
    })
      .catch(function(e) { console.warn('stats chunk failed', key, e); delete pending[key]; });
  }
  function touch(key) { var i = lru.indexOf(key); if (i >= 0) lru.splice(i, 1); lru.push(key); }
  function evict() {
    var inUse = {};
    [scope.primary, scope.compare].forEach(function(s) { if (s.on && s.state) inUse[s.state] = true; if (s.on && s.metro) inUse[s.metro] = true; });
    while (lru.length > MAX_STATES) {
      var drop = lru.find(function(k) { return !inUse[k]; });
      if (!drop) break;
      lru.splice(lru.indexOf(drop), 1); delete stateData[drop];
      Object.keys(subsetCache).forEach(function(ck) { if (ck.indexOf(drop + ':') === 0) delete subsetCache[ck]; });
    }
  }
  function subset(key, kind, idx) {
    var ck = key + ':' + kind + ':' + idx;
    if (subsetCache[ck]) return subsetCache[ck];
    var d = stateData[key], sel = [];
    var col = d[kind];
    for (var i = 0; i < col.length; i++) if (col[i] === idx) sel.push(i);
    var out = {};
    Object.keys(d).forEach(function(c) {
      if (c === 'county' || c === 'place') return;
      var a = d[c], b = new Array(sel.length);
      for (var j = 0; j < sel.length; j++) b[j] = a[sel[j]];
      out[c] = b;
    });
    subsetCache[ck] = out;
    return out;
  }
  function dataFor(s) {
    var lv = levelOf(s);
    if (lv === 'none') return null;
    if (lv === 'nation') { fetchState('Nation'); return stateData['Nation'] || window.__statsData.metro; }  // res-8 sample once loaded
    if (lv === 'metro') { fetchState(s.metro); if (!stateData[s.metro]) return null; touch(s.metro); return stateData[s.metro]; }
    fetchState(s.state);
    if (!stateData[s.state]) return null;
    touch(s.state);
    if (lv === 'state') return stateData[s.state];
    return subset(s.state, lv, lv === 'county' ? s.county : s.place);
  }
  window.__scopeStateKey = function() { return scope.primary.state; };
  window.__stateAt = stateAt;   // state / community whose polygon holds a lon/lat (the top bar's score samples the map centre)
  window.__scopeCountyId = function() { var s = scope.primary, st = s.state && byKey[s.state]; return (st && s.county != null && st.counties[s.county]) ? st.counties[s.county].id : null; };
  window.__scopePlaceId = function() { var s = scope.primary, st = s.state && byKey[s.state]; return (st && s.place != null && st.places[s.place]) ? st.places[s.place].id : null; };
  // Names of the municipalities / provinces / counties from the index, for the click popups of census polygons (`cell_popup.py`).
  window.__geoNameOf = function(level, id) {
    if (!INDEX) return null;
    id = String(id);
    var look = function(kind, key) {
      for (var i = 0; i < INDEX.states.length; i++) { var arr = INDEX.states[i][kind] || []; for (var j = 0; j < arr.length; j++) if (String(arr[j].id) === key) return arr[j].name; }
      return null;
    };
    if (level === 'section' || level === 'district') {
      var pl = look('places', id.slice(0, 5));
      return pl ? pl + ' - district ' + id.slice(5, 7) + (level === 'section' ? ' section ' + id.slice(7) : '') : null;
    }
    if (level === 'province' || level === 'county') return look('counties', id);
    if (level === 'municipality' || level === 'place') return look('places', id);
    return null;
  };
  window.__scopeMetro = function() { var m = scope.primary.on && scope.primary.metro && byMetro[scope.primary.metro]; return m || null; };   // {key, name, members, ...} of the active metropolitan area
  window.__scopeLevel = function() { return levelOf(scope.primary); };   // 'nation' | 'state' | 'county' | 'place' | 'none'
  window.__scopeData = function(area) {
    if (area === 'core') return dataFor(scope.compare);
    return dataFor(scope.primary) || window.__statsData.metro;  // nation while a state file is still loading
  };

  // ---- re-render -----------------------------------------------------------------------------------------------
  var rerenderTimer = null;
  function rerender() {
    renderButtons();
    clearTimeout(rerenderTimer);
    rerenderTimer = setTimeout(function() {
      var panel = document.getElementById('statsPanel');
      if (panel && panel.style.display !== 'none' && window.__renderActiveTab) window.__renderActiveTab();
    }, 120);
  }
  window.__scopeRerender = rerender;

  // ---- pickers -------------------------------------------------------------------------------------------------
  function norm(t) { return (t || '').toLowerCase(); }
  function renderPop(side) {
    var s = scope[side], pop = document.querySelector('.scopeSide[data-side="' + side + '"] .scopePop');
    var kind = pickerOpen[side], lv = levelOf(s);
    var html = '';
    if (side === 'compare') {   // the comparison is off until activated here; its level then defaults to the nation
      html += '<label style="display:flex;align-items:center;gap:6px;font:12px sans-serif;margin-bottom:' + (s.on ? '8' : '0') + 'px;cursor:pointer;">' +
        '<input type="checkbox" data-act="cmpToggle" ' + (s.on ? 'checked' : '') + '> Compare with another level</label>';
      if (!s.on) { pop.innerHTML = html; return; }
    }
    html += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px;">';
    var btn = function(id, text, active, disabled, title) {
      return '<button type="button" data-act="' + id + '" ' + (disabled ? 'disabled ' : '') + 'title="' + (title || '') + '" ' +
        'style="padding:4px 8px;border:1px solid #c9ced6;border-radius:4px;font:12px sans-serif;cursor:' + (disabled ? 'not-allowed' : 'pointer') + ';' +
        'background:' + (active ? '#2b6cb0' : '#f5f6f8') + ';color:' + (active ? '#fff' : (disabled ? '#aaa' : '#333')) + ';">' + text + '</button>';
    };
    html += btn('nation', 'Nation', lv === 'nation', false);
    var ES = CFG.nation_name === 'Spain', LBL = ES ? {state: 'Autonomous community', county: 'Province', place: 'Municipality'} : {state: 'State', county: 'County', place: 'Place'};
    var first = ES ? 'select an autonomous community first' : 'select a state first';
    html += btn('state', LBL.state, !!s.state && s.on, false);
    html += btn('county', LBL.county, lv === 'county', !s.state || !s.on, first);
    html += btn('place', LBL.place, lv === 'place', !s.state || !s.on, first);
    if (INDEX && INDEX.metros && INDEX.metros.length) html += btn('metro', 'Metro area', lv === 'metro', false);
    html += '</div>';
    if (kind) {
      html += '<input type="search" class="scopeFilter" autocomplete="off" placeholder="Search ' + (LBL[kind] || kind).toLowerCase() + ' (type a name)..." ' +
        'style="width:100%;box-sizing:border-box;padding:6px;border:1px solid #c9ced6;border-radius:4px;font:12px sans-serif;">' +
        '<div class="scopeResults" style="margin-top:4px;"></div><div class="scopeHint" style="color:#888;font:11px sans-serif;margin-top:3px;"></div>';
    }
    pop.innerHTML = html;
    var f = pop.querySelector('.scopeFilter');
    if (f) {
      var items;
      if (kind === 'state') items = INDEX.states.map(function(x) { return {i: x.key, name: x.name, pop: x.pop || 0}; });
      else if (kind === 'metro') items = INDEX.metros.map(function(x) { return {i: x.key, name: x.name, pop: x.pop || 0}; });
      else items = byKey[s.state][kind === 'county' ? 'counties' : 'places'].map(function(x, i) { return {i: i, name: x.name, pop: x.pop || 0}; });
      var box = pop.querySelector('.scopeResults'), hint = pop.querySelector('.scopeHint'), cur = 0, shown = [];
      var fmtPop = function(p) { return p >= 1e6 ? (p / 1e6).toFixed(1) + 'M' : p >= 1e3 ? Math.round(p / 1e3) + 'k' : String(Math.round(p)); };
      var paint = function() {
        var q = norm(f.value).trim(), ranked = items.filter(function(it) { return !q || norm(it.name).indexOf(q) >= 0; });
        // best match first: names starting with the text, then word starts, then anywhere; ties by population
        var rank = function(it) { var n = norm(it.name); return !q ? 3 : n.indexOf(q) === 0 ? 0 : n.indexOf(' ' + q) >= 0 ? 1 : 2; };
        ranked.sort(function(a, b) { return rank(a) - rank(b) || b.pop - a.pop || (a.name < b.name ? -1 : 1); });
        shown = ranked.slice(0, 8); cur = Math.min(cur, Math.max(shown.length - 1, 0));
        box.innerHTML = shown.length ? shown.map(function(it, n) {
          return '<div class="scopeItem" data-i="' + it.i + '" style="display:flex;justify-content:space-between;gap:8px;padding:4px 6px;cursor:pointer;border-radius:3px;font:12px sans-serif;' +
            (n === cur ? 'background:#e6effa;' : '') + '"><span>' + it.name + '</span></div>';
        }).join('') : '<div style="padding:4px 6px;color:#888;font:12px sans-serif;">No match</div>';
        hint.textContent = ranked.length > shown.length ? 'Showing ' + shown.length + ' of ' + ranked.length + ' - keep typing to narrow' : ranked.length + ' result' + (ranked.length === 1 ? '' : 's');
      };
      f.addEventListener('input', function() { cur = 0; paint(); });
      f.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowDown') { cur = Math.min(cur + 1, shown.length - 1); paint(); e.preventDefault(); }
        else if (e.key === 'ArrowUp') { cur = Math.max(cur - 1, 0); paint(); e.preventDefault(); }
        else if (e.key === 'Enter' && shown[cur]) { pickItem(side, shown[cur].i); e.preventDefault(); }
        else if (e.key === 'Escape') { pickerOpen[side] = null; renderPop(side); }
      });
      paint(); f.focus();
    }
  }
  function pickItem(side, value) {
    var s = scope[side], kind = pickerOpen[side];
    if (kind === 'metro') { s.metro = value; s.state = null; s.county = null; s.place = null; }
    else if (kind === 'state') { s.state = value; s.metro = null; s.county = null; s.place = null; }
    else if (kind === 'county') { s.county = parseInt(value, 10); s.place = null; }
    else { s.place = parseInt(value, 10); s.county = null; }
    s.on = true; pickerOpen[side] = null; scope.manual = true;
    renderPop(side); rerender();
  }
  function onPopClick(side, e) {
    var s = scope[side];
    var item = e.target.closest('.scopeItem');
    if (item) { pickItem(side, item.dataset.i); return; }
    var b = e.target.closest('button[data-act]');
    if (!b || b.disabled) return;
    var act = b.dataset.act;
    scope.manual = true;
    if (act === 'nation') { scope[side] = blank(true); pickerOpen[side] = null; }
    else if (act === 'metro') {
      if (s.metro) { scope[side] = blank(true); pickerOpen[side] = null; } else { pickerOpen[side] = 'metro'; }
    } else if (act === 'state') {
      if (s.state && s.on) { scope[side] = blank(true); pickerOpen[side] = null; }   // clicking an active state deactivates it
      else { pickerOpen[side] = 'state'; }
    } else if (act === 'county') {
      if (s.county != null) { s.county = null; pickerOpen[side] = null; } else { pickerOpen[side] = 'county'; }
    } else if (act === 'place') {
      if (s.place != null) { s.place = null; pickerOpen[side] = null; } else { pickerOpen[side] = 'place'; }
    }
    renderPop(side); rerender();
  }
  // Population and daily vehicle-km of a scope, from the index (vkm = transit vehicle-km per day inside the area).
  function totalsOf(s) {
    var lv = levelOf(s), st = s.state && byKey[s.state], e;
    if (lv === 'none') return null;
    if (lv === 'nation') { e = {pop: 0, vkm: 0}; INDEX.states.forEach(function(x) { e.pop += x.pop || 0; e.vkm += x.vkm || 0; }); }
    else if (lv === 'metro') e = byMetro[s.metro];
    else if (lv === 'state') e = st;
    else e = (lv === 'county' ? st.counties[s.county] : st.places[s.place]);
    return e;
  }
  function totalsLine(s) {
    var e = totalsOf(s);
    if (!e || !e.pop) return '';
    var f = function(v) { return v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : Math.round(v).toLocaleString('en-US'); };
    return '<div style="font:11px sans-serif;font-weight:normal;color:#666;margin-top:2px;white-space:normal;">Pop. ' + f(e.pop) +
      ' &middot; ' + f(e.vkm || 0) + ' veh-km/day &middot; ' + f((e.vkm || 0) / e.pop * 1000) + ' veh-km per 1,000 pop</div>';
  }
  function renderButtons() {
    document.querySelector('.scopeSide[data-side="primary"] .scopeMainBtn').innerHTML = 'Level: <b>' + labelOf(scope.primary) + '</b> &#9662;';
    document.querySelector('.scopeSide[data-side="compare"] .scopeMainBtn').innerHTML = scope.compare.on
      ? 'Compare with: <b>' + labelOf(scope.compare) + '</b> &#9662;'
      : 'Compare with &#9662;';   // nothing selected until the user opts in
  }

  // ---- which state(s) are on screen ----------------------------------------------------------------------------
  function inRing(x, y, ring) {
    var inside = false;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      var xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }
  function stateAt(x, y) {
    for (var k = 0; k < INDEX.states.length; k++) {
      var st = INDEX.states[k], b = st.bbox;
      if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue;
      for (var r = 0; r < st.rings.length; r++) if (inRing(x, y, st.rings[r])) return st.key;
    }
    return null;
  }
  // Samples a 7x7 grid over the view: the state holding most samples is "the state on screen" when it holds at least
  // half of them (a z7 view of Wyoming also shows slivers of its neighbours); otherwise the view covers several states.
  function statesInView() {
    var b = map.getBounds(), w = b.getWest(), e = b.getEast(), s = b.getSouth(), n = b.getNorth(), count = {}, total = 0;
    for (var i = 0; i <= 6; i++) for (var j = 0; j <= 6; j++) {
      total++;
      var k = stateAt(w + (e - w) * (i + 0.5) / 7, s + (n - s) * (j + 0.5) / 7);
      if (k) count[k] = (count[k] || 0) + 1;
    }
    var best = null;
    Object.keys(count).forEach(function(k) { if (!best || count[k] > count[best]) best = k; });
    return {dominant: best, share: best ? count[best] / total : 0};
  }
  // Edit mode (draw lines / With edits) only exists when the view is inside one state, never at nation level.
  var editOk = true;
  function setEditAllowed(ok) {
    editOk = ok;
    var btn = document.getElementById('routeEditToggleBtn'), tb = document.getElementById('route-toolbar');
    if (!ok && tb && tb.classList.contains('rt-open') && btn) btn.click();   // leave edit mode (stops map clicks adding stops)
    if (btn) btn.style.display = ok ? '' : 'none';
    if (tb) tb.style.display = ok ? '' : 'none';
    var top = document.getElementById('topBarScenarioSelect');
    if (!ok && top && top.value !== '') { top.value = ''; top.dispatchEvent(new Event('change', {bubbles: true})); }
    syncScenario();
  }
  var lastClass = null;
  var EDIT_MIN_ZOOM = 7;   // lines can only be drawn once zoomed into a state (stops / population cells of the state are loaded)
  function editLevelOk() { return !!(pinned || (lastClass && lastClass.indexOf('single') === 0)) && map.getZoom() >= EDIT_MIN_ZOOM; }
  map.on('zoom', function() { var ok = editLevelOk(); if (INDEX && ok !== editOk) setEditAllowed(ok); });
  var pinned = null;   // state opened by a deep link: kept until the user first moves the map
  function onView() {
    if (!INDEX) return;
    if (pinned) { lastClass = 'single:' + pinned; setEditAllowed(editLevelOk()); ['primary', 'compare'].forEach(renderPop); rerender(); return; }
    var z = map.getZoom(), v = statesInView();
    if (z >= CFG.min_zoom + 1) { var c0 = map.getCenter(), cs = stateAt(c0.lng, c0.lat); if (cs) v = {dominant: cs, share: 1}; }   // zoomed in: the state under the map centre, not the one covering most of the view
    var single = z >= CFG.min_zoom && v.dominant && (v.share >= 0.5 || z >= CFG.min_zoom + 1);   // one zoom level in, the state under most of the view is chosen even if shared
    var cls = single ? 'single:' + v.dominant : 'multi';
    if (cls === lastClass) return;
    var wasSingle = lastClass && lastClass.indexOf('single') === 0;
    lastClass = cls;
    if (!single) {                       // low zoom / several states: nation, no comparison
      scope.primary = blank(true); if (!scope.compare.on) scope.compare = blank(false); scope.manual = false;
      pickerOpen = {primary: null, compare: null};
    } else if (!scope.manual || !wasSingle) {
      scope.primary = blank(true); scope.primary.state = v.dominant; scope.manual = false;
    }
    setEditAllowed(editLevelOk());
    ['primary', 'compare'].forEach(renderPop);
    rerender();
  }

  // ---- scenario: every tab has its own Scenario selectors; "With edits" does not exist at nation level ----
  var SCEN_IDS = ['dist', 'reg', 'anova'].reduce(function(a, p) { return a.concat([p + 'ScenarioSelect', p + 'SecondaryScenarioSelect']); }, []);
  function syncScenario() {
    SCEN_IDS.forEach(function(id) {
      var el = document.getElementById(id);
      if (!el) return;
      [].forEach.call(el.options, function(o) { o.disabled = !editOk && o.value !== ''; });
      if (!editOk && el.value !== '') el.value = '';
    });
  }
  // the compare checkbox is one switch for the three tabs
  function mirrorCompare(source) {
    ['dist', 'reg', 'anova'].forEach(function(p) { var c = document.getElementById(p + 'CompareEnable'); if (c && c !== source) c.checked = source.checked; });
  }

  function init() {
    var bar = document.getElementById('scopeBar');
    if (!bar) return;
    document.querySelectorAll('.scopeSide').forEach(function(sideEl) {
      var side = sideEl.dataset.side, pop = sideEl.querySelector('.scopePop');
      sideEl.querySelector('.scopeMainBtn').addEventListener('click', function() {
        var open = pop.style.display !== 'none';
        document.querySelectorAll('.scopePop').forEach(function(p) { p.style.display = 'none'; });
        if (!open) { pop.style.display = 'block'; renderPop(side); }
      });
      pop.addEventListener('click', function(e) { onPopClick(side, e); });
      pop.addEventListener('change', function(e) {
        if (!e.target || e.target.getAttribute('data-act') !== 'cmpToggle') return;
        scope.compare = blank(e.target.checked);   // on: nation by default; off: nothing
        pickerOpen.compare = null; scope.manual = true;
        renderPop('compare'); rerender();
      });
    });
    // The comparison of the nationwide map is driven by the "Compare with" button above (checkbox + level), so the per-tab
    // Compare-with control of the city maps (checkbox + popover) does not exist here.
    ['dist', 'reg', 'anova'].forEach(function(p) {
      var c = document.getElementById(p + 'CompareEnable');
      var w = c && (c.closest('.compareWithWrap') || c.parentElement);
      if (w) w.style.display = 'none';
      var btn = document.querySelector('.compareWithBtn[data-target="' + p + 'ComparePopover"]');
      if (btn) { var bw = btn.closest('.compareWithWrap') || btn.parentElement; if (bw) bw.style.display = 'none'; }
    });
    document.querySelectorAll('.statsTabBtn').forEach(function(b) {
      b.addEventListener('click', function() {
        var hide = (b.dataset.tab === 'discretization' || b.dataset.tab === 'metadata');
        bar.style.display = hide ? 'none' : 'block';
      });
    });
    // Per-tab controls of the nationwide map: the Area (metro / core) and Place selectors do not apply (the scope bar picks
    // nation / state / county / place); the main Scenario, the Compare-with checkbox, Column and Scenario stay.
    ['dist', 'reg', 'anova'].forEach(function(p) {
      var aw = document.getElementById(p + 'AreaWrap'); if (aw) aw.style.display = 'none';
      ['CompareAreaSelect', 'ComparePlaceSelect'].forEach(function(n) { var el = document.getElementById(p + n); if (el) el.parentElement.style.display = 'none'; });
    });
    var oldRow = document.getElementById('statsAreaBtnRow');
    if (oldRow) oldRow.style.display = 'none';
    renderButtons(); syncScenario();
  }

  // ---- popup data: the res-8 cell under a point, with percentiles among the cells of its county, its state and the nation --------
  var nationQ = null, nationQTried = false, cellIdx = {};
  var nationQPromise = null;
  function loadNationQ() {
    if (nationQTried) return nationQPromise; nationQTried = true;
    nationQPromise = fetch(CFG.dir + '/nation_quantiles.json').then(function(r) { return r.ok ? r.json() : null; }).then(function(j) { nationQ = j; }).catch(function() {});
    return nationQPromise;
  }
  function whenState(key) {
    return new Promise(function(resolve) {
      fetchState(key);
      var t0 = Date.now();
      (function poll() { if (stateData[key]) return resolve(stateData[key]); if (Date.now() - t0 > 30000) return resolve(null); setTimeout(poll, 120); })();
    });
  }
  function res9(lng, lat) {   // the res-9 popup files (transitlos.map.popup_cells): county / province, state / community and nation tables
    if (!window.__popup9 || typeof h3 === 'undefined') return Promise.resolve(null);
    var P = window.__popup9(CFG.dir + '/popup9/');
    return P.row(lng, lat).then(function(o) {
      if (!o) return null;
      var sk = o.ids.state, st = sk ? byKey[sk] : null, list = [];
      if (o.ids.county) {
        var cn = null; if (st) (st.counties || []).forEach(function(c) { if (String(c.id) === String(o.ids.county)) cn = c.name; });
        list.push([cn || (CFG.labels && CFG.labels.county) || 'County', 'county_' + o.ids.county]);
      }
      if (sk) list.push([st ? st.name : sk, 'state_' + sk]);
      list.push([CFG.nation_name || 'Nation', 'nation']);
      return Promise.all(list.map(function(x) { return P.q(x[1]); })).then(function(tabs) {
        var scopes = [];
        list.forEach(function(x, i) { var t = tabs[i]; if (t) scopes.push({name: x[0], pct: function(k, v) { return t[k] ? window.__pctInQuantiles(t[k], v) : null; }}); });
        return {cell: o.cell, row: o.row, scopes: scopes};
      });
    }).catch(function() { return null; });
  }
  function installCellProvider() {
    window.__cellPopupProvider = function(lng, lat) {
      return res9(lng, lat).then(function(r) { return r || res8(lng, lat); });
    };
  }
  function res8(lng, lat) {
    {
      var qp = loadNationQ();
      var key = stateAt(lng, lat);
      if (!key || typeof h3 === 'undefined') return Promise.resolve(null);
      return Promise.all([whenState(key), qp]).then(function(res) {
        var d = res[0];
        if (!d) return null;
        var cell = h3.latLngToCell(lat, lng, 8);
        var m = cellIdx[key];
        if (!m) { m = cellIdx[key] = {}; d.h3_cell.forEach(function(c, i) { m[c] = i; }); }
        var i = m[cell];
        if (i === undefined) {   // a click on a coarser circle can land on an unpopulated cell: nearest populated neighbour (<= 2 rings)
          var best = null, bd = 1e9, kc = Math.cos(lat * Math.PI / 180);
          h3.gridDisk(cell, 2).forEach(function(c) {
            if (m[c] === undefined) return;
            var ll = h3.cellToLatLng(c), dd = (ll[0] - lat) * (ll[0] - lat) + (ll[1] - lng) * (ll[1] - lng) * kc * kc;
            if (dd < bd) { bd = dd; best = c; }
          });
          if (best) { cell = best; i = m[best]; }
        }
        if (i === undefined) return null;
        var row = {};
        Object.keys(d).forEach(function(k) { if (k !== 'h3_cell' && k !== 'county' && k !== 'place' && Array.isArray(d[k])) row[k] = d[k][i]; });
        var st = byKey[key], scopes = [], ci = d.county[i];
        if (ci >= 0) {
          var sub = subset(key, 'county', ci);
          scopes.push({name: st.counties[ci].name, pct: function(k, v) { return sub[k] ? window.__pctInArray(sub[k], v) : null; }});
        }
        scopes.push({name: st.name, pct: function(k, v) { return d[k] ? window.__pctInArray(d[k], v) : null; }});
        scopes.push({name: 'Nation', pct: function(k, v) { return nationQ && nationQ[k] ? window.__pctInQuantiles(nationQ[k], v) : null; }});
        return {cell: cell, row: row, scopes: scopes};
      });
    }
  }

  // ---- layout: keep floating panels from covering each other ----------------------------------------------------
  // The line editor toolbar (left) and the stats panel (left) would overlap: the stats panel moves to the right of the
  // toolbar while the editor is open. The address box drops below the top bar on narrow windows.
  function layout() {
    var tb = document.getElementById('route-toolbar'), sp = document.getElementById('statsPanel'), ad = document.getElementById('addressSearch');
    if (tb && sp) {
      var open = tb.classList.contains('rt-open') && getComputedStyle(tb).display !== 'none';
      var left = open ? (Math.round(tb.getBoundingClientRect().right) + 10) + 'px' : '10px';
      sp.style.left = left;
      var sb = document.getElementById('statsButton');   // the round stats button sits under the toolbar on short windows
      if (sb) sb.style.left = left;
    }
    if (ad) { var narrow = window.innerWidth < 900; ad.style.top = narrow ? '56px' : '10px'; ad.style.left = narrow ? '10px' : 'calc(var(--tl-left, 0px) + 156px)'; }
  }
  setInterval(layout, 300);
  window.addEventListener('resize', layout);
  // Development overlay: flagged-cell borders only read at cell scale (nationwide they merge into a solid wash and are costly to draw).
  if (CFG.dev_min_zoom) {
    var applyDevZoom = function() {
      map.getStyle().layers.forEach(function(l) { if (/development/.test(l.id)) map.setLayerZoomRange(l.id, Math.max(CFG.dev_min_zoom, window.__devMinZoom || 0), 24); });
    };
    if (map.isStyleLoaded()) applyDevZoom(); else map.once('load', applyDevZoom);
  }
  fetch(CFG.dir + '/index.json').then(function(r) { return r.json(); }).then(function(idx) {
    INDEX = idx; idx.states.forEach(function(s) { byKey[s.key] = s; }); (idx.metros || []).forEach(function(m) { byMetro[m.key] = m; });
    init();
    installCellProvider();
    // Deep link: /<...>/Massachusetts, /<...>/new_york or ?state=Texas opens the map on that state.
    var wanted = (new URLSearchParams(location.search).get('state') || decodeURIComponent(location.pathname.split('/').filter(Boolean).pop() || ''))
      .toLowerCase().replace(/[^a-z]/g, '');
    var hit = wanted && idx.states.find(function(x) { return x.key.toLowerCase().replace(/[^a-z]/g, '') === wanted || x.name.toLowerCase().replace(/[^a-z]/g, '') === wanted; });
    if (hit) {
      scope.primary = blank(true); scope.primary.state = hit.key; scope.manual = true;
      map.fitBounds([[hit.bbox[0], hit.bbox[1]], [hit.bbox[2], hit.bbox[3]]], {padding: 40, duration: 0});
      pinned = hit.key;
      ['mousedown', 'wheel', 'touchstart'].forEach(function(ev) {
        map.getCanvas().addEventListener(ev, function() { pinned = null; }, {once: true, passive: true});
      });
    }
    map.on('moveend', onView);
    onView();
  }).catch(function(e) { console.warn('stats scope index failed', e); });
})();
"""


ADDRESS_HTML = """
<form id="addressSearch" autocomplete="off" style="position:absolute;top:10px;left:150px;z-index:7;display:flex;gap:4px;
  background:white;padding:4px 6px;border-radius:20px;box-shadow:0 1px 6px rgba(0,0,0,.25);font:12px sans-serif;align-items:center;">
  <input id="addressInput" type="search" placeholder="Search an address or place..." style="border:none;outline:none;width:230px;font:12px sans-serif;padding:3px 6px;">
  <button type="submit" style="border:none;background:#2b6cb0;color:#fff;border-radius:14px;padding:4px 10px;cursor:pointer;font:12px sans-serif;">Go</button>
  <span id="addressMsg" style="color:#b00;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
</form>
"""

ADDRESS_JS = r"""
(function() {
  var DEF_ZOOM = __ZOOM__, DEF_COUNTRY = '__COUNTRY__', map = window.__mainMap, marker = null, timer = null, seq = 0;
  var form = document.getElementById('__FORM__'), input = document.getElementById('__INPUT__'), msg = document.getElementById('__MSG__');
  if (!form || !input || !map) return;
  // a page can override the defaults at any time (the US map: zoom 6, US results only)
  function zoom() { return window.__addressZoom || DEF_ZOOM; }
  function country() { return window.__addressCountry !== undefined ? window.__addressCountry : DEF_COUNTRY; }
  var list = document.createElement('div');
  list.style.cssText = 'display:none;position:absolute;left:0;right:0;top:100%;margin-top:4px;background:#fff;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.3);z-index:20;max-height:240px;overflow-y:auto;font:12px sans-serif;min-width:260px;';
  if (getComputedStyle(form).position === 'static') form.style.position = 'relative';
  form.appendChild(list);
  function hide() { list.style.display = 'none'; list.innerHTML = ''; }
  function clearMarker() { if (marker) { marker.remove(); marker = null; } }
  function goTo(lng, lat, label) {
    msg.textContent = ''; hide(); clearMarker();
    marker = new maplibregl.Marker({color: '#d62728'}).setLngLat([lng, lat]).setPopup(new maplibregl.Popup({offset: 24}).setText(label || '')).addTo(map);
    map.flyTo({center: [lng, lat], zoom: zoom(), duration: 1200});
    if (window.__openCellPopup) {   // then open the popup (all stats shown) of the res-8 hexagon the address is on
      var fired = false, show = function() { if (fired) return; fired = true; window.__openCellPopup(lng, lat, {expanded: true, title: label ? 'Address: ' + label : null}); };
      map.once('moveend', show); setTimeout(show, 1800);
    }
  }
  function labelOf(f) {
    var p = f.properties || {}, street = [p.street, p.housenumber].filter(Boolean).join(' ');
    var parts = [p.name, street && street !== p.name ? street : null, p.city || p.district, p.state, p.country].filter(Boolean);
    return parts.filter(function(x, i) { return parts.indexOf(x) === i; }).join(', ');
  }
  function suggest(q) {
    var id = ++seq;
    var url = 'https://photon.komoot.io/api/?limit=6&q=' + encodeURIComponent(q) + (country() === 'us' ? '&bbox=-180,15,-60,72' : '');
    fetch(url).then(function(r) { return r.json(); }).then(function(res) {
      if (id !== seq || !input.value.trim()) return;
      var feats = (res.features || []).filter(function(f) { return country() !== 'us' || !f.properties || f.properties.countrycode === 'US'; });
      if (!feats.length) { hide(); return; }
      list.innerHTML = '';
      feats.forEach(function(f) {
        var d = document.createElement('div'), t = labelOf(f);
        d.textContent = t; d.style.cssText = 'padding:6px 10px;cursor:pointer;border-bottom:1px solid #eee;white-space:normal;color:#222;';
        d.addEventListener('mouseenter', function() { d.style.background = '#e6effa'; });
        d.addEventListener('mouseleave', function() { d.style.background = ''; });
        d.addEventListener('mousedown', function(e) { e.preventDefault(); input.value = t; goTo(f.geometry.coordinates[0], f.geometry.coordinates[1], t); });
        list.appendChild(d);
      });
      list.style.display = 'block';
    }).catch(function() { hide(); });
  }
  input.addEventListener('input', function() {
    var q = input.value.trim();
    clearTimeout(timer);
    if (!q) { seq++; hide(); clearMarker(); msg.textContent = ''; return; }   // cleared box: marker goes away
    if (q.length < 3) { hide(); return; }
    timer = setTimeout(function() { suggest(q); }, 250);
  });
  input.addEventListener('keydown', function(e) { if (e.key === 'Escape') hide(); });
  input.addEventListener('blur', function() { setTimeout(hide, 150); });
  form.addEventListener('submit', function(e) {
    e.preventDefault(); hide();
    var q = input.value.trim();
    if (!q) { clearMarker(); msg.textContent = ''; return; }
    msg.style.color = '#666'; msg.textContent = '...';
    fetch('https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1' + (country() ? '&countrycodes=' + country() : '') + '&q=' + encodeURIComponent(q), {headers: {'Accept': 'application/json'}})
      .then(function(r) { return r.json(); })
      .then(function(res) {
        if (!res.length) { msg.style.color = '#b00'; msg.textContent = 'Not found'; return; }
        goTo(parseFloat(res[0].lon), parseFloat(res[0].lat), res[0].display_name);
      })
      .catch(function() { msg.style.color = '#b00'; msg.textContent = 'Search failed'; });
  });
})();
"""


def address_js(zoom, form="addressSearch", inp="addressInput", msg="addressMsg", country="us") -> str:
    """The address search script (suggestions as you type via Photon, OSM Nominatim on Go, marker cleared with the box).

    Args:
        zoom: Zoom after a search.
        form: Form element id.
        inp: Input element id.
        msg: Message element id.
        country: Country code that biases the search.
    """
    return (ADDRESS_JS.replace("__ZOOM__", str(zoom)).replace("__COUNTRY__", country)
            .replace("__FORM__", form).replace("__INPUT__", inp).replace("__MSG__", msg))


def replace_scope_js_in_saved_html(path: str, cfg: dict) -> None:
    """Development helper: swap the scope + address scripts of an already built page for the current ones (no rebuild).

    Args:
        path: Built page.
        cfg: Scope configuration.

    Raises:
        ValueError: If the page has no scope script.
    """
    import re

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    new = SCOPE_JS.replace("__CFG__", json.dumps(cfg))
    html, n = re.subn(r"\n\(function\(\) \{\n  var CFG = \{.*?\n\}\)\(\);\n", lambda m: new, html, count=1, flags=re.S)
    if n != 1:
        raise ValueError("scope script not found")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def inject_scope_into_saved_html(path: str, cfg: dict) -> str:
    """Replace the Metro/City-core buttons and per-tab compare logic of the saved page's stats panel with the scope bar.

    Args:
        path: Built page.
        cfg: Scope configuration.

    Returns:
        The path.

    Raises:
        ValueError: If the page has no stats panel.
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    marker_row = '<div id="statsAreaBtnRow"'
    if marker_row not in html:
        raise ValueError(f"{path}: stats panel not found (inject it first)")
    html = html.replace(marker_row, SCOPE_BAR_HTML + marker_row, 1)
    body_marker = "</body>"
    try:   # the nation's display name (the scope bar's "Nation" label) follows the data: US page vs Spain page
        import os as _os0

        _idx = json.load(open(_os0.path.join(_os0.path.dirname(_os0.path.abspath(path)), cfg["dir"], "index.json")))
        cfg.setdefault("nation_name", "United States" if "Alabama" in [st["name"] for st in _idx["states"]] else "Spain")
    except Exception:  # noqa: BLE001
        pass
    block = "<script>\n" + SCOPE_JS.replace("__CFG__", json.dumps(cfg)) + "\n</script>\n"
    try:   # Ranking tab (state / county / place units); the table is computed once from the stats chunks next to the page
        import os as _os
        from .ranking_tab import ranking_script

        chunk_dir = _os.path.join(_os.path.dirname(_os.path.abspath(path)), cfg["dir"])
        names = [st["name"] for st in json.load(open(_os.path.join(chunk_dir, "index.json")))["states"]]
        block += ranking_script(chunk_dir, cfg["dir"], "us" if "Alabama" in names else "spain")
    except Exception as exc:  # noqa: BLE001
        print(f"[scope_panel] ranking tab skipped: {exc!r}")
    try:   # the res-9 popup files, from the per-state sources the stats-chunk step wrote (transitlos.map.popup_cells)
        import os as _os3

        from .popup_cells import build_popup_from_sources

        _cd = _os3.path.join(_os3.path.dirname(_os3.path.abspath(path)), cfg["dir"])
        if _os3.path.isdir(_os3.path.join(_cd, "popup_src")):
            print(f"[scope_panel] popup9: {build_popup_from_sources(_cd)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[scope_panel] popup9 skipped: {exc!r}")
    try:   # dotted metropolitan-area boundaries (own checkbox), when the study wrote `metro_lines.geojson` next to the page
        import os as _os2

        if _os2.path.isfile(_os2.path.join(_os2.path.dirname(_os2.path.abspath(path)), "metro_lines.geojson")):
            from .metro_areas import metro_lines_script

            block += metro_lines_script("metro_lines.geojson")
    except Exception as exc:  # noqa: BLE001
        print(f"[scope_panel] metro lines skipped: {exc!r}")
    html = html.replace(body_marker, block + body_marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
