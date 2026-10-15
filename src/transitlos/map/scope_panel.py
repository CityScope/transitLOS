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
  <div style="margin-top:6px;font:12px sans-serif;">Scenario
    <select id="scopeScenarioSelect" style="margin-left:4px;min-width:130px;"></select>
  </div>
</div>
"""

SCOPE_JS = r"""
(function() {
  var CFG = __CFG__;
  var map = window.__mainMap;
  var INDEX = null, byKey = {}, stateData = {}, pending = {}, subsetCache = {}, lru = [];
  var MAX_STATES = 4;   // state tables kept in memory (the nation table is always kept)
  var scope = {
    primary: {on: true, state: null, county: null, place: null},
    compare: {on: false, state: null, county: null, place: null},
    manual: false,
  };
  var pickerOpen = {primary: null, compare: null};
  window.__scopeActive = true;

  function blank(on) { return {on: on, state: null, county: null, place: null}; }
  function levelOf(s) {
    if (!s.on) return 'none';
    if (s.place != null) return 'place';
    if (s.county != null) return 'county';
    if (s.state) return 'state';
    return 'nation';
  }
  function labelOf(s) {
    var lv = levelOf(s), st = s.state && byKey[s.state];
    if (lv === 'none') return 'None';
    if (lv === 'nation') return 'United States';
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
    [scope.primary, scope.compare].forEach(function(s) { if (s.on && s.state) inUse[s.state] = true; });
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
    if (lv === 'nation') { fetchState('Nation'); return stateData['Nation'] || window.__statsData.metro; }  // res-7 sample once loaded
    fetchState(s.state);
    if (!stateData[s.state]) return null;
    touch(s.state);
    if (lv === 'state') return stateData[s.state];
    return subset(s.state, lv, lv === 'county' ? s.county : s.place);
  }
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
    var html = '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px;">';
    var btn = function(id, text, active, disabled, title) {
      return '<button type="button" data-act="' + id + '" ' + (disabled ? 'disabled ' : '') + 'title="' + (title || '') + '" ' +
        'style="padding:4px 8px;border:1px solid #c9ced6;border-radius:4px;font:12px sans-serif;cursor:' + (disabled ? 'not-allowed' : 'pointer') + ';' +
        'background:' + (active ? '#2b6cb0' : '#f5f6f8') + ';color:' + (active ? '#fff' : (disabled ? '#aaa' : '#333')) + ';">' + text + '</button>';
    };
    if (side === 'compare') html += btn('none', 'None', lv === 'none', false);
    html += btn('nation', 'Nation', lv === 'nation', false);
    html += btn('state', 'State', !!s.state && s.on, false);
    html += btn('county', 'County', lv === 'county', !s.state || !s.on, 'select a state first');
    html += btn('place', 'Place', lv === 'place', !s.state || !s.on, 'select a state first');
    html += '</div>';
    if (kind) {
      html += '<input type="search" class="scopeFilter" autocomplete="off" placeholder="Search ' + kind + ' (type a name)..." ' +
        'style="width:100%;box-sizing:border-box;padding:6px;border:1px solid #c9ced6;border-radius:4px;font:12px sans-serif;">' +
        '<div class="scopeResults" style="margin-top:4px;"></div><div class="scopeHint" style="color:#888;font:11px sans-serif;margin-top:3px;"></div>';
    }
    pop.innerHTML = html;
    var f = pop.querySelector('.scopeFilter');
    if (f) {
      var items;
      if (kind === 'state') items = INDEX.states.map(function(x) { return {i: x.key, name: x.name, pop: x.pop || 0}; });
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
            (n === cur ? 'background:#e6effa;' : '') + '"><span>' + it.name + '</span><span style="color:#999;">' + (it.pop ? fmtPop(it.pop) : '') + '</span></div>';
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
    if (kind === 'state') { s.state = value; s.county = null; s.place = null; }
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
    if (act === 'none') { scope[side] = blank(false); pickerOpen[side] = null; }
    else if (act === 'nation') { scope[side] = blank(true); pickerOpen[side] = null; }
    else if (act === 'state') {
      if (s.state && s.on) { scope[side] = blank(true); pickerOpen[side] = null; }   // clicking an active state deactivates it
      else { pickerOpen[side] = 'state'; }
    } else if (act === 'county') {
      if (s.county != null) { s.county = null; pickerOpen[side] = null; } else { pickerOpen[side] = 'county'; }
    } else if (act === 'place') {
      if (s.place != null) { s.place = null; pickerOpen[side] = null; } else { pickerOpen[side] = 'place'; }
    }
    renderPop(side); rerender();
  }
  function renderButtons() {
    document.querySelector('.scopeSide[data-side="primary"] .scopeMainBtn').innerHTML = 'Level: <b>' + labelOf(scope.primary) + '</b> &#9662;';
    document.querySelector('.scopeSide[data-side="compare"] .scopeMainBtn').innerHTML = 'Compare with: <b>' + labelOf(scope.compare) + '</b> &#9662;';
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
  function onView() {
    if (!INDEX) return;
    var z = map.getZoom(), v = statesInView();
    var single = z >= CFG.min_zoom && v.dominant && v.share >= 0.5;
    var cls = single ? 'single:' + v.dominant : 'multi';
    if (cls === lastClass) return;
    var wasSingle = lastClass && lastClass.indexOf('single') === 0;
    lastClass = cls;
    if (!single) {                       // low zoom / several states: nation, no comparison
      scope.primary = blank(true); scope.compare = blank(false); scope.manual = false;
      pickerOpen = {primary: null, compare: null};
    } else if (!scope.manual || !wasSingle) {
      scope.primary = blank(true); scope.primary.state = v.dominant; scope.manual = false;
    }
    setEditAllowed(!!single);
    ['primary', 'compare'].forEach(renderPop);
    rerender();
  }

  // ---- scenario ------------------------------------------------------------------------------------------------
  function syncScenario() {
    var top = document.getElementById('topBarScenarioSelect'), mine = document.getElementById('scopeScenarioSelect');
    if (!top || !mine) return;
    if (mine.getAttribute('data-html') !== top.innerHTML) { mine.innerHTML = top.innerHTML; mine.setAttribute('data-html', top.innerHTML); }
    if (mine.value !== top.value) mine.value = top.value;
    mine.querySelectorAll('option').forEach(function(o) { o.disabled = !editOk && o.value !== ''; });
  }
  setInterval(syncScenario, 600);

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
    });
    document.getElementById('scopeScenarioSelect').addEventListener('change', function(e) {
      var top = document.getElementById('topBarScenarioSelect');
      if (!top) return;
      top.value = e.target.value;
      top.dispatchEvent(new Event('change', {bubbles: true}));
      rerender();
    });
    document.querySelectorAll('.statsTabBtn').forEach(function(b) {
      b.addEventListener('click', function() {
        var hide = (b.dataset.tab === 'discretization' || b.dataset.tab === 'metadata');
        bar.style.display = hide ? 'none' : 'block';
      });
    });
    // The old per-tab "Compare with" popovers: scenario / area / place choices are replaced by this bar; only
    // Distribution keeps its "Column" overlay.
    ['reg', 'anova', 'dist'].forEach(function(p) {
      var wrap = document.querySelector('#' + p + 'ComparePopover');
      if (!wrap) return;
      wrap.querySelectorAll('select').forEach(function(sel) {
        if (sel.id !== p + 'OverlaySelect') sel.parentElement.style.display = 'none';
      });
      if (p !== 'dist' && wrap.parentElement) wrap.parentElement.style.display = 'none';
      if (p === 'dist') { var cb = document.querySelector('.compareWithBtn[data-target="distComparePopover"]'); if (cb) cb.innerHTML = 'Overlay column <span style="float:right;color:#888;">&#9662;</span>'; }
    });
    var oldRow = document.getElementById('statsAreaBtnRow');
    if (oldRow) oldRow.style.display = 'none';
    renderButtons(); syncScenario();
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
    if (ad) { var narrow = window.innerWidth < 900; ad.style.top = narrow ? '56px' : '10px'; ad.style.left = narrow ? '10px' : '150px'; }
  }
  setInterval(layout, 300);
  window.addEventListener('resize', layout);
  // Development overlay: flagged-cell borders only read at cell scale (nationwide they merge into a solid wash and are costly to draw).
  if (CFG.dev_min_zoom) {
    var applyDevZoom = function() {
      map.getStyle().layers.forEach(function(l) { if (/development/.test(l.id)) map.setLayerZoomRange(l.id, CFG.dev_min_zoom, 24); });
    };
    if (map.isStyleLoaded()) applyDevZoom(); else map.once('load', applyDevZoom);
  }
  fetch(CFG.dir + '/index.json').then(function(r) { return r.json(); }).then(function(idx) {
    INDEX = idx; idx.states.forEach(function(s) { byKey[s.key] = s; });
    init();
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
  var ZOOM = __ZOOM__, map = window.__mainMap, marker = null;
  var form = document.getElementById('addressSearch'), input = document.getElementById('addressInput'), msg = document.getElementById('addressMsg');
  if (!form || !map) return;
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    var q = input.value.trim();
    if (!q) { msg.textContent = ''; return; }
    msg.style.color = '#666'; msg.textContent = 'Searching...';
    fetch('https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=us&q=' + encodeURIComponent(q), {headers: {'Accept': 'application/json'}})
      .then(function(r) { return r.json(); })
      .then(function(res) {
        if (!res.length) { msg.style.color = '#b00'; msg.textContent = 'Not found'; return; }
        var lng = parseFloat(res[0].lon), lat = parseFloat(res[0].lat);
        msg.textContent = '';
        if (marker) marker.remove();
        marker = new maplibregl.Marker({color: '#d62728'}).setLngLat([lng, lat]).setPopup(new maplibregl.Popup({offset: 24}).setText(res[0].display_name)).addTo(map);
        map.flyTo({center: [lng, lat], zoom: ZOOM, duration: 1200});
      })
      .catch(function() { msg.style.color = '#b00'; msg.textContent = 'Search failed'; });
  });
})();
"""


def replace_scope_js_in_saved_html(path: str, cfg: dict) -> None:
    """Development helper: swap the scope + address scripts of an already built page for the current ones (no rebuild)."""
    import re

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    new = SCOPE_JS.replace("__CFG__", json.dumps(cfg))
    html, n = re.subn(r"\n\(function\(\) \{\n  var CFG = \{.*?\n\}\)\(\);\n", lambda m: new, html, count=1, flags=re.S)
    if n != 1:
        raise ValueError("scope script not found")
    if cfg.get("address_zoom"):
        addr = ADDRESS_JS.replace("__ZOOM__", str(cfg["address_zoom"]))
        html, n2 = re.subn(r"\n\(function\(\) \{\n  var ZOOM = .*?\n\}\)\(\);\n", lambda m: addr, html, count=1, flags=re.S)
        if n2 != 1:
            raise ValueError("address script not found")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def inject_scope_into_saved_html(path: str, cfg: dict) -> str:
    """Replace the Metro/City-core buttons and per-tab compare logic of the saved page's stats panel with the scope bar."""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    marker_row = '<div id="statsAreaBtnRow"'
    if marker_row not in html:
        raise ValueError(f"{path}: stats panel not found (inject it first)")
    html = html.replace(marker_row, SCOPE_BAR_HTML + marker_row, 1)
    body_marker = "</body>"
    block = "<script>\n" + SCOPE_JS.replace("__CFG__", json.dumps(cfg)) + "\n</script>\n"
    if cfg.get("address_zoom"):
        block = ADDRESS_HTML + block + "<script>\n" + ADDRESS_JS.replace("__ZOOM__", str(cfg["address_zoom"])) + "\n</script>\n"
    html = html.replace(body_marker, block + body_marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
