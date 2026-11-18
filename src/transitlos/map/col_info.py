"""Small "i" buttons next to every column name shown on a MapLibre page (census panel, selectors, regression, distribution, popups).

`COL_INFO_JS` reads `window.__columnMeta` (`{column: {name, source, level, description, unit, original}}`, emitted by `col_info_script`
from the same rows as the Metadata tab, see `transitlos.map.build._column_metadata_rows`). A MutationObserver scans the panels that show
column names (stats panel, popups, selectors, legend) and appends a tiny "i" to every element whose text is exactly a known column label; a
click opens a small floating panel with the column's source, native level and description. A label with no metadata simply gets no button.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable

# Derived columns that have no census source but do appear as selectable / plotted names.
_EXTRA_META = {
    "level_of_service": {"name": "Level of service", "source": "Computed by this study (transitLOS)", "level": "H3 cell",
                         "description": "Composite 0-100 score of how well public transport serves a place: speed, headway and walking access combined."},
    "population": {"name": "Population", "source": "Census / WorldPop (as available for the study area)", "level": "see the Metadata tab",
                   "description": "Residents counted in the cell or area."},
}


def column_meta_json(metadata_rows: Iterable[Dict], extra: Dict[str, Dict] = None) -> str:
    """Column metadata (name, source, levels, year, description...) as JSON for the info buttons.

    Args:
        metadata_rows: One dict per column (``column``, ``name``, ``source``, ``level``...).
        extra: Extra ``{column: metadata}`` for shown columns without a row.

    Returns:
        JSON text.
    """
    out: Dict[str, Dict] = {k: dict(v) for k, v in _EXTRA_META.items()}
    for k, v in (extra or {}).items():   # shown columns without a metadata row (e.g. shares folded into their count's row)
        out[k] = dict(v)
    for r in metadata_rows or []:
        col = r.get("column")
        if not col:
            continue
        native = r.get("native_levels") or ([r["level"]] if r.get("level") else [])
        out[col] = {
            "name": str(r.get("name") or col),
            "source": str(r.get("source") or ""),
            "level": ", ".join(str(x) for x in native),
            "description": str(r.get("description") or ""),
            "unit": str(r.get("unit") or ""),
            "original": str(r.get("original_name") or ""),
        }
    return json.dumps(out, ensure_ascii=False).replace("</", "<\\/")


COL_INFO_JS = r"""
(function() {
  if (window.__colInfoInit) return;
  window.__colInfoInit = true;
  var META = window.__columnMeta || {};
  var ALLOWED = '#statsPanel, .maplibregl-popup, #layer-switcher, #top-center-bar, #downloadPanel';
  var st = document.createElement('style');
  st.textContent =
    '.colInfoBtn{display:inline-block;box-sizing:border-box;width:13px;height:13px;margin-left:4px;border:1px solid #9aa3ad;border-radius:50%;' +
    'font:italic 600 9px/11px Georgia,serif;text-align:center;color:#6b7580;background:#fff;cursor:pointer;vertical-align:middle;opacity:.7;' +
    'font-style:italic;text-indent:0;white-space:normal;letter-spacing:0;text-transform:none}' +
    '.colInfoBtn:hover{opacity:1;color:#2b6cb0;border-color:#2b6cb0}' +
    '.colInfoBtn.colInfoAbs{position:absolute;right:20px;top:50%;margin-top:-6px;background:#fff}' +
    '#colInfoPanel{position:fixed;z-index:30000;max-width:290px;min-width:200px;background:#fff;border:1px solid #c9ced6;border-radius:6px;' +
    'box-shadow:0 3px 12px rgba(0,0,0,.28);padding:8px 10px;font:12px/1.35 sans-serif;color:#222;display:none}' +
    '#colInfoPanel .ciT{font-weight:600;margin-bottom:4px;padding-right:14px}#colInfoPanel .ciX{position:absolute;right:6px;top:3px;cursor:pointer;color:#888;font-size:14px}' +
    '#colInfoPanel .ciK{color:#777;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em;margin-top:5px}#colInfoPanel .ciD{color:#444}';
  document.head.appendChild(st);

  function norm(t) { return (t || '').toLowerCase().replace(/\s+/g, ' ').replace(/[:*]+$/, '').trim(); }
  function strip(t) { return t.replace(/\s*\([^)]*\)\s*$/, '').trim(); }
  var byLabel = {};
  function index() {
    META = window.__columnMeta || META;
    Object.keys(META).forEach(function(col) {
      var m = META[col];
      [m.name, strip(m.name || ''), col, m.original].forEach(function(k) {
        var raw = k; k = norm(k); if (k && !byLabel[k]) byLabel[k] = col;
        // translated display labels (e.g. the Spanish switch): canonical column -> every translation of its English label
        var D = window.__I18N && window.__I18N.es;
        if (D && raw) { var t = D[k] !== undefined ? D[k] : D[raw]; if (t) { t = norm(t); if (t && !byLabel[t]) byLabel[t] = col; } }
      });
    });
  }
  var __idx0 = index;
  index = function() {   // also: every Python display label, and the label without a trailing "share" (rows show "Female (%)" for "Female share")
    __idx0();
    var L = window.__PY_FIELD_LABELS || {};
    function add(label, col) {
      var base = norm(label); if (!base) return;
      [base, base.replace(/ share$/, ''), base.replace(/ share$/, ' (%)')].forEach(function(k) { if (k && !byLabel[k]) byLabel[k] = col; });
    }
    Object.keys(L).forEach(function(col) { add(L[col], col); });
    var LL = window.__LOOKUP_LABELS || {};   // the exact (disambiguated) labels the tab rows show
    Object.keys(LL).forEach(function(col) { var k = norm(LL[col]); if (k) byLabel[k] = col; });
    Object.keys(META).forEach(function(col) { add(META[col].name || '', col); });
  };
  index();
  var STOP = {share: 1, rate: 1, '%': 1, '(%)': 1, the: 1, of: 1, and: 1, mean: 1, median: 1};
  function toks(t) {
    var o = {}, n = 0;
    norm(t).replace(/[(),:·\/%]+/g, ' ').split(' ').forEach(function(x) { if (x && !STOP[x]) { o[x] = 1; n++; } });
    return {set: o, n: n};
  }
  var fuzzyCache = null;
  function fuzzy(n) {   // reworded row labels ("Turnout, national election" vs "General (national) election: turnout"): unique best token overlap
    if (!fuzzyCache) { fuzzyCache = Object.keys(META).map(function(col) { return [col, toks(META[col].name || '')]; }); }
    var a = toks(n); if (a.n < 2) return null;
    var best = 0, second = 0, bc = null;
    fuzzyCache.forEach(function(e) {
      var b = e[1], inter = 0; Object.keys(a.set).forEach(function(k) { if (b.set[k]) inter++; });
      var j = inter / Math.min(a.n, b.n);   // containment: the shorter wording is (nearly) all inside the longer one
      if (j > best) { second = best; best = j; bc = e[0]; } else if (j > second) second = j;
    });
    return (best >= 0.8 && best - second >= 0.15) ? bc : null;
  }
  function lookup(text) {
    var n = norm(text);
    if (!n || n.length < 3 || n.length > 90) return null;
    var c = byLabel[n] || byLabel[strip(n)];
    if (c) return c;
    var c2 = fuzzy(n); if (c2) return c2;
    var w = n.split(' ');   // a leading source word ("WorldPop Population density") in front of a known label
    for (var k = 1; k <= 2 && k < w.length; k++) {
      var rest = w.slice(k).join(' ');
      c = byLabel[rest] || byLabel[strip(rest)];
      if (c) return c;
    }
    return null;
  }
  function ownText(el) {
    var t = '';
    for (var i = 0; i < el.childNodes.length; i++) if (el.childNodes[i].nodeType === 3) t += el.childNodes[i].nodeValue;
    return t;
  }
  function decorate(el) {
    if (el.nodeType !== 1 || el.closest('#colInfoPanel, script, style, select, option, input, textarea, .statsTabBtn, .colInfoBtn, .__csel-wrap, .__csel-list, thead, th, #mapLegend, #placeRankList')) return;
    if (!el.hasAttribute('data-info-col') && el.closest('[data-info-col]')) return;   // inside a row that names its column: that row's button only
    var forced = el.getAttribute('data-info-col');   // list rows that name their column (regression / ANOVA lists): no text matching needed
    if (forced) {
      if (el.querySelector(':scope > .colInfoBtn')) return;
      var fb = document.createElement('span');
      fb.className = 'colInfoBtn'; fb.textContent = 'i'; fb.title = 'About this column'; fb.setAttribute('data-col', forced);
      var firstB = null;   // after the label (before the value in brackets) when the row is "label <b>(value)</b>"
      for (var j = 0; j < el.childNodes.length; j++) { var cn = el.childNodes[j]; if (cn.nodeType === 1 && /^\(/.test(cn.textContent || '')) { firstB = cn; break; } }
      if (firstB) el.insertBefore(fb, firstB); else el.appendChild(fb);
      return;
    }
    for (var i = 0; i < el.children.length; i++) if (!el.children[i].classList.contains('colInfoBtn') && !/^(B|I|STRONG|EM|SPAN)$/.test(el.children[i].tagName)) return;
    if (el.querySelector(':scope > .colInfoBtn')) return;
    var text = ownText(el);
    if (!text.trim() && el.children.length === 1 && el.firstElementChild.children.length === 0) text = el.firstElementChild.textContent;   // <td><b>label</b></td>
    var col = lookup(text);
    if (!col) return;
    var b = document.createElement('span');
    b.className = 'colInfoBtn' + (el.classList.contains('__csel-btn') ? ' colInfoAbs' : '');
    b.textContent = 'i'; b.title = 'About this column'; b.setAttribute('data-col', col);
    el.appendChild(b);
  }
  function scan(root) {
    if (root.nodeType === 3) root = root.parentElement;
    if (!root || root.nodeType !== 1 || !root.closest(ALLOWED)) return;
    if (root.closest('#colInfoPanel')) return;
    var nodes = root.querySelectorAll('div,span,b,td,th,label,li,a,p,strong');
    decorate(root);
    for (var i = 0; i < nodes.length && i < 4000; i++) decorate(nodes[i]);
  }
  var pending = [], timer = null;
  var mo = new MutationObserver(function(muts) {
    muts.forEach(function(m) {
      if (m.target.closest && m.target.closest('#colInfoPanel')) return;
      m.addedNodes.forEach(function(n) {
        if (n.nodeType === 1 && n.classList.contains('colInfoBtn')) return;
        pending.push(n);
      });
    });
    if (pending.length && !timer) timer = setTimeout(function() {
      var list = pending; pending = []; timer = null;
      if (!Object.keys(byLabel).length || (window.__columnMeta && window.__columnMeta !== META)) index();
      list.forEach(function(n) { try { scan(n); } catch (e) {} });
    }, 120);
  });
  mo.observe(document.body, {childList: true, subtree: true});
  setTimeout(function() { index(); document.querySelectorAll(ALLOWED).forEach(scan); }, 600);
  window.__colInfoScan = function() { index(); document.querySelectorAll(ALLOWED).forEach(scan); };

  // ---- panel ----
  var panel = null, openFor = null;
  function closePanel() { if (panel) panel.style.display = 'none'; openFor = null; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function(c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  function show(btn) {
    var col = btn.getAttribute('data-col'), m = META[col] || {};
    if (document.documentElement.lang === 'es' && window.__columnMetaEs && window.__columnMetaEs[col]) m = window.__columnMetaEs[col];
    if (!panel) { panel = document.createElement('div'); panel.id = 'colInfoPanel'; document.body.appendChild(panel); }
    var none = '&mdash;';
    panel.innerHTML = '<span class="ciX" title="Close">&times;</span><div class="ciT">' + esc(m.name || btn.getAttribute('data-col')) + '</div>' +
      '<div class="ciK">Source</div><div class="ciD">' + (m.source ? esc(m.source) : none) + '</div>' +
      '<div class="ciK">Native level</div><div class="ciD">' + (m.level ? esc(m.level) : none) + '</div>' +
      '<div class="ciK">Description</div><div class="ciD">' + (m.description ? esc(m.description) : none) + '</div>';
    panel.style.display = 'block';
    var r = btn.getBoundingClientRect(), w = panel.offsetWidth, h = panel.offsetHeight;
    var x = Math.min(Math.max(6, r.left), window.innerWidth - w - 6), y = r.bottom + 6;
    if (y + h > window.innerHeight - 6) y = Math.max(6, r.top - h - 6);
    panel.style.left = x + 'px'; panel.style.top = y + 'px';
    openFor = btn;
  }
  document.addEventListener('click', function(e) {
    var b = e.target.closest && e.target.closest('.colInfoBtn');
    if (b) { e.stopImmediatePropagation(); e.preventDefault(); if (openFor === b && panel && panel.style.display === 'block') closePanel(); else show(b); return; }
    if (e.target.closest && e.target.closest('.ciX')) { closePanel(); return; }
    if (panel && panel.style.display === 'block' && !e.target.closest('#colInfoPanel')) closePanel();
  }, true);
  ['mousedown', 'mouseup', 'pointerdown', 'touchstart'].forEach(function(ev) {
    document.addEventListener(ev, function(e) { if (e.target.closest && e.target.closest('.colInfoBtn')) e.stopPropagation(); }, true);
  });
  document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closePanel(); });
})();
"""


def col_info_script(metadata_rows: Iterable[Dict], extra: Dict[str, Dict] = None, labels: Dict[str, str] = None, lookup_labels: Dict[str, str] = None) -> str:
    """`<script>` block: the metadata lookup plus the info-button machinery (runs once per page).

    `labels` is `{column: display name}` computed in Python so the JS panels show exactly the same readable names.

    Args:
        metadata_rows: One dict per column.
        extra: Extra ``{column: metadata}``.
        labels: ``{column: display name}``.
        lookup_labels: ``{column: label}`` for coded values.
    """
    labels_json = json.dumps(labels or {}, ensure_ascii=False).replace("</", "<\\/")
    lookup_json = json.dumps(lookup_labels or {}, ensure_ascii=False).replace("</", "<\\/")
    return f"<script>\nwindow.__PY_FIELD_LABELS = {labels_json};\nwindow.__LOOKUP_LABELS = {lookup_json};\nwindow.__columnMeta = {column_meta_json(metadata_rows, extra)};\n</script>\n<script>{COL_INFO_JS}</script>\n"
