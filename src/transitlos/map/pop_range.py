"""Shared population range filter for the Ranking / Place-rank tabs.

`POP_RANGE_JS` defines `window.__popRange(host, opts)`: two editable drop-down boxes (Min / Max population) with the presets
0, 1k, 5k, 10k, 50k, 100k, 500k (Max also "No limit"); the user can open the list or type their own number (``5k``, ``2.5M``,
``12000``). `opts` = {max: domain maximum, lo, hi: current values (hi == null means "no upper limit"), onChange(lo, hi)}.
The returned object has `set(max, lo, hi)`.
"""

POP_RANGE_JS = r"""
window.__popRange = window.__popRange || function(host, opts) {
  var PRESETS = [0, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5];
  var dmax = Math.max(1, opts.max || 1), lo = opts.lo || 0, hi = opts.hi == null ? null : opts.hi;
  function fmt(v) {
    if (v >= 1e6) return (+(v / 1e6).toFixed(2)) + 'M';
    if (v >= 1e3) return (+(v / 1e3).toFixed(1)) + 'k';
    return String(Math.round(v));
  }
  function parse(s) {
    var m = String(s).trim().toLowerCase().replace(/[, ]/g, '').match(/^(\d*\.?\d+)\s*([km]?)/);
    if (!m) return null;
    return parseFloat(m[1]) * (m[2] === 'k' ? 1e3 : m[2] === 'm' ? 1e6 : 1);
  }
  if (!document.getElementById('prStyle')) {
    var st = document.createElement('style'); st.id = 'prStyle';
    st.textContent =
      '.prWrap{font:12px system-ui,sans-serif;color:#374151;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:6px 8px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;}' +
      '.prWrap .prLbl{font-weight:600;color:#111827;}' +
      '.prWrap .prSub{color:#6b7280;font-size:11px;}' +
      '.prBox{position:relative;display:inline-flex;align-items:center;background:#fff;border:1px solid #cbd5e1;border-radius:6px;height:26px;transition:border-color .12s,box-shadow .12s;}' +
      '.prBox:focus-within{border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.2);}' +
      '.prBox input{width:64px;border:0;outline:0;background:transparent;padding:0 2px 0 8px;font:12px system-ui,sans-serif;color:#111827;text-align:right;height:24px;}' +
      '.prBox button{border:0;background:transparent;cursor:pointer;color:#6b7280;padding:0 7px;height:24px;font-size:9px;border-left:1px solid #eef0f3;}' +
      '.prBox button:hover{color:#111827;background:#f3f4f6;border-radius:0 5px 5px 0;}' +
      '.prMenu{position:absolute;z-index:99999;top:28px;left:0;min-width:100%;background:#fff;border:1px solid #d1d5db;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,.15);padding:4px;display:none;}' +
      '.prMenu div{padding:4px 10px;cursor:pointer;border-radius:5px;text-align:right;white-space:nowrap;}' +
      '.prMenu div:hover{background:#eff6ff;color:#1d4ed8;}' +
      '.prMenu div.sel{font-weight:600;color:#1d4ed8;}' +
      '.prReset{border:1px solid #cbd5e1;background:#fff;border-radius:6px;height:26px;padding:0 8px;cursor:pointer;color:#4b5563;font:12px system-ui,sans-serif;}' +
      '.prReset:hover{background:#f3f4f6;color:#111827;}';
    document.head.appendChild(st);
  }
  host.innerHTML =
    '<div class="prWrap"><span class="prLbl">Population</span>' +
    '<span class="prSub">min</span><span class="prBox" data-k="lo"><input type="text" inputmode="decimal" aria-label="Minimum population"><button type="button" tabindex="-1" aria-label="Show options">&#9660;</button><div class="prMenu"></div></span>' +
    '<span class="prSub">max</span><span class="prBox" data-k="hi"><input type="text" inputmode="decimal" aria-label="Maximum population"><button type="button" tabindex="-1" aria-label="Show options">&#9660;</button><div class="prMenu"></div></span>' +
    '<button type="button" class="prReset" title="Reset">Reset</button></div>';
  var boxes = {}, closeAll = function() { Object.keys(boxes).forEach(function(k) { boxes[k].menu.style.display = 'none'; }); };
  ['lo', 'hi'].forEach(function(k) {
    var el = host.querySelector('.prBox[data-k="' + k + '"]');
    boxes[k] = {el: el, inp: el.querySelector('input'), btn: el.querySelector('button'), menu: el.querySelector('.prMenu')};
  });
  function emit() { opts.onChange(lo, hi); }
  function paint() {
    if (document.activeElement !== boxes.lo.inp) boxes.lo.inp.value = fmt(lo);
    if (document.activeElement !== boxes.hi.inp) boxes.hi.inp.value = hi == null ? 'No limit' : fmt(hi);
  }
  function items(k) {
    var list = PRESETS.filter(function(v) { return v > 0 || k === 'lo'; }).map(function(v) { return {v: v, t: fmt(v)}; });
    if (k === 'hi') list.push({v: null, t: 'No limit'});
    return list;
  }
  function choose(k, v) {
    if (k === 'lo') { lo = v; if (hi != null && lo > hi) hi = lo; }
    else { hi = v; if (hi != null && hi < lo) lo = hi; }
    closeAll(); paint(); emit();
  }
  function fillMenu(k) {
    var cur = k === 'lo' ? lo : hi;
    boxes[k].menu.innerHTML = items(k).map(function(it, i) {
      return '<div data-i="' + i + '"' + (it.v === cur ? ' class="sel"' : '') + '>' + it.t + '</div>';
    }).join('');
  }
  ['lo', 'hi'].forEach(function(k) {
    var b = boxes[k];
    b.btn.addEventListener('click', function(e) {
      e.stopPropagation(); var open = b.menu.style.display === 'block'; closeAll();
      if (!open) { fillMenu(k); b.menu.style.display = 'block'; }
    });
    b.menu.addEventListener('mousedown', function(e) { e.preventDefault(); });
    b.menu.addEventListener('click', function(e) {
      var d = e.target.closest('[data-i]'); if (d) choose(k, items(k)[+d.getAttribute('data-i')].v);
    });
    b.inp.addEventListener('focus', function() { b.inp.select(); });
    b.inp.addEventListener('keydown', function(e) { if (e.key === 'Enter') b.inp.blur(); else if (e.key === 'Escape') { paint(); b.inp.blur(); } });
    b.inp.addEventListener('change', function() {
      var v = parse(b.inp.value);
      if (k === 'hi' && (v === null || v >= dmax)) choose('hi', null);
      else if (v === null) paint();
      else choose(k, Math.round(v));
    });
    b.inp.addEventListener('blur', paint);
  });
  document.addEventListener('click', closeAll);
  host.querySelector('.prReset').addEventListener('click', function() { lo = 0; hi = null; paint(); emit(); });
  paint();
  return { set: function(max, l, h) { dmax = Math.max(1, max || 1); lo = l || 0; hi = h == null ? null : h; paint(); } };
};
"""
