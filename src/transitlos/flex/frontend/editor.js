/* Scoring editors for the map: choose the SHAPE and the parameters of every scoring function and see the curve.
 *
 *   FlexEditor.build(container, params, onChange)   builds the whole 'Scoring' panel inside `container`;
 *                                                   `params` is the JSON of ScoreParams (meta.json "params"), edited in place;
 *                                                   onChange(params) is called after every edit (the viewer recomputes).
 *   FlexEditor.families                             the shape families (same formulas as transitlos.flex.shapes)
 *
 * Every curve has a family (default analytic | exponential | power | logistic | steps | linear | gaussian | hard buffer | custom points) with its own
 * parameters, plus a plot where, in 'custom' mode, points can be dragged (click the plot to add one, double-click a point to remove it).
 * A non-default family is sampled into a table (kind 'table', 512 points) -- exactly what the Python/C scoring evaluates.
 */
(function (root) {
  const GRID = {
    headway_curve: { lo: 1, hi: 720, scale: 'log', n: 512, xlabel: 'headway (min)', title: 'Headway score H', ymax: 1 },
    speed_curve: { lo: 0, hi: 150, scale: 'lin', n: 512, xlabel: 'speed (km/h)', title: 'Speed score V', ymax: 1.2 },
    walk_curve: { lo: 0, hi: 40, scale: 'lin', n: 1024, xlabel: 'walk u = (t - plateau) / t0', title: 'Walk decay D0(u)', ymax: 1 },
  };
  const clamp01 = (x) => Math.max(0, x);
  const F = {
    headway_curve: {
      default: { label: 'Default (log-linear elasticity)', params: [['h_sat', 'saturation (min)', 1, 30, 0.5, 5], ['h_p', 'elasticity p', -1, 1, 0.01, -0.0658], ['h_q', 'elasticity q', 0, 0.6, 0.005, 0.1563]], analytic: true },
      exponential: { label: 'Exponential', params: [['sat', 'full score up to (min)', 1, 30, 0.5, 5], ['scale', 'decay scale (min)', 5, 200, 1, 40]], fn: (h, a) => h <= a.sat ? 1 : Math.exp(-(h - a.sat) / a.scale) },
      power: { label: 'Power', params: [['sat', 'full score up to (min)', 1, 30, 0.5, 5], ['exponent', 'exponent', 0.1, 2, 0.05, 0.6]], fn: (h, a) => h <= a.sat ? 1 : Math.pow(a.sat / h, a.exponent) },
      logistic: { label: 'Logistic', params: [['mid', 'midpoint (min)', 5, 180, 1, 30], ['steep', 'steepness', 0.02, 0.5, 0.01, 0.12]], fn: (h, a) => 1 / (1 + Math.exp(a.steep * (h - a.mid))) },
      steps: { label: 'Steps', params: [['h1', 'score 1.0 up to (min)', 1, 60, 1, 5], ['h2', 'score 0.9 up to', 2, 90, 1, 10], ['h3', 'score 0.7 up to', 5, 120, 1, 20], ['h4', 'score 0.5 up to', 10, 180, 1, 30], ['h5', 'score 0.25 up to', 20, 300, 5, 60], ['h6', 'score 0.1 up to', 30, 720, 5, 120]],
               fn: (h, a) => h <= a.h1 ? 1 : h <= a.h2 ? 0.9 : h <= a.h3 ? 0.7 : h <= a.h4 ? 0.5 : h <= a.h5 ? 0.25 : h <= a.h6 ? 0.1 : 0 },
      custom: { label: 'Custom points', points: [[2, 1], [10, 0.85], [30, 0.55], [60, 0.35], [180, 0.1], [600, 0.02]] },
    },
    speed_curve: {
      default: { label: 'Default (Hill)', params: [['v_anchor', 'anchor: score 1.0 at (km/h)', 5, 80, 1, 30], ['v_bonus', 'ceiling bonus', 0, 0.5, 0.01, 0.15], ['v_n', 'shape exponent n', 0.5, 5, 0.1, 2]], analytic: true },
      linear: { label: 'Linear', params: [['v0', 'zero at (km/h)', 0, 40, 1, 5], ['v1', 'full at (km/h)', 10, 120, 1, 40], ['ceiling', 'ceiling', 0.5, 1.2, 0.01, 1]], fn: (v, a) => a.ceiling * Math.min(1, clamp01((v - a.v0) / (a.v1 - a.v0))) },
      logistic: { label: 'Logistic', params: [['mid', 'midpoint (km/h)', 5, 80, 1, 20], ['steep', 'steepness', 0.05, 0.6, 0.01, 0.2], ['ceiling', 'ceiling', 0.5, 1.2, 0.01, 1]], fn: (v, a) => a.ceiling / (1 + Math.exp(-a.steep * (v - a.mid))) },
      steps: { label: 'Steps', params: [['v1', '0.2 from (km/h)', 1, 40, 1, 10], ['v2', '0.5 from', 5, 60, 1, 20], ['v3', '0.8 from', 10, 80, 1, 30], ['v4', '1.0 from', 15, 120, 1, 45]], fn: (v, a) => v >= a.v4 ? 1 : v >= a.v3 ? 0.8 : v >= a.v2 ? 0.5 : v >= a.v1 ? 0.2 : 0 },
      custom: { label: 'Custom points', points: [[0, 0], [10, 0.45], [20, 0.8], [30, 1], [60, 1.1], [150, 1.15]] },
    },
    walk_curve: {
      default: { label: 'Default ((1+u)^-p)', params: [['p_shape', 'exponent p', 0.2, 4, 0.05, 1]], analytic: true },
      exponential: { label: 'Exponential', params: [['scale', 'scale', 0.2, 5, 0.05, 1]], fn: (u, a) => Math.exp(-u / a.scale) },
      gaussian: { label: 'Gaussian', params: [['scale', 'scale', 0.2, 5, 0.05, 1]], fn: (u, a) => Math.exp(-Math.pow(u / a.scale, 2)) },
      linear: { label: 'Linear to zero', params: [['umax', 'zero at u =', 0.5, 20, 0.1, 3]], fn: (u, a) => Math.min(1, clamp01(1 - u / a.umax)) },
      logistic: { label: 'Logistic', params: [['mid', 'midpoint u', 0.2, 10, 0.1, 2], ['steep', 'steepness', 0.2, 8, 0.1, 2]], fn: (u, a) => Math.min(1, (1 / (1 + Math.exp(a.steep * (u - a.mid)))) / (1 / (1 + Math.exp(-a.steep * a.mid)))) },
      buffer: { label: 'Hard buffer (plateau radius)', params: [], fn: (u) => u <= 0 ? 1 : 0 },
      custom: { label: 'Custom points', points: [[0, 1], [0.5, 0.7], [1, 0.45], [3, 0.12], [10, 0.02], [40, 0]] },
    },
  };

  function analytic(key, p) {   // value of the default analytic curve for the current parameters
    if (key === 'headway_curve') { const eps = p.h_p + p.h_q * Math.log(p.h_sat); return (h) => { if (h <= p.h_sat) return 1; const lr = Math.log(h / p.h_sat); return Math.exp(-eps * lr - (p.h_q / 2) * lr * lr); }; }
    if (key === 'speed_curve') { const m = 1 + p.v_bonus, kk = p.v_anchor * Math.pow(m - 1, 1 / p.v_n), kn = Math.pow(kk, p.v_n); return (v) => { if (v <= 0) return 0; const vn = Math.pow(v, p.v_n); return m * vn / (vn + kn); }; }
    return (u) => Math.pow(1 + u, -p.p_shape);
  }
  function interp(points, x) {
    const P = points.slice().sort((a, b) => a[0] - b[0]);
    if (x <= P[0][0]) return P[0][1];
    for (let i = 1; i < P.length; i++) if (x <= P[i][0]) { const t = (x - P[i - 1][0]) / (P[i][0] - P[i - 1][0] || 1); return P[i - 1][1] * (1 - t) + P[i][1] * t; }
    return P[P.length - 1][1];
  }
  function gridX(key) {   // sample positions identical to Curve.from_function
    const g = GRID[key], xs = new Float64Array(g.n);
    for (let i = 0; i < g.n; i++) { const t = i / (g.n - 1); xs[i] = g.scale === 'log' ? Math.exp(Math.log(g.lo) + t * (Math.log(g.hi) - Math.log(g.lo))) : g.lo + t * (g.hi - g.lo); }
    return xs;
  }
  /* sample a function into the table curve the scorer uses */
  function toTable(key, f) {
    const g = GRID[key], xs = gridX(key), y = new Array(g.n);
    for (let i = 0; i < g.n; i++) y[i] = Math.max(0, f(xs[i]));
    return { kind: 'table', lo: g.lo, hi: g.hi, scale: g.scale, y };
  }

  /* one parameter: "name  [number]" and a slider under it; typing a number or moving the slider both apply */
  function sliderRow(parent, label, mn, mx, st, value, onSet) {
    const row = el('div', { class: 'sr srow' }, parent);
    el('span', { class: 'sname' }, row, label);
    const num = el('input', { type: 'number', min: mn, max: mx, step: st, class: 'snum' }, row);
    const rng = el('input', { type: 'range', min: mn, max: mx, step: st }, parent);
    const fmt = (v) => (+v).toFixed(st < 0.01 ? 3 : st < 0.1 ? 2 : st < 1 ? 1 : 0);
    num.value = fmt(value); rng.value = value;
    rng.addEventListener('input', () => { num.value = fmt(rng.value); onSet(+rng.value); });
    num.addEventListener('change', () => { const v = +num.value; if (!isFinite(v)) return; rng.value = v; onSet(v); });
    return { num, rng };
  }
  function svg(tag, attrs, parent) { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
  function el(tag, attrs, parent, text) { const e = document.createElement(tag); for (const k in attrs || {}) e.setAttribute(k, attrs[k]); if (text) e.textContent = text; if (parent) parent.appendChild(e); return e; }

  function tableFn(c) {   // evaluate a table curve exactly like score.js
    const y = c.y, n = y.length, lg = c.scale === 'log', a = lg ? Math.log(c.lo) : c.lo, b = lg ? Math.log(c.hi) : c.hi;
    return (x) => { let pos = ((lg ? Math.log(Math.max(x, 1e-300)) : x) - a) / (b - a) * (n - 1); pos = Math.min(n - 1, Math.max(0, pos)); let i = Math.floor(pos); if (i > n - 2) i = n - 2; const f = pos - i; return y[i] * (1 - f) + y[i + 1] * f; };
  }
  function samplePoints(key, fn, n) {   // a few editable points along a curve (log-spaced for headway)
    const g = GRID[key], out = [];
    for (let i = 0; i < n; i++) { const t = i / (n - 1); const x = g.scale === 'log' ? Math.exp(Math.log(g.lo) + t * (Math.log(g.hi) - Math.log(g.lo))) : g.lo + t * (g.hi - g.lo); out.push([x, Math.max(0, fn(x))]); }
    return out;
  }

  function curveEditor(container, key, params, onChange, state) {
    const g = GRID[key], box = el('div', { class: 'ed' }, container);
    // per-editor copy of the families (custom points must not leak between editors or reloads); a preset's own table is shown as 'loaded'
    const fam = {};
    for (const k in F[key]) fam[k] = Object.assign({}, F[key][k], F[key][k].points ? { points: F[key][k].points.map((p) => p.slice()) } : {});
    if (state.table) fam.loaded = { label: 'Curve of the preset (table)', fn: tableFn(state.table), loaded: true };
    el('div', { class: 'edt' }, box, g.title);
    const sel = el('select', {}, box);
    for (const k in fam) el('option', { value: k }, sel, fam[k].label);
    const fromLoaded = () => { if (state.table) fam.custom.points = samplePoints(key, tableFn(state.table), 9); };
    const sliders = el('div', { class: 'sl' }, box);
    const W = 280, H = 130, L = 30, B = 22, T = 6, R = 6, pw = W - L - R, ph = H - T - B;
    const sv = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'plot' }, box);
    const X = (x) => L + (g.scale === 'log' ? (Math.log(x) - Math.log(g.lo)) / (Math.log(g.hi) - Math.log(g.lo)) : (x - g.lo) / (g.hi - g.lo)) * pw;
    const Y = (y) => T + (1 - y / g.ymax) * ph;
    const invX = (px) => { const t = (px - L) / pw; return g.scale === 'log' ? Math.exp(Math.log(g.lo) + t * (Math.log(g.hi) - Math.log(g.lo))) : g.lo + t * (g.hi - g.lo); };
    const invY = (py) => (1 - (py - T) / ph) * g.ymax;
    svg('rect', { x: L, y: T, width: pw, height: ph, class: 'pbg' }, sv);
    for (const yy of [0, 0.5, 1]) { svg('line', { x1: L, x2: L + pw, y1: Y(yy), y2: Y(yy), class: 'grid' }, sv); svg('text', { x: L - 4, y: Y(yy) + 3, 'text-anchor': 'end', class: 'tick' }, sv).textContent = yy; }
    const xt = g.scale === 'log' ? [1, 5, 10, 30, 60, 180, 720] : (key === 'speed_curve' ? [0, 30, 60, 90, 120, 150] : [0, 10, 20, 30, 40]);
    for (const xx of xt) { svg('text', { x: X(xx), y: H - 10, 'text-anchor': 'middle', class: 'tick' }, sv).textContent = xx; }
    svg('text', { x: L + pw / 2, y: H - 0, 'text-anchor': 'middle', class: 'tick' }, sv).textContent = g.xlabel;
    const path = svg('path', { class: 'curve', fill: 'none' }, sv), ref = svg('path', { class: 'refcurve', fill: 'none' }, sv), dots = svg('g', {}, sv);
    const refF = analytic(key, root.__defaultParams || params);

    function currentFn() {
      const f = fam[state.family];
      if (f.analytic) return analytic(key, params);
      if (f.loaded) return f.fn;
      if (f.points) return (x) => interp(f.points, x);
      return (x) => f.fn(x, state.a);
    }
    function draw() {
      const fn = currentFn(), xs = gridX(key); let d = '', dr = '';
      for (let i = 0; i < xs.length; i += 2) { d += (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(Math.min(g.ymax, fn(xs[i]))).toFixed(1); dr += (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ' ' + Y(Math.min(g.ymax, refF(xs[i]))).toFixed(1); }
      path.setAttribute('d', d); ref.setAttribute('d', dr);
      dots.innerHTML = '';
      const f = fam[state.family];
      if (f.points) f.points.forEach((pt, i) => {
        const c = svg('circle', { cx: X(pt[0]), cy: Y(pt[1]), r: 5, class: 'pt' }, dots);
        c.addEventListener('pointerdown', (ev) => { ev.preventDefault(); c.setPointerCapture(ev.pointerId); c._drag = true; });
        c.addEventListener('pointermove', (ev) => { if (!c._drag) return; const r = sv.getBoundingClientRect(), sx = W / r.width; pt[0] = Math.min(g.hi, Math.max(g.lo, invX((ev.clientX - r.left) * sx))); pt[1] = Math.min(g.ymax, Math.max(0, invY((ev.clientY - r.top) * sx))); draw(); });
        c.addEventListener('pointerup', () => { c._drag = false; commit(); });
        c.addEventListener('dblclick', () => { if (f.points.length > 2) { f.points.splice(i, 1); draw(); commit(); } });
      });
    }
    function commit() {
      const f = fam[state.family];
      if (f.analytic) params[key] = { kind: 'default', lo: 0, hi: 1, scale: 'lin', y: [] };
      else if (f.loaded) params[key] = JSON.parse(JSON.stringify(state.table));
      else params[key] = toTable(key, currentFn());
      onChange(params);
    }
    function buildSliders() {
      sliders.innerHTML = '';
      const f = fam[state.family];
      for (const [k, label, mn, mx, st, def] of (f.params || [])) {
        const get = () => (f.analytic ? params[k] : state.a[k]);
        const set = (v) => { if (f.analytic) params[k] = v; else state.a[k] = v; };
        if (!f.analytic && state.a[k] === undefined) state.a[k] = def;
        sliderRow(sliders, label, mn, mx, st, get(), (v) => { set(v); draw(); commit(); });
      }
      if (f.points) el('div', { class: 'hint' }, sliders, 'drag points; click the plot to add one; double-click a point to remove it');
      if (f.loaded) el('div', { class: 'hint' }, sliders, 'the fitted curve of the preset; choose "Custom points" to edit it point by point, or another shape');
    }
    sel.addEventListener('change', () => { if (sel.value === 'custom' && state.family === 'loaded') fromLoaded(); state.family = sel.value; buildSliders(); draw(); commit(); });
    sv.addEventListener('click', (ev) => {
      const f = fam[state.family]; if (!f.points || ev.target.tagName === 'circle') return;
      const r = sv.getBoundingClientRect(), sx = W / r.width, x = invX((ev.clientX - r.left) * sx), y = invY((ev.clientY - r.top) * sx);
      if (x >= g.lo && x <= g.hi && y >= 0 && y <= g.ymax) { f.points.push([x, Math.min(g.ymax, Math.max(0, y))]); draw(); commit(); }
    });
    sel.value = state.family; buildSliders(); draw();
    return { redraw: draw, select: (fam_) => { sel.value = fam_; state.family = fam_; buildSliders(); draw(); } };
  }

  function build(container, params, onChange) {
    container.innerHTML = '';
    root.__defaultParams = JSON.parse(JSON.stringify(params));
    const out = {};
    const changed = (pp) => { onChange(pp); if (out.refresh) out.refresh(); };   // every edit also refreshes the walking chart and the examples
    const notify = () => changed(params);
    // aggregation + weights + modes
    const agg = el('div', { class: 'ed' }, container);
    el('div', { class: 'edt' }, agg, 'How the three scores combine');
    const aggSel = el('select', {}, agg);
    [['product', 'product: mode × speed × headway (weights as exponents)'], ['nested', 'nested: (mode^wm · speed^ws)^(1/(wm+ws)) · headway^wh'], ['power:1', 'arithmetic mean'], ['power:0', 'geometric mean'], ['power:-1', 'harmonic mean'], ['power:-4', 'close to the minimum (p = -4)']].forEach(([v, t]) => el('option', { value: v }, aggSel, t));
    aggSel.value = params.agg === 'power_mean' ? 'power:' + params.agg_p : (params.agg === 'product' ? 'product' : 'nested');
    aggSel.addEventListener('change', () => { const v = aggSel.value; if (v === 'nested' || v === 'product') params.agg = v; else { params.agg = 'power_mean'; params.agg_p = +v.split(':')[1]; } notify(); });
    for (const [k, label] of [['w_mode', 'Weight of mode (0 = ignore)'], ['w_speed', 'Weight of speed (0 = ignore)'], ['w_headway', 'Weight of headway (0 = ignore)']]) {
      sliderRow(agg, label, 0, 3, 0.1, params[k], (v) => { params[k] = v; notify(); });
    }
    const mm = el('div', { class: 'sr' }, agg); el('span', {}, mm, 'Value of each mode (0-1): ');
    ['bus', 'tram', 'rail'].forEach((m, i) => { el('span', {}, mm, ' ' + m + ' '); const n = el('input', { type: 'number', min: 0, max: 1.2, step: 0.05, style: 'width:52px' }, mm); n.value = params.mrc[i]; n.addEventListener('input', () => { params.mrc[i] = +n.value; notify(); }); });
    // curves
    out.editors = {};
    for (const key of ['headway_curve', 'speed_curve', 'walk_curve']) {
      const isT = params[key].kind === 'table';
      const st = { family: isT ? 'loaded' : 'default', a: {}, table: isT ? JSON.parse(JSON.stringify(params[key])) : null };
      out.editors[key] = curveEditor(container, key, params, changed, st);
    }
    // walking
    const wk = el('div', { class: 'ed' }, container);
    el('div', { class: 'edt' }, wk, 'Walking');
    const nums = [['walk_speed', 'Walking speed (m/s)', 0.5, 2, 0.05], ['plateau_m', 'No decay up to (m)', 0, 1000, 10], ['k', 'Walk further to better stops, k (0 = off)', 0, 8, 0.1]];
    for (const [k, label, mn, mx, st] of nums) {
      sliderRow(wk, label, mn, mx, st, params[k], (v) => { params[k] = v; notify(); });
    }
    const t0 = el('div', { class: 'sr' }, wk); el('span', {}, t0, 'Walk time scale t0 (min) per mode: ');
    ['bus', 'tram', 'rail'].forEach((m, i) => { el('span', {}, t0, ' ' + m + ' '); const n = el('input', { type: 'number', min: 0.1, max: 60, step: 0.1, style: 'width:62px' }, t0); n.value = params.t0_base[i]; n.addEventListener('input', () => { params.t0_base[i] = +n.value; notify(); }); });
    // what the settings mean: walking factor by distance for each mode, and example scores
    const wplot = el('div', {}, wk);
    const ex = el('div', { class: 'ed' }, container);
    el('div', { class: 'edt' }, ex, 'What these settings give (score 0-100)');
    const exBody = el('div', {}, ex);
    out.refresh = function () { drawWalkMetres(wplot, params); drawExamples(exBody, params); };
    out.refresh();
    return out;
  }

  function walkFactor(p, mode, dist, q) {   // same formula as score.js cellScores
    const tp = p.plateau_m / p.walk_speed / 60, t = dist / p.walk_speed / 60;
    if (t <= tp) return 1;
    const u = (t - tp) / (p.t0_base[mode] * (1 + p.k * q));
    return p.walk_curve && p.walk_curve.kind === 'table' ? tableFn(p.walk_curve)(u) : Math.pow(1 + u, -p.p_shape);
  }
  function drawWalkMetres(box, p) {
    box.innerHTML = '';
    const W = 280, H = 120, L = 30, B = 22, T = 6, R = 6, pw = W - L - R, ph = H - T - B, dmax = 2000;
    const sv = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'plot' }, box);
    const X = (d) => L + d / dmax * pw, Y = (y) => T + (1 - y) * ph;
    svg('rect', { x: L, y: T, width: pw, height: ph, class: 'pbg' }, sv);
    for (const yy of [0, 0.5, 1]) { svg('line', { x1: L, x2: L + pw, y1: Y(yy), y2: Y(yy), class: 'grid' }, sv); svg('text', { x: L - 4, y: Y(yy) + 3, 'text-anchor': 'end', class: 'tick' }, sv).textContent = yy; }
    for (const d of [0, 400, 800, 1200, 1600, 2000]) svg('text', { x: X(d), y: H - 10, 'text-anchor': 'middle', class: 'tick' }, sv).textContent = d;
    svg('text', { x: L + pw / 2, y: H, 'text-anchor': 'middle', class: 'tick' }, sv).textContent = 'walking distance (m), stop score 0.5';
    const col = ['#2a78d6', '#eb6834', '#1baf7a'];
    ['bus', 'tram', 'rail'].forEach((m, i) => {
      let dpath = '';
      for (let d = 0; d <= dmax; d += 20) dpath += (d ? 'L' : 'M') + X(d).toFixed(1) + ' ' + Y(Math.min(1, walkFactor(p, i, d, 0.5))).toFixed(1);
      svg('path', { d: dpath, fill: 'none', stroke: col[i], 'stroke-width': 2 }, sv);
      svg('text', { x: L + pw - 2, y: T + 10 + 10 * i, 'text-anchor': 'end', class: 'tick', fill: col[i] }, sv).textContent = m;
    });
  }
  function drawExamples(box, p) {
    if (!root.FlexScore) return;
    const S = root.FlexScore.prepare(p), rows = [['bus', 0, 10, 20], ['bus', 0, 30, 20], ['bus', 0, 60, 20], ['bus', 0, 120, 20], ['tram', 1, 10, 22], ['rail', 2, 5, 40], ['rail', 2, 15, 40]];
    const mode = Int8Array.from(rows.map((r) => r[1])), h = Float32Array.from(rows.map((r) => r[2])), v = Float32Array.from(rows.map((r) => r[3]));
    const q = root.FlexScore.stopScores(S, mode, h, v, rows.length, new Float32Array(rows.length));
    let html = '<table class="extab"><tr><th></th><th>at stop</th><th>200 m</th><th>400 m</th><th>800 m</th></tr>';
    rows.forEach((r, i) => {
      html += `<tr><td>${r[0]} every ${r[2]} min</td>` + [0, 200, 400, 800].map((d) => `<td>${(100 * q[i] * walkFactor(p, r[1], d, q[i])).toFixed(1)}</td>`).join('') + '</tr>';
    });
    box.innerHTML = html + '</table><div class="hint">speed 20 km/h bus, 22 tram, 40 rail</div>';
  }
  root.FlexEditor = { build, families: F, toTable, GRID };
})(typeof window !== 'undefined' ? window : globalThis);
