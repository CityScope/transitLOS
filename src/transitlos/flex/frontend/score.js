/* Front-end scoring: the SAME definition as flex/scoring.py (numpy) and flexcore.c (C) -- parameters, curve SHAPES and aggregation -- so the map can
 * recompute scores after the user changes anything. `params` is the JSON of ScoreParams (see ScoreParams.to_json / the meta.json of an export):
 *   {mrc:[bus,tram,rail], v_anchor, v_bonus, v_n, h_sat, h_p, h_q, walk_speed, plateau_m, t0_base:[..3], k, p_shape, w_mode, w_speed, w_headway,
 *    agg:"nested"|"product"|"power_mean", agg_p, headway_curve|speed_curve|walk_curve: {kind:"default"|"table", lo, hi, scale:"lin"|"log", y:[...]}}
 * Typed arrays only; prepare() once per parameter change (precomputes tables/constants), then stopScores() over [S*P] and cellScores() over [nc*K].
 */
(function (root) {
  function prep(c) {
    if (!c || c.kind !== 'table') return null;
    const y = Float64Array.from(c.y), n = y.length, log = c.scale === 'log';
    const a = log ? Math.log(c.lo) : c.lo, b = log ? Math.log(c.hi) : c.hi;
    return { y, n, log, a, span: b - a };
  }
  function tab(t, x) {
    let pos = (t.log ? Math.log(x < 1e-300 ? 1e-300 : x) : x) - t.a;
    pos = pos / t.span * (t.n - 1);
    if (pos < 0) pos = 0; else if (pos > t.n - 1) pos = t.n - 1;
    let i = Math.floor(pos); if (i > t.n - 2) i = t.n - 2;
    const f = pos - i;
    return t.y[i] * (1 - f) + t.y[i + 1] * f;
  }
  function prepare(p) {
    const m = 1 + p.v_bonus, kk = p.v_anchor * Math.pow(m - 1, 1 / p.v_n);
    return {
      p, m, kn: Math.pow(kk, p.v_n), eps: p.h_p + p.h_q * Math.log(p.h_sat), wms: p.w_mode + p.w_speed,
      wsum: p.w_mode + p.w_speed + p.w_headway,
      th: prep(p.headway_curve), tv: prep(p.speed_curve), tw: prep(p.walk_curve),
      ws: p.walk_speed, tp: p.plateau_m / p.walk_speed / 60,
    };
  }
  function stopScores(S, mode, h, v, n, out) {
    const p = S.p;
    for (let i = 0; i < n; i++) {
      if (mode[i] < 0) { out[i] = 0; continue; }
      const mr = p.mrc[mode[i]];
      let sp;
      if (v[i] <= 0) sp = 0; else if (S.tv) sp = tab(S.tv, v[i]); else { const vn = Math.pow(v[i], p.v_n); sp = S.m * vn / (vn + S.kn); }
      let f;
      if (S.th) f = tab(S.th, h[i]);
      else if (h[i] <= p.h_sat) f = 1;
      else { const lr = Math.log(h[i] / p.h_sat); f = Math.exp(-S.eps * lr - (p.h_q / 2) * lr * lr); }
      if (mr <= 0 || sp <= 0 || f <= 0) { out[i] = 0; continue; }
      let r;
      if (p.agg === 'power_mean') {
        if (S.wsum <= 0) { out[i] = 0; continue; }
        const w0 = p.w_mode / S.wsum, w1 = p.w_speed / S.wsum, w2 = p.w_headway / S.wsum;
        r = p.agg_p === 0 ? Math.exp(w0 * Math.log(mr) + w1 * Math.log(sp) + w2 * Math.log(f))
                          : Math.pow(w0 * Math.pow(mr, p.agg_p) + w1 * Math.pow(sp, p.agg_p) + w2 * Math.pow(f, p.agg_p), 1 / p.agg_p);
      } else if (p.agg === 'product') {   // mode^wm * speed^ws * headway^wh
        r = Math.pow(mr, p.w_mode) * Math.pow(sp, p.w_speed) * Math.pow(f, p.w_headway);
      } else {
        const core = S.wms === 0 ? 1 : Math.pow(Math.pow(mr, p.w_mode) * Math.pow(sp, p.w_speed), 1 / S.wms);
        r = core * Math.pow(f, p.w_headway);
      }
      out[i] = r > 1 ? 1 : r;
    }
    return out;
  }
  /* cell value 0-100 = max over its stations and their profiles of stop_score * walk decay. np[s] = profiles of station s. */
  function cellScores(S, cellStation, cellDist, nc, K, P, np, mode, ss, outLos, outSlot) {
    const p = S.p, t0 = p.t0_base, kq = p.k, pp = p.p_shape, tp = S.tp;
    for (let c = 0; c < nc; c++) {
      let best = 0, bs = -1;
      for (let k = 0; k < K; k++) {
        const s = cellStation[c * K + k];
        if (s < 0) continue;
        const t = cellDist[c * K + k] / S.ws / 60, base = s * P, cnt = np[s];
        for (let j = 0; j < cnt; j++) {
          const q = ss[base + j];
          let val = q;
          if (t > tp) {
            const u = (t - tp) / (t0[mode[base + j]] * (1 + kq * q));
            val = q * (S.tw ? tab(S.tw, u) : Math.pow(1 + u, -pp));
          }
          if (val > best) { best = val; bs = k; }
        }
      }
      outLos[c] = best * 100; if (outSlot) outSlot[c] = bs;
    }
    return outLos;
  }
  /* Catchment: the sum of the census counts (counts[nc*ncol]) of the cells each station WINS (the station of the winning slot of the cell).
   * totals[S*ncol] and won[S] (number of cells won) are filled; cells without a station are not attributed. */
  function catchment(cellStation, slot, nc, K, counts, ncol, S, totals, won) {
    totals.fill(0); won.fill(0);
    for (let c = 0; c < nc; c++) {
      const sl = slot[c];
      if (sl < 0) continue;
      const s = cellStation[c * K + sl];
      won[s]++;
      for (let j = 0; j < ncol; j++) totals[s * ncol + j] += counts[c * ncol + j];
    }
  }
  root.FlexScore = { prepare, stopScores, cellScores, catchment };
})(typeof window !== 'undefined' ? window : globalThis);
