/* How well does a scoring explain the ACS transit share?  Estimated in the browser from a small random sample of block groups (web/sample/): every edit
 * re-scores the sample cells with the same engine as the map, builds each block group's predictor  f = sum(cell pop * cell score) / census pop / 100
 * (every cell of the block group, H3 res 10; people beyond walking reach count 0), and fits
 *   share = a + b f^gamma   by population-weighted least squares;  loss = weighted mean squared error,  R2 = 1 - loss / weighted variance of the share.
 * Set 0 is the nation, sets 1.. are metro areas (their own block groups). */
(function (root) {
  const get = (p, T) => fetch(p).then((r) => { if (!r.ok) throw new Error(p + ' ' + r.status); return r.arrayBuffer(); }).then((b) => new T(b));
  const F = { ready: false };
  F.load = async function (base) {
    // the sample has its own version (always checked): its files are cached for as long as that version stands
    let ver = '0'; try { ver = (await (await fetch(base + 'version.txt?t=' + Date.now(), { cache: 'no-store' })).text()).trim() || '0'; } catch (e) {}
    F.version = ver; const q = '?v=' + encodeURIComponent(ver);
    const meta = await (await fetch(base + 'meta.json' + q)).json();
    const [mode, h, v, np, cst, cdist, cpop, cbg, bg, sbg, sw] = await Promise.all([
      get(base + 'stop_mode.bin' + q, Int8Array), get(base + 'stop_h.bin' + q, Float32Array), get(base + 'stop_v.bin' + q, Float32Array), get(base + 'stop_np.bin' + q, Int32Array),
      get(base + 'cell_station.bin' + q, Int32Array), get(base + 'cell_dist.bin' + q, Uint16Array), get(base + 'cell_pop.bin' + q, Float32Array), get(base + 'cell_bg.bin' + q, Int32Array),
      get(base + 'bg.bin' + q, Float32Array), get(base + 'set_bg.bin' + q, Int32Array), get(base + 'set_w.bin' + q, Float32Array)]);
    Object.assign(F, { meta, mode, h, v, np, cst, cdist, cpop, cbg, bg, sbg, sw });
    F.ss = new Float32Array(meta.S * meta.P); F.los = new Float32Array(meta.nc); F.f = new Float64Array(meta.nbg);
    F.start = new Int32Array(meta.nbg + 1);
    for (let c = 0; c < meta.nc; c++) F.start[cbg[c] + 1]++;
    for (let b = 0; b < meta.nbg; b++) F.start[b + 1] += F.start[b];
    F.ready = true;
    return F;
  };
  /* -> [{r2, loss}] one entry per set (0 = nation) */
  F.evaluate = function (params, gamma) {
    const m = F.meta, S = FlexScore.prepare(params), bg = F.bg;
    FlexScore.stopScores(S, F.mode, F.h, F.v, m.S * m.P, F.ss);
    FlexScore.cellScores(S, F.cst, F.cdist, m.nc, m.K, m.P, F.np, F.mode, F.ss, F.los, null);
    for (let b = 0; b < m.nbg; b++) {
      let num = 0, den = 0;
      for (let c = F.start[b]; c < F.start[b + 1]; c++) { num += F.cpop[c] * F.los[c]; den += F.cpop[c]; }
      // the block group's score: population-weighted mean over ALL its cells, divided by its census population (people beyond walking reach have
      // no cell and count 0; the cells' own population when that is larger)
      const P = Math.max(bg[b * 4], den);
      F.f[b] = P > 0 ? (num / P) / 100 : 0;
    }
    return m.sets.map((s) => {
      let W = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
      for (let i = s.o; i < s.o + s.n; i++) {
        const b = F.sbg[i], w = F.sw[i] * bg[b * 4], x = Math.pow(Math.max(F.f[b], 0), gamma), y = Math.min(bg[b * 4 + 2], bg[b * 4 + 1]) / bg[b * 4 + 1];
        W += w; sx += w * x; sy += w * y; sxx += w * x * x; sxy += w * x * y; syy += w * y * y;
      }
      const mx = sx / W, my = sy / W, vx = sxx / W - mx * mx, vy = syy / W - my * my, cxy = sxy / W - mx * my;
      const loss = vx > 1e-15 ? vy - cxy * cxy / vx : vy;
      return { r2: vy > 0 ? 1 - loss / vy : 0, loss };
    });
  };
  root.FlexFit = F;
})(window);
