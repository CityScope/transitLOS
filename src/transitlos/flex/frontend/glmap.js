/* Minimal WebGL2 map for the flexible-scoring viewer (no dependencies): H3 hexagons coloured by a per-cell value, stops as circles of chosen size,
 * pan/zoom, and picking. Coordinates are Web Mercator kilometres relative to the map centre (float32 stays precise to ~1 m).
 *
 *   const m = new FlexMap(canvas);
 *   m.setCells(geom, nc);            // geom: Float32Array [nc*14] = centre lon,lat then 6 vertices lon,lat
 *   m.setValues(Float32Array nc, 'score'|'diff');   // per-cell value (score 0-100; diff -100..100; < -1e3 = hidden)
 *   m.setStops(lonlat, S); m.setStopSizes(Float32Array S px, selectedIndex);
 *   m.fit(bbox); m.draw(); m.pickStop(px, py), m.pickCell(px, py)
 */
(function (root) {
  const R = 6378.137;
  const mx = (lon) => lon * Math.PI / 180 * R, my = (lat) => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)) * R;
  const VS_CELL = `#version 300 es
  in vec2 a_pos; in float a_val; uniform vec2 u_off; uniform float u_scale; uniform vec2 u_res; out float v_val;
  void main(){ vec2 p=(a_pos-u_off)*u_scale; gl_Position=vec4(p.x/(u_res.x*0.5), p.y/(u_res.y*0.5), 0.0, 1.0); v_val=a_val; }`;
  const FS_CELL = `#version 300 es
  precision mediump float; in float v_val; uniform int u_mode; out vec4 o;
  vec3 ramp(float t){ t=clamp(t,0.0,1.0); vec3 a=vec3(0.84,0.19,0.15), b=vec3(0.99,0.68,0.38), c=vec3(1.0,1.0,0.75), d=vec3(0.65,0.85,0.42), e=vec3(0.10,0.59,0.31);
    if(t<0.25) return mix(a,b,t/0.25); if(t<0.5) return mix(b,c,(t-0.25)/0.25); if(t<0.75) return mix(c,d,(t-0.5)/0.25); return mix(d,e,(t-0.75)/0.25); }
  void main(){ if(v_val < -1000.0) discard;
    vec3 col; if(u_mode==1){ float t=clamp(v_val/50.0,-1.0,1.0); col = t<0.0 ? mix(vec3(0.97),vec3(0.17,0.38,0.76),-t) : mix(vec3(0.97),vec3(0.80,0.15,0.15),t); } else col = ramp(v_val/100.0);
    o=vec4(col,0.86); }`;
  const VS_CPT = `#version 300 es
  in vec2 a_pos; in float a_val; uniform vec2 u_off; uniform float u_scale; uniform vec2 u_res; uniform float u_size; out float v_val;
  void main(){ vec2 p=(a_pos-u_off)*u_scale; gl_Position=vec4(p.x/(u_res.x*0.5), p.y/(u_res.y*0.5), 0.0, 1.0); gl_PointSize=u_size; v_val=a_val; }`;
  const VS_STOP = `#version 300 es
  in vec2 a_pos; in float a_size; in float a_sel; uniform vec2 u_off; uniform float u_scale; uniform vec2 u_res; out float v_sel;
  void main(){ vec2 p=(a_pos-u_off)*u_scale; gl_Position=vec4(p.x/(u_res.x*0.5), p.y/(u_res.y*0.5), 0.0, 1.0); gl_PointSize=a_size; v_sel=a_sel; }`;
  const FS_STOP = `#version 300 es
  precision mediump float; in float v_sel; out vec4 o;
  void main(){ vec2 c=gl_PointCoord*2.0-1.0; float r=dot(c,c); if(r>1.0) discard; vec3 col = v_sel>0.5 ? vec3(1.0,0.85,0.1) : vec3(1.0); float edge=smoothstep(0.62,0.78,r); o=vec4(mix(col, vec3(0.1), edge), 1.0); }`;

  function prog(gl, vs, fs) {
    const mk = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
    const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p;
  }
  class FlexMap {
    constructor(canvas) {
      this.cv = canvas; const gl = this.gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
      if (!gl) throw new Error('WebGL2 not available');
      this.pc = prog(gl, VS_CELL, FS_CELL); this.ps = prog(gl, VS_STOP, FS_STOP); this.pp = prog(gl, VS_CPT, FS_CELL);
      this.cx = 0; this.cy = 0; this.scale = 1; this.ox = 0; this.oy = 0; this.nc = 0; this.S = 0;
      this.stopSize = 0; this._bind();
    }
    _bind() {
      const cv = this.cv;
      cv.addEventListener('wheel', (e) => { e.preventDefault(); const r = cv.getBoundingClientRect(), dpr = cv.width / r.width;
        const px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr, k = Math.exp(-e.deltaY * 0.0015);
        const [wx, wy] = this.unproject(px, py); this.scale *= k; const [wx2, wy2] = this.unproject(px, py); this.cx += wx - wx2; this.cy += wy - wy2; this.draw(); }, { passive: false });
      let drag = null;
      cv.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; cv.setPointerCapture(e.pointerId); });
      cv.addEventListener('pointermove', (e) => { if (!drag) return; const r = cv.getBoundingClientRect(), dpr = cv.width / r.width, dx = (e.clientX - drag.x) * dpr, dy = (e.clientY - drag.y) * dpr;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; if (!drag.moved) return; this.cx -= dx / this.scale; this.cy += dy / this.scale; drag.x = e.clientX; drag.y = e.clientY; this.draw(); });
      cv.addEventListener('pointerup', (e) => { const was = drag; drag = null; if (was && !was.moved && this.onClick) { const r = cv.getBoundingClientRect(), dpr = cv.width / r.width; this.onClick((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr, e); } });
    }
    unproject(px, py) { return [this.cx + (px - this.cv.width / 2) / this.scale, this.cy - (py - this.cv.height / 2) / this.scale]; }
    resize() { const r = this.cv.getBoundingClientRect(), d = devicePixelRatio || 1; this.cv.width = Math.max(2, Math.round(r.width * d)); this.cv.height = Math.max(2, Math.round(r.height * d)); this.draw(); }
    setCells(geom, nc) {
      const gl = this.gl; this.nc = nc;
      if (nc === 0) return;
      let minx = 1e30, maxx = -1e30, miny = 1e30, maxy = -1e30;
      for (let i = 0; i < nc; i++) { const x = mx(geom[i * 14]), y = my(geom[i * 14 + 1]); if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y; }
      this.ox = (minx + maxx) / 2; this.oy = (miny + maxy) / 2; this.ext = [minx - this.ox, miny - this.oy, maxx - this.ox, maxy - this.oy];
      const pos = new Float32Array(nc * 7 * 2), idx = new Uint32Array(nc * 18); this.centres = new Float32Array(nc * 2);
      for (let i = 0; i < nc; i++) {
        for (let j = 0; j < 7; j++) { pos[(i * 7 + j) * 2] = mx(geom[i * 14 + 2 * j]) - this.ox; pos[(i * 7 + j) * 2 + 1] = my(geom[i * 14 + 2 * j + 1]) - this.oy; }
        this.centres[i * 2] = pos[i * 14]; this.centres[i * 2 + 1] = pos[i * 14 + 1];
        for (let j = 0; j < 6; j++) { const o = i * 18 + j * 3; idx[o] = i * 7; idx[o + 1] = i * 7 + 1 + j; idx[o + 2] = i * 7 + 1 + (j + 1) % 6; }
      }
      this.posBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
      this.idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
      this.valBuf = gl.createBuffer(); this.valArr = new Float32Array(nc * 7); this.mode = 0;
      this.cenBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.cenBuf); gl.bufferData(gl.ARRAY_BUFFER, this.centres, gl.STATIC_DRAW); this.valCellBuf = gl.createBuffer();
      // grid index on centres for picking (cells ~0.5 km)
      const cs = 0.5; this.gs = cs; this.gw = Math.ceil((this.ext[2] - this.ext[0]) / cs) + 2; this.gh = Math.ceil((this.ext[3] - this.ext[1]) / cs) + 2;
      this.gridHead = new Int32Array(this.gw * this.gh).fill(-1); this.gridNext = new Int32Array(nc);
      for (let i = 0; i < nc; i++) { const gx = Math.floor((this.centres[i * 2] - this.ext[0]) / cs) + 1, gy = Math.floor((this.centres[i * 2 + 1] - this.ext[1]) / cs) + 1, k = gy * this.gw + gx; this.gridNext[i] = this.gridHead[k]; this.gridHead[k] = i; }
    }
    setValues(vals, mode) {
      if (!this.nc) return; const gl = this.gl, a = this.valArr; this.mode = mode === 'diff' ? 1 : 0;
      for (let i = 0, k = 0; i < this.nc; i++) { const v = vals[i]; for (let j = 0; j < 7; j++) a[k++] = v; }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.valBuf); gl.bufferData(gl.ARRAY_BUFFER, a, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.valCellBuf); gl.bufferData(gl.ARRAY_BUFFER, vals, gl.DYNAMIC_DRAW); this.draw();
    }
    setStops(lonlat, S, group) {   // group (optional, int32 [S]): the station each drawn point belongs to; a selection marks every point of the station
      const gl = this.gl; this.S = S; this.group = group || null; const p = new Float32Array(S * 2);
      for (let i = 0; i < S; i++) { p[i * 2] = mx(lonlat[i * 2]) - this.ox; p[i * 2 + 1] = my(lonlat[i * 2 + 1]) - this.oy; }
      this.stopPos = p; this.stopBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.stopBuf); gl.bufferData(gl.ARRAY_BUFFER, p, gl.STATIC_DRAW);
      this.sizeBuf = gl.createBuffer(); this.selBuf = gl.createBuffer(); this.sizes = new Float32Array(S).fill(5); this.sel = new Float32Array(S);
    }
    setStopSizes(sizes, selected) {
      if (!this.S) return; const gl = this.gl; this.sizes = sizes; this.sel.fill(0); if (selected >= 0) { if (this.group) { for (let i = 0; i < this.S; i++) if (this.group[i] === selected) this.sel[i] = 1; } else this.sel[selected] = 1; }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf); gl.bufferData(gl.ARRAY_BUFFER, sizes, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.selBuf); gl.bufferData(gl.ARRAY_BUFFER, this.sel, gl.DYNAMIC_DRAW); this.draw();
    }
    fit(bbox) {
      const e = bbox ? [mx(bbox[0]) - this.ox, my(bbox[1]) - this.oy, mx(bbox[2]) - this.ox, my(bbox[3]) - this.oy] : this.ext; if (!e) return;
      this.cx = (e[0] + e[2]) / 2; this.cy = (e[1] + e[3]) / 2; this.scale = 0.92 * Math.min(this.cv.width / Math.max(e[2] - e[0], 0.5), this.cv.height / Math.max(e[3] - e[1], 0.5)); this.draw();
    }
    draw() {
      if (this.onView) this.onView();
      const gl = this.gl, cv = this.cv; gl.viewport(0, 0, cv.width, cv.height); gl.clearColor(0.93, 0.93, 0.93, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      const cellPx = 0.13 * this.scale;   // a res-10 cell is ~130 m across
      if (this.nc && this.valBuf && cellPx < 4) {   // zoomed out: a point per cell (sub-pixel triangles would not show)
        const p = this.pp; gl.useProgram(p);
        gl.uniform2f(gl.getUniformLocation(p, 'u_off'), this.cx, this.cy); gl.uniform1f(gl.getUniformLocation(p, 'u_scale'), this.scale); gl.uniform2f(gl.getUniformLocation(p, 'u_res'), cv.width, cv.height);
        gl.uniform1i(gl.getUniformLocation(p, 'u_mode'), this.mode); gl.uniform1f(gl.getUniformLocation(p, 'u_size'), Math.max(2.0, cellPx * 1.4));
        const ap = gl.getAttribLocation(p, 'a_pos'), av = gl.getAttribLocation(p, 'a_val');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cenBuf); gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.valCellBuf); gl.enableVertexAttribArray(av); gl.vertexAttribPointer(av, 1, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, this.nc);
      } else if (this.nc && this.valBuf) {
        const p = this.pc; gl.useProgram(p);
        gl.uniform2f(gl.getUniformLocation(p, 'u_off'), this.cx, this.cy); gl.uniform1f(gl.getUniformLocation(p, 'u_scale'), this.scale); gl.uniform2f(gl.getUniformLocation(p, 'u_res'), cv.width, cv.height); gl.uniform1i(gl.getUniformLocation(p, 'u_mode'), this.mode);
        const ap = gl.getAttribLocation(p, 'a_pos'), av = gl.getAttribLocation(p, 'a_val');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf); gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.valBuf); gl.enableVertexAttribArray(av); gl.vertexAttribPointer(av, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf); gl.drawElements(gl.TRIANGLES, this.nc * 18, gl.UNSIGNED_INT, 0);
      }
      if (this.S && this.sizeBuf && this.showStops !== false) {
        const p = this.ps; gl.useProgram(p);
        gl.uniform2f(gl.getUniformLocation(p, 'u_off'), this.cx, this.cy); gl.uniform1f(gl.getUniformLocation(p, 'u_scale'), this.scale); gl.uniform2f(gl.getUniformLocation(p, 'u_res'), cv.width, cv.height);
        for (const [name, buf, sz] of [['a_pos', this.stopBuf, 2], ['a_size', this.sizeBuf, 1], ['a_sel', this.selBuf, 1]]) { const l = gl.getAttribLocation(p, name); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, sz, gl.FLOAT, false, 0, 0); }
        gl.drawArrays(gl.POINTS, 0, this.S);
      }
    }
    pickStop(px, py) {   // nearest stop within its drawn radius (min 8 px)
      if (!this.S) return -1; const [wx, wy] = this.unproject(px, py); let best = -1, bd = 1e30;
      for (let i = 0; i < this.S; i++) { const dx = (this.stopPos[i * 2] - wx) * this.scale, dy = (this.stopPos[i * 2 + 1] - wy) * this.scale, d = Math.hypot(dx, dy), r = Math.max(8, this.sizes[i] / 2 + 2);
        if (d <= r && d < bd) { bd = d; best = i; } }
      return best;
    }
    pickCell(px, py) {
      if (!this.nc) return -1; const [wx, wy] = this.unproject(px, py), gx = Math.floor((wx - this.ext[0]) / this.gs) + 1, gy = Math.floor((wy - this.ext[1]) / this.gs) + 1; let best = -1, bd = 1e30;
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const x = gx + a, y = gy + b; if (x < 0 || y < 0 || x >= this.gw || y >= this.gh) continue;
        for (let i = this.gridHead[y * this.gw + x]; i >= 0; i = this.gridNext[i]) { const d = Math.hypot(this.centres[i * 2] - wx, this.centres[i * 2 + 1] - wy); if (d < bd) { bd = d; best = i; } } }
      return bd < 0.09 ? best : -1;   // within ~90 m of a cell centre (res-10 cells are ~130 m across)
    }
    project(lon, lat) { return [(mx(lon) - this.ox - this.cx) * this.scale + this.cv.width / 2, this.cv.height / 2 - (my(lat) - this.oy - this.cy) * this.scale]; }
  }
  root.FlexMap = FlexMap;
})(typeof window !== 'undefined' ? window : globalThis);
