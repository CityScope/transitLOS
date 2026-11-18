"""ctypes wrapper of `flexcore.c` (compiled on first import, recompiled when the C file changes). No numba: the installed NumPy is too new for it."""
from __future__ import annotations

import ctypes as C
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scoring import DEFAULT, ScoreParams

HERE = Path(__file__).resolve().parent
SRC = HERE / "flexcore.c"


def _build() -> Path:
    """Compile flexcore.c next to the module (or into ~/.cache/transitlos when the package folder is read-only). Needs gcc (OpenMP optional)."""
    import os
    import tempfile
    for folder in (HERE, Path(os.environ.get("TRANSITLOS_CACHE", Path.home() / ".cache" / "transitlos"))):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            lib = folder / "_flexcore.so"
            if not lib.exists() or lib.stat().st_mtime < SRC.stat().st_mtime:
                tmp = Path(tempfile.mkstemp(suffix=".so", dir=folder)[1])
                for flags in (["-fopenmp"], []):   # OpenMP when available, serial otherwise
                    r = subprocess.run(["gcc", "-O3", "-march=native", *flags, "-shared", "-fPIC", "-o", str(tmp), str(SRC), "-lm"], capture_output=True)
                    if r.returncode == 0:
                        break
                else:
                    raise RuntimeError("could not compile flexcore.c: " + r.stderr.decode()[:500])
                tmp.replace(lib)
            return lib
        except OSError:
            continue
    raise RuntimeError("no writable folder for the compiled flexcore library")


_lib = C.CDLL(str(_build()))



def _p(a):
    return a.ctypes.data_as(C.c_void_p)


@dataclass(frozen=True)
class Config:
    """Cap / floor hyperparameters (all in one place; saved with the results)."""
    tau_band: float = 0.30     # tolerance band of the extremes, on the normalised 0..1 subset score (tuned on synthetic data: 0.3 beat 0.1-0.7)
    gamma: float = 2.0         # quality exponent of the diversity fill
    delta: float = 0.05        # a candidate closer than this (normalised) to a chosen one is a near-copy: not added
    dom_tau_h: float = 0.10    # dominance tolerance: headway
    dom_tau_v: float = 0.05    # dominance tolerance: speed
    # Optional floors, OFF by default: nothing is dropped for being "bad". Dominance removes what is objectively worse than something kept, and the cap
    # (P, K) keeps a diverse set that prefers good values. Set e.g. h_max=60 to also refuse patterns with a headway above 60 minutes.
    h_max: float = 1e9         # floor: a profile slower than this headway (min) is not stored
    v_min: float = 0.0         #        ... or slower than this speed (km/h)
    s_min: float = 0.0         #        ... or with a default stop score below this
    p_min: float = 0.0         # a station whose default PTLOS at the cell is below this is not stored for the cell
    P: int = 10                # profiles per station
    K: int = 15                # stations per cell
    Kc: int = 24               # labels per node in the search
    Kcand: int = 48            # candidates per cell before the cap
    max_walk_m: float = 1500.0
    cell_radius_m: float = 66.0   # H3 res 10 edge length (~ centre-to-vertex radius)

    def array(self) -> np.ndarray:
        """The configuration as one float64 vector for the C core.

        Returns:
            The vector.
        """
        return np.array([self.tau_band, self.gamma, self.delta, self.dom_tau_h, self.dom_tau_v, self.h_max, self.v_min, self.s_min, self.p_min], dtype=np.float64)


def sp_array(p: ScoreParams = DEFAULT) -> np.ndarray:
    """The scoring as one float64 vector for the C core (layout documented at the top of the scoring section of flexcore.c).

    Args:
        p: Scoring parameters.

    Returns:
        float64 vector.
    """
    head = [*p.mrc, p.v_anchor, p.v_bonus, p.v_n, p.h_sat, p.h_p, p.h_q, p.walk_speed, p.plateau_m, *p.t0_base, p.k, p.p_shape, p.w_mode, p.w_speed, p.w_headway,
            {"power_mean": 1.0, "product": 2.0}.get(p.agg, 0.0), p.agg_p]
    tables, pos = [], 39
    for c in (p.headway_curve, p.speed_curve, p.walk_curve):
        if c.kind == "table":
            head += [1.0, c.lo, c.hi, 1.0 if c.scale == "log" else 0.0, float(c.n), float(pos)]
            tables.append(np.asarray(c.y, dtype=np.float64)); pos += c.n
        else:
            head += [0.0, 0.0, 1.0, 0.0, 2.0, float(pos)]
    return np.concatenate([np.asarray(head, dtype=np.float64)] + tables) if tables else np.asarray(head, dtype=np.float64)


AX_DIST, AX_HEAD, AX_MODE, AX_SPEED = 1, 2, 4, 8


def select_items(mode, h, v, d, q, cap: int, axes: int, cfg: Config = Config()) -> np.ndarray:
    """The cap on generic items (tests): indexes chosen, in selection order.

    Args:
        mode: Mode per item (0 bus, 1 tram, 2 rail).
        h: Headway (min).
        v: Speed (km/h).
        d: Walking distance (m).
        q: Quality (score) per item.
        cap: Items to keep.
        axes: Number of axes used for diversity.
        cfg: Caps and floors.

    Returns:
        Indices of the chosen items.
    """
    mode = np.ascontiguousarray(mode, dtype=np.int32); h = np.ascontiguousarray(h, dtype=np.float64); v = np.ascontiguousarray(v, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64); q = np.ascontiguousarray(q, dtype=np.float64)
    out = np.full(max(cap, 1), -1, dtype=np.int32)
    c = _lib.select_items(C.c_int(len(mode)), _p(mode), _p(h), _p(v), _p(d), _p(q), C.c_int(axes), C.c_int(cap), _p(cfg.array()), _p(out))
    return out[:c].copy()


def select_profiles(off, mode, h, v, cfg: Config = Config(), params: ScoreParams = DEFAULT):
    """Profiles of many stations. `off[ns+1]` delimits the candidates (mode 0/1/2, headway min, speed km/h) of each station.
    Returns (idx[ns, P] into the candidate arrays or -1, count[ns], dropped[ns]).

    Args:
        off: ``[stations + 1]`` offsets into the candidate arrays.
        mode: Candidate mode.
        h: Candidate headway (min).
        v: Candidate speed (km/h).
        cfg: Caps and floors.
        params: Scoring parameters.
    """
    off = np.ascontiguousarray(off, dtype=np.int32); mode = np.ascontiguousarray(mode, dtype=np.int32)
    h = np.ascontiguousarray(h, dtype=np.float32); v = np.ascontiguousarray(v, dtype=np.float32)
    ns = len(off) - 1
    idx = np.full((ns, cfg.P), -1, dtype=np.int32); n = np.zeros(ns, dtype=np.int32); dr = np.zeros(ns, dtype=np.int32)
    ca, sp = cfg.array(), sp_array(params)
    _lib.select_profiles(C.c_int(ns), _p(off), _p(mode), _p(h), _p(v), C.c_int(cfg.P), _p(sp), _p(ca), _p(idx), _p(n), _p(dr))
    return idx, n, dr


class Profiles:
    """Station profile arrays in the layout the C code reads: [S, P] int8 mode (-1 empty), float32 headway and speed, plus summaries."""

    def __init__(self, mode, h, v):
        self.mode = np.ascontiguousarray(mode, dtype=np.int8)
        self.h = np.ascontiguousarray(h, dtype=np.float32)
        self.v = np.ascontiguousarray(v, dtype=np.float32)
        self.S, self.P = self.mode.shape
        valid = self.mode >= 0
        self.n = np.ascontiguousarray(valid.sum(1), dtype=np.int32)
        self.smode = np.ascontiguousarray(np.where(valid, self.mode, -1).max(1), dtype=np.int8)
        self.shmin = np.ascontiguousarray(np.where(valid, self.h, np.inf).min(1), dtype=np.float32)
        self.svmax = np.ascontiguousarray(np.where(valid, self.v, 0).max(1), dtype=np.float32)

    def valid_mask(self):
        """True for the filled profile slots.

        Returns:
            Boolean array ``[S, P]``.
        """
        return self.mode >= 0

    def args(self):
        """The profile arrays as C arguments, in the order the C core expects.

        Returns:
            Tuple of ctypes arguments.
        """
        return (C.c_int(self.P), _p(self.n), _p(self.mode), _p(self.h), _p(self.v), _p(self.smode), _p(self.shmin), _p(self.svmax))


def multilabel_search(indptr, adj, w, src_node, src_off, src_station, prof: Profiles, cfg: Config = Config()):
    """Returns (lab_station[n,Kc], lab_dist[n,Kc], lab_n[n], lab_cap[n] (labels refused at that node because it was full), stats[stored, dropped_by_dominance, node_cap_hits]).

    Args:
        indptr: CSR row offsets of the street graph.
        adj: CSR neighbours.
        w: CSR edge lengths (m).
        src_node: Source node per snapped station end.
        src_off: Walking distance from the station to that node (m).
        src_station: Station index per source.
        prof: Station profiles.
        cfg: Caps and floors.
    """
    n = len(indptr) - 1
    indptr = np.ascontiguousarray(indptr, dtype=np.int32); adj = np.ascontiguousarray(adj, dtype=np.int32); w = np.ascontiguousarray(w, dtype=np.float32)
    sn = np.ascontiguousarray(src_node, dtype=np.int32); so = np.ascontiguousarray(src_off, dtype=np.float32); ss = np.ascontiguousarray(src_station, dtype=np.int32)
    ls = np.full((n, cfg.Kc), -1, dtype=np.int32); ld = np.zeros((n, cfg.Kc), dtype=np.float32); ln = np.zeros(n, dtype=np.int32); st = np.zeros(3, dtype=np.int64)
    lc = np.zeros(n, dtype=np.uint8)
    ca = cfg.array()
    _lib.multilabel_search(C.c_int(n), _p(indptr), _p(adj), _p(w), C.c_int(len(sn)), _p(sn), _p(so), _p(ss), C.c_int(cfg.Kc), C.c_float(cfg.max_walk_m),
                           prof.args()[0], prof.args()[1], prof.args()[2], prof.args()[3], prof.args()[4], prof.args()[5], prof.args()[6], prof.args()[7],
                           _p(ca), _p(ls), _p(ld), _p(ln), _p(lc), _p(st))
    return ls, ld, ln, lc, st


def process_cells(rec_off, ru, rv, ra, rb, rl, lab, prof: Profiles, cfg: Config = Config(), params: ScoreParams = DEFAULT):
    """Stations per cell. `lab` = multilabel_search()[:4]. Returns (station[nc,K], dist_m[nc,K] uint16 (65535 empty), flag[nc], cropped[nc]).
    flag: bit0 the cap was needed, bit1 the candidate list limit was hit, bit2 the search refused labels at a node touching the cell.
    cropped = stations GENUINELY cut (non-dominated, above the floors, but over the cap or the list limit); stations dropped by floors or dominance are not counted.

    Args:
        rec_off: Offsets of each cell's edge records.
        ru: Edge start node per record.
        rv: Edge end node per record.
        ra: Entry position along the edge (m).
        rb: Exit position along the edge (m).
        rl: Edge length (m).
        lab: ``multilabel_search()[:4]``.
        prof: Station profiles.
        cfg: Caps and floors.
        params: Scoring parameters.
    """
    ls, ld, ln, lc = lab
    nc = len(rec_off) - 1
    ro = np.ascontiguousarray(rec_off, dtype=np.int32); u = np.ascontiguousarray(ru, dtype=np.int32); v = np.ascontiguousarray(rv, dtype=np.int32)
    a = np.ascontiguousarray(ra, dtype=np.float32); b = np.ascontiguousarray(rb, dtype=np.float32); l = np.ascontiguousarray(rl, dtype=np.float32)
    ost = np.full((nc, cfg.K), -1, dtype=np.int32); od = np.full((nc, cfg.K), 65535, dtype=np.uint16); of = np.zeros(nc, dtype=np.uint8); ocr = np.zeros(nc, dtype=np.uint8)
    ca, sp = cfg.array(), sp_array(params)
    pa = prof.args()
    _lib.process_cells(C.c_int(nc), _p(ro), _p(u), _p(v), _p(a), _p(b), _p(l), C.c_float(cfg.cell_radius_m), _p(ls), _p(ld), _p(ln), _p(lc), C.c_int(cfg.Kc), C.c_int(cfg.K),
                       C.c_int(cfg.Kcand), pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7], _p(sp), _p(ca), _p(ost), _p(od), _p(of), _p(ocr))
    return ost, od, of, ocr


def ptlos_array(dist, mode, h, v, params: ScoreParams = DEFAULT) -> np.ndarray:
    """Level-of-service value (stop score x walk decay) of each (distance, mode, headway, speed), computed by the C core.

    Args:
        dist: Walking distance (m).
        mode: Mode (0 bus, 1 tram, 2 rail).
        h: Headway (min).
        v: Speed (km/h).
        params: Scoring parameters.

    Returns:
        float64 array.
    """
    dist = np.ascontiguousarray(dist, dtype=np.float64); mode = np.ascontiguousarray(mode, dtype=np.int32)
    h = np.ascontiguousarray(h, dtype=np.float64); v = np.ascontiguousarray(v, dtype=np.float64)
    out = np.empty(len(dist)); sp = sp_array(params)
    _lib.ptlos_array(C.c_int(len(dist)), _p(dist), _p(mode), _p(h), _p(v), _p(sp), _p(out))
    return out


def recompute_cells(cell_station, cell_dist, prof: Profiles, params: ScoreParams = DEFAULT):
    """The fast recomputation: (los[nc] float32 on 0-100, slot[nc] int8 winning station slot, profile[nc] int8 winning profile) under `params`.

    Args:
        cell_station: ``[cells, K]`` station per slot.
        cell_dist: ``[cells, K]`` walking distance (m).
        prof: Station profiles.
        params: Scoring parameters.
    """
    st = np.ascontiguousarray(cell_station, dtype=np.int32); di = np.ascontiguousarray(cell_dist, dtype=np.uint16)
    nc, K = st.shape
    los = np.zeros(nc, dtype=np.float32); slot = np.full(nc, -1, dtype=np.int8); pidx = np.full(nc, -1, dtype=np.int8)
    sp = sp_array(params)
    _lib.recompute_cells(C.c_int(nc), C.c_int(K), C.c_int(prof.P), _p(st), _p(di), _p(prof.n), _p(prof.mode), _p(prof.h), _p(prof.v), _p(sp), _p(los), _p(slot), _p(pidx))
    return los, slot, pidx


def recompute_stops(prof: Profiles, params: ScoreParams = DEFAULT) -> np.ndarray:
    """Stop score of every (station, profile) slot, shape [S, P] float32 (0 for empty slots).

    Args:
        prof: Station profiles.
        params: Scoring parameters.
    """
    out = np.zeros((prof.S, prof.P), dtype=np.float32)
    sp = sp_array(params)
    _lib.recompute_stops(C.c_int(prof.S), C.c_int(prof.P), _p(prof.mode), _p(prof.h), _p(prof.v), _p(sp), _p(out))
    return out


def recompute_cells_fast(cell_station, cell_dist, prof: Profiles, params: ScoreParams = DEFAULT, ss: np.ndarray | None = None):
    """Like recompute_cells, but the stop scores are computed once (or passed in `ss`, shape [S, P]) and each cell only does the walk decay.

    Args:
        cell_station: ``[cells, K]`` station per slot.
        cell_dist: ``[cells, K]`` walking distance (m).
        prof: Station profiles.
        params: Scoring parameters.
        ss: Optional precomputed stop scores ``[S, P]``.

    Returns:
        ``(los, slot, profile)`` as :func:`recompute_cells`.
    """
    st = np.ascontiguousarray(cell_station, dtype=np.int32); di = np.ascontiguousarray(cell_dist, dtype=np.uint16)
    nc, K = st.shape
    if ss is None:
        ss = recompute_stops(prof, params)
    ss = np.ascontiguousarray(ss, dtype=np.float32)
    los = np.zeros(nc, dtype=np.float32); slot = np.full(nc, -1, dtype=np.int8); pidx = np.full(nc, -1, dtype=np.int8)
    sp = sp_array(params)
    _lib.recompute_cells_ss(C.c_int(nc), C.c_int(K), C.c_int(prof.P), _p(st), _p(di), _p(prof.n), _p(prof.mode), _p(ss), _p(sp), _p(los), _p(slot), _p(pidx))
    return los, slot, pidx
