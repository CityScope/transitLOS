"""Cell scoring as plain array code that runs on a CUDA GPU (CuPy) when one is usable, else on the CPU (NumPy): the same definition as scoring.py / flexcore.c / score.js.

    ev = XPData(subs, nbg)            # uploads the sample once (cells, stations, block-group index); subs = objects with prof, cst, cdist, bg, pop
    num = ev.numerator(params)        # [nbg]  sum over each block group's cells of  population * level of service  (float64, on the host)

The device is chosen by `transitlos.flex.backend` (CuPy if a CUDA device works, else NumPy); cells are processed in chunks, so the full US sample fits on any card.
"""
import numpy as np

from . import backend as _backend


def backend():
    """The array library of the current device (see :func:`transitlos.flex.backend.xp`).

    Returns:
        The ``cupy`` or ``numpy`` module.
    """
    return _backend.xp()


def _table(c):
    return None if not c or c.get("kind") != "table" else c


class XPData:
    """Station profiles and cell stations of several areas held on the CPU or GPU, to recompute block-group scores fast for
    many parameter sets (the fits).

    Args:
        subs: Per-area data (profiles, cell stations, distances, block group and population per cell).
        nbg: Number of block groups.
        xp: Array library; ``None`` = :func:`backend`.
        chunk_cells: Cells per chunk (memory).
    """
    def __init__(self, subs, nbg, xp=None, chunk_cells=None):
        self.xp = xp or backend()
        self.nbg = nbg
        xp = self.xp
        self.parts = []
        for s in subs:
            pr = s.prof
            self.parts.append({
                "mode": xp.asarray(np.asarray(pr.mode, np.int8)), "h": xp.asarray(np.asarray(pr.h, np.float32)), "v": xp.asarray(np.asarray(pr.v, np.float32)),
                "n": xp.asarray(np.asarray(pr.n, np.int32)), "P": int(pr.P),
                "cst": xp.asarray(np.ascontiguousarray(s.cst, np.int32)), "cd": xp.asarray(np.ascontiguousarray(s.cdist, np.uint16)),
                "bg": xp.asarray(np.asarray(s.bg, np.int64)), "pop": xp.asarray(np.asarray(s.pop, np.float64)), "nc": int(len(s.bg)), "K": int(s.cst.shape[1])})
        self.chunk = chunk_cells or (400_000 if xp is not np else 150_000)

    # ------------------------------------------------------------------------------------------------------------------ curves
    def _tab(self, t, x):
        xp = self.xp
        y, n, log = t["y"], t["n"], t["log"]
        pos = (xp.log(xp.maximum(x, 1e-300)) if log else x) - t["a"]
        pos = xp.clip(pos / t["span"] * (n - 1), 0, n - 1)
        i = xp.minimum(xp.floor(pos).astype(xp.int64), n - 2)
        f = pos - i
        return y[i] * (1 - f) + y[i + 1] * f

    def _prep(self, p):
        xp = self.xp
        def tab(c):
            c = _table(c)
            if c is None:
                return None
            log = c["scale"] == "log"
            a, b = (np.log(c["lo"]), np.log(c["hi"])) if log else (c["lo"], c["hi"])
            return {"y": xp.asarray(np.asarray(c["y"], np.float64)), "n": len(c["y"]), "log": log, "a": a, "span": b - a}
        m = 1 + p["v_bonus"]
        kk = p["v_anchor"] * (m - 1) ** (1 / p["v_n"])
        return {"p": p, "m": m, "kn": kk ** p["v_n"], "eps": p["h_p"] + p["h_q"] * np.log(p["h_sat"]), "wms": p["w_mode"] + p["w_speed"],
                "wsum": p["w_mode"] + p["w_speed"] + p["w_headway"], "th": tab(p.get("headway_curve")), "tv": tab(p.get("speed_curve")), "tw": tab(p.get("walk_curve")),
                "ws": p["walk_speed"], "tp": p["plateau_m"] / p["walk_speed"] / 60}

    def stop_scores(self, S, part):
        """Stop score of every (station, profile) slot of one area.

        Args:
            S: Prepared parameters.
            part: One area's data.

        Returns:
            ``[S, P]`` array.
        """
        xp, p = self.xp, S["p"]
        mode, h, v = part["mode"], part["h"].astype(xp.float64), part["v"].astype(xp.float64)
        mrc = xp.asarray(np.asarray(p["mrc"], np.float64))
        mr = mrc[xp.maximum(mode, 0)]
        if S["tv"] is not None:
            sp = self._tab(S["tv"], v)
        else:
            vn = xp.power(xp.maximum(v, 0), p["v_n"])
            sp = S["m"] * vn / (vn + S["kn"])
        sp = xp.where(v <= 0, 0.0, sp)
        if S["th"] is not None:
            f = self._tab(S["th"], h)
        else:
            lr = xp.log(xp.maximum(h, 1e-300) / p["h_sat"])
            f = xp.where(h <= p["h_sat"], 1.0, xp.exp(-S["eps"] * lr - (p["h_q"] / 2) * lr * lr))
        bad = (mode < 0) | (mr <= 0) | (sp <= 0) | (f <= 0)
        mr_, sp_, f_ = xp.maximum(mr, 1e-300), xp.maximum(sp, 1e-300), xp.maximum(f, 1e-300)
        if p["agg"] == "power_mean":
            w0, w1, w2 = p["w_mode"] / S["wsum"], p["w_speed"] / S["wsum"], p["w_headway"] / S["wsum"]
            ap = p["agg_p"]
            r = xp.exp(w0 * xp.log(mr_) + w1 * xp.log(sp_) + w2 * xp.log(f_)) if ap == 0 else xp.power(w0 * xp.power(mr_, ap) + w1 * xp.power(sp_, ap) + w2 * xp.power(f_, ap), 1 / ap)
        elif p["agg"] == "product":
            r = xp.power(mr_, p["w_mode"]) * xp.power(sp_, p["w_speed"]) * xp.power(f_, p["w_headway"])
        else:
            core = 1.0 if S["wms"] == 0 else xp.power(xp.power(mr_, p["w_mode"]) * xp.power(sp_, p["w_speed"]), 1 / S["wms"])
            r = core * xp.power(f_, p["w_headway"])
        return xp.where(bad, 0.0, xp.minimum(r, 1.0)).astype(xp.float32)

    def cell_scores(self, S, part, lo, hi):
        """Level of service of the cells ``lo:hi`` of one area (best station and profile per cell).

        Args:
            S: Prepared parameters.
            part: One area's data.
            lo: First cell.
            hi: Last cell (exclusive).

        Returns:
            Array of cell scores.
        """
        xp, p = self.xp, S["p"]
        P, K = part["P"], part["K"]
        ss = part["ss"]
        st = part["cst"][lo:hi]                       # [n, K]
        valid_st = st >= 0
        sti = xp.maximum(st, 0)
        q = ss[sti]                                   # [n, K, P]
        pr = xp.arange(P, dtype=xp.int32)[None, None, :]
        ok = valid_st[:, :, None] & (pr < part["n"][sti][:, :, None])
        mode = xp.maximum(part["mode"][sti], 0)      # [n, K, P]
        t = part["cd"][lo:hi].astype(xp.float32)[:, :, None] / S["ws"] / 60.0
        t0 = xp.asarray(np.asarray(p["t0_base"], np.float32))[mode]
        u = (t - S["tp"]) / (t0 * (1 + p["k"] * q))
        if S["tw"] is not None:
            d = self._tab(S["tw"], u.astype(xp.float64)).astype(xp.float32)
        else:
            d = xp.power(1 + xp.maximum(u, 0), -p["p_shape"])
        val = xp.where(t > S["tp"], q * d, q)
        val = xp.where(ok, val, 0.0)
        return val.reshape(hi - lo, -1).max(axis=1) * 100.0

    # ------------------------------------------------------------------------------------------------------------------ public
    def numerator(self, params):
        """params: the ScoreParams as a dict (to_json). Returns the numerator of every block group's predictor (numpy float64 [nbg]).

        Args:
            params: Scoring parameters as a dict.
        """
        xp = self.xp
        S = self._prep(params)
        num = xp.zeros(self.nbg, dtype=xp.float64)
        for part in self.parts:
            part["ss"] = self.stop_scores(S, part)
            for lo in range(0, part["nc"], self.chunk):
                hi = min(lo + self.chunk, part["nc"])
                los = self.cell_scores(S, part, lo, hi).astype(xp.float64)
                bg = part["bg"][lo:hi]
                ok = bg >= 0
                num += xp.bincount(bg[ok], weights=(part["pop"][lo:hi] * los)[ok], minlength=self.nbg)
        return xp.asnumpy(num) if xp is not np else num
