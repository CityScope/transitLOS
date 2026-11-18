"""The array-code scoring (GPU when CUDA is usable, NumPy otherwise) gives the C core's block-group numerators, and the device choice falls back to the CPU."""
import json
import types

import numpy as np

from transitlos.flex import backend, core, scoring as sc, xp_scoring


def _sample(seed=0, S=300, nc=4000, K=8, P=5, nbg=60):
    rng = np.random.default_rng(seed)
    mode = rng.integers(0, 3, (S, P)).astype(np.int8)
    mode[rng.random((S, P)) < 0.3] = -1
    mode[:, 0] = np.where(mode[:, 0] < 0, 0, mode[:, 0])          # every station has at least one profile, profiles are packed first
    for i in range(S):
        k = int((mode[i] >= 0).sum()); mode[i, :k] = np.maximum(mode[i, :k], 0); mode[i, k:] = -1
    h = rng.uniform(2, 120, (S, P)).astype(np.float32); v = rng.uniform(5, 60, (S, P)).astype(np.float32)
    prof = core.Profiles(mode, h, v)
    cst = rng.integers(-1, S, (nc, K)).astype(np.int32)
    cdist = rng.integers(0, 2500, (nc, K)).astype(np.uint16)
    bg = rng.integers(-1, nbg, nc).astype(np.int64)
    pop = rng.uniform(0, 80, nc)
    return types.SimpleNamespace(prof=prof, cst=cst, cdist=cdist, bg=bg, pop=pop), nbg


def test_numpy_matches_c_core():
    sub, nbg = _sample()
    for p in (sc.DEFAULT, sc.DEFAULT.with_(h_sat=9.0, k=2.0, w_headway=1.7), sc.DEFAULT.with_(agg="power_mean", agg_p=-1.0, w_mode=0.6)):
        los, _, _ = core.recompute_cells_fast(sub.cst, sub.cdist, sub.prof, p)
        ok = sub.bg >= 0
        ref = np.bincount(sub.bg[ok], weights=sub.pop[ok] * los[ok].astype(np.float64), minlength=nbg)
        ev = xp_scoring.XPData([sub], nbg, xp=np)
        got = ev.numerator(json.loads(p.to_json()))
        assert np.allclose(got, ref, rtol=2e-4, atol=1e-2 * max(ref.max(), 1) * 1e-3)


def test_device_falls_back_to_cpu(monkeypatch):
    backend._cache.clear()
    monkeypatch.setenv("TRANSITLOS_DEVICE", "cpu")
    assert backend.device() == "cpu" and backend.xp() is np
    backend._cache.clear()
    monkeypatch.setenv("TRANSITLOS_DEVICE", "auto")
    monkeypatch.setitem(__import__("sys").modules, "cupy", None)   # an import error must mean 'cpu', not a crash
    assert backend.device() == "cpu"
    backend._cache.clear()
