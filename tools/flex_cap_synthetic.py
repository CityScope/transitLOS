"""Synthetic test of the CAP: does the kept set still contain (nearly) the winner after the scoring function is changed?

For many random groups of candidate stations (mode, headway, speed, distance) the cap keeps K. Then the scoring function is changed in ways the selection
never saw -- different speed/headway curves, walk decay, mode premiums reversed or removed, and axes switched OFF entirely (weight 0: e.g. mode no longer
matters) -- and the best PTLOS among the kept set is compared with the best among ALL candidates (regret, in 0-100 score points).

Baselines: the K best by the default score ("top-K"), the K nearest, K random. Run:  python tools/flex_cap_synthetic.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from transitlos.flex import core, scoring as sc  # noqa: E402

AXES = core.AX_DIST | core.AX_HEAD | core.AX_MODE | core.AX_SPEED


def make_group(rng, n):
    mode = rng.choice(3, size=n, p=[0.6, 0.15, 0.25])
    base_v = np.array([16.0, 20.0, 38.0])[mode]
    v = np.clip(base_v * rng.lognormal(0, 0.35, n), 6, 90)
    h = np.clip(np.array([25.0, 12.0, 8.0])[mode] * rng.lognormal(0, 0.8, n), 2, 58)
    d = rng.uniform(40, 1400, n)
    return mode.astype(np.int32), h, v, d


def nondominated(mode, h, v, d):
    keep = np.ones(len(mode), bool)
    for i in range(len(mode)):
        better = (d <= d[i]) & (mode >= mode[i]) & (h <= h[i]) & (v >= v[i]) & ((d < d[i]) | (mode > mode[i]) | (h < h[i]) | (v > v[i]))
        keep[i] = not better.any()
    return keep


def random_params(rng, i):
    kind = ['axis_off', 'weights', 'curves', 'mode_neutral'][i % 4]
    return _random_params(rng, i), kind


def _random_params(rng, i):
    """Scoring functions the selection has never seen. Every 4th one switches an axis off."""
    p = sc.DEFAULT
    mrc = list(p.mrc)
    r = rng.random()
    if r < 0.25:
        mrc = [1.0, 1.0, 1.0]                       # mode no longer matters
    elif r < 0.4:
        mrc = [1.0, 0.894, 0.8]                      # reversed: bus preferred
    elif r < 0.6:
        mrc = [rng.uniform(0.3, 1), rng.uniform(0.3, 1), rng.uniform(0.3, 1)]
    w = dict(w_mode=1.0, w_speed=1.0, w_headway=1.0)
    if i % 4 == 0:
        w[rng.choice(list(w))] = 0.0                 # one axis switched off
    elif i % 4 == 1:
        for k in w:
            w[k] = float(rng.choice([0.5, 1.0, 2.0]))
    return p.with_(mrc=tuple(mrc), v_anchor=rng.uniform(12, 50), v_n=rng.uniform(1, 3), v_bonus=rng.uniform(0, 0.4), h_sat=rng.uniform(3, 15),
                   walk_speed=rng.uniform(0.9, 1.6), plateau_m=rng.uniform(50, 400), k=rng.uniform(0.5, 5),
                   t0_base=tuple(np.asarray(p.t0_base) * rng.uniform(0.3, 2.5)), **w)


def run(n_groups=400, K=6, n_params=60, seed=0, n_max=70, cfg=None, quiet=False):
    cfg = cfg or core.Config()
    rng = np.random.default_rng(seed)
    pk = [(sc.DEFAULT, 'default')] + [random_params(rng, i) for i in range(n_params)]
    params = [p for p, _ in pk]; kinds = np.array([k for _, k in pk])
    names = ["flex cap", f"top-{K} by default", f"{K} nearest", f"{K} random"]
    reg = {k: [] for k in names}
    sizes, t_sel = [], 0.0
    for g in range(n_groups):
        n = int(rng.integers(K + 4, n_max))
        mode, h, v, d = make_group(rng, n)
        keep = nondominated(mode, h, v, d)
        mode, h, v, d = mode[keep], h[keep], v[keep], d[keep]
        n = len(mode)
        if n <= K:
            continue
        order = np.argsort(d, kind="stable")
        mode, h, v, d = mode[order], h[order], v[order], d[order]
        q = sc.ptlos(d, mode, h, v)
        t0 = time.perf_counter()
        sel = core.select_items(mode, h, v, d, q, K, AXES, cfg)
        t_sel += time.perf_counter() - t0
        sizes.append(n)
        picks = {names[0]: sel, names[1]: np.argsort(-q, kind="stable")[:K], names[2]: np.arange(K), names[3]: rng.choice(n, K, replace=False)}
        for p in params:
            full = sc.ptlos(d, mode, h, v, p)
            best = full.max()
            for k, idx in picks.items():
                reg[k].append((best - full[idx].max()) * 100.0)
    if quiet:
        return {k: np.array(v) for k, v in reg.items()}
    print(f"K={K}: {len(sizes)} groups with more than K non-dominated candidates (mean {np.mean(sizes):.1f}); selection time {t_sel / len(sizes) * 1e6:.0f} us/group")
    print(f"{'method':22s} {'mean regret':>12s} {'p99':>8s} {'max':>8s} {'share exact':>12s}")
    for k in names:
        r = np.array(reg[k])
        print(f"{k:22s} {r.mean():12.3f} {np.percentile(r, 99):8.2f} {r.max():8.2f} {np.mean(r < 1e-9):12.3f}")
    npar = len(params)
    for kind in ['default', 'axis_off', 'weights', 'curves', 'mode_neutral']:
        m = np.tile(kinds == kind, len(sizes))
        print(f"   by change '{kind}': " + "  ".join(f"{k.split(' ')[0]}={np.array(reg[k])[m].mean():.3f}/{np.array(reg[k])[m].max():.1f}" for k in names))
    return {k: np.array(v) for k, v in reg.items()}


if __name__ == "__main__":
    for K, nmax in ((10, 70), (10, 250), (6, 250), (4, 250)):
        print(f"--- groups of up to {nmax} candidates")
        run(K=K, n_max=nmax, n_groups=300)
        print()
