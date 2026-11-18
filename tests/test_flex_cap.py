"""The cap (equilibrium + tolerant extremes + diversity): basic properties and the synthetic 'scoring changed afterwards' study."""
import importlib.util
from pathlib import Path

import numpy as np

from transitlos.flex import core, scoring as sc

_spec = importlib.util.spec_from_file_location("flex_cap_synthetic", Path(__file__).resolve().parent.parent / "tools" / "flex_cap_synthetic.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_cap_basic_properties():
    rng = np.random.default_rng(3)
    for _ in range(50):
        mode, h, v, d = S.make_group(rng, 40)
        q = sc.ptlos(d, mode, h, v)
        sel = core.select_items(mode, h, v, d, q, 8, S.AXES)
        assert 1 <= len(sel) <= 8 and len(set(sel)) == len(sel)
        assert sel[0] == int(np.argmax(q))                                   # the equilibrium is always first
        assert (core.select_items(mode, h, v, d, q, 8, S.AXES) == sel).all()  # deterministic


def test_cap_survives_axis_switched_off():
    r = S.run(K=10, n_max=250, n_groups=120, n_params=40, seed=5, quiet=True)
    assert r["flex cap"].mean() < r["top-10 by default"].mean()
    assert r["flex cap"].mean() < 0.15                                        # score points (0-100)


def test_nearest_is_always_kept_exactly():
    rng = np.random.default_rng(11)
    for _ in range(100):
        mode, h, v, d = S.make_group(rng, 60)
        q = sc.ptlos(d, mode, h, v)
        for cap in (2, 3, 5):
            sel = core.select_items(mode, h, v, d, q, cap, S.AXES)
            assert int(np.argmin(d)) in set(sel.tolist())
