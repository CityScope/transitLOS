"""transitlos.flex.scoring / core: the recomputable scoring reproduces transitlos.scoring, and numpy, C (and JSON) agree for every shape."""
import itertools

import numpy as np
import pytest

from transitlos.flex import core, scoring as sc, shapes as sh
from transitlos.scoring.accessibility_score import accessibility_score
from transitlos.scoring.frequency import frequency_score
from transitlos.scoring.mrc import mrc_score
from transitlos.scoring.speed import speed_score
from transitlos.scoring.stop_score import stop_score as ref_stop_score

RT = {0: 3, 1: 0, 2: 2}
MN = {0: "bus", 1: "tram", 2: "rail"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 1500
    return rng.integers(0, 3, n), rng.uniform(1, 400, n), rng.uniform(0.5, 120, n), rng.uniform(0, 3000, n)


def test_default_equals_transitlos(data):
    mode, h, v, d = data
    ref = np.array([ref_stop_score(mrc_score(route_type=RT[m], headway_cv=None), speed_score(s), frequency_score(x)) for m, x, s in zip(mode, h, v)])
    assert np.abs(ref - sc.stop_score(mode, h, v)).max() < 1e-12
    refp = np.array([accessibility_score(s, dd / 1.3 / 60, MN[m]) for s, dd, m in zip(ref, d, mode)])
    assert np.abs(refp - sc.ptlos(d, mode, h, v)).max() < 1e-12
    assert np.abs(refp - core.ptlos_array(d, mode, h, v)).max() < 1e-12


@pytest.mark.parametrize("name,p", [("harmonic", -1.0), ("geometric", 0.0)])
def test_legacy_aggregations_are_power_means(data, name, p):
    mode, h, v, _ = data
    ref = np.array([ref_stop_score(mrc_score(route_type=RT[m], headway_cv=None), speed_score(s), frequency_score(x), mode=name) for m, x, s in zip(mode, h, v)])
    pp = sc.DEFAULT.with_(agg="power_mean", agg_p=p, w_mode=1 / 3, w_speed=1 / 3, w_headway=1 / 3)
    assert np.abs(ref - sc.stop_score(mode, h, v, pp)).max() < 1e-12
    assert np.abs(ref - core.ptlos_array(np.zeros(len(h)), mode, h, v, pp)).max() < 1e-12


def test_every_shape_numpy_equals_c(data):
    mode, h, v, d = data
    H = [sh.headway_as_table(), sh.headway_exponential(), sh.headway_power(), sh.headway_logistic(), sh.headway_step(), sh.headway_piecewise([(2, 1), (15, .8), (60, .3), (300, .05)])]
    V = [sh.speed_as_table(), sh.speed_linear(), sh.speed_logistic(), sh.speed_step(), sh.speed_piecewise([(0, 0), (20, .5), (60, 1)])]
    W = [sh.walk_offset_power(), sh.walk_exponential(), sh.walk_gaussian(), sh.walk_linear(), sh.walk_logistic(), sh.walk_hard_buffer(), sh.walk_piecewise([(0, 1), (1, .7), (5, .1)])]
    for hc, vc, wc in itertools.product(H, V, W):
        p = sc.DEFAULT.with_(headway_curve=hc, speed_curve=vc, walk_curve=wc)
        assert np.abs(sc.ptlos(d, mode, h, v, p) - core.ptlos_array(d, mode, h, v, p)).max() < 1e-9


def test_table_of_default_shapes_is_close_to_analytic(data):
    mode, h, v, d = data
    p = sc.DEFAULT.with_(headway_curve=sh.headway_as_table(), speed_curve=sh.speed_as_table(), walk_curve=sh.walk_offset_power())
    assert np.abs(sc.ptlos(d, mode, h, v) - sc.ptlos(d, mode, h, v, p)).max() < 5e-3


def test_axis_switched_off_and_json_roundtrip(data):
    mode, h, v, d = data
    off = sc.DEFAULT.with_(w_mode=0.0, mrc=(1.0, 0.5, 0.1))
    a = sc.stop_score(mode, h, v, off)
    assert np.allclose(a, sc.stop_score(np.zeros_like(mode), h, v, off))          # mode no longer matters
    p = sc.DEFAULT.with_(headway_curve=sh.headway_exponential(), walk_curve=sh.walk_gaussian(), agg="power_mean", agg_p=-1.0)
    q = sc.ScoreParams.from_json(p.to_json())
    assert np.allclose(sc.ptlos(d, mode, h, v, p), sc.ptlos(d, mode, h, v, q))
