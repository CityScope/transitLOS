"""Recomputable scoring: ONE parameter object (`ScoreParams`) that fully defines how stops and walking are scored, in numpy.

The same definition is evaluated by the C core (`flexcore.c`) and by the browser (`frontend/score.js`); `to_json` / `from_json` move it between them.

What can change after the data is processed
  * parameters      anchors, ceilings, elasticity, walking speed, plateau, t0 per mode, coupling k, the MRC value of each mode, an axis weight
                    (0 switches an axis OFF: mode, speed, headway)
  * SHAPES          each of the three curves can be replaced by another family or by any table of points (see `shapes.py`):
                      headway curve  H(headway min)            default = transitlos' log-linear-elasticity curve
                      speed curve    V(speed km/h)             default = Hill curve anchored at 30 km/h
                      walk curve     D0(u), u = (t - plateau)/t0(stop score, mode)     default = (1+u)^-p
  * aggregation     "nested"      min(1, ( MRC^wm * V^ws )^(1/(wm+ws)) * H^wh )   (default; = sqrt(MRC*V)*H with weights 1)
                    "product"     mode^w_mode * speed^w_speed * headway^w_headway
                    "power_mean"  generalised weighted mean of (MRC, V, H) with exponent `agg_p` (1 arithmetic, 0 geometric, -1 harmonic, large negative ~ min)

stop_score = aggregation(MRC(mode), V(speed), H(headway));   PTLOS(stop, d) = stop_score * D(walk time to the stop; stop_score, mode)
Modes are integers: 0 bus, 1 tram, 2 rail (the tier of the profile, as in transitlos.scoring.direction_subset).

Default parameters and shapes are read from `transitlos.scoring.parameters`, so with `ScoreParams()` this reproduces `transitlos.scoring` exactly.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Sequence, Tuple

import numpy as np

MODES = ("bus", "tram", "rail")


def _defaults():
    from ..scoring.parameters import REGIONS
    r = REGIONS["global"]
    return dict(mrc=(r.mr_mode["bus"], r.mr_mode["tram"], r.mr_mode["rail"]), v_anchor=r.v_anchor_kmh, v_bonus=r.v_ceiling_bonus, v_n=r.v_shape_n,
                h_sat=r.h_saturation_minutes, h_p=r.h_elasticity_p, h_q=r.h_elasticity_q, walk_speed=r.walking_speed_mps, plateau_m=r.t_plateau_distance_m,
                t0_base=(r.t0_base_minutes["bus"], r.t0_base_minutes["tram"], r.t0_base_minutes["rail"]), k=r.k_quality, p_shape=r.p_shape)


_D = _defaults()


@dataclass(frozen=True)
class Curve:
    """Shape of one 1-D scoring function.

    kind "default": the analytic shape of transitlos (its parameters live in `ScoreParams`).
    kind "table":   `y` sampled on a uniform grid of `n` points over [lo, hi] (`scale` "lin" or "log" in x), linear interpolation, clamped outside.
                    Any shape can be expressed this way (see `shapes.py`); numpy, C and JavaScript interpolate identically.
    """
    kind: str = "default"
    lo: float = 0.0
    hi: float = 1.0
    scale: str = "lin"
    y: Tuple[float, ...] = ()

    @property
    def n(self) -> int:
        """Number of table points.

        Returns:
            ``len(y)``.
        """
        return len(self.y)

    @staticmethod
    def from_function(f: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, n: int = 512, scale: str = "lin") -> "Curve":
        """Sample ``f`` on ``n`` points over ``[lo, hi]`` (linear or log spacing) into a table curve; negative values
        become 0.

        Args:
            f: Function of an array.
            lo: Lowest x.
            hi: Highest x.
            n: Number of points.
            scale: ``"lin"`` or ``"log"`` spacing in x.

        Returns:
            A table :class:`Curve`.
        """
        x = np.linspace(lo, hi, n) if scale == "lin" else np.exp(np.linspace(math.log(lo), math.log(hi), n))
        y = np.clip(np.asarray(f(x), dtype=np.float64), 0.0, None)
        return Curve("table", float(lo), float(hi), scale, tuple(float(v) for v in y))

    def eval(self, x):
        """Table lookup (kind "table" only).

        Args:
            x: Values (any shape).

        Returns:
            Interpolated scores (clamped outside ``[lo, hi]``).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        n = len(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = (x - self.lo) / (self.hi - self.lo) if self.scale == "lin" else (np.log(np.maximum(x, 1e-300)) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        pos = np.clip(pos * (n - 1), 0, n - 1)
        i = np.minimum(np.floor(pos).astype(np.int64), n - 2)
        fr = pos - i
        return y[i] * (1 - fr) + y[i + 1] * fr


DEFAULT_CURVE = Curve()


@dataclass(frozen=True)
class ScoreParams:
    """Every parameter of the transit level-of-service score: mode values (MRC), the speed, headway and walking curves, and
    how the three stop scores combine.

    Immutable: change a copy with :meth:`with_`. ``to_json`` / ``from_json`` save and load it (the browser editor uses
    the same JSON).
    """
    mrc: Tuple[float, float, float] = _D["mrc"]            # MRC value of bus, tram, rail
    v_anchor: float = _D["v_anchor"]                        # speed curve (default shape)
    v_bonus: float = _D["v_bonus"]
    v_n: float = _D["v_n"]
    h_sat: float = _D["h_sat"]                              # headway curve (default shape)
    h_p: float = _D["h_p"]
    h_q: float = _D["h_q"]
    walk_speed: float = _D["walk_speed"]                    # m/s
    plateau_m: float = _D["plateau_m"]                      # walking distance with no decay
    t0_base: Tuple[float, float, float] = _D["t0_base"]     # minutes, bus tram rail: scale of the walk decay
    k: float = _D["k"]                                      # t0 = t0_base * (1 + k * stop_score); 0 = no coupling
    p_shape: float = _D["p_shape"]                          # walk curve (default shape) exponent
    w_mode: float = 1.0
    w_speed: float = 1.0
    w_headway: float = 1.0
    agg: str = "nested"
    agg_p: float = 0.0
    headway_curve: Curve = DEFAULT_CURVE
    speed_curve: Curve = DEFAULT_CURVE
    walk_curve: Curve = DEFAULT_CURVE

    def with_(self, **kw) -> "ScoreParams":
        """A copy with some fields changed.

        Args:
            **kw: Fields to change.

        Returns:
            The new parameters.
        """
        return replace(self, **kw)

    def to_json(self) -> str:
        """The parameters as JSON text.

        Returns:
            JSON text.
        """
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "ScoreParams":
        """Parameters from JSON text written by :meth:`to_json`.

        Args:
            s: JSON text.

        Returns:
            The parameters.
        """
        d = json.loads(s)
        for key in ("mrc", "t0_base"):
            d[key] = tuple(d[key])
        for key in ("headway_curve", "speed_curve", "walk_curve"):
            c = d.get(key) or {}
            c["y"] = tuple(c.get("y", ()))
            d[key] = Curve(**c)
        return ScoreParams(**d)


DEFAULT = ScoreParams()


# ---------------------------------------------------------------------------------------------------------------- the functions
def mrc(mode, p: ScoreParams = DEFAULT):
    """Mode value (MRC) of each mode.

    Args:
        mode: Mode (0 bus, 1 tram, 2 rail).
        p: Scoring parameters.

    Returns:
        float64 array.
    """
    return np.asarray(p.mrc, dtype=np.float64)[np.asarray(mode, dtype=np.int64)]


def speed_score(v, p: ScoreParams = DEFAULT):
    """Speed score (0 to ``1 + v_bonus``): the speed curve, or the default saturating shape anchored at ``v_anchor``.

    Args:
        v: Speed (km/h).
        p: Scoring parameters.

    Returns:
        float64 array (0 for speed 0).
    """
    v = np.asarray(v, dtype=np.float64)
    if p.speed_curve.kind == "table":
        return np.where(v <= 0, 0.0, p.speed_curve.eval(v))
    m = 1.0 + p.v_bonus
    kk = p.v_anchor * (m - 1.0) ** (1.0 / p.v_n)
    return m * v ** p.v_n / (v ** p.v_n + kk ** p.v_n)


def headway_score(h, p: ScoreParams = DEFAULT):
    """Headway score (1 at or below ``h_sat``, falling with the headway): the headway curve, or the default log-quadratic
    shape.

    Args:
        h: Headway (min).
        p: Scoring parameters.

    Returns:
        float64 array.
    """
    h = np.asarray(h, dtype=np.float64)
    if p.headway_curve.kind == "table":
        return p.headway_curve.eval(h)
    eps_sat = p.h_p + p.h_q * math.log(p.h_sat)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(np.maximum(h, 1e-9) / p.h_sat)
        out = np.exp(-eps_sat * lr - (p.h_q / 2.0) * lr * lr)
    return np.where(h <= p.h_sat, 1.0, out)


def stop_score(mode, h, v, p: ScoreParams = DEFAULT):
    """Score of a stop pattern from its mode, headway and speed, combined as set by ``p.agg`` (nested, power mean or
    product).

    Args:
        mode: Mode (0 bus, 1 tram, 2 rail).
        h: Headway (min).
        v: Speed (km/h).
        p: Scoring parameters.

    Returns:
        float64 array, 0 to 1.
    """
    m, s, f = mrc(mode, p), speed_score(v, p), headway_score(h, p)
    zero = (m <= 0) | (s <= 0) | (f <= 0)
    if p.agg == "power_mean":
        w = np.array([p.w_mode, p.w_speed, p.w_headway], dtype=np.float64)
        wsum = w.sum()
        w = w / wsum if wsum > 0 else w
        if p.agg_p == 0.0:
            out = np.exp(w[0] * np.log(np.maximum(m, 1e-300)) + w[1] * np.log(np.maximum(s, 1e-300)) + w[2] * np.log(np.maximum(f, 1e-300)))
        else:
            out = (w[0] * np.maximum(m, 1e-300) ** p.agg_p + w[1] * np.maximum(s, 1e-300) ** p.agg_p + w[2] * np.maximum(f, 1e-300) ** p.agg_p) ** (1.0 / p.agg_p)
        out = np.minimum(1.0, out) if wsum > 0 else np.zeros_like(f)
    elif p.agg == "product":   # mode^wm * speed^ws * headway^wh: every factor multiplies (the data-driven fits use this)
        out = np.minimum(1.0, m ** p.w_mode * s ** p.w_speed * f ** p.w_headway)
    else:
        wms = p.w_mode + p.w_speed
        core = np.ones_like(f) if wms == 0 else (m ** p.w_mode * s ** p.w_speed) ** (1.0 / wms)
        out = np.minimum(1.0, core * f ** p.w_headway)
    return np.where(zero, 0.0, out)


def walk_decay(dist_m, ss, mode, p: ScoreParams = DEFAULT):
    """Share of a stop's score kept after walking ``dist_m``: 1 up to the plateau, then the walk curve of the walking time
    over ``t0`` (``t0`` grows with the stop score when ``k`` > 0).

    Args:
        dist_m: Walking distance (m).
        ss: Stop score.
        mode: Mode (0 bus, 1 tram, 2 rail).
        p: Scoring parameters.

    Returns:
        float64 array, 0 to 1.
    """
    t = np.asarray(dist_m, dtype=np.float64) / p.walk_speed / 60.0
    tp = p.plateau_m / p.walk_speed / 60.0
    t0 = np.asarray(p.t0_base, dtype=np.float64)[np.asarray(mode, dtype=np.int64)] * (1.0 + p.k * np.asarray(ss, dtype=np.float64))
    u = np.maximum(t - tp, 0.0) / t0
    d = p.walk_curve.eval(u) if p.walk_curve.kind == "table" else (1.0 + u) ** (-p.p_shape)
    return np.where(t <= tp, 1.0, d)


def ptlos(dist_m, mode, h, v, p: ScoreParams = DEFAULT):
    """Level of service a place gets from one stop pattern: stop score x walk decay.

    Args:
        dist_m: Walking distance (m).
        mode: Mode (0 bus, 1 tram, 2 rail).
        h: Headway (min).
        v: Speed (km/h).
        p: Scoring parameters.

    Returns:
        float64 array, 0 to 1.
    """
    ss = stop_score(mode, h, v, p)
    return ss * walk_decay(dist_m, ss, mode, p)
