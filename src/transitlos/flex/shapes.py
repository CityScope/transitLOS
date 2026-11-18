"""Library of alternative shapes for the three scoring curves (`ScoreParams.headway_curve / speed_curve / walk_curve`).

Every function returns a `Curve` (a sampled table), so the numpy, C and JavaScript evaluators share it exactly. Use `piecewise(...)` or
`Curve.from_function(f, lo, hi)` for any shape that is not listed. Domains: headway minutes (log grid 1-720), speed km/h (linear 0-150), walk
`u` = (walking time - plateau) / t0 (linear 0-40).

    from transitlos.flex import shapes, ScoreParams
    p = ScoreParams(headway_curve=shapes.headway_exponential(sat=5, scale=40), speed_curve=shapes.speed_linear(0, 40),
                    walk_curve=shapes.walk_gaussian(), agg="power_mean", agg_p=-1)
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .scoring import Curve, ScoreParams, DEFAULT, headway_score, speed_score

# -- grids ------------------------------------------------------------------------------------------------------------------
H_LO, H_HI, H_N = 1.0, 720.0, 512      # headway, minutes, log spacing
V_LO, V_HI, V_N = 0.0, 150.0, 512      # speed km/h
U_LO, U_HI, U_N = 0.0, 40.0, 1024      # walk u


def _headway(f) -> Curve:
    return Curve.from_function(f, H_LO, H_HI, H_N, "log")


def _speed(f) -> Curve:
    return Curve.from_function(f, V_LO, V_HI, V_N, "lin")


def _walk(f) -> Curve:
    return Curve.from_function(f, U_LO, U_HI, U_N, "lin")


def piecewise(points: Sequence[Tuple[float, float]], lo: float, hi: float, scale: str, n: int) -> Curve:
    """Curve through (x, score) points (linear in x between points, flat beyond the first and last).

    Args:
        points: ``(x, score)`` points.
        lo: Lowest x of the table.
        hi: Highest x of the table.
        scale: ``"lin"`` or ``"log"``.
        n: Table points.

    Returns:
        A table :class:`Curve`.
    """
    xs, ys = zip(*sorted(points))
    return Curve.from_function(lambda x: np.interp(x, xs, ys), lo, hi, n, scale)


# -- headway: H(minutes) ------------------------------------------------------------------------------------------------------
def headway_default() -> Curve:
    """The default analytic headway curve.

    Returns:
        A default :class:`Curve`.
    """
    return Curve()


def headway_as_table(p: ScoreParams = DEFAULT) -> Curve:
    """The default analytic shape sampled as a table (for testing and as a starting point to edit).

    Args:
        p: Scoring parameters whose default shape is sampled.

    Returns:
        A table :class:`Curve`.
    """
    return _headway(lambda x: headway_score(x, p.with_(headway_curve=Curve())))


def headway_exponential(sat: float = 5.0, scale: float = 40.0) -> Curve:
    """Headway curve 1 up to ``sat`` minutes, then ``exp(-(h - sat) / scale)``.

    Args:
        sat: Headway (min) with full score.
        scale: Decay scale (min).

    Returns:
        A table :class:`Curve`.
    """
    return _headway(lambda h: np.where(h <= sat, 1.0, np.exp(-(h - sat) / scale)))


def headway_power(sat: float = 5.0, exponent: float = 0.6) -> Curve:
    """Headway curve 1 up to ``sat`` minutes, then ``(sat / h) ** exponent``.

    Args:
        sat: Headway (min) with full score.
        exponent: Decay exponent.

    Returns:
        A table :class:`Curve`.
    """
    return _headway(lambda h: np.where(h <= sat, 1.0, (sat / h) ** exponent))


def headway_logistic(mid: float = 30.0, steepness: float = 0.12) -> Curve:
    """Logistic headway curve, half score at ``mid`` minutes.

    Args:
        mid: Headway (min) at half score.
        steepness: Steepness per minute.

    Returns:
        A table :class:`Curve`.
    """
    return _headway(lambda h: 1.0 / (1.0 + np.exp(steepness * (h - mid))))


def headway_step(levels: Sequence[Tuple[float, float]] = ((5, 1.0), (10, 0.9), (20, 0.7), (30, 0.5), (60, 0.25), (120, 0.1))) -> Curve:
    """Staircase: score `s` while headway <= `h` (the last level applies beyond with 0).

    Args:
        levels: ``(headway, score)`` steps.

    Returns:
        A table :class:`Curve`.
    """
    lv = sorted(levels)
    def f(h):
        out = np.zeros_like(h)
        for hh, s in reversed(lv):
            out = np.where(h <= hh, s, out)
        return out
    return _headway(f)


def headway_piecewise(points: Sequence[Tuple[float, float]]) -> Curve:
    """Headway curve through ``(headway, score)`` points.

    Args:
        points: ``(headway minutes, score)`` points.

    Returns:
        A table :class:`Curve`.
    """
    return piecewise(points, H_LO, H_HI, "log", H_N)


# -- speed: V(km/h) -----------------------------------------------------------------------------------------------------------
def speed_default() -> Curve:
    """The default analytic speed curve.

    Returns:
        A default :class:`Curve`.
    """
    return Curve()


def speed_as_table(p: ScoreParams = DEFAULT) -> Curve:
    """The default speed shape sampled as a table (to test or edit).

    Args:
        p: Scoring parameters whose default shape is sampled.

    Returns:
        A table :class:`Curve`.
    """
    return _speed(lambda x: speed_score(x, p.with_(speed_curve=Curve())))


def speed_linear(v0: float = 5.0, v1: float = 40.0, ceiling: float = 1.0) -> Curve:
    """Speed curve rising linearly from 0 at ``v0`` to ``ceiling`` at ``v1`` km/h.

    Args:
        v0: Speed with score 0.
        v1: Speed with full score.
        ceiling: Highest score.

    Returns:
        A table :class:`Curve`.
    """
    return _speed(lambda v: ceiling * np.clip((v - v0) / (v1 - v0), 0.0, 1.0))


def speed_logistic(mid: float = 20.0, steepness: float = 0.2, ceiling: float = 1.0) -> Curve:
    """Logistic speed curve, half of ``ceiling`` at ``mid`` km/h.

    Args:
        mid: Speed at half score.
        steepness: Steepness per km/h.
        ceiling: Highest score.

    Returns:
        A table :class:`Curve`.
    """
    return _speed(lambda v: ceiling / (1.0 + np.exp(-steepness * (v - mid))))


def speed_step(levels: Sequence[Tuple[float, float]] = ((10, 0.2), (20, 0.5), (30, 0.8), (45, 1.0))) -> Curve:
    """Score `s` once speed >= `v` (levels sorted by v).

    Args:
        levels: ``(speed, score)`` steps.

    Returns:
        A table :class:`Curve`.
    """
    lv = sorted(levels)
    def f(v):
        out = np.zeros_like(v)
        for vv, s in lv:
            out = np.where(v >= vv, s, out)
        return out
    return _speed(f)


def speed_piecewise(points: Sequence[Tuple[float, float]]) -> Curve:
    """Speed curve through ``(speed, score)`` points.

    Args:
        points: ``(speed km/h, score)`` points.

    Returns:
        A table :class:`Curve`.
    """
    return piecewise(points, V_LO, V_HI, "lin", V_N)


# -- walking: D0(u), u = (t - plateau)/t0 ---------------------------------------------------------------------------------------
def walk_default() -> Curve:
    """The default walking curve ``(1 + u) ** -p_shape``.

    Returns:
        A default :class:`Curve`.
    """
    return Curve()


def walk_offset_power(p: float = 1.0) -> Curve:
    """Walking curve ``(1 + u) ** -p`` (u = walking time beyond the plateau over ``t0``).

    Args:
        p: Exponent.

    Returns:
        A table :class:`Curve`.
    """
    return _walk(lambda u: (1.0 + u) ** (-p))


def walk_exponential(scale: float = 1.0) -> Curve:
    """Walking curve ``exp(-u / scale)``.

    Args:
        scale: Decay scale in units of ``t0``.

    Returns:
        A table :class:`Curve`.
    """
    return _walk(lambda u: np.exp(-u / scale))


def walk_gaussian(scale: float = 1.0) -> Curve:
    """Walking curve ``exp(-(u / scale) ** 2)``.

    Args:
        scale: Decay scale in units of ``t0``.

    Returns:
        A table :class:`Curve`.
    """
    return _walk(lambda u: np.exp(-(u / scale) ** 2))


def walk_linear(u_max: float = 3.0) -> Curve:
    """Walking curve falling linearly to 0 at ``u_max``.

    Args:
        u_max: Value of u with score 0.

    Returns:
        A table :class:`Curve`.
    """
    return _walk(lambda u: np.clip(1.0 - u / u_max, 0.0, 1.0))


def walk_logistic(mid: float = 2.0, steepness: float = 2.0) -> Curve:
    """Logistic walking curve, scaled to 1 at u = 0.

    Args:
        mid: u at half score.
        steepness: Steepness.

    Returns:
        A table :class:`Curve`.
    """
    f = lambda u: 1.0 / (1.0 + np.exp(steepness * (u - mid)))
    return _walk(lambda u: np.clip(f(u) / f(np.zeros(1))[0], 0.0, 1.0))


def walk_hard_buffer() -> Curve:
    """Everything inside the plateau distance scores fully, nothing beyond it (use `plateau_m` as the buffer radius)."""
    return _walk(lambda u: np.where(u <= 0.0, 1.0, 0.0))


def walk_piecewise(points: Sequence[Tuple[float, float]]) -> Curve:
    """Walking curve through ``(u, score)`` points.

    Args:
        points: ``(u, score)`` points.

    Returns:
        A table :class:`Curve`.
    """
    return piecewise(points, U_LO, U_HI, "lin", U_N)
