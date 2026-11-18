"""Speed score `V(speed)`: converts a stop's average operating speed into a score.

    V(v) = M * v^n / (v^n + K^n),    K = anchor * (M - 1)^(1/n)

a smooth increasing curve with `V(anchor) = 1` and `V -> M = 1 + bonus` as `v -> infinity` (no hard cap). With the defaults
(anchor 30 km/h, bonus 0.15, n = 2): `K ~= 11.62`, `V(30) = 1`, `V(100) ~= 1.135`.

The speed is the commercial (operating) speed, dwell time at stops included, not the cruising speed between stops.
"""

from __future__ import annotations

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def speed_score(
    avg_speed_kmh: float,
    region: str = "global",
    anchor_kmh: float | None = None,
    ceiling_bonus: float | None = None,
    shape_n: float | None = None,
) -> float:
    """Compute `V(speed)` from average operating speed.

    Args:
        avg_speed_kmh: Average operating (commercial) speed in km/h --
            i.e. distance/time averaged over the whole route, INCLUDING
            dwell time at stops, not pure between-stops cruising speed. Must
            be >= 0.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region now shares identical, universal
            parameters (see `parameters.py`'s module docstring).
        anchor_kmh: Speed at which `V` reaches exactly 1.0. Defaults to the
            region's `v_anchor_kmh` (30.0).
        ceiling_bonus: The curve's asymptotic ceiling above 1.0, approached
            as avg_speed_kmh -> infinity, never hit at any specific finite
            speed and NOT pinned to any named high-speed benchmark (an
            earlier same-day revision tried pinning it to German
            Regional-Express's speed and was reverted -- see module
            docstring, Revision 4). Defaults to the region's
            `v_ceiling_bonus` (0.15), an explicit design-judgment choice.
        shape_n: Hill-function exponent controlling how quickly the curve
            rises below the anchor (higher n => lower scores at low speed).
            Defaults to the region's `v_shape_n` (2.0).

    Returns:
        A float, strictly increasing in avg_speed_kmh, exactly 0.0 only at
        avg_speed_kmh == 0, asymptotically approaching (but never reaching)
        `1 + ceiling_bonus` as avg_speed_kmh -> infinity.

    Raises:
        ValueError: If avg_speed_kmh is negative.
    """
    if avg_speed_kmh < 0:
        raise ValueError(f"avg_speed_kmh must be >= 0, got {avg_speed_kmh}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    anchor = anchor_kmh if anchor_kmh is not None else params.v_anchor_kmh
    bonus = ceiling_bonus if ceiling_bonus is not None else params.v_ceiling_bonus
    n = shape_n if shape_n is not None else params.v_shape_n

    # Hill-function (generalized/power-law Michaelis-Menten) curve
    # V(v) = M*v^n/(v^n+K^n): M=1+bonus is the asymptote (an explicit
    # design-judgment ceiling, not fitted to any second point -- see module
    # docstring, Revision 4, for why a second fitted point was tried twice
    # and dropped both times), K is solved so V(anchor)=1.0 exactly, and n
    # is a fixed shape parameter chosen to keep the curve from rising too
    # steeply at low speed.
    m = 1.0 + bonus
    k = anchor * (m - 1.0) ** (1.0 / n)
    return m * avg_speed_kmh**n / (avg_speed_kmh**n + k**n)
