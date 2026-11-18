"""Numeric default parameters of the transitLOS scoring formulas (one place for every constant).

``REGIONS`` maps every region key to the same universal parameters; it is kept so that callers passing ``region=`` keep working.
``UNVALIDATED_REGIONS`` is empty for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegionParameters:
    """Container for the scoring model's default parameters.

    Attributes:
        weights: (w_mrc, w_v, w_h) weights of `MRC`, `V` and `H` in `stop_score`'s weighted geometric mean (equal thirds).
        walking_speed_mps: Walking speed, m/s (1.3).
        t_plateau_distance_m: Walk distance (m) at or below which `D(t) = 1` whatever the stop score (200 m, ~2.56 min).
        t0_base_minutes: Per-mode base scale (minutes) of `D`'s decay before the `(1 + k * stop_score)` scaling: bus 6.6, rail 9.4,
            tram 14.55.
        k_quality: Quality coupling `k` in `t0(stop_score) = t0_base(mode) * (1 + k * stop_score)` (2.5).
        p_shape: Shape exponent `p` in `D(t) = (1 + (t - t_plateau) / t0)^(-p)` (1.0, all modes).
        h_saturation_minutes: Headway (min) at or below which `H(headway) = 1` (5 min).
        h_elasticity_p: Intercept `p` of the local headway elasticity `eps(headway) = p + q * ln(headway)`.
        h_elasticity_q: Slope `q` of the same elasticity.
        v_anchor_kmh: Speed (km/h) at which `V(v) = M * v^n / (v^n + K^n)` equals 1 (30 km/h).
        v_ceiling_bonus: Asymptotic ceiling of `V` above 1: `V -> M = 1 + bonus` as `v -> infinity` (0.15).
        v_shape_n: Hill exponent `n` of `V` (2.0: keeps `V` low near walking speed).
        mr_mode: `MR_mode(mode)` lookup: rail 1.000, tram / light rail / BRT 0.894, bus 0.800.
        comfort_scores: `C(occupancy_status)` lookup keyed by GTFS-realtime `VehiclePosition.OccupancyStatus` (optional, not used
            by the default pipeline).
    """

    weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    walking_speed_mps: float = 1.3
    t_plateau_distance_m: float = 200.0
    t0_base_minutes: dict[str, float] = field(
        default_factory=lambda: {"bus": 6.6, "rail": 9.4, "tram": 14.55}
    )
    k_quality: float = 2.5
    p_shape: float = 1.0
    h_saturation_minutes: float = 5.0
    h_elasticity_p: float = -0.06581572485740672
    h_elasticity_q: float = 0.15634627853115843
    v_anchor_kmh: float = 30.0
    v_ceiling_bonus: float = 0.15
    v_shape_n: float = 2.0
    mr_mode: dict[str, float] = field(
        default_factory=lambda: {"rail": 1.000, "tram": 0.894, "bus": 0.800}
    )
    comfort_scores: dict[str, float] = field(
        default_factory=lambda: {
            "EMPTY": 1.00,
            "MANY_SEATS_AVAILABLE": 1.00,
            "FEW_SEATS_AVAILABLE": 0.85,
            "STANDING_ROOM_ONLY": 0.55,
            "CRUSHED_STANDING_ROOM_ONLY": 0.40,
            "FULL": 0.00,
            "NOT_ACCEPTING_PASSENGERS": 0.00,
        }
    )


#: Region lookup: every key maps to the same universal parameters (kept for callers passing `region=`).
REGIONS: dict[str, RegionParameters] = {
    "global": RegionParameters(),
    "europe": RegionParameters(),
    "north_america": RegionParameters(),
    "global_south": RegionParameters(),
}

#: No region differs from "global": empty, kept for import compatibility.
UNVALIDATED_REGIONS: frozenset[str] = frozenset()
