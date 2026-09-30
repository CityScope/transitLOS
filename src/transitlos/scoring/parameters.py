"""Numeric default parameters for the transitLOS scoring formulas.

Every constant here traces back to a specific row/section of the finalized
``transitLOS/documentation/scoring/scoring.md`` ("Transit Quality-of-Service
Scoring: Master Synthesis") rather than being scattered as a magic number
through the logic files.

**2026-09-28 model rewrite.** The previous version of this module kept a
``REGIONS`` dict (``"global"``/``"europe"``/``"north_america"``/
``"global_south"``) with genuinely different per-region defaults (walking
speed, walk-distance thresholds, etc.), reflecting the *prior* scoring
model. scoring.md's finalized model (§3.3 "Universal / global-fallback
defaults") explicitly re-examined every constant in the new four-function
model (``H``, ``V``, ``MRC``, ``D``) and found no defensible region-specific
override for any of them: walking speed is fixed at the single 1.3 m/s
general-adult figure everywhere (§2.1 -- the 1.15 m/s "North American MUTCD"
figure is explicitly rejected there as a conservative crosswalk-design value,
not a central-tendency behavioral one), the headway-elasticity fit (§2.2),
speed ramp (§2.4), and MR_mode/MR_CV/comfort tables (§2.3) are all presented
as universal literature-derived or reasoned-design constants with no
regional variant. §3.4 documents this as a known, flagged **gap** (no
Global-South-specific override exists), not as "the same value is
deliberately chosen for every region" -- so ``REGIONS`` is kept only as a
**backward-compatible parameter-lookup shim** (every key maps to identical
values) for callers that still pass a ``region=`` kwarg; it no longer
encodes any actual regional differentiation. ``UNVALIDATED_REGIONS`` is now
empty for the same reason: there is no longer a region whose parameters
differ from "global" at all, so there is nothing to warn about differently
than every other region already implicitly is (the Global-South gap noted
above applies equally to every key, and is instead called out here, once,
in prose).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegionParameters:
    """Container for the scoring model's default parameters.

    Attributes:
        weights: Default (w_mrc, w_v, w_h) equal weights for `MRC`, `V`,
            `H` in `stop_score`'s weighted geometric mean. scoring.md §1.2:
            three independent literature searches (VoT-ratio, MCDA/AHP,
            conjoint/BWS) found no adoptable differential-weighting scheme,
            so equal thirds is the stated result of that search.
        walking_speed_mps: General-adult walking speed, m/s. scoring.md
            §2.1: 1.3 m/s (Fitzpatrick, Brewer & Turner 2006's "commonly
            cited normal range" midpoint, corroborated by TfL PTAL's 1.33
            m/s convention and a 2024 meta-analysis's pooled 1.26-1.37 m/s).
        t_plateau_distance_m: Walk distance (m) at/below which `D(t)=1.0`
            exactly, regardless of stop_score. scoring.md §2.1 "Anchor":
            200 m, a stated design choice (not stop_score-scaled), converts
            to ~2.56 min at 1.3 m/s.
        t0_base_minutes: Per-mode base scale (minutes) for `D`'s decay,
            before the `(1 + k*stop_score)` quality scaling. scoring.md
            §2.1 Hyperparameters / Anchor: bus ~6.6 min (midpoint of
            4.8-8.4 min range), rail/metro ~9.4 min, tram/LRT ~14.55 min
            (midpoint of 14.1-15.0 min range), from El-Geneidy et al.
            (2014)/Wu & El-Geneidy (2026) real walking-distance percentiles.
        k_quality: Quality-coupling constant in
            `t0(stop_score) = t0_base(mode) * (1 + k * stop_score)`.
            scoring.md §2.1: k=2.5, a point estimate derived from Rose,
            Mulley, Tsai & Hensher (2013)/Mulley et al. (2018)'s stated-
            preference walk-tolerance-vs-headway finding, honest sensitivity
            band 1.55-3.5.
        p_shape: Shape exponent `p` in `D(t) = (1 + (t-t_plateau)/t0)^(-p)`.
            scoring.md §2.1 Hyperparameters: p=1.0, a reasoned simplification
            (no source fits this exact form and reports a p), held fixed
            across modes.
        h_saturation_minutes: Headway (min) at/below which `H(headway)=1.0`.
            scoring.md §2.2: 5 min.
        h_elasticity_p: Intercept `p` of the log-linear local-elasticity fit
            `eps(headway) = p + q*ln(headway)`. scoring.md §2.2
            Hyperparameters: a least-squares fit to TCRP Report 95 Ch.9
            Table 9-2 (Lago, Mayworm & McEnroe 1981)'s three geometric-mean
            band-representative points (7.07, 22.36, 70.71 min).
        h_elasticity_q: Slope `q` of the same fit. scoring.md §2.2.
        v_anchor_kmh: Speed (km/h) at which the smooth curve `V(v) = M*v^n/
            (v^n+K^n)` (scoring.md §2.4, 2026-09-29 SECOND SHAPE-FIX/RE-
            ANCHOR revision) reaches exactly 1.0. scoring.md §2.4: 30 km/h --
            re-derived directly against real metro/subway COMMERCIAL speed
            (distance/time INCLUDING dwell time at stops, not cruising
            speed): four major dense-network metros independently converge
            on 30-33 km/h (London Underground ~33, Paris Metro legacy lines
            ~30-35, New York City Subway ~30, Tokyo Metro ~30), and UITP's
            general figure for dense-network metro commercial speed is
            25-45 km/h. Replaces the prior 25 km/h anchor, which was
            BRT-corridor-derived (ITDP), not subway/metro-derived, per an
            explicit user request to re-ground this anchor in subway data
            specifically.
        v_ceiling_bonus: The curve's asymptotic ceiling, expressed as extra
            credit above 1.0: the curve approaches `M = 1+bonus` as
            `v -> infinity`, never a hard cap and no longer pinned to any
            specific finite speed (see the 2026-09-29 THIRD SHAPE-FIX note
            in `speed.py` for why a second fitted point was tried twice --
            first an arbitrary "100 km/h," then German Regional-Express's
            "80 km/h" -- and dropped both times). scoring.md §2.4: +0.15,
            explicitly flagged as the single weakest-grounded constant in
            the whole document -- "the retained middle of a plausible
            0-50% range," not a value that beat named numeric rivals, and
            (as of this revision) honestly NOT claimed to be pinned to any
            specific named high-speed benchmark -- a pure design-judgment
            asymptote, magnitude unchanged across all four 2026-09-29
            revisions. Sanity-checked (not re-derived) against real HSR
            commercial speeds -- Shinkansen ~220 km/h, TGV Paris-Lyon
            ~200-207 km/h, China's Beijing-Shanghai HSR ~292 km/h -- all of
            which evaluate to V~=1.146-1.148 under this curve, within 0.2
            points of the asymptote. See scoring.md §2.4 Hyperparameters and
            `parameters/07...md` §4k.
        v_shape_n: Hill-function exponent `n` in `V(v) = M*v^n/(v^n+K^n)`.
            scoring.md §2.4: n=2.0, a reasoned design choice (not a
            fitted/derived value -- no source specifies this exponent)
            picked to correct a real low-speed defect found in this
            function's `n=1` (plain Michaelis-Menten) predecessor, which put
            `V(5)~=0.49-0.59` across re-anchorings -- an unreasonably high
            score for a speed barely above walking pace. `n=2` (a standard
            "Hill coefficient of 2" shape, not an arbitrary decimal) pulls
            the curve's rise back down near walking-pace speeds while
            preserving smoothness (still a single closed-form, infinitely
            differentiable function, not a piecewise construction) and the
            `V(anchor)=1.0` anchor. This diagnosis and fix carried forward
            unchanged through the anchor's later re-derivation (25->30 km/h)
            -- the low-speed shape problem and the anchor's value are
            independent design questions.
        mr_mode: Per-mode-category `MR_mode(mode)` categorical lookup.
            scoring.md §2.3(a): rail/subway/metro/commuter rail = 1.000
            (anchor); tram/light rail/dedicated-ROW BRT = 0.894 (geometric-
            mean interpolation, `sqrt(1.000*0.800)`); local bus (mixed
            traffic) = 0.800 (Wardman, Chintakayala & de Jong 2016, Table 7,
            "metro valued/bus valued" ratio).
        comfort_scores: `C(occupancy_status)` categorical lookup keyed by
            GTFS-realtime `VehiclePosition.OccupancyStatus` enum values.
            scoring.md §2.3(c). Optional refinement, not wired into the
            main pipeline by default (real-time occupancy data is rarely
            available in this project's actual pipeline).
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


#: Backward-compatible region lookup -- see module docstring. Every key maps
#: to identical, universal parameters; kept only so callers still passing a
#: `region=` kwarg keep working.
REGIONS: dict[str, RegionParameters] = {
    "global": RegionParameters(),
    "europe": RegionParameters(),
    "north_america": RegionParameters(),
    "global_south": RegionParameters(),
}

#: No region's parameters differ from "global" any more (see module
#: docstring) -- kept as an empty set, not removed, for import compatibility.
UNVALIDATED_REGIONS: frozenset[str] = frozenset()
