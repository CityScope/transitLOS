"""Numeric default parameters for the transitLOS scoring formulas, keyed by region.

Every constant here traces back to a specific row/section of
``report/scoring/SCORING.md`` (or, where SCORING.md defers to it, one of the
``report/scoring/parameters/*.md`` files) rather than being scattered as a
magic number through the logic files. Values are grouped into a
``RegionParameters`` dataclass and collected in the ``REGIONS`` dict, keyed by
region string: ``"global"``, ``"europe"``, ``"north_america"``, and
``"global_south"``.

Important honesty note (SCORING.md §3.3): the literature reviewed contains
essentially one usable source for Global-South/LMIC parameters (a
value-of-time study), and nothing for walk-distance thresholds, walking
speed, headway elasticity, or decay-function parameters in that context.
``REGIONS["global_south"]`` therefore reuses the global defaults verbatim and
must be treated as an **unvalidated extrapolation**, not a calibrated
regional parameter set. Code that consumes ``region="global_south"`` (see
``accessibility_score.py``) surfaces a runtime warning for this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegionParameters:
    """Container for one region's set of default scoring parameters.

    Attributes:
        weights: Default (w1, w2, w3, w4) weights for mode, speed, frequency,
            and reliability in ``stop_score``'s weighted geometric mean.
            `mode`+`speed` together carry the same combined share the
            original 3-factor `mode_speed` term had (1/3, split evenly into
            two explicit 1/6 factors, per the user's explicit request for a
            separate mode score alongside speed) -- `frequency`/
            `reliability` are unchanged at 1/3 each. Equal weighting within
            that split is a researcher's design choice, not empirically
            derived. See SCORING.md §2, rows "w1"/"w2"/"w3".
        headway_saturation_minutes: The headway (minutes) at/below which
            ``frequency_score`` is exactly 1.0. **Changed 2026-08-11 (5.0,
            uniform across regions) at the user's explicit request** -- this
            is now deliberately a stricter "excellent service" ceiling for
            the *score*, not the same thing as the literature's ~10-12 min
            random-vs-scheduled-arrival threshold (SCORING.md §2, row
            "Random- vs. scheduled-arrival threshold"; Bowman & Turnquist
            1981; Ingvardson et al. 2018) -- that behavioral threshold is
            about when passengers *stop consulting a schedule*, which is a
            different question from when a service should be scored as
            *maximally* good. 5 minutes matches common "turn-up-and-go"/
            high-frequency-network conventions (e.g. many BRT Gold/Platinum
            standards, metro-frequency service definitions) and is a
            **researcher's design choice**, not itself literature-derived --
            flagged here rather than presented as the same number as the
            behavioral threshold above. This is also the anchor point used
            by ``frequency.py``'s saturating curve shape (see
            parameters/05_headway_score_function.md Step 3).
        headway_decay_scale_minutes: The "tau"-like scale (minutes) governing
            how quickly frequency_score falls off for headways above the
            saturation point. NOT a literature-derived constant -- flagged
            explicitly in parameters/05_headway_score_function.md Step 2 as
            an engineering approximation, proposed range 15-25 min. Exposed
            here as a tunable default, never silently hardcoded elsewhere.
        max_useful_headway_minutes: Headway (minutes) beyond which
            frequency_score is treated as effectively zero (used only to
            bound illustrative examples/tests, not a hard cutoff in the
            function itself, per SCORING.md's "continuous, not hard cutoff"
            principle).
        walking_speed_mps: Typical adult walking speed in m/s, used by
            downstream walk-time computations (not by walk_access.py
            directly, since walk_access.py takes walk *time* as input, but
            kept here for callers converting distance to time). See
            SCORING.md §2, row "Walking speed (general adult)" and its
            regional overrides in §3.
        walk_t0_base_minutes: Base scale parameter `t0_base` (minutes) for
            the offset-power walk-decay form `D(t) = (1 + t/t0)^(-p)`
            implemented in `walk_access.py`. **Recalibrated 2026-08-12, at
            the user's explicit request** ("this decay is way too fast"):
            the previous 4.5/0.8/4.0 (`t0_base`/`k`/`p`) combination,
            combined with `stop_score` scaling `t0` itself, collapsed to
            ~10-25% of full value within one 250 m step past the plateau --
            far steeper than either the directly-measured literature or
            published real-world precedent (see `walk_decay_shape_p`
            below). The decay curve is now also **decoupled from
            `stop_score`** (no more `k`/stop_score-scaled `t0`): mode/speed/
            frequency/reliability already discount `stop_score` itself
            (`stop_score.py`), so `accessibility_score` folding a *second*,
            stop-quality-dependent walk-tolerance boost on top of that was
            double-counting stop quality. `D(t)` is now a single,
            stop-independent function of walk time alone, and
            `accessibility_score = stop_score * D(t)` is a plain product
            (report/scoring/parameters/11_walk_time_valuation_and_decay.md
            §11).

            The new default (9.0 min) is calibrated so `D(800 m) ≈ 0.5` --
            TCRP 95's own 800 m *rail* planning threshold, deliberately
            chosen over the shorter 400 m *bus* threshold, at the user's
            explicit direction: since the decay curve is now shared across
            every mode (not scaled per-stop by `stop_score` anymore), a bus
            stop's typically lower `stop_score` (lower `mode_score`/
            `speed_score`) already discounts its `accessibility_score`
            appropriately on its own -- anchoring the *shared* curve to the
            bus threshold would additionally penalize rail/metro stops
            (which riders demonstrably walk much farther to reach, file 11
            §9.2's real percentile data: rail median 527-785 m vs. bus
            210-402 m) for distances their real-world catchment comfortably
            covers. Anchoring to the longer, rail-calibrated threshold
            avoids double-penalizing the mode that already tolerates -- and
            is already scored for -- longer walks. Sanity-checked against
            Walk Score/Transit Score's published (if not fully
            open-sourced) decay benchmark (file 11 §4: full value to ~400 m,
            ~12% remaining at 1600 m, ~0% by 2400 m) -- the new curve gives
            D(1600 m) ≈ 0.28 (more generous than that benchmark, consistent
            with anchoring to rail rather than bus) vs. the old curve's
            D(1600 m) ≈ 0.006 (see file 11 §11 for the full comparison
            table).
        walk_decay_shape_p: Shape parameter `p` in
            `D(t) = (1 + t/t0)^(-p)`. Controls how quickly the hazard rate
            `p/(t0+t)` starts high and decays -- unlike negative-exponential
            decay, this hazard rate *falls* with distance rather than
            staying constant, which is the literature-grounded property this
            form was chosen for (file 11 §8: Kanafani 1983 via Iacono/
            El-Geneidy 2010; Chen 2015). **Recalibrated 2026-08-12** from 4.0
            to 1.2 alongside `walk_t0_base_minutes` above -- the old p=4.0
            produced a much steeper near-field drop-off than either the
            literature's own R²-validated exponential fits or the Walk
            Score/Transit Score precedent; p=1.2 (closer to a near-hyperbolic
            log-logistic shape) matches both far better while preserving the
            theoretically-preferred declining-hazard property over plain
            exponential decay. Still a tunable design parameter, not an
            independently fitted value (file 11 §8.1/§11).
        walk_distance_threshold_m: Nominal "soft" walk-distance threshold(s)
            in meters used only as a reference point for choosing
            walk_beta_per_minute defaults; SCORING.md §2/§3 rows on
            walk-distance thresholds (TCRP 95 bus/rail flat thresholds for
            North America; ARE/ÖROK frequency-tiered bands for Europe).
        walk_distance_saturation_m: Walk distance (meters) at/below which
            `walk_access.walk_decay` returns exactly 1.0 -- **added
            2026-08-11 at the user's explicit request** (250 m, uniform
            across regions). A researcher's design choice distinct from
            `walk_distance_threshold_m` above (that one is where
            accessibility becomes merely *acceptable* per TCRP/ARE-ÖROK
            planning convention; this one is where it should be scored as
            *maximal*) -- see `walk_access.py`'s module docstring for the
            full justification.
        speed_saturation_kmh: Operating speed (km/h) at which
            `speed.speed_score` reaches exactly 1.0 -- 70 km/h, uniform
            across regions. Set at a realistic upper bound for urban transit
            lines, since urban service does not get meaningfully faster than
            this. **Capped at 1.0 above this point, 2026-08-12, at the
            user's explicit request** ("stop score cannot be above 1") --
            an earlier design (2026-08-11) left `speed_score` deliberately
            uncapped so exceptionally fast service (e.g. intercity rail)
            could still be distinguished from an ordinary rapid line, but
            that let `stop_score` (a weighted geometric mean of four
            sub-scores) exceed 1.0 whenever `speed_score` did, breaking the
            invariant that `stop_score`/`accessibility_score` are both
            genuinely bounded to [0, 1]. See `speed.py`'s module docstring.
        mode_speed_breakpoints_kmh: Ascending km/h breakpoints
            (slow/ordinary/semi-rapid boundaries) used by mode_speed.py's
            banded classification. SCORING.md §2 row "Mode/speed
            classification" is explicit that the classification *approach*
            (speed-based, not mode premiums) is literature-grounded. The
            outer two breakpoints are now anchored to the ITDP BRT Standard
            2024's verbatim "Low Commercial Speeds" deduction table
            (parameters/07_speed_based_service_classification.md §1,
            resolved 2026-08-10 against the full PDF): ITDP applies its
            steepest penalty band below 11 km/h and zero penalty above
            20 km/h, so those are used here as the bottom and top
            breakpoints. The middle breakpoint (ordinary/semi-rapid
            boundary) still has no direct ITDP/ARE/ÖROK numeric anchor and
            remains the report author's own interpolated choice (roughly
            midway between the two ITDP-anchored points).
        reliability_cv_scale: Scale parameter governing how quickly
            reliability_score falls off as headway coefficient-of-variation
            (CV) increases. SCORING.md §2 row "Reliability metric" is
            explicit that headway CV is the literature-consistent building
            block, but the exact CV -> score mapping is a design choice
            (file 04; file 15 §4).
    """

    weights: tuple[float, float, float, float] = (1.0 / 6, 1.0 / 6, 1.0 / 3, 1.0 / 3)
    headway_saturation_minutes: float = 5.0
    headway_decay_scale_minutes: float = 20.0
    max_useful_headway_minutes: float = 120.0
    walking_speed_mps: float = 1.3
    walk_t0_base_minutes: float = 9.0
    walk_decay_shape_p: float = 1.2
    walk_distance_threshold_m: float = 400.0
    walk_distance_saturation_m: float = 250.0
    mode_speed_breakpoints_kmh: tuple[float, float, float] = (11.0, 15.5, 20.0)
    speed_saturation_kmh: float = 70.0
    reliability_cv_scale: float = 0.5


REGIONS: dict[str, RegionParameters] = {
    # SCORING.md §2: global defaults, applicable absent a region-specific
    # override table entry.
    "global": RegionParameters(),
    # SCORING.md §3.1: Europe overrides walk-distance bands (frequency-tiered
    # 300/500/750/1000 m, ARE/ÖROK) and uses Wardman-family IVT multipliers
    # (not modeled numerically here since IVT valuation isn't part of these
    # six formulas). `headway_saturation_minutes` is left at the global 5.0
    # design-choice ceiling (2026-08-11) rather than region-varying, since it
    # no longer represents the region-varying random-arrival *behavioral*
    # threshold (still 10-12 min per the literature, just no longer what
    # this parameter drives) -- see the field's docstring in
    # `RegionParameters`.
    "europe": RegionParameters(
        headway_decay_scale_minutes=20.0,
        walking_speed_mps=1.3,
        walk_distance_threshold_m=500.0,
        mode_speed_breakpoints_kmh=(11.0, 15.5, 20.0),
        # walk_t0_base_minutes/walk_decay_shape_p
        # deliberately NOT overridden here -- 2026-08-11, at the user's
        # explicit request, the offset-power decay *shape* uses the global
        # params by default everywhere; only walking speed and the soft
        # walk-distance-threshold reference point vary by region, per file
        # 11's finding that no region-specific decay-rate literature exists
        # (report/scoring/parameters/11_walk_time_valuation_and_decay.md §3).
    ),
    # SCORING.md §3.2: North America uses flat 400m/800m mode-based
    # thresholds (TCRP 95) and the Frei & Mahmassani (2013) disaggregate,
    # Chicago-specific headway elasticity estimate -- prefer the lower-
    # magnitude, more precisely-measured stop-level threshold.
    # `headway_saturation_minutes` left at the global 5.0 ceiling, same
    # reasoning as Europe above.
    "north_america": RegionParameters(
        headway_decay_scale_minutes=20.0,
        walking_speed_mps=1.15,  # MUTCD conservative design speed, §3.2
        walk_distance_threshold_m=400.0,
        # walk_t0_base_minutes/walk_decay_shape_p left
        # at the global default -- see the note in "europe" above.
        mode_speed_breakpoints_kmh=(11.0, 15.5, 20.0),
    ),
    # SCORING.md §3.3: Global South / LMIC. The literature reviewed supports
    # only a value-of-time override (not modeled by these six geometry-only
    # functions) and explicitly provides NO override for walk-distance
    # thresholds, walking speed, headway elasticity, or decay parameters.
    # Reusing the global defaults here is therefore an UNVALIDATED
    # EXTRAPOLATION, not a calibrated regional parameter set -- callers are
    # warned at runtime (see accessibility_score.py / stop_score.py) rather
    # than having this silently pass as equally well-grounded.
    "global_south": RegionParameters(),
}

#: Region keys that SCORING.md §3.3 flags as lacking any regional literature
#: base -- used by scoring functions to decide whether to emit a warning.
UNVALIDATED_REGIONS = frozenset({"global_south"})
