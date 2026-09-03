"""Headway/frequency scoring: converts a service headway into a [0, 1] score.

Implements a monotonic, saturating (diminishing-returns) function of headway,
per SCORING.md §2 row "Headway-score function" and
parameters/05_headway_score_function.md. SCORING.md is explicit that no
single validated closed-form headway-score function exists in the
literature, and that the fabricated "Universal Frequency Score" construct
referenced in early notes could not be verified as real and must not be
used. (Note, 2026-08-10: "Welding" turned out to be a real author --
Welding, P.I. (1957), "The instability of close interval service," Opnl
Res. Q. 8, 133-148, cited in Henderson, Kwong & Adkins (1991) -- but
Welding's actual formula is E(w) = sum(h_i^2) / (2 * sum(h_i)), not the
E(w) = (H/2)(1+CV^2) form previously guessed at; that (H/2)(1+CV^2) form is
instead attributable to Bowman & Turnquist (1981), cited below. Neither
formula is implemented here -- this module still uses a shape-based design
choice, not a literature closed form.) What *is* literature-grounded is only the
function's *shape*: score should be flat/near-ceiling below the
random-vs-scheduled-arrival threshold (~10-12 min, Bowman & Turnquist 1981;
Ingvardson et al. 2018) and fall off smoothly above it (TCRP 95 Ch.9;
Balcombe et al. 2004 diminishing-returns elasticity). This module implements
that hybrid shape -- a flat "frequent enough" plateau followed by
exponential-style decay -- as a clearly-labeled design choice, consistent
with parameters/05_headway_score_function.md Step 3's recommended hybrid.
"""

from __future__ import annotations

import math

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def frequency_score(
    headway_minutes: float,
    region: str = "global",
    saturation_minutes: float | None = None,
    decay_scale_minutes: float | None = None,
) -> float:
    """Compute a [0, 1] frequency score from a service headway.

    Below `saturation_minutes` the score is 1.0 (turn-up-and-go service,
    where passengers arrive effectively at random rather than consulting a
    schedule -- SCORING.md §2 "Random- vs. scheduled-arrival threshold").
    Above it, score decays smoothly (exponential-style) as headway grows,
    reflecting diminishing marginal value of further frequency improvements
    once service is already frequent (Balcombe et al. 2004; TCRP 95 Ch.9).

    This exact functional form (flat plateau + exponential tail) is a
    design choice recommended, but not literature-mandated, by
    parameters/05_headway_score_function.md Step 3 -- no source provides a
    validated closed-form headway->score function. Do not treat this as an
    empirically fitted curve.

    Args:
        headway_minutes: Scheduled or average headway in minutes. Must be
            >= 0.
        region: Region key into `parameters.REGIONS` used to select default
            saturation/decay parameters if not explicitly overridden. One of
            "global", "europe", "north_america", "global_south".
        saturation_minutes: Headway below which score is 1.0. Defaults to
            the region's `headway_saturation_minutes`.
        decay_scale_minutes: Decay scale for headways above the saturation
            point. Defaults to the region's `headway_decay_scale_minutes`.
            NOT a literature-derived constant (parameters/05 Step 2);
            exposed here so callers can recalibrate it.

    Returns:
        A float in [0, 1]; 1.0 means "as frequent as service needs to be
        scored," 0.0 means effectively unusably infrequent.

    Raises:
        ValueError: If headway_minutes is negative.
    """
    if headway_minutes < 0:
        raise ValueError(f"headway_minutes must be >= 0, got {headway_minutes}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    sat = saturation_minutes if saturation_minutes is not None else params.headway_saturation_minutes
    scale = decay_scale_minutes if decay_scale_minutes is not None else params.headway_decay_scale_minutes

    if headway_minutes <= sat:
        return 1.0

    excess = headway_minutes - sat
    return math.exp(-excess / scale)
