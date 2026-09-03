"""Builds a UrbanAccessAnalyzer-compatible distance_matrix from transitlos's own
D(t; stop_score) decay, with output discretized onto a fixed {0.0, 0.1, ..., 1.0}
level_of_service grid -- replacing `UrbanAccessAnalyzer.isochrones.default_distance_matrix`'s
own step-function decay, at the user's explicit request (2026-08-12).

Why this exists (read `level_of_service.py`'s module docstring first): the
previous pipeline deliberately never called `walk_access.walk_decay` because
`AccessibilityAnalyzer.default_distance_matrix` already applies its own
(unrelated) rank-based falloff, and stacking both would double-count walk
decay. This module resolves that by building the `distance_matrix` argument
`AccessibilityAnalyzer.run` expects *from* `transitlos.scoring`'s own
`accessibility_score`, so the isochrone engine now runs transitLOS's
literature-grounded decay instead of `default_distance_matrix`'s generic one.

"Intelligent discretization" (the user's phrase) means two distinct things,
both implemented here:

1. **Output side**: every level_of_service this module ever produces -- both
   inside the matrix and (via `ceil_to_grid`) as a post-processing safety
   net on the final per-edge result -- is ceiling-rounded onto the fixed
   grid `{0.0, 0.1, 0.2, ..., 1.0}` (e.g. 0.91 -> 1.0, not rounded to
   nearest). This is a deliberate design choice for output legibility, not
   a literature-grounded number.
2. **Input side (run-count minimization)**: rather than accept whatever
   flat `walk_distance_steps` convention the caller passes, this module
   *derives* distance thresholds by inverting `walk_decay` at the midpoint
   of each of the 10 output-grid bands (0.05, 0.15, ..., 0.95) for a
   reference stop_score -- so every distance column that gets computed maps
   onto an output-visible boundary, and no finer. Separately, every
   distinct `stop_score` present is collapsed into a bucket sharing one
   representative value whenever two scores would produce an *identical*
   ceiling-discretized output at every chosen distance -- i.e. buckets are
   made as coarse as possible without losing any information the final
   output can actually distinguish. Since `UrbanAccessAnalyzer`'s own
   `isochrones.distance_matrix_to_processing_order` /
   `compute_node_access` already batch every POI sharing one
   `(level_of_service, distance)` pair into a single multi-source Dijkstra call
   (see that module's docstring), collapsing rows here directly collapses
   the number of underlying isochrone/shortest-path runs -- this *is* "as
   little as possible" isochrone-run count and "stops grouped together as
   much as possible", not a separate optimization layered on top.

Processing order (informational, not implemented here): `UrbanAccessAnalyzer`
already processes `distance_matrix_to_processing_order`'s rows best-rank
(highest level_of_service) first and reconciles each node/edge to the best tier
it qualifies for across all rows -- see `isochrones.py`'s module docstring,
step 3: "A node's final level of service is the best (lowest rank) tier for
which it has positive remaining distance". This already guarantees every
street segment ends up with the maximum level_of_service any stop/distance
combination can justify, regardless of the order rows are *submitted* in
this module's matrix -- no additional ordering logic is needed here.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import polars as pl

from .accessibility_score import accessibility_score
from .parameters import REGIONS
from .walk_access import walk_time_for_decay
from ._warnings import warn_if_unvalidated_region

#: Output grid this project discretizes every level_of_service onto.
OUTPUT_GRID_STEP = 0.1


def ceil_to_grid(value: float, step: float = OUTPUT_GRID_STEP) -> float:
    """Ceiling-round a scalar level_of_service onto the `{0, step, 2*step, ..., 1.0}` grid.

    E.g. `ceil_to_grid(0.91) == 1.0`, `ceil_to_grid(0.0) == 0.0`,
    `ceil_to_grid(0.30) == 0.3` (exact multiples are not pushed to the next
    tier -- the small epsilon subtraction guards against float error like
    `0.3 / 0.1 == 2.9999999999999996` incorrectly ceiling-ing to 0.4).

    Args:
        value: The raw level_of_service, expected in `[0, 1]` (any
            floating-point overshoot just above 1.0 is clipped to the
            grid's max).
        step: Grid spacing. Defaults to `OUTPUT_GRID_STEP` (0.1).

    Returns:
        A float on the grid, clipped to `[0.0, 1.0]`.
    """
    return float(min(1.0, max(0.0, math.ceil(value / step - 1e-9) * step)))


def ceil_to_grid_array(values: np.ndarray, step: float = OUTPUT_GRID_STEP) -> np.ndarray:
    """Vectorized `ceil_to_grid`, for a final safety-net pass over a whole level_of_service column."""
    arr = np.asarray(values, dtype=float)
    out = np.ceil(arr / step - 1e-9) * step
    return np.clip(out, 0.0, 1.0).round(6)


def _target_grid_midpoints() -> list[float]:
    """10 target D(t)-equivalent access-score values: midpoints of each 0.1-wide output tier."""
    return [round(0.05 + OUTPUT_GRID_STEP * i, 2) for i in range(10)]


def build_intelligent_distance_matrix(
    stop_scores: Sequence[float],
    region: str = "global",
    max_distance_m: float | None = None,
) -> tuple[pl.DataFrame, dict[float, float], list[float]]:
    """Build a distance_matrix + stop_score-bucket map from transitlos's own decay.

    Args:
        stop_scores: The distinct (or a representative sample of) stop_score
            values actually present in the stops being scored. Only their
            range matters -- used to pick a reference stop_score for
            distance-step placement and to build the bucket mapping.
        region: Region key forwarded to `accessibility_score`/`walk_decay`
            (see `parameters.py` -- every region uses the same decay-shape
            defaults unless explicitly overridden).
        max_distance_m: Optional hard outer walk-distance cutoff (meters).
            Since D(t) is a gentle, asymptotically-declining curve rather
            than a hard cutoff, its own natural output-tier thresholds can
            extend arbitrarily far (the lowest 0.05-D0.1 tier's threshold
            is several kilometers out) -- an isochrone search out to that
            distance for every stop is expensive and, past some point, not
            meaningfully different from "no access". When given, any
            natural threshold beyond `max_distance_m` is dropped, and
            `max_distance_m` itself is appended as the final distance step
            (its own real, ceiling-discretized D(t) value, not silently
            truncated to whatever tier happened to fall just under it) so
            isochrones still search the *entire* requested radius, just no
            further.

    Returns:
        `(distance_matrix, bucket_map, distance_steps_m)`:

        - `distance_matrix`: a `polars.DataFrame` with one row per bucket
          representative `poi_score` and one column per
          `str(distance_in_meters)`, values ceiling-discretized onto the
          output grid -- directly usable as
          `AccessibilityAnalyzer.run(distance_matrix, poi_score_col=...)`'s
          first argument.
        - `bucket_map`: `{original_stop_score: bucket_representative}` for
          every distinct input stop_score -- callers must remap their own
          stops' score column through this before calling `.run()`, so the
          join against `distance_matrix["poi_score"]` lines up exactly (see
          `level_of_service.py`).
        - `distance_steps_m`: the chosen distance thresholds in meters
          (ascending), for logging/inspection.
    """
    warn_if_unvalidated_region(region)
    scores = np.unique(np.asarray(
        [s for s in stop_scores if s is not None and not (isinstance(s, float) and math.isnan(s)) and s > 0],
        dtype=float,
    ))
    if len(scores) == 0:
        return pl.DataFrame({"poi_score": []}), {}, []

    params = REGIONS[region]

    # 1. Distance thresholds: invert D(t) directly at each output-tier
    # midpoint, plus an explicit plateau step (t_sat, D(t)=1.0 exactly) so
    # nodes genuinely at/near a stop register the stop's true max score
    # (ceiling-discretized) rather than being capped at whatever the
    # nearest post-plateau column happens to resolve to. Simpler than the
    # previous "reference stop_score" scaffolding: since D(t) no longer
    # depends on stop_score at all (2026-08-12 decoupling, see
    # `walk_access.py`), and `stop_score` is now itself always <= 1.0
    # (`speed.py`'s saturation cap), the same distance thresholds are
    # correct for every stop regardless of its own score.
    t_sat = params.walk_distance_saturation_m / params.walking_speed_mps / 60.0
    times_min = sorted({t_sat} | {
        walk_time_for_decay(target, region=region)
        for target in _target_grid_midpoints()
    })
    # Rounded only to 6 decimal places (sub-millimeter) -- just enough to
    # avoid float repr noise in the column names, nowhere near coarse enough
    # to shift a value across a ceil_to_grid boundary the way rounding to a
    # whole/tenth meter could (that used to make a bucket's stored cell
    # value disagree with recomputing accessibility_score from the rounded
    # distance -- a real, if tiny in practice, discrepancy).
    distance_steps_m = [round(t * params.walking_speed_mps * 60.0, 6) for t in times_min]

    if max_distance_m is not None:
        kept = [(t_m, d_m) for t_m, d_m in zip(times_min, distance_steps_m) if d_m <= max_distance_m]
        times_min = [t_m for t_m, _ in kept]
        distance_steps_m = [d_m for _, d_m in kept]
        if not distance_steps_m or distance_steps_m[-1] < max_distance_m:
            times_min.append(max_distance_m / params.walking_speed_mps / 60.0)
            distance_steps_m.append(round(float(max_distance_m), 6))

    # 2. Bucket stop_score values: consecutive (ascending) scores collapse
    # into one bucket whenever they produce an identical ceiling-discretized
    # output signature across every chosen distance step -- see module
    # docstring, point 2.
    def signature(s: float) -> tuple[float, ...]:
        return tuple(ceil_to_grid(accessibility_score(s, t_min, region=region)) for t_min in times_min)

    bucket_map: dict[float, float] = {}
    matrix_rows: list[dict] = []
    current_sig: tuple[float, ...] | None = None
    current_group: list[float] = []

    def _flush():
        if not current_group:
            return
        rep = current_group[-1]  # bucket's own max score -- any member works, all share `current_sig`
        row = {"poi_score": float(rep)}
        for d_m, val in zip(distance_steps_m, current_sig):
            row[str(d_m)] = float(val)
        matrix_rows.append(row)
        for cs in current_group:
            bucket_map[float(cs)] = float(rep)

    for s in scores:  # ascending
        sig = signature(s)
        if sig != current_sig:
            _flush()
            current_sig = sig
            current_group = [s]
        else:
            current_group.append(s)
    _flush()

    distance_matrix = pl.DataFrame(matrix_rows) if matrix_rows else pl.DataFrame({"poi_score": []})
    return distance_matrix, bucket_map, distance_steps_m
