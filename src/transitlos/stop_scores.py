"""Vectorized application of `transitlos.scoring` to a per-stop table.

Takes the raw per-stop table produced by `transitlos.stops.download_and_prepare_stops`
(columns `headway_minutes`, `headway_cv`, `avg_speed_kmh`, `route_type`) and
adds the `mrc_score`, `speed_score`, `frequency_score`, `mode_category`, and
`stop_score` columns from `transitlos.scoring`.

**2026-09-28 model rewrite**: the prior four-factor
(`mode_score`/`speed_score`/`frequency_score`/`reliability_score`) design is
replaced by the finalized three-factor model (`mrc_score`/`speed_score`/
`frequency_score` -- see `transitlos.scoring.mrc`'s module docstring for why
mode and reliability/comfort are now one combined axis). `stop_score` is
also no longer independently walk-decayed here: downstream callers now pass
`stop_score` *and* `mode_category` into `walk_access.walk_decay`/
`accessibility_score.accessibility_score`, since `D(t; stop_score)`'s scale
is itself mode- and stop_score-dependent (see `walk_access.py`).

Honesty note: `headway_cv` coming out of `transitlos.stops` is currently
always null (pyGTFSHandler's `get_headway_at_stops` only exposes a mean
headway, not its distribution). Rather than silently treating every stop as
perfectly regular (which would make `mrc.MR_CV(0) == 1.0` for every stop,
erasing the real, literature-grounded mode-categorical gap), this module
only feeds a real, observed `headway_cv` into `mrc_score`'s optional (b)
refinement -- rows with a null `headway_cv` fall back to the primary (a)
mode-categorical default (`MR_mode`) instead, leaving a
`reliability_score_is_assumed` boolean column so callers can tell which rows
had a real CV input.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from .scoring import frequency_score, mode_category, mrc_score, speed_score, stop_score


def compute_stop_scores(
    stops_df: Union[pd.DataFrame, "gpd.GeoDataFrame"],
    region: str = "global",
    weights: Optional[tuple[float, float, float]] = None,
    default_headway_cv: Optional[float] = None,
    stop_score_mode: str = "nested",
) -> Union[pd.DataFrame, "gpd.GeoDataFrame"]:
    """Add `mrc_score`/`speed_score`/`frequency_score`/`stop_score` columns.

    Args:
        stops_df: A pandas DataFrame or GeoDataFrame with `headway_minutes`,
            `headway_cv`, `avg_speed_kmh`, and (optionally) `route_type`
            columns (as produced by
            `transitlos.stops.download_and_prepare_stops`). Rows with a null
            `headway_minutes` or `avg_speed_kmh` get `NaN` scores. Rows with
            a missing/absent `route_type` fall back to the "bus" mode
            category (see `transitlos.scoring.mrc.mode_category`).
        region: Region key forwarded to every `transitlos.scoring` call.
            Kept for backward compatibility -- every region now shares
            identical, universal parameters (see `scoring/parameters.py`).
        weights: Only meaningful when `stop_score_mode` is `"harmonic"` or
            `"geometric"` (the legacy flat three-way mean) -- forwarded to
            `stop_score`. Must be left `None` for the default
            `stop_score_mode="nested"` (see `scoring.stop_score`'s
            docstring, scoring.md Section 1.2 -- the nested default has no
            weight parameter by construction).
        stop_score_mode: Forwarded to `scoring.stop_score` as `mode`.
            Defaults to `"nested"` (`sqrt(MRC*V)*H`, 2026-09-29 default).
            Pass `"harmonic"`/`"geometric"` to reproduce the prior flat
            generalized-power-mean forms, e.g. for before/after comparisons.
        default_headway_cv: **No longer substituted into `mrc_score`** (see
            module docstring's honesty note -- doing so would silently
            erase the mode-categorical default for every stop). Kept as an
            accepted, unused parameter for backward-compatibility only;
            passing a non-`None` value has no effect. Use
            `reliability_score_is_assumed` to identify rows using the
            mode-categorical fallback.

    Returns:
        `stops_df` (same type, a copy) with `mode_category`, `mrc_score`,
        `speed_score`, `frequency_score`, `stop_score`, and
        `reliability_score_is_assumed` columns added.
    """
    df = stops_df.copy()
    df["reliability_score_is_assumed"] = df["headway_cv"].isna()
    route_type = df["route_type"] if "route_type" in df.columns else pd.Series([None] * len(df), index=df.index)

    def _score_row(headway, cv, speed, rtype):
        if pd.isna(headway) or pd.isna(speed):
            return pd.Series(
                {"mode_category": None, "mrc_score": float("nan"), "speed_score": float("nan"),
                 "frequency_score": float("nan"), "stop_score": float("nan")}
            )
        rt = None if pd.isna(rtype) else rtype
        mc = mode_category(rt)
        cv_value = None if pd.isna(cv) else float(cv)
        mrc = mrc_score(route_type=rt, headway_cv=cv_value, region=region)
        sp = speed_score(float(speed), region=region)
        fs = frequency_score(float(headway), region=region)
        ss = stop_score(mrc, sp, fs, weights=weights, region=region, mode=stop_score_mode)
        return pd.Series(
            {"mode_category": mc, "mrc_score": mrc, "speed_score": sp, "frequency_score": fs, "stop_score": ss}
        )

    scores = pd.DataFrame(
        [
            _score_row(h, cv, s, rt)
            for h, cv, s, rt in zip(df["headway_minutes"], df["headway_cv"], df["avg_speed_kmh"], route_type)
        ],
        index=df.index,
    )
    for col in scores.columns:
        df[col] = scores[col]
    return df
