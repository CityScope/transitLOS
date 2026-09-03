"""Vectorized application of `transitlos.scoring` to a per-stop table.

Takes the raw per-stop table produced by `transitlos.stops.download_and_prepare_stops`
(columns `headway_minutes`, `headway_cv`, `avg_speed_kmh`, `route_type`) and
adds the `mode_score`, `speed_score`, `frequency_score`, `reliability_score`,
and `stop_score` columns from `transitlos.scoring`.

Honesty note: `headway_cv` coming out of `transitlos.stops` is currently
always null (pyGTFSHandler's `get_headway_at_stops` only exposes a mean
headway, not its distribution). Rather than silently treating every stop as
perfectly regular, this module fills missing `headway_cv` with an explicit
`default_headway_cv` parameter (default `0.0`, i.e. "assume regular service
absent better information") and leaves a `reliability_score_is_assumed`
boolean column so callers can tell which rows had a real reliability input.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from .scoring import frequency_score, mode_score, reliability_score, speed_score, stop_score


def compute_stop_scores(
    stops_df: Union[pd.DataFrame, "gpd.GeoDataFrame"],
    region: str = "global",
    weights: Optional[tuple[float, float, float, float]] = None,
    default_headway_cv: float = 0.0,
) -> Union[pd.DataFrame, "gpd.GeoDataFrame"]:
    """Add `mode_score`/`speed_score`/`frequency_score`/`reliability_score`/`stop_score` columns.

    Args:
        stops_df: A pandas DataFrame or GeoDataFrame with `headway_minutes`,
            `headway_cv`, `avg_speed_kmh`, and (optionally) `route_type`
            columns (as produced by
            `transitlos.stops.download_and_prepare_stops`). Rows with a null
            `headway_minutes` or `avg_speed_kmh` get `NaN` scores. Rows with
            a missing/absent `route_type` fall back to the "bus" mode
            category (see `transitlos.scoring.mode.mode_category`).
        region: Region key forwarded to every `transitlos.scoring` call
            (`"global"`, `"europe"`, `"north_america"`, `"global_south"`).
        weights: Optional `(w1, w2, w3, w4)` weights forwarded to
            `stop_score` (mode, speed, frequency, reliability respectively).
            Defaults to the region's own default weights if omitted.
        default_headway_cv: Value substituted for null `headway_cv` entries
            before scoring reliability. Defaults to `0.0` (assume perfectly
            regular service when no variance data is available) -- see the
            module docstring's honesty note.

    Returns:
        `stops_df` (same type, a copy) with `mode_score`, `speed_score`,
        `frequency_score`, `reliability_score`, `stop_score`, and
        `reliability_score_is_assumed` columns added.
    """
    df = stops_df.copy()
    df["reliability_score_is_assumed"] = df["headway_cv"].isna()
    headway_cv = df["headway_cv"].fillna(default_headway_cv)
    route_type = df["route_type"] if "route_type" in df.columns else pd.Series([None] * len(df), index=df.index)

    def _score_row(headway, cv, speed, rtype):
        if pd.isna(headway) or pd.isna(speed):
            return pd.Series(
                {"mode_score": float("nan"), "speed_score": float("nan"), "frequency_score": float("nan"),
                 "reliability_score": float("nan"), "stop_score": float("nan")}
            )
        mo = mode_score(None if pd.isna(rtype) else rtype)
        sp = speed_score(float(speed), region=region)
        fs = frequency_score(float(headway), region=region)
        rs = reliability_score(float(cv), region=region)
        ss = stop_score(mo, sp, fs, rs, weights=weights, region=region)
        return pd.Series(
            {"mode_score": mo, "speed_score": sp, "frequency_score": fs, "reliability_score": rs, "stop_score": ss}
        )

    scores = pd.DataFrame(
        [
            _score_row(h, cv, s, rt)
            for h, cv, s, rt in zip(df["headway_minutes"], headway_cv, df["avg_speed_kmh"], route_type)
        ],
        index=df.index,
    )
    for col in scores.columns:
        df[col] = scores[col]
    return df
