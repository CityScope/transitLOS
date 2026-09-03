"""Pure population-weighted equity statistics over a combined access/population table.

No I/O, no dependency on any other `transitlos` module or the sibling
packages -- only `numpy`/`pandas`. Takes whatever
`transitlos.population.cross_with_population` (or any DataFrame with the
same two columns) produces.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted mean, ignoring rows with a null value or non-positive weight."""
    mask = ~np.isnan(values) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted median: the value at which cumulative weight crosses 50%."""
    mask = ~np.isnan(values) & (weights > 0)
    if not mask.any():
        return float("nan")
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    cutoff = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, cutoff))
    return float(v[min(idx, len(v) - 1)])


def _gini_of_access(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted Gini coefficient of `values` (0 = perfectly equal access).

    Standard weighted Gini via the Lorenz curve: sort by value, compute the
    weight- and value-weighted cumulative shares, and take
    `1 - 2 * area under the Lorenz curve`.
    """
    mask = ~np.isnan(values) & (weights > 0)
    if mask.sum() < 2:
        return float("nan")
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]

    cum_w = np.cumsum(w)
    cum_wv = np.cumsum(w * v)
    total_w, total_wv = cum_w[-1], cum_wv[-1]
    if total_wv == 0:
        return 0.0

    lorenz_x = np.concatenate([[0.0], cum_w / total_w])
    lorenz_y = np.concatenate([[0.0], cum_wv / total_wv])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = trapezoid(lorenz_y, lorenz_x)
    return float(1.0 - 2.0 * area)


def equity_statistics(
    combined_gdf: pd.DataFrame,
    score_col: str = "level_of_service",
    population_col: str = "population",
    groups: Optional[Union[str, Sequence[str]]] = None,
    thresholds: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict:
    """Compute population-weighted access-equity statistics.

    Args:
        combined_gdf: DataFrame/GeoDataFrame with `score_col` and
            `population_col` columns (e.g. from
            `transitlos.population.cross_with_population`).
        score_col: Column holding the per-row level of service.
        population_col: Column holding the per-row population count, used
            as weights throughout.
        groups: Optional column name (or list of column names) to also
            break statistics down by, in addition to the overall figures.
        thresholds: Score thresholds used to report the population share
            above/below each one.

    Returns:
        A dict with keys `"overall"` (a dict of `weighted_mean`,
        `weighted_median`, `gini`, and `pct_population_above_threshold`/
        `pct_population_below_threshold` sub-dicts keyed by threshold), and,
        if `groups` was given, `"by_group"`: a `pandas.DataFrame` indexed by
        the group column(s) with the same statistics per group.
    """
    values = combined_gdf[score_col].to_numpy(dtype=float)
    weights = combined_gdf[population_col].to_numpy(dtype=float)

    def _stats_for(v: np.ndarray, w: np.ndarray) -> dict:
        total_w = np.nansum(w[~np.isnan(v) & (w > 0)])
        above = {}
        below = {}
        for t in thresholds:
            mask = ~np.isnan(v) & (w > 0)
            above[t] = float(np.sum(w[mask & (v >= t)]) / total_w) if total_w > 0 else float("nan")
            below[t] = float(np.sum(w[mask & (v < t)]) / total_w) if total_w > 0 else float("nan")
        return {
            "weighted_mean": _weighted_mean(v, w),
            "weighted_median": _weighted_median(v, w),
            "gini": _gini_of_access(v, w),
            "pct_population_above_threshold": above,
            "pct_population_below_threshold": below,
        }

    result = {"overall": _stats_for(values, weights)}

    if groups is not None:
        group_cols = [groups] if isinstance(groups, str) else list(groups)
        rows = []
        for key, group_df in combined_gdf.groupby(group_cols):
            v = group_df[score_col].to_numpy(dtype=float)
            w = group_df[population_col].to_numpy(dtype=float)
            stats = _stats_for(v, w)
            row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            row.update(
                {
                    "weighted_mean": stats["weighted_mean"],
                    "weighted_median": stats["weighted_median"],
                    "gini": stats["gini"],
                }
            )
            for t in thresholds:
                row[f"pct_above_{t}"] = stats["pct_population_above_threshold"][t]
                row[f"pct_below_{t}"] = stats["pct_population_below_threshold"][t]
            rows.append(row)
        result["by_group"] = pd.DataFrame(rows).set_index(group_cols)

    return result
