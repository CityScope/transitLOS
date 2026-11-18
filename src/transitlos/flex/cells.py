"""Stations of every H3 cell on a street network: snap -> multi-label search -> edge/cell distances -> floors, dominance, cap.

    net = CsrNetwork.load(network_cache_folder)               # or CsrNetwork(nodes, edges, crs) from a StreetNetwork's tables
    res = compute_cell_stops(net, stations_lat, stations_lon, prof, cfg)
    los, slot, profile = recompute(res, prof, params)         # any scoring, 5-10 ms per 10^4 cells
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import core
from .scoring import DEFAULT, ScoreParams


@dataclass
class CellStops:
    """The stations each H3 cell can walk to: up to K per cell, with the walking distance, plus flags and search
    statistics.
    """
    cells: np.ndarray        # uint64 H3 ids, sorted
    station: np.ndarray      # [nc, K] int32 station index (-1 empty)
    dist: np.ndarray         # [nc, K] uint16 metres (65535 empty)
    flag: np.ndarray         # [nc] uint8: bit0 the cap was needed, bit1 candidate list limit hit, bit2 search refused labels near the cell
    cropped: np.ndarray      # [nc] uint8: stations genuinely cropped (non-dominated, above the floors, over the cap) -- not those dropped by floors/dominance
    stats: dict              # timings and search statistics


def compute_cell_stops(net, lat, lon, prof: core.Profiles, cfg: core.Config = core.Config(), params: ScoreParams = DEFAULT, resolution: int = 10,
                       max_snap_m: float = 300.0) -> CellStops:
    """`net` = UrbanAccessAnalyzer.csr_network.CsrNetwork; stations at (lat, lon) with `prof` profiles (same order).

    Args:
        net: Street network (``UrbanAccessAnalyzer.csr_network.CsrNetwork``).
        lat: Station latitudes.
        lon: Station longitudes.
        prof: Station profiles, in the same order as the stations.
        cfg: Caps and floors.
        params: Scoring parameters (used by the cap).
        resolution: H3 resolution of the cells.
        max_snap_m: Stations farther than this from a street are not used.

    Returns:
        The :class:`CellStops`.
    """
    t = {}
    t0 = time.perf_counter()
    sn, so, ss, ok = net.snap(lat, lon, max_snap_m)
    t["snap_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ls, ld, ln, lc, st = core.multilabel_search(net.indptr, net.adj, net.w, sn, so, ss, prof, cfg)
    t["search_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    reached = ln > 0   # only edges with a reached end can give a cell a station
    cells, rec_off, ru, rv, ra, rb, rl = net.edge_cell_records(resolution, edge_mask=reached[net.eu] | reached[net.ev])
    t["edge_cells_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cs, cd, cf, cr = core.process_cells(rec_off, ru, rv, ra, rb, rl, (ls, ld, ln, lc), prof, cfg, params)
    t["cells_s"] = time.perf_counter() - t0
    stats = dict(t, stations=int(len(lat)), stations_snapped=int(ok.sum()), labels=int(st[0]), dropped_by_dominance=int(st[1]), node_cap_hits=int(st[2]),
                 cells=int(len(cells)), cells_with_stations=int((cs >= 0).any(1).sum()), cells_capped=int((cf & 1).sum()), cells_with_cropped=int((cr > 0).sum()),
                 stations_cropped=int(cr.sum()), max_cropped_in_a_cell=int(cr.max()) if len(cr) else 0)
    return CellStops(cells, cs, cd, cf, cr, stats)


def recompute(res: CellStops, prof: core.Profiles, params: ScoreParams = DEFAULT):
    """Cell scores (0-100) under `params`: (los[nc], winning station slot[nc], winning profile[nc]).

    Args:
        res: Result of :func:`compute_cell_stops`.
        prof: Station profiles.
        params: Scoring parameters.
    """
    return core.recompute_cells_fast(res.station, res.dist, prof, params)
