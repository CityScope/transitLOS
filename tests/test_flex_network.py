"""Multi-label search on a small street grid: stations whose shortest paths OVERLAP are both reported with correct distances; dominance prunes the worse."""
import numpy as np
import polars as pl
import pytest
import shapely

pytest.importorskip("h3ronpy")
from transitlos.flex import core, scoring as sc
from UrbanAccessAnalyzer.csr_network import CsrNetwork

CRS = "EPSG:32618"
X0, Y0 = 327000.0, 4690000.0        # Boston area, metres


def grid(n=7, step=100.0):
    nodes = pl.DataFrame({"node_id": list(range(n * n)), "x": [X0 + (i % n) * step for i in range(n * n)], "y": [Y0 + (i // n) * step for i in range(n * n)]})
    u, v, geom = [], [], []
    for i in range(n * n):
        for j in ((i + 1) if i % n < n - 1 else None, (i + n) if i // n < n - 1 else None):
            if j is None:
                continue
            u.append(i); v.append(j)
            geom.append(shapely.to_wkb(shapely.LineString([(X0 + (i % n) * step, Y0 + (i // n) * step), (X0 + (j % n) * step, Y0 + (j // n) * step)])))
    return CsrNetwork(nodes, pl.DataFrame({"u": u, "v": v, "length_m": [step] * len(u), "geometry_wkb": geom}), CRS)


def profiles(rows):
    """rows = [[(mode, headway, speed), ...] per station]"""
    P = max(len(r) for r in rows)
    m = np.full((len(rows), P), -1); h = np.zeros((len(rows), P)); v = np.zeros((len(rows), P))
    for i, r in enumerate(rows):
        for j, (a, b, c) in enumerate(r):
            m[i, j], h[i, j], v[i, j] = a, b, c
    return core.Profiles(m, h, v)


def test_overlapping_paths_both_stations_reported():
    net = grid()
    # A = bus with the better headway/speed, B = rail (better mode, worse headway/speed): neither dominates the other. Both sit on the bottom row west of
    # node 24 (row 3, col 3), so the shortest paths of A and B to it overlap along the same streets
    prof = profiles([[(0, 4.0, 30.0)], [(2, 20.0, 25.0)]])
    src_node = np.array([0, 1]); src_off = np.zeros(2, dtype=np.float32); src_station = np.array([0, 1])
    cfg = core.Config(max_walk_m=1500.0)
    ls, ld, ln, lc, st = core.multilabel_search(net.indptr, net.adj, net.w, src_node, src_off, src_station, prof, cfg)
    assert set(ls[0, :ln[0]]) == {0, 1}                        # both labels exist at A's own node
    assert dict(zip(ls[24, :ln[24]], ld[24, :ln[24]])).get(1) == pytest.approx(100 * (2 + 3))   # B -> node 24: 2 columns + 3 rows
    assert dict(zip(ls[24, :ln[24]], ld[24, :ln[24]])).get(0) == pytest.approx(100 * (3 + 3))   # A -> node 24 (overlapping path): also reported, own distance


def test_dominated_station_is_pruned():
    net = grid()
    prof = profiles([[(2, 5.0, 40.0)], [(0, 20.0, 15.0)]])      # station 0 (rail, 5 min, 40 km/h) is closer AND better than station 1
    ls, ld, ln, lc, st = core.multilabel_search(net.indptr, net.adj, net.w, np.array([0, 2]), np.zeros(2, dtype=np.float32), np.array([0, 1]), prof, core.Config())
    assert 1 not in set(ls[7, :ln[7]])                        # node 7 is next to station 0: the farther, worse station 1 is dominated there
    assert st[1] > 0
