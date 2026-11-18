import numpy as np
import pytest

from transitlos.flex import catchment as ct


def test_winner_and_totals_conserve_counts():
    cs = np.array([[0, 1], [2, -1], [1, 0], [-1, -1]], dtype=np.int32)
    slot = np.array([1, 0, 0, -1], dtype=np.int8)
    w = ct.winner_station(cs, slot)
    assert w.tolist() == [1, 2, 1, -1]
    counts = np.array([[10.0, 1], [20, 2], [30, 3], [40, 4]])
    tot = ct.stop_catchment(w, counts, 3)
    assert tot.tolist() == [[0, 0], [40, 4], [20, 2]]                    # cell 3 has no winner: not attributed
    assert tot.sum(0).tolist() == (counts[:3].sum(0)).tolist()
    assert ct.cells_won(w, 3).tolist() == [0, 2, 1]


def test_block_counts_spread_over_cells():
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("h3ronpy")
    import h3
    from shapely.geometry import box
    from h3ronpy.vector import coordinates_to_cells
    # a ~1 km square block with 1000 people, and a tiny block (inside one cell) with 50
    big = box(-71.100, 42.350, -71.088, 42.359)
    tiny = box(-71.0501, 42.3501, -71.0500, 42.3502)
    blocks = gpd.GeoDataFrame({"population": [1000.0, 50.0]}, geometry=[big, tiny], crs=4326)
    xs, ys = np.meshgrid(np.linspace(-71.101, -71.087, 60), np.linspace(42.349, 42.360, 60))
    cells = np.unique(np.asarray(coordinates_to_cells(ys.ravel(), xs.ravel(), 10)).astype(np.uint64))
    tc = np.asarray(coordinates_to_cells(np.array([42.35015]), np.array([-71.05005]), 10)).astype(np.uint64)
    cells = np.unique(np.r_[cells, tc])
    out = ct.block_counts_to_cells(blocks, cells, ["population"])
    assert out["population"].sum() == pytest.approx(1050.0, rel=0.15)    # uniform density: about everything lands on the served cells
    tiny_cell = int(np.flatnonzero(cells == tc[0])[0])
    assert out["population"].iloc[tiny_cell] >= 50.0
