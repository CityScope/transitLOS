"""The map (frontend/viewer.html) on a synthetic dataset, in headless Chromium: JavaScript recomputation == Python for edited shapes/params, census catchments, click popups.
Skipped when playwright / chromium / h3ronpy are not available."""
import importlib.util
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h3ronpy")
pw = pytest.importorskip("playwright.sync_api")
from transitlos.flex import catchment as ct, core, scoring as sc, shapes as sh
from transitlos.flex.cells import compute_cell_stops
from transitlos.flex.export import export_for_browser

_spec = importlib.util.spec_from_file_location("netgrid", Path(__file__).resolve().parent / "test_flex_network.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    net = G.grid(n=12, step=90.0)
    prof = G.profiles([[(0, 12.0, 20.0), (0, 40.0, 30.0)], [(2, 6.0, 35.0)], [(1, 15.0, 22.0)]])
    from pyproj import Transformer
    tr = Transformer.from_crs(G.CRS, 4326, always_xy=True)
    nodes = [5, 60, 110]
    lon, lat = tr.transform(np.array([G.X0 + (n % 12) * 90 for n in nodes]), np.array([G.Y0 + (n // 12) * 90 for n in nodes]))
    res = compute_cell_stops(net, lat, lon, prof, core.Config())
    rng = np.random.default_rng(0)
    counts = pd.DataFrame({"population": rng.integers(0, 500, len(res.cells)).astype(float), "jobs": rng.integers(0, 300, len(res.cells)).astype(float)})
    web = tmp_path_factory.mktemp("web")
    export_for_browser(web, lat, lon, prof, res, sc.DEFAULT, counts=counts, count_labels={"population": "Population", "jobs": "Jobs"}, extra_meta={"name": "synthetic"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(web)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port, res, prof, counts
    srv.shutdown()


def test_viewer_matches_python(site):
    port, res, prof, counts = site
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True, args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
        except Exception as exc:   # no browser installed
            pytest.skip(f"chromium not available: {exc}")
        pg = b.new_page(viewport={"width": 1300, "height": 800})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:{port}/viewer.html")
        pg.evaluate("window.__ready")

        def agree(py):
            js = np.array(pg.evaluate("__api.los()"), dtype=np.float64)
            los, slot, _ = core.recompute_cells_fast(res.station, res.dist, prof, py)
            tot = ct.stop_catchment(ct.winner_station(res.station, slot), counts.to_numpy(), prof.S)
            jt = np.array(pg.evaluate("__api.tot()")).reshape(prof.S, -1)
            assert np.abs(js - los).max() < 1e-3
            assert np.abs(jt - tot).max() <= 1e-6 * max(tot.max(), 1)

        agree(sc.DEFAULT)
        pg.evaluate("__api.buildEditors()")
        sels = pg.query_selector_all("#editors .ed select")
        sels[1].select_option("exponential"); pg.wait_for_timeout(100); pg.evaluate("__api.schedule()")
        py = sc.DEFAULT.with_(headway_curve=sh.headway_exponential(5, 40)); agree(py)
        sels[3].select_option("gaussian"); pg.wait_for_timeout(100); pg.evaluate("__api.schedule()")
        py = py.with_(walk_curve=sh.walk_gaussian(1.0)); agree(py)
        pg.evaluate("(()=>{const p=__api.getParams(); p.w_mode=0; p.w_speed=0; __api.setParams(p)})()")
        agree(py.with_(w_mode=0.0, w_speed=0.0))
        # click the stop that wins the most population: the popup shows its count
        pg.evaluate("__api.setParams(JSON.parse(JSON.stringify(__api.meta().params)))")
        los, slot, _ = core.recompute_cells_fast(res.station, res.dist, prof, sc.DEFAULT)
        tot = ct.stop_catchment(ct.winner_station(res.station, slot), counts.to_numpy(), prof.S)
        s = int(tot[:, 0].argmax())
        x, y = pg.evaluate(f"__api.stopXY({s})"); pg.evaluate(f"__api.click({x}, {y})")
        assert pg.evaluate("__api.popupKind()") == f"stop:{s}" and f"{int(round(tot[s, 0])):,}" in pg.evaluate("__api.popup()")
        assert not errs
        b.close()
