import os

import geopandas as gpd
import shapely

from transitlos import network as net


class _FakeNetwork:
    def simplify(self, *a, **k):
        return self

    def crop(self, *a, **k):
        return self

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        open(os.path.join(folder, "meta.txt"), "w").write("x")


def test_cache_by_aoi_uses_one_subfolder_per_area_and_reuses_it(tmp_path, monkeypatch):
    built = []

    def fake_from_pbf(cls, *a, **k):
        built.append(1)
        return _FakeNetwork()

    monkeypatch.setattr(net.StreetNetwork, "from_pbf", classmethod(fake_from_pbf))
    monkeypatch.setattr(net.StreetNetwork, "load", classmethod(lambda cls, folder: ("loaded", folder)))
    a = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 0.01, 0.01)], crs=4326)
    b = gpd.GeoDataFrame(geometry=[shapely.box(1, 1, 1.01, 1.01)], crs=4326)

    net.prepare_street_network(a, cache_dir=str(tmp_path), pbf_path="x.pbf", cache_by_aoi=True)
    net.prepare_street_network(b, cache_dir=str(tmp_path), pbf_path="x.pbf", cache_by_aoi=True)
    entries = sorted(p.name for p in tmp_path.iterdir())
    assert len(built) == 2 and len(entries) == 2 and all(e.startswith("network_") for e in entries)  # one per area

    again = net.prepare_street_network(a, cache_dir=str(tmp_path), pbf_path="x.pbf", cache_by_aoi=True)
    assert len(built) == 2 and again[0] == "loaded"  # same area: from the cache, not rebuilt
