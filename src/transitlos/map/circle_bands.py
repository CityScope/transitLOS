"""Zoom bands of the circle view, from how far apart the circles of each H3 resolution are on screen.

A resolution takes over where its neighbouring cell centres are ~6 px apart (at the map's latitude) and keeps the view until the next finer
resolution reaches that spacing, so a view never shows a sparse handful of circles nor a solid carpet of them. Each band starts no earlier
than the first zoom its tiles exist at (an earlier start would draw nothing); the radius grows with zoom inside the band (the page's
`__applyCircleRadius`). Bands may start at fractional zooms.

    apply_to_page(path)   # rewrites a built page's circle layers and `__circleZoomBands`
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

#: Distance between neighbouring H3 cell centres (km), sqrt(2 * mean cell area / sqrt(3)).
SPACING_KM = {0: 1121.0, 1: 419.6, 2: 158.3, 3: 119.6, 4: 45.2, 5: 17.09, 6: 6.46, 7: 2.441, 8: 0.9227, 9: 0.3487, 10: 0.1318,
              11: 0.04982, 12: 0.01883, 13: 0.007116, 14: 0.002689, 15: 0.001016}
TARGET_PX = 6.0


def start_zoom(res: int, lat: float) -> float:
    """Zoom at which cells of `res` are TARGET_PX apart at latitude `lat`.

    Args:
        res: H3 resolution.
        lat: Latitude (degrees).

    Returns:
        Zoom level.
    """
    return math.log2(TARGET_PX * 40075.0 * math.cos(math.radians(lat)) / (256.0 * SPACING_KM[res]))


def circle_bands(resolutions: Iterable[int], lat: float, tile_min_zoom: Optional[Dict[int, int]] = None) -> Dict[int, tuple]:
    """`{res: (min_zoom, max_zoom_exclusive)}`; the coarsest resolution starts at 0 and the finest runs to 24.

    Args:
        resolutions: H3 resolutions drawn as circles.
        lat: Latitude of the map.
        tile_min_zoom: Optional ``{res: first zoom with tiles}``.
    """
    res = sorted(resolutions); tile_min_zoom = tile_min_zoom or {}
    lo = {}
    for i, r in enumerate(res):
        if i == 0:
            lo[r] = 0.0
            continue
        z = max(start_zoom(r, lat), float(tile_min_zoom.get(r, 0)), lo[res[i - 1]] + 0.5)
        lo[r] = round(z, 1)
    return {r: (lo[r], lo[res[i + 1]] if i + 1 < len(res) else 24.0) for i, r in enumerate(res)}


def _header_min_zoom_and_lat(path: Path):
    from geohierarchy.maps.maplibre.pmtiles_meta import read_header

    h = read_header(path)
    return (h["minzoom"], (h["bounds"][1] + h["bounds"][3]) / 2 if h and "bounds" in h else None) if h else (None, None)


def apply_to_page(page) -> dict:
    """Sets every circle layer's zoom range and the page's `__circleZoomBands` (inclusive ints are [lo, hi_exclusive - 1]) from the
    circle tiles next to the page.

    Args:
        page: Path of the built page.

    Returns:
        ``{res: (min zoom, max zoom)}`` applied (empty if the page has no circle tiles).
    """
    page = Path(page); s = page.read_text(encoding="utf-8")
    tiles = {}
    for sid, r, rel in re.findall(r'"(circles:h3_(\d+)(?:__chunk\d+)?)": \{"type": "vector", "tiles": \["pmtiles://([^"]+?\.pmtiles)/', s):
        tiles.setdefault(int(r), []).append(page.parent / rel)
    if not tiles:
        return {}
    tile_min, lats = {}, []
    for r, files in tiles.items():
        mins = []
        for f in files:
            mz, lat = _header_min_zoom_and_lat(f)
            if mz is not None:
                mins.append(mz)
            if lat is not None and r == min(tiles):
                lats.append(lat)
        tile_min[r] = min(mins) if mins else 0
    lat = sum(lats) / len(lats) if lats else 40.0
    bands = circle_bands(tiles, lat, tile_min)
    n = 0
    for r, (lo, hi) in bands.items():
        pat = r'("id": "circles:h3_%d_circle(?:__chunk\d+)?", "type": "circle", "source": "[^"]+", "source-layer": "[^"]+", "minzoom": )[\d.]+(, "maxzoom": )[\d.]+' % r
        s, k = re.subn(pat, lambda m: f"{m.group(1)}{lo:g}{m.group(2)}{hi:g}", s); n += k
    m = re.search(r"__circleZoomBands = (\{[^;]*\});", s)
    if m:
        s = s[:m.start(1)] + json.dumps({str(r): [lo, round(hi - 1, 1)] for r, (lo, hi) in bands.items()}) + s[m.end(1):]
    page.write_text(s, encoding="utf-8")
    return {"lat": round(lat, 1), "layers": n, "bands": bands}
