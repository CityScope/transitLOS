"""Files for the browser: the compact binary tables the front-end needs to recompute scores and catchments (see frontend/).

    export_for_browser(out_dir, lat, lon, prof, res, params, counts=None, count_labels=None)

writes   meta.json                 sizes, bbox, the default ScoreParams (JSON), the shape catalog, the names of the census count columns
         stop_{mode,h,v,np}.bin    station profiles  int8 / float32 / float32 [S*P], int32 [S];   stop_lonlat.bin float32 [S*2]
         cell_station.bin, cell_dist.bin   int32 [nc*K], uint16 [nc*K]
         cell_counts.bin           float32 [nc*ncol]: census counts of each cell (population, jobs, ...), the numbers summed per winning stop
         cell_geom.bin             float32 [nc*14]: centre then 6 vertices, lon,lat (the demo viewer draws these; a production map gets geometry from tiles)
         viewer.html, score.js, editor.js, glmap.js     copied from frontend/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import shapes
from .scoring import DEFAULT, Curve, ScoreParams

FRONTEND = Path(__file__).resolve().parent / "frontend"


def shape_catalog() -> dict:
    """Presets for each curve, as tables (JSON), keyed by name -- what a UI offers in its 'shape' drop-downs."""
    def c(x):
        return {"kind": x.kind, "lo": x.lo, "hi": x.hi, "scale": x.scale, "y": list(x.y)}
    return {
        "headway_curve": {"default (log-linear elasticity)": c(Curve()), "exponential (scale 40 min)": c(shapes.headway_exponential()), "power (exponent 0.6)": c(shapes.headway_power()),
                          "logistic (mid 30 min)": c(shapes.headway_logistic()), "steps (5/10/20/30/60/120 min)": c(shapes.headway_step())},
        "speed_curve": {"default (Hill, anchor 30 km/h)": c(Curve()), "linear 5-40 km/h": c(shapes.speed_linear()), "logistic (mid 20 km/h)": c(shapes.speed_logistic()),
                        "steps (10/20/30/45 km/h)": c(shapes.speed_step())},
        "walk_curve": {"default ((1+u)^-1)": c(Curve()), "exponential": c(shapes.walk_exponential()), "gaussian": c(shapes.walk_gaussian()), "linear to u=3": c(shapes.walk_linear()),
                       "logistic (mid u=2)": c(shapes.walk_logistic()), "hard buffer (plateau radius)": c(shapes.walk_hard_buffer())},
    }


def publish_frontend(out) -> None:
    """Copies the scripts and writes the stamped, version-addressed viewer page into `out`.

    Args:
        out: Output folder of the map.
    """
    out = Path(out)
    for f in ("score.js", "editor.js", "glmap.js", "fit.js"):
        shutil.copy(FRONTEND / f, out / f)
    from transitlos.map.buildstamp import write_stamped
    import hashlib, re
    page = (FRONTEND / "viewer.html").read_text(encoding="utf-8")
    page = re.sub(r'<script src="([\w.-]+\.js)"', lambda m: f'<script src="{m.group(1)}?v={hashlib.sha1((out / m.group(1)).read_bytes()).hexdigest()[:10]}"', page)   # a changed script is a new URL
    write_stamped(out / "viewer.html", page)


def export_for_browser(out_dir, lat, lon, prof, res, params: ScoreParams = DEFAULT, geometry: bool = True, counts=None, count_labels=None, extra_meta=None, points=None) -> dict:
    """`points` (optional DataFrame: stop_lon, stop_lat, station, stop_name): the real stop_ids drawn on the map (each coloured/sized/selected as its station).

    Args:
        out_dir: Output folder (created).
        lat: Station latitudes.
        lon: Station longitudes.
        prof: Station profiles.
        res: Cell stations (:class:`CellStops`).
        params: Default scoring parameters.
        geometry: Also write the cell geometry.
        counts: Optional ``[cells, columns]`` counts.
        count_labels: Names of the count columns.
        extra_meta: Extra fields for ``meta.json``.
        points: Optional real stops (see above).

    Returns:
        The ``meta.json`` contents.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S, P = prof.S, prof.P
    nc, K = res.station.shape
    prof.mode.astype(np.int8).tofile(out / "stop_mode.bin")
    prof.h.astype(np.float32).tofile(out / "stop_h.bin")
    prof.v.astype(np.float32).tofile(out / "stop_v.bin")
    prof.n.astype(np.int32).tofile(out / "stop_np.bin")
    res.station.astype(np.int32).tofile(out / "cell_station.bin")
    res.dist.astype(np.uint16).tofile(out / "cell_dist.bin")
    np.column_stack([np.asarray(lon), np.asarray(lat)]).astype(np.float32).tofile(out / "stop_lonlat.bin")
    cols = list(counts.columns) if counts is not None else []
    if counts is not None:
        counts.to_numpy(dtype=np.float32).tofile(out / "cell_counts.bin")
    meta = {"S": int(S), "P": int(P), "nc": int(nc), "K": int(K), "params": json.loads(params.to_json()), "catalog": shape_catalog(),
            "count_columns": cols, "count_labels": [(count_labels or {}).get(c, c) for c in cols],
            "stations_lonlat": [float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))] if len(lat) else None}
    if extra_meta:
        meta.update(extra_meta)
    if geometry:
        import h3
        g = np.zeros((nc, 14), dtype=np.float32)
        for i, c in enumerate(res.cells):
            cid = f"{int(c):x}"
            la, lo = h3.cell_to_latlng(cid)
            g[i, 0], g[i, 1] = lo, la
            b = h3.cell_to_boundary(cid)   # (lat, lon) pairs; pentagons have 5 vertices: the last one is repeated
            for j in range(6):
                lt, ln = b[j % len(b)]
                g[i, 2 + 2 * j], g[i, 3 + 2 * j] = ln, lt
        g.tofile(out / "cell_geom.bin")
        meta["bbox"] = [float(g[:, 0::2].min()), float(g[:, 1::2].min()), float(g[:, 0::2].max()), float(g[:, 1::2].max())]
    if points is not None and len(points):
        np.column_stack([points.stop_lon.to_numpy(), points.stop_lat.to_numpy()]).astype(np.float32).tofile(out / "pt_lonlat.bin")
        points.station.to_numpy().astype(np.int32).tofile(out / "pt_station.bin")
        (out / "pt_names.json").write_text(json.dumps([str(x).replace("_", " ") for x in points.stop_name]))
        meta["points"] = int(len(points))
    (out / "meta.json").write_text(json.dumps(meta))
    publish_frontend(out)
    return meta
