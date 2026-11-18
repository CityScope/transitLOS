"""One row per real stop_id (at its own location) with ITS routes and mode, plus the scores of the station it belongs to.

The scoring works per parent station (a group of stops: `best_subset_stop_scores`), but the map must show every stop where it really is:
a rail platform and the bus stops around it are separate points, each with its own mode (emoji), its own route list (colours included)
and, in the popup, the station-level data. `stop_table_from_feed` builds that table from the same `Feed` the scoring used:

    stop_id, stop_name, stop_lat, stop_lon, parent_station,
    mode_category        mode OF THIS STOP: rail > tram > bus among the routes that serve this stop_id,
    all_routes           this stop's route labels (comma separated, rail first),
    route_info           "label~#color~#textcolor~mode~kept(1/0);..." for the same routes (colours exactly as the route lines draw them),
    + the station's score columns (`stop_score`, `headway_minutes`, ... `scoring_mode`) joined on `parent_station`.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import polars as pl

from .map.routes import MODE_FALLBACK_COLOR, _hex_color
from .scoring.mrc import mode_category

_MODE_ORDER = {"rail": 0, "tram": 1, "bus": 2}
_HEX = re.compile(r"^[0-9a-fA-F]{6}$")


def _color(v) -> str:
    v = str(v or "").strip().lstrip("#")
    return "#" + v.upper() if _HEX.match(v) else ""


def _label(short, long, rid) -> str:
    s = str(short or "").strip() or str(long or "").strip() or str(rid or "").strip()
    return re.sub(r"[;~]", " ", s)


def stop_table_from_feed(feed, stations: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The stop-level table of `feed` (see module docstring); `stations` = the scored station table (joined on `parent_station`).

    Args:
        feed: A loaded pyGTFSHandler Feed.
        stations: Optional scored station table.

    Returns:
        One row per stop.
    """
    routes = feed.routes.lf.collect()
    keep = [c for c in ("route_id", "route_short_name", "route_long_name", "route_type", "route_color", "route_text_color") if c in routes.columns]
    rt = routes.select(keep).with_columns(pl.col("route_id").cast(pl.Utf8))
    rows = rt.to_dicts()
    info = {}
    for r in rows:
        try:
            mode = "tram" if int(r.get("route_type")) == 702 else mode_category(int(r.get("route_type")))
        except (TypeError, ValueError):
            mode = "bus"
        # Same colour rule as the route lines (`map.routes.build_route_lines`): the GTFS colour, else the mode's fallback colour.
        info[str(r["route_id"])] = (_label(r.get("route_short_name"), r.get("route_long_name"), r["route_id"]),
                                    _hex_color(r.get("route_color"), MODE_FALLBACK_COLOR.get(mode, MODE_FALLBACK_COLOR["bus"])),
                                    _hex_color(r.get("route_text_color"), "#FFFFFF"), mode, str(r["route_id"]))
    served = (
        feed.stop_times.lf.select("trip_id", "stop_id").unique()
        .join(feed.trips.lf.select("trip_id", "route_id"), on="trip_id", how="inner")
        .select(pl.col("stop_id").cast(pl.Utf8), pl.col("route_id").cast(pl.Utf8)).unique().collect()
    )
    per_stop: dict = {}
    for sid, rid in zip(served["stop_id"].to_list(), served["route_id"].to_list()):
        if rid in info:
            per_stop.setdefault(sid, set()).add(rid)
    stops = feed.stops.lf.collect()
    cols = [c for c in ("stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station") if c in stops.columns]
    st = stops.select(cols).with_columns(pl.col("stop_id").cast(pl.Utf8)).to_pandas()
    # Routes the station's best-scoring subset keeps (`route_ids`, `;`-separated); the others are marked "0" in `route_info`.
    kept: dict = {}
    if stations is not None and not stations.empty and "route_ids" in stations.columns:
        for ps, rids in zip(stations["parent_station"].astype(str), stations["route_ids"]):
            kept[ps] = set(str(rids).split(";")) if isinstance(rids, str) else None
    out_rows = []
    for sid, name, lat, lon, parent in zip(st["stop_id"], st["stop_name"], st["stop_lat"], st["stop_lon"], st.get("parent_station", st["stop_id"])):
        rids = per_stop.get(str(sid))
        if not rids:
            continue
        items = sorted({info[r] for r in rids}, key=lambda t: (_MODE_ORDER.get(t[3], 3), t[0]))
        keep = kept.get(str(parent))
        out_rows.append({
            "stop_id": str(sid), "stop_name": name, "stop_lat": float(lat), "stop_lon": float(lon), "parent_station": str(parent),
            "mode_category": items[0][3],
            "all_routes": ",".join(t[0] for t in items),
            "route_info": ";".join("~".join(t[:4] + (("0" if keep is not None and t[4] not in keep else "1"),)) for t in items),
        })
    table = pd.DataFrame(out_rows)
    if table.empty or stations is None or stations.empty:
        return table
    cols = [c for c in stations.columns if c not in ("stop_lat", "stop_lon", "stop_name", "geometry", "mode_category", "route_ids")]
    return table.merge(stations[cols].drop_duplicates("parent_station"), on="parent_station", how="left")
