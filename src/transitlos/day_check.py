"""Is a day a typical day of a GTFS feed? A day is EITHER a holiday OR a normal day, never a mix: a feed is scored on a shared analysis
date only when that date runs the feed's normal volume of trips for that weekday.

`day_volume_ratio(gtfs_dir, day)`: trips active on `day` / median trips active on the same weekday 1-3 weeks before and after (inside the
feed's calendar). `is_typical_day(...)`: ratio within [0.8, 1.25] (a feed's own holiday removing service, or a holiday running the Sunday
service on top of the weekday one, falls outside). Reads only calendar.txt, calendar_dates.txt and trips.txt (fast, no stop_times).
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Dict, Optional, Set

_WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _rows(path: Path):
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


class _Cal:
    def __init__(self, d: Path):
        self.cal = _rows(d / "calendar.txt")
        self.exc: Dict[str, list] = {}
        for r in _rows(d / "calendar_dates.txt"):
            self.exc.setdefault(r.get("date", ""), []).append((r.get("service_id", ""), r.get("exception_type", "")))
        self.trips: Dict[str, int] = {}
        for r in _rows(d / "trips.txt"):
            s = r.get("service_id", "")
            self.trips[s] = self.trips.get(s, 0) + 1

    def active(self, day: date) -> Set[str]:
        ymd, wd = day.strftime("%Y%m%d"), _WD[day.weekday()]
        out = {r["service_id"] for r in self.cal if r.get(wd) == "1" and r.get("start_date", "") <= ymd <= r.get("end_date", "99999999")}
        for sid, t in self.exc.get(ymd, []):
            if t == "1":
                out.add(sid)
            elif t == "2":
                out.discard(sid)
        return out

    def volume(self, day: date) -> int:
        return sum(self.trips.get(s, 0) for s in self.active(day))


def day_volume_ratio(gtfs_dir, day: date, weeks: int = 3) -> Optional[float]:
    """Trips on `day` over the median of the same weekday `weeks` weeks before / after (days with no trips at all ignored); None if the
    neighbours have no service either (nothing to compare with).

    Args:
        gtfs_dir: Unzipped GTFS folder.
        day: The day checked.
        weeks: Weeks before and after to compare with.

    Returns:
        The ratio, or None.
    """
    c = _Cal(Path(gtfs_dir))
    v = c.volume(day)
    nb = [c.volume(day + timedelta(days=7 * k)) for k in range(-weeks, weeks + 1) if k]
    nb = [x for x in nb if x > 0]
    if not nb:
        return None
    m = median(nb)
    return v / m if m else None


def is_typical_day(gtfs_dir, day: date, lo: float = 0.8, hi: float = 1.25) -> bool:
    """Whether ``day`` has a normal amount of service: its trips are between ``lo`` and ``hi`` times the same weekday of
    the weeks around it (True when there is nothing to compare with).

    Args:
        gtfs_dir: Unzipped GTFS folder.
        day: The day checked.
        lo: Lowest accepted ratio.
        hi: Highest accepted ratio.

    Returns:
        True for a typical day.
    """
    r = day_volume_ratio(gtfs_dir, day)
    return r is None or lo <= r <= hi
