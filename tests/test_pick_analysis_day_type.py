"""`pick_analysis_date(day_type=...)`: a working weekday (default), a Saturday or a Sunday."""

from datetime import date

import pytest

from transitlos import stops


class _FakeFeed:
    """Only what `pick_analysis_date` needs before its best-effort trip-count check."""

    def __init__(self):
        self.calendar = type("C", (), {"min_date": date(2026, 11, 1), "max_date": date(2026, 12, 31)})()
        self.asked = []

    def get_representative_date(self, start_date=None, end_date=None, date_type="weekday", route_types=None):
        self.asked.append(date_type)
        return {"weekday": date(2026, 11, 18), "saturday": date(2026, 11, 21), "sunday": date(2026, 11, 22)}[date_type]


@pytest.mark.parametrize("day_type,expected", [("weekday", date(2026, 11, 18)), ("saturday", date(2026, 11, 21)), ("sunday", date(2026, 11, 22))])
def test_day_type_reaches_the_representative_date(day_type, expected):
    feed = _FakeFeed()
    with pytest.warns(UserWarning):   # the trip-count check needs a real feed and is skipped here
        assert stops.pick_analysis_date(feed, day_type=day_type) == expected
    assert feed.asked[0] == day_type


def test_unknown_day_type():
    with pytest.raises(ValueError):
        stops.pick_analysis_date(_FakeFeed(), day_type="holiday")
