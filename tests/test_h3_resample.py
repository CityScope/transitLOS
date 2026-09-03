"""Tests for transitlos.h3_resample.level_of_service_to_h3.

Covers the transit-specific policy layered on top of
`geohierarchy.edges.edges_to_h3_by_distance`: max-score-per-cell, the
nearest-street fallback, and full coverage via the "no access" sentinel.
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from transitlos.h3_resample import level_of_service_to_h3


@pytest.fixture
def access_gdf():
    # Two streets in one corner of the AOI below, with different scores --
    # any cell touched by both should read the higher (max), not a blend.
    return gpd.GeoDataFrame(
        {
            "level_of_service": [0.3, 0.9],
            "geometry": [
                LineString([(0.0005, 0.0005), (0.0015, 0.0015)]),
                LineString([(0.0006, 0.0006), (0.0016, 0.0016)]),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def aoi():
    return gpd.GeoDataFrame({"geometry": [box(0, 0, 0.05, 0.05)]}, crs="EPSG:4326")


def test_every_cell_gets_a_value(access_gdf, aoi):
    result = level_of_service_to_h3(
        access_gdf, resolution=8, aoi=aoi, fallback_radius_multiplier=5.0
    )
    assert result["level_of_service"].notna().all()
    assert (result["level_of_service"] >= 0).all()


def test_near_cells_get_the_max_not_the_mean(access_gdf, aoi):
    result = level_of_service_to_h3(
        access_gdf, resolution=8, aoi=aoi, fallback_radius_multiplier=5.0
    )
    touched = result[result["level_of_service"] > 0]
    assert not touched.empty
    # 0.9 (the max of the two streets) must appear somewhere; 0.6 (a mean)
    # must not, proving max-not-mean semantics.
    assert (touched["level_of_service"] == 0.9).any()
    assert not (touched["level_of_service"].round(2) == 0.6).any()


def test_far_cells_fall_back_to_no_access_sentinel(access_gdf, aoi):
    result_no_fallback = level_of_service_to_h3(
        access_gdf, resolution=8, aoi=aoi, fallback_radius_multiplier=None
    )
    zero_count_no_fallback = (result_no_fallback["level_of_service"] == 0.0).sum()

    result_with_fallback = level_of_service_to_h3(
        access_gdf,
        resolution=8,
        aoi=aoi,
        fallback_radius_multiplier=5.0,
        no_access_value=0.0,
    )
    zero_count_with_fallback = (result_with_fallback["level_of_service"] == 0.0).sum()

    # Every cell is populated either way (both fill remaining gaps with the
    # sentinel), but the fallback should recover real scores for some cells
    # that would otherwise have fallen back straight to the sentinel.
    assert result_no_fallback["level_of_service"].notna().all()
    assert result_with_fallback["level_of_service"].notna().all()
    assert zero_count_with_fallback < zero_count_no_fallback
    assert len(result_no_fallback) == len(result_with_fallback)


def test_custom_no_access_value(access_gdf, aoi):
    result = level_of_service_to_h3(
        access_gdf,
        resolution=8,
        aoi=aoi,
        fallback_radius_multiplier=1.0,
        no_access_value=-1.0,
    )
    assert (result["level_of_service"] == -1.0).any()


def test_empty_access_gdf_raises():
    empty = gpd.GeoDataFrame({"level_of_service": [], "geometry": []}, crs="EPSG:4326")
    with pytest.raises(ValueError):
        level_of_service_to_h3(empty, resolution=8)
