import datetime

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import trackintel as ti
from trackintel.analysis.location_identification import pre_filter_locations


@pytest.fixture
def example_staypoints():
    """Staypoints to load into the database."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-01 07:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-02 23:00:00", tz="utc")  # for time testin bigger gap
    one_hour = datetime.timedelta(hours=1)

    list_dict = [
        {"user_id": 0, "started_at": t1, "finished_at": t2, "geometry": p1, "location_id": 0},
        {"user_id": 0, "started_at": t2, "finished_at": t3, "geometry": p1, "location_id": 0},
        {"user_id": 0, "started_at": t3, "finished_at": t4, "geometry": p2, "location_id": 1},  # longest duration
        {"user_id": 1, "started_at": t4, "finished_at": t4 + one_hour, "geometry": p1, "location_id": 0},
    ]
    stps = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    stps.index.name = "id"
    assert stps.as_staypoints
    assert "location_id" in stps.columns
    return stps


@pytest.fixture
def default_kwargs():
    kw = {
        "agg_level": "dataset",
        "thresh_min_sp": 0,
        "thresh_min_loc": 0,
        "thresh_sp_at_loc": 0,
        "thresh_loc_time": 0,
        "thresh_loc_period": pd.Timedelta("0h")
    }
    return kw


class TestPre_Filter:
    """Tests for the function `pre_filter_locations()`."""

    def test_no_kw(self, example_staypoints, default_kwargs):
        """Test that nothing gets filtered if all parameters are set to zero."""
        assert(all(pre_filter_locations(example_staypoints, **default_kwargs)))

    def test_thresh_min_sp(self, example_staypoints, default_kwargs):
        """Test the minimum staypoint per user parameter."""
        default_kwargs["thresh_min_sp"] = 2
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["user_id"] == 0))

    def test_thresh_min_loc(self, example_staypoints, default_kwargs):
        """Test the minimum location per user parameter."""
        default_kwargs["thresh_min_loc"] = 2
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["user_id"] == 0))

    def test_thresh_sp_at_loc(self, example_staypoints, default_kwargs):
        """Test the minimum staypoint per location parameter."""
        default_kwargs["thresh_sp_at_loc"] = 2
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["location_id"] == 0))

    def test_tresh_loc_time(self, example_staypoints, default_kwargs):
        """Test the minimum duration per location parameter."""
        default_kwargs["thresh_loc_time"] = 14
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["location_id"] == 1))

    def test_loc_period(self, example_staypoints, default_kwargs):
        """Test the minimum period per location parameter."""
        default_kwargs["thresh_loc_period"] = pd.Timedelta("2d")
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["location_id"] == 0))

    def test_agg_level(self, example_staypoints, default_kwargs):
        """Test if different aggregation works."""
        default_kwargs["agg_level"] = "user"
        default_kwargs["thresh_loc_period"] = pd.Timedelta("1d")
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints[["user_id", "location_id"]] == [0, 1]).all(axis=1))
