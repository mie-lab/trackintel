import datetime

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import trackintel as ti
from trackintel.analysis.location_identification import pre_filter_locations, _freq_transform, freq_recipe, _freq_assign


@pytest.fixture
def example_staypoints():
    """Staypoints to test pre_filter."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)

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
    sps = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    sps.index.name = "id"
    assert sps.as_staypoints
    assert "location_id" in sps.columns
    return sps


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

    def test_agg_level_error(self, example_staypoints, default_kwargs):
        """Test if ValueError is raised for unknown agg_level."""
        default_kwargs["agg_level"] = "unknown"
        with pytest.raises(ValueError):
            pre_filter_locations(example_staypoints, **default_kwargs)


@pytest.fixture
def example_freq():
    """Example staypoints with 4 location for 2 users with [3, 2, 1, 1] staypoint per location."""
    p1 = Point(8.5067847, 47.1)  # geometry isn't used
    list_dict = [
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 0},
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 0},
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 0},
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 1},
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 1},
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 2},
        {"user_id": 0, "duration": 1, "g": p1, "location_id": 3},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 0},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 0},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 0},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 1},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 1},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 2},
        {"user_id": 1, "duration": 1, "g": p1, "location_id": 3},
    ]
    sps = gpd.GeoDataFrame(data=list_dict, geometry="g", crs="EPSG:4326")
    sps.index.name = "id"
    assert "location_id" in sps.columns
    return sps


class TestFreq_Recipe:
    """Test freq_recipe."""
    def test_default_labels(self, example_freq):
        """Test recipe with default labels"""
        freq = freq_recipe(example_freq)
        example_freq["activity_label"] = None
        example_freq.loc[example_freq["location_id"] == 0, "activity_label"] = "home"
        example_freq.loc[example_freq["location_id"] == 1, "activity_label"] = "work"
        assert freq["activity_label"].count() == example_freq["activity_label"].count()
        assert_geodataframe_equal(example_freq, freq)

    def test_custom_labels(self, example_freq):
        """Test recipe with custom label of a different length"""
        custom_label = "doing_nothing"
        freq = freq_recipe(example_freq, "doing_nothing")
        example_freq["activity_label"] = None
        example_freq.loc[example_freq["location_id"] == 0, "activity_label"] = custom_label
        assert freq["activity_label"].count() == example_freq["activity_label"].count()
        assert_geodataframe_equal(example_freq, freq)

    def test_creation_duration(self, example_freq):
        """Test if function can handle no "duration" column."""
        del example_freq["duration"]
        times = pd.date_range("1971-01-01", periods=len(example_freq)+1, freq="H")
        example_freq["started_at"] = times[:-1]
        example_freq["finished_at"] = times[1:]
        freq = freq_recipe(example_freq)
        example_freq["activity_label"] = None
        example_freq.loc[example_freq["location_id"] == 0, "activity_label"] = "home"
        example_freq.loc[example_freq["location_id"] == 1, "activity_label"] = "work"
        assert freq["activity_label"].count() == example_freq["activity_label"].count()
        assert_geodataframe_equal(example_freq, freq)


class Test_Freq_Transform:
    """Test help function _freq_transform."""
    def test_function(self):
        list_dict = [
            {"location_id": 0, "duration": 1},
            {"location_id": 0, "duration": 1},
            {"location_id": 1, "duration": 1},
        ]
        df = pd.DataFrame(list_dict)
        freq = _freq_transform(df, "work")
        sol = pd.Series(["work", "work", None])
        assert freq.equals(sol)


class Test_Freq_Assign:
    """Test help function _freq_assign."""
    def test_function(self):
        """Test function with simple input."""
        dur = pd.Series([9, 0, 8, 1, 7, 6, 5])
        labels = ("label1", "label2", "label3")
        freq_sol = np.array([labels[0], None, labels[1], None, labels[2], None, None])
        freq = _freq_assign(dur, *labels)
        assert all(freq == freq_sol)
