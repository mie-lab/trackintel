import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import trackintel as ti
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point
from trackintel.analysis.location_identification import (
    _freq_assign,
    _freq_transform,
    freq_method,
    location_identifier,
    pre_filter_locations,
    osna_method,
    _osna_label_funct,
)


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
    spts = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    spts.index.name = "id"
    assert spts.as_staypoints
    assert "location_id" in spts.columns
    return spts


@pytest.fixture
def default_kwargs():
    kw = {
        "agg_level": "dataset",
        "thresh_sp": 0,
        "thresh_loc": 0,
        "thresh_sp_at_loc": 0,
        "thresh_loc_time": "0h",
        "thresh_loc_period": "0h",
    }
    return kw


class TestPre_Filter:
    """Tests for the function `pre_filter_locations()`."""

    def test_no_kw(self, example_staypoints, default_kwargs):
        """Test that nothing gets filtered if all parameters are set to zero."""
        assert all(pre_filter_locations(example_staypoints, **default_kwargs))

    def test_thresh_sp(self, example_staypoints, default_kwargs):
        """Test the minimum staypoint per user parameter."""
        default_kwargs["thresh_sp"] = 2
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["user_id"] == 0))

    def test_thresh_loc(self, example_staypoints, default_kwargs):
        """Test the minimum location per user parameter."""
        default_kwargs["thresh_loc"] = 2
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["user_id"] == 0))

    def test_thresh_sp_at_loc(self, example_staypoints, default_kwargs):
        """Test the minimum staypoint per location parameter."""
        default_kwargs["thresh_sp_at_loc"] = 2
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["location_id"] == 0))

    def test_tresh_loc_time(self, example_staypoints, default_kwargs):
        """Test the minimum duration per location parameter."""
        default_kwargs["thresh_loc_time"] = "14h"
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert all(f == (example_staypoints["location_id"] == 1))

    def test_loc_period(self, example_staypoints, default_kwargs):
        """Test the minimum period per location parameter."""
        default_kwargs["thresh_loc_period"] = "2d"
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
    list_dict = [
        {"user_id": 0, "location_id": 0},
        {"user_id": 0, "location_id": 0},
        {"user_id": 0, "location_id": 0},
        {"user_id": 0, "location_id": 1},
        {"user_id": 0, "location_id": 1},
        {"user_id": 0, "location_id": 2},
        {"user_id": 0, "location_id": 3},
        {"user_id": 1, "location_id": 0},
        {"user_id": 1, "location_id": 0},
        {"user_id": 1, "location_id": 0},
        {"user_id": 1, "location_id": 1},
        {"user_id": 1, "location_id": 1},
        {"user_id": 1, "location_id": 2},
        {"user_id": 1, "location_id": 3},
    ]
    p1 = Point(8.5067847, 47.1)  # geometry isn't used
    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 01:00:00", tz="utc")
    for d in list_dict:
        d["started_at"] = t1
        d["finished_at"] = t2
        d["geom"] = p1

    spts = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    spts.index.name = "id"
    assert "location_id" in spts.columns
    assert spts.as_staypoints
    return spts


class TestFreq_method:
    """Test freq_method."""

    def test_default_labels(self, example_freq):
        """Test method with default labels"""
        freq = freq_method(example_freq)
        example_freq["activity_label"] = None
        example_freq.loc[example_freq["location_id"] == 0, "activity_label"] = "home"
        example_freq.loc[example_freq["location_id"] == 1, "activity_label"] = "work"
        assert freq["activity_label"].count() == example_freq["activity_label"].count()
        assert_geodataframe_equal(example_freq, freq)

    def test_custom_labels(self, example_freq):
        """Test method with custom label of a different length"""
        custom_label = "doing_nothing"
        freq = freq_method(example_freq, "doing_nothing")
        example_freq["activity_label"] = None
        example_freq.loc[example_freq["location_id"] == 0, "activity_label"] = custom_label
        assert freq["activity_label"].count() == example_freq["activity_label"].count()
        assert_geodataframe_equal(example_freq, freq)

    def test_duration(self, example_freq):
        """Test if function can handle only "duration" column and no columns "started_at", "finished_at"."""
        example_freq["duration"] = example_freq["finished_at"] - example_freq["started_at"]
        del example_freq["finished_at"]
        del example_freq["started_at"]
        freq = freq_method(example_freq)
        example_freq["activity_label"] = None
        example_freq.loc[example_freq["location_id"] == 0, "activity_label"] = "home"
        example_freq.loc[example_freq["location_id"] == 1, "activity_label"] = "work"
        assert freq["activity_label"].count() == example_freq["activity_label"].count()
        assert_geodataframe_equal(example_freq, freq)


class Test_Freq_Transform:
    """Test help function _freq_transform."""

    def test_function(self):
        """Test if groupby assign works."""
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


class TestLocation_Identifier:
    """Test function `location_identifier`"""

    def test_unkown_method(self, example_staypoints):
        """Test if ValueError is risen if method is unknown"""
        with pytest.raises(ValueError):
            location_identifier(example_staypoints, method="UNKNOWN", pre_filter=False)

    def test_no_location_column(self, example_staypoints):
        """Test if key error is risen if no column `location_id`."""
        with pytest.raises(KeyError):
            del example_staypoints["location_id"]
            location_identifier(example_staypoints)

    def test_pre_filter(self, example_freq, default_kwargs):
        """Test if function calls pre_filter correctly."""
        default_kwargs["agg_level"] = "user"
        default_kwargs["thresh_sp_at_loc"] = 2
        li = location_identifier(example_freq, method="FREQ", pre_filter=True, **default_kwargs)
        f = pre_filter_locations(example_freq, **default_kwargs)
        example_freq.loc[f, "activity_label"] = freq_method(example_freq[f])["activity_label"]
        assert_geodataframe_equal(li, example_freq)

    def test_freq_method(self, example_freq):
        """Test if function calls freq method correctly."""
        li = location_identifier(example_freq, method="FREQ", pre_filter=False)
        fr = freq_method(example_freq)
        assert_geodataframe_equal(li, fr)

    def test_osna_method(self, example_osna):
        """Test if function calls osna method correctly."""
        li = location_identifier(example_osna, method="OSNA", pre_filter=False)
        osna = osna_method(example_osna)
        assert_geodataframe_equal(li, osna)


@pytest.fixture
def example_osna():
    """Example staypoints with 2 location for 2 users with 4 different times."""
    weekday = "2021-05-19 "
    weekend = "2021-05-22 "
    t_rest = pd.Timestamp(weekday + "07:00:00", tz="utc")
    t_work = pd.Timestamp(weekday + "18:00:00", tz="utc")
    t_weekend = pd.Timestamp(weekend + "00:00:00", tz="utc")
    t_leisure = pd.Timestamp(weekday + "01:00:00", tz="utc")
    h = pd.Timedelta("1h")
    list_dict = [
        {"user_id": 0, "location_id": 0, "started_at": t_rest},
        {"user_id": 0, "location_id": 0, "started_at": t_leisure},
        {"user_id": 0, "location_id": 0, "started_at": t_work},  # (0, 0) 1 rest + 1 leisure + 1 work
        {"user_id": 0, "location_id": 1, "started_at": t_rest},
        {"user_id": 0, "location_id": 1, "started_at": t_work},
        {"user_id": 0, "location_id": 1, "started_at": t_work},  # (0, 1) 1 rest + 2 work + 1 weekend
        {"user_id": 0, "location_id": 1, "started_at": t_weekend},
        {"user_id": 0, "location_id": 2, "started_at": t_leisure},  # (0, 2) 1 leisure
        {"user_id": 1, "location_id": 0, "started_at": t_rest},
        {"user_id": 1, "location_id": 0, "started_at": t_leisure},
        {"user_id": 1, "location_id": 0, "started_at": t_work},  # (1, 0) 1 rest + 1 leisure + 1 work
        {"user_id": 1, "location_id": 1, "started_at": t_leisure},
        {"user_id": 1, "location_id": 1, "started_at": t_leisure},
        {"user_id": 1, "location_id": 1, "started_at": t_work},
        {"user_id": 1, "location_id": 1, "started_at": t_work},  # (1, 1) 2 leisure + 2 work
        {"user_id": 1, "location_id": 2, "started_at": t_leisure},  # (1, 2) 1 leisure
    ]  # I am not happy with this fixture as we test for a lot of things at the same time.
    p = Point(8.5, 47.1)  # geometry isn't used
    for d in list_dict:
        d["finished_at"] = d["started_at"] + h
        d["geom"] = p
    spts = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    spts.index.name = "id"
    assert "location_id" in spts.columns
    assert spts.as_staypoints
    return spts


class TestOsna_Method:
    """Test `osna_method`"""

    def test_default(self, example_osna):
        """Test with no changes to test data."""
        osna = osna_method(example_osna)
        example_osna.loc[example_osna["location_id"] == 0, "activity_label"] = "home"
        example_osna.loc[example_osna["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(example_osna, osna)

    def test_overlap(self, example_osna):
        """Test if overlap of home and work location next work location is taken."""
        # for that lets add 2 work times to location 0
        t = pd.Timestamp("2021-05-19 12:00:00", tz="utc")
        h = pd.to_timedelta("1h")
        p = Point(0, 0)
        list_dict = [
            {"user_id": 0, "location_id": 0, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 0, "location_id": 0, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 1, "location_id": 0, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 1, "location_id": 0, "started_at": t, "finished_at": t + h, "geom": p},
        ]
        spts = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        spts.index.name = "id"
        spts = example_osna.append(spts)
        example_osna.loc[example_osna["location_id"] == 0, "activity_label"] = "home"
        example_osna.loc[example_osna["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(osna_method(spts).iloc[:-4], example_osna)

    def test_weekends_ignored(self, example_osna):
        """Test that weekends aren't included in analysis."""
        # just add a bunch of weekends to user 0 at location 0
        t = pd.Timestamp("2021-05-22 12:00:00", tz="utc")  # saturday
        h = pd.to_timedelta("1h")
        p = Point(0, 0)
        list_dict = [
            {"user_id": 0, "location_id": 2, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 0, "location_id": 2, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 0, "location_id": 2, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 0, "location_id": 2, "started_at": t, "finished_at": t + h, "geom": p},
        ]
        spts = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        spts.index.name = "id"
        spts = example_osna.append(spts)
        example_osna.loc[example_osna["location_id"] == 0, "activity_label"] = "home"
        example_osna.loc[example_osna["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(osna_method(spts).iloc[:-4], example_osna)


class Test_Osna_Label_Funct:
    """Test function `_osna_label_funct`"""
    def test_weekend(self):
        """Test if weekend only depends on day and not time."""
        t1 = pd.Timestamp("2021-05-22 01:00:00")
        t2 = pd.Timestamp("2021-05-22 07:00:00")
        t3 = pd.Timestamp("2021-05-22 08:00:00")
        t4 = pd.Timestamp("2021-05-22 20:00:00")
        assert _osna_label_funct(t1) == "weekend"
        assert _osna_label_funct(t2) == "weekend"
        assert _osna_label_funct(t3) == "weekend"
        assert _osna_label_funct(t4) == "weekend"

    def test_weekday(self):
        """Test the different labels on a weekday."""
        t1 = pd.Timestamp("2021-05-20 01:00:00")
        t2 = pd.Timestamp("2021-05-20 02:00:00")
        t3 = pd.Timestamp("2021-05-20 08:00:00")
        t4 = pd.Timestamp("2021-05-20 19:00:00")
        t5 = pd.Timestamp("2021-05-20 18:59:59")
        assert _osna_label_funct(t1) == "leisure"
        assert _osna_label_funct(t2) == "rest"
        assert _osna_label_funct(t3) == "work"
        assert _osna_label_funct(t4) == "leisure"
        assert _osna_label_funct(t5) == "work"
