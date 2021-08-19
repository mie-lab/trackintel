import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import trackintel as ti
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal, assert_index_equal
from shapely.geometry import Point
from trackintel.analysis.location_identification import (
    _freq_assign,
    _freq_transform,
    _osna_label_timeframes,
    freq_method,
    location_identifier,
    osna_method,
    pre_filter_locations,
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
    sp = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    sp.index.name = "id"
    assert sp.as_staypoints
    assert "location_id" in sp.columns
    return sp


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

    def test_non_continous_index(self, example_staypoints, default_kwargs):
        """Test if function works with non-continous index."""
        # issue-#247
        example_staypoints.index = [0, 999, 1, 15]
        f = pre_filter_locations(example_staypoints, **default_kwargs)
        assert_index_equal(f.index, example_staypoints.index)


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

    sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    sp.index.name = "id"
    assert "location_id" in sp.columns
    assert sp.as_staypoints
    return sp


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

    def test_more_labels_than_entries(self):
        dur = pd.Series([9, 0])
        labels = ("label1", "label2", "label3")
        freq_sol = np.array([labels[0], labels[1]])
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
    """Generate example staypoints with 3 location for 1 user within 3 different timeframes."""
    weekday = "2021-05-19 "
    t_rest = pd.Timestamp(weekday + "07:00:00", tz="utc")
    t_work = pd.Timestamp(weekday + "18:00:00", tz="utc")
    t_leisure = pd.Timestamp(weekday + "01:00:00", tz="utc")
    h = pd.Timedelta("1h")

    list_dict = [
        {"user_id": 0, "location_id": 0, "started_at": t_rest},
        {"user_id": 0, "location_id": 0, "started_at": t_leisure},
        {"user_id": 0, "location_id": 0, "started_at": t_work},
        {"user_id": 0, "location_id": 1, "started_at": t_rest},
        {"user_id": 0, "location_id": 1, "started_at": t_work},
        {"user_id": 0, "location_id": 1, "started_at": t_work},
        {"user_id": 0, "location_id": 2, "started_at": t_leisure},
    ]

    p = Point(8.0, 47.0)  # geometry isn't used
    for d in list_dict:
        d["finished_at"] = d["started_at"] + h
        d["geom"] = p

    sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    sp.index.name = "id"
    assert "location_id" in sp.columns
    assert sp.as_staypoints
    return sp


class TestOsna_Method:
    """Test for the osna_method() function."""

    def test_default(self, example_osna):
        """Test with no changes to test data."""
        osna = osna_method(example_osna)
        example_osna.loc[example_osna["location_id"] == 0, "activity_label"] = "home"
        example_osna.loc[example_osna["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(example_osna, osna)

    def test_overlap(self, example_osna):
        """Test if overlap of home and work location the 2nd location is taken as work location."""
        # add 2 work times to location 0,
        # location 0 would be the most stayed location for both home and work
        t = pd.Timestamp("2021-05-19 12:00:00", tz="utc")
        h = pd.to_timedelta("1h")
        p = Point(0.0, 0.0)
        list_dict = [
            {"user_id": 0, "location_id": 0, "started_at": t, "finished_at": t + h, "geom": p},
            {"user_id": 0, "location_id": 0, "started_at": t, "finished_at": t + h, "geom": p},
        ]
        sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        sp.index.name = "id"
        sp = example_osna.append(sp)

        result = osna_method(sp).iloc[:-2]
        example_osna.loc[example_osna["location_id"] == 0, "activity_label"] = "home"
        example_osna.loc[example_osna["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(result, example_osna)

    def test_only_weekends(self, example_osna):
        """Test if an "empty df" warning rises if only weekends are included."""
        weekend = "2021-05-22"  # a saturday

        def _insert_weekend(dt, day=weekend):
            """Take datetime and return new datetime with same time but new day."""
            time = dt.time().strftime("%H:%M:%S")
            new_dt = " ".join((day, time))
            return pd.Timestamp(new_dt, tz=dt.tz)

        # replace all days with weekends --> no label in data.
        example_osna["started_at"] = example_osna["started_at"].apply(_insert_weekend)
        example_osna["finished_at"] = example_osna["finished_at"].apply(_insert_weekend)

        # check if warning is raised if all points are excluded (weekend)
        with pytest.warns(UserWarning):
            result = osna_method(example_osna)

        # activity_label column is all pd.NA
        example_osna["activity_label"] = pd.NA
        assert_geodataframe_equal(result, example_osna)

    def test_two_users(self, example_osna):
        """Test if two users are handled correctly."""
        two_user = example_osna.append(example_osna)
        two_user.iloc[len(example_osna) :, 0] = 1  # second user gets id 1
        result = osna_method(two_user)
        two_user.loc[two_user["location_id"] == 0, "activity_label"] = "home"
        two_user.loc[two_user["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(result, two_user)

    def test_leisure_weighting(self):
        """Test if leisure has the weight given in the paper."""
        weight_rest = 0.739
        weight_leis = 0.358
        ratio = weight_rest / weight_leis
        ratio += 0.01  # tip the scale in favour of leisure
        weekday = "2021-05-19 "
        t_rest = pd.Timestamp(weekday + "07:00:00", tz="utc")
        t_work = pd.Timestamp(weekday + "18:00:00", tz="utc")
        t_leis = pd.Timestamp(weekday + "01:00:00", tz="utc")
        h = pd.Timedelta("1h")

        list_dict = [
            {"user_id": 0, "location_id": 0, "started_at": t_rest, "finished_at": t_rest + h},
            {"user_id": 0, "location_id": 1, "started_at": t_leis, "finished_at": t_leis + ratio * h},
            {"user_id": 0, "location_id": 2, "started_at": t_work, "finished_at": t_work + h},
        ]
        p = Point(8.0, 47.0)  # geometry isn't used
        for d in list_dict:
            d["geom"] = p
        sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        sp.index.name = "id"
        result = osna_method(sp)
        sp.loc[sp["location_id"] == 1, "activity_label"] = "home"
        sp.loc[sp["location_id"] == 2, "activity_label"] = "work"
        assert_geodataframe_equal(sp, result)

    def test_only_one_work_location(self):
        """Test if only one work location of a user can be handled."""
        t_work = pd.Timestamp("2021-07-14 18:00:00", tz="utc")
        h = pd.Timedelta("1h")
        p = Point(0.0, 0.0)  # not used
        list_dict = [{"user_id": 0, "location_id": 0, "started_at": t_work, "finished_at": t_work + h, "g": p}]
        sp = gpd.GeoDataFrame(data=list_dict, geometry="g")
        sp.index.name = "id"
        result = osna_method(sp)
        sp["activity_label"] = "work"
        assert_geodataframe_equal(result, sp)

    def test_only_one_rest_location(self):
        """Test if only one rest location of a user can be handled."""
        t_rest = pd.Timestamp("2021-07-14 07:00:00", tz="utc")
        h = pd.Timedelta("1h")
        p = Point(0.0, 0.0)  # not used
        list_dict = [{"user_id": 0, "location_id": 0, "started_at": t_rest, "finished_at": t_rest + h, "g": p}]
        sp = gpd.GeoDataFrame(data=list_dict, geometry="g")
        sp.index.name = "id"
        result = osna_method(sp)
        sp["activity_label"] = "home"
        assert_geodataframe_equal(result, sp)

    def test_only_one_leisure_location(self):
        """Test if only one leisure location of a user can be handled."""
        t_leis = pd.Timestamp("2021-07-14 01:00:00", tz="utc")
        h = pd.Timedelta("1h")
        p = Point(0.0, 0.0)  # not used
        list_dict = [{"user_id": 0, "location_id": 0, "started_at": t_leis, "finished_at": t_leis + h, "g": p}]
        sp = gpd.GeoDataFrame(data=list_dict, geometry="g")
        sp.index.name = "id"
        result = osna_method(sp)
        sp["activity_label"] = "home"
        assert_geodataframe_equal(result, sp)

    def test_prior_activity_label(self, example_osna):
        """Test that prior activity_label column does not corrupt output."""
        example_osna["activity_label"] = np.arange(len(example_osna))
        result = osna_method(example_osna)
        del example_osna["activity_label"]
        example_osna.loc[example_osna["location_id"] == 0, "activity_label"] = "home"
        example_osna.loc[example_osna["location_id"] == 1, "activity_label"] = "work"
        assert_geodataframe_equal(example_osna, result)

    def test_multiple_users_with_only_one_location(self):
        """Test that function can handle multiple users with only one location."""
        t_leis = pd.Timestamp("2021-07-14 01:00:00", tz="utc")
        t_work = pd.Timestamp("2021-07-14 18:00:00", tz="utc")
        h = pd.Timedelta("1h")
        list_dict = [
            {"user_id": 0, "location_id": 0, "started_at": t_leis, "finished_at": t_leis + h},
            {"user_id": 0, "location_id": 1, "started_at": t_work, "finished_at": t_work + h},
            {"user_id": 1, "location_id": 0, "started_at": t_leis, "finished_at": t_leis + h},
            {"user_id": 2, "location_id": 0, "started_at": t_work, "finished_at": t_work + h},
        ]
        sp = pd.DataFrame(list_dict)
        sp.index.name = "id"
        result = osna_method(sp)
        sp["activity_label"] = ["home", "work", "home", "work"]
        assert_frame_equal(sp, result)


class Test_osna_label_timeframes:
    """Test for the _osna_label_timeframes() function."""

    def test_weekend(self):
        """Test if weekend only depends on day and not time."""
        t1 = pd.Timestamp("2021-05-22 01:00:00")
        t2 = pd.Timestamp("2021-05-22 07:00:00")
        t3 = pd.Timestamp("2021-05-22 08:00:00")
        t4 = pd.Timestamp("2021-05-22 20:00:00")
        assert _osna_label_timeframes(t1) == "weekend"
        assert _osna_label_timeframes(t2) == "weekend"
        assert _osna_label_timeframes(t3) == "weekend"
        assert _osna_label_timeframes(t4) == "weekend"

    def test_weekday(self):
        """Test the different labels on a weekday."""
        t1 = pd.Timestamp("2021-05-20 01:00:00")
        t2 = pd.Timestamp("2021-05-20 02:00:00")
        t3 = pd.Timestamp("2021-05-20 08:00:00")
        t4 = pd.Timestamp("2021-05-20 19:00:00")
        t5 = pd.Timestamp("2021-05-20 18:59:59")
        assert _osna_label_timeframes(t1) == "leisure"
        assert _osna_label_timeframes(t2) == "rest"
        assert _osna_label_timeframes(t3) == "work"
        assert _osna_label_timeframes(t4) == "leisure"
        assert _osna_label_timeframes(t5) == "work"
