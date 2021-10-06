import os
import datetime

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import trackintel as ti


@pytest.fixture
def testdata_sp_tpls_geolife_long():
    """Generate sp and tpls sequences of the original pfs for subsequent testing."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

    tpls["type"] = "tripleg"
    sp["type"] = "staypoint"
    sp_tpls = sp.append(tpls, ignore_index=True).sort_values(by="started_at")
    return sp_tpls


@pytest.fixture
def testdata_all_geolife_long():
    """Generate sp, tpls and trips of the original pfs for subsequent testing."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
    sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

    return sp, tpls, trips


def get_test_sp(start_time, duration):
    """Generate test staypoints for tracking duration given start_time and duration."""
    p1 = Point(8.5067847, 47.4)

    # we generate three records, duration = 0, negative duration and positive duration
    list_dict = [
        {"user_id": 0, "started_at": start_time, "finished_at": start_time, "geom": p1},
        {"user_id": 0, "started_at": start_time, "finished_at": start_time - duration, "geom": p1},
        {"user_id": 0, "started_at": start_time, "finished_at": start_time + duration, "geom": p1},
    ]
    sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    sp.index.name = "id"

    return sp


class TestTemporal_tracking_quality:
    """Tests for the temporal_tracking_quality() function."""

    def test_tracking_quality_all(self, testdata_sp_tpls_geolife_long):
        """Test if the calculated total tracking quality is correct."""
        sp_tpls = testdata_sp_tpls_geolife_long

        # calculate tracking quality for a sample user
        user_0 = sp_tpls.loc[sp_tpls["user_id"] == 0]
        extent = (user_0["finished_at"].max() - user_0["started_at"].min()).total_seconds()
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="all")

        assert quality_manual == quality.loc[quality["user_id"] == 0, "quality"].values[0]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_day(self, testdata_sp_tpls_geolife_long):
        """Test if the calculated tracking quality per day is correct."""
        sp_tpls = testdata_sp_tpls_geolife_long
        splitted_records = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="day")

        # get the day relative to the start day
        start_date = splitted_records["started_at"].min().date()
        splitted_records["day"] = splitted_records["started_at"].apply(lambda x: (x.date() - start_date).days)
        # calculate tracking quality of the first day for the first user
        user_0 = splitted_records.loc[(splitted_records["user_id"] == 0) & (splitted_records["day"] == 0)]
        extent = 60 * 60 * 24
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent
        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="day")

        assert quality_manual == quality.iloc[0]["quality"]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_week(self, testdata_sp_tpls_geolife_long):
        """Test if the calculated tracking quality per week is correct."""
        sp_tpls = testdata_sp_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="day")

        # get the day relative to the start day
        start_date = splitted["started_at"].min().date()
        splitted["week"] = splitted["started_at"].apply(lambda x: (x.date() - start_date).days // 7)

        # calculate tracking quality of the first week for the first user
        user_0 = splitted.loc[splitted["user_id"] == 0]
        extent = 60 * 60 * 24 * 7
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="week")

        assert quality_manual == quality.loc[(quality["user_id"] == 0), "quality"].values[0]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_weekday(self, testdata_sp_tpls_geolife_long):
        """Test if the calculated tracking quality per weekday is correct."""
        sp_tpls = testdata_sp_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="day")

        # get the day relative to the start day
        start_date = splitted["started_at"].min().date()
        splitted["week"] = splitted["started_at"].apply(lambda x: (x.date() - start_date).days // 7)

        splitted["weekday"] = splitted["started_at"].dt.weekday

        # calculate tracking quality of the first week for the first user
        user_0 = splitted.loc[(splitted["user_id"] == 0) & (splitted["weekday"] == 3)]
        extent = (60 * 60 * 24) * (user_0["week"].max() - user_0["week"].min() + 1)
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="weekday")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["weekday"] == 3), "quality"].values[0]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_hour(self, testdata_sp_tpls_geolife_long):
        """Test if the calculated tracking quality per hour is correct."""
        sp_tpls = testdata_sp_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="hour")

        # get the day relative to the start day
        start_date = splitted["started_at"].min().date()
        splitted["day"] = splitted["started_at"].apply(lambda x: (x.date() - start_date).days)
        # get the hour of the record
        splitted["hour"] = splitted["started_at"].dt.hour

        # calculate tracking quality of an hour for the first user
        user_0 = splitted.loc[(splitted["user_id"] == 0) & (splitted["hour"] == 2)]
        extent = (60 * 60) * (user_0["day"].max() - user_0["day"].min() + 1)
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="hour")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["hour"] == 2), "quality"].values[0]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_error(self, testdata_sp_tpls_geolife_long):
        """Test if the an error is raised when passing unknown 'granularity' to temporal_tracking_quality()."""
        sp_tpls = testdata_sp_tpls_geolife_long

        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity=12345)
        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality.temporal_tracking_quality(sp_tpls, granularity="random")

    def test_tracking_quality_wrong_datamodel(self):
        """Test if the a keyerror is raised when passing incorrect datamodels."""
        # read positionfixes and feed to temporal_tracking_quality()
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        with pytest.raises(KeyError):
            ti.analysis.tracking_quality.temporal_tracking_quality(pfs)

        # generate locations and feed to temporal_tracking_quality()
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")
        _, locs = sp.as_staypoints.generate_locations(
            method="dbscan", epsilon=10, num_samples=0, distance_metric="haversine", agg_level="dataset"
        )
        with pytest.raises(KeyError):
            ti.analysis.tracking_quality.temporal_tracking_quality(locs)

    def test_tracking_quality_user_error(self, testdata_sp_tpls_geolife_long):
        """Test if the an error is raised when passing unknown 'granularity' to _get_tracking_quality_user()."""
        sp_tpls = testdata_sp_tpls_geolife_long
        user_0 = sp_tpls.loc[sp_tpls["user_id"] == 0]

        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality._get_tracking_quality_user(user_0, granularity=12345)
        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality._get_tracking_quality_user(user_0, granularity="random")

    def test_staypoints_accessors(self, testdata_all_geolife_long):
        """Test tracking_quality calculation from staypoints accessor."""
        sp, _, _ = testdata_all_geolife_long

        # for staypoints
        sp_quality_accessor = sp.as_staypoints.temporal_tracking_quality()
        sp_quality_method = ti.analysis.tracking_quality.temporal_tracking_quality(sp)
        pd.testing.assert_frame_equal(sp_quality_accessor, sp_quality_method)

    def test_triplegs_accessors(self, testdata_all_geolife_long):
        """Test tracking_quality calculation from triplegs accessor."""
        _, tpls, _ = testdata_all_geolife_long

        # for triplegs
        tpls_quality_accessor = tpls.as_triplegs.temporal_tracking_quality()
        tpls_quality_method = ti.analysis.tracking_quality.temporal_tracking_quality(tpls)
        pd.testing.assert_frame_equal(tpls_quality_accessor, tpls_quality_method)

    def test_trips_accessors(self, testdata_all_geolife_long):
        """Test tracking_quality calculation from trips accessor."""
        _, _, trips = testdata_all_geolife_long

        # for trips
        trips_quality_accessor = trips.as_trips.temporal_tracking_quality()
        trips_quality_method = ti.analysis.tracking_quality.temporal_tracking_quality(trips)
        pd.testing.assert_frame_equal(trips_quality_accessor, trips_quality_method)

    def test_non_positive_duration_warning(self):
        """Test the function can handle non positive duration records without running into infinite loop."""
        t = pd.Timestamp("1971-01-01 00:00:00", tz="utc")  # duration 0 at midnight
        negative_one_hour = datetime.timedelta(hours=1)  # negative duration
        sp = get_test_sp(t, negative_one_hour)

        warn_string = "The input dataframe does not contain any record with positive duration. Please check."
        with pytest.warns(UserWarning, match=warn_string):
            ti.analysis.tracking_quality.temporal_tracking_quality(sp.iloc[:-1])

    def test_absolute_extent(self):
        """Test the absolute date is correctly generated for both granularity day and week."""
        # we test two examples: one at midnight the other at midday
        t_ls = [pd.Timestamp("1971-01-01 00:00:00", tz="utc"), pd.Timestamp("1971-01-01 12:00:00", tz="utc")]
        ten_days = pd.Timedelta(days=10)
        # the midnight record loses one day after the split
        last_start_date_after_split_ls = [t_ls[0] + ten_days - pd.Timedelta(days=1), t_ls[0] + ten_days]
        for t, last_start_date_after_split in zip(t_ls, last_start_date_after_split_ls):
            sp = get_test_sp(t, ten_days)

            quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp, granularity="day")
            # get the "date" of the last record and compare to the last "date" in data
            assert quality.values[-1][-2].day == last_start_date_after_split.day

            quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp, granularity="week")
            # get the "week" of the last record and compare to the last "week" in data
            assert quality.values[-1][-2].week == last_start_date_after_split.week

    def test_non_positive_duration_filtered(self):
        """Test the non positive duration records are filtered and do not affect the result."""
        t = pd.Timestamp("1971-01-01 00:00:00", tz="utc")  # duration 0 at midnight
        one_hour = datetime.timedelta(hours=1)
        sp = get_test_sp(t, one_hour)

        granularity_ls = ["all", "day", "week", "weekday", "hour"]
        correct_quality_ls = [1, 1 / 24, 1 / 24 / 7, 1 / 24, 1]

        for granularity, correct_quality in zip(granularity_ls, correct_quality_ls):
            quality = ti.analysis.tracking_quality.temporal_tracking_quality(sp, granularity=granularity)
            # get the "quality" of the last record and compare to the correct_quality
            assert quality.values[-1][-1] == correct_quality


class TestSplit_overlaps:
    """Tests for the _split_overlaps() function."""

    def test_split_overlaps_days(self, testdata_sp_tpls_geolife_long):
        """Test if _split_overlaps() function can split records that span several days."""
        sp_tpls = testdata_sp_tpls_geolife_long

        # some of the records span several day
        multi_day_records = sp_tpls["finished_at"].dt.day - sp_tpls["started_at"].dt.day
        assert (multi_day_records > 0).any()

        # split the records according to day
        sp_tpls.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="day")

        # no record spans several days after the split
        multi_day_records = (splitted["finished_at"] - pd.to_timedelta("1s")).dt.day - splitted["started_at"].dt.day
        assert (multi_day_records == 0).all()

    def test_split_overlaps_hours(self, testdata_sp_tpls_geolife_long):
        """Test if _split_overlaps() function can split records that span several hours."""
        sp_tpls = testdata_sp_tpls_geolife_long

        # some of the records span several hours
        hour_diff = sp_tpls["finished_at"].dt.hour - sp_tpls["started_at"].dt.hour
        assert (hour_diff > 0).any()

        # split the records according to hour
        sp_tpls.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="hour")

        # no record spans several hours after the split
        hour_diff = (splitted["finished_at"] - pd.to_timedelta("1s")).dt.hour - splitted["started_at"].dt.hour
        assert (hour_diff == 0).all()

    def test_split_overlaps_hours_case2(self, testdata_sp_tpls_geolife_long):
        """Test if _split_overlaps() function can split record that have the same hour but different days."""
        sp_tpls = testdata_sp_tpls_geolife_long

        # get the first two records
        head2 = sp_tpls.head(2).copy()
        # construct the finished_at exactly one day after started_at
        head2["finished_at"] = head2.apply(lambda x: x["started_at"].replace(day=x["started_at"].day + 1), axis=1)

        # the records have the same hour
        hour_diff = (head2["finished_at"] - pd.to_timedelta("1s")).dt.hour - head2["started_at"].dt.hour
        assert (hour_diff == 0).all()
        # but have different days
        day_diff = (head2["finished_at"] - pd.to_timedelta("1s")).dt.day - head2["started_at"].dt.day
        assert (day_diff > 0).all()

        # split the records according to hour
        head2.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(head2, granularity="hour")

        # no record has different days after the split
        day_diff = (splitted["finished_at"] - pd.to_timedelta("1s")).dt.day - splitted["started_at"].dt.day
        assert (day_diff == 0).all()

    def test_split_overlaps_duration(self, testdata_sp_tpls_geolife_long):
        """Test if the column 'duration' gets updated after using the _split_overlaps() function."""
        sp_tpls = testdata_sp_tpls_geolife_long
        # initiate the duration column
        sp_tpls["duration"] = sp_tpls["finished_at"] - sp_tpls["started_at"]
        sp_tpls.reset_index(inplace=True)

        # split the records according to day
        splitted_day = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="day")

        # split the records according to hour
        splitted_hour = ti.analysis.tracking_quality._split_overlaps(sp_tpls, granularity="hour")

        # test "duration" is recalculated after the split
        assert splitted_day["duration"].sum() == sp_tpls["duration"].sum()
        assert splitted_hour["duration"].sum() == sp_tpls["duration"].sum()

    def test_max_iter_warning(self):
        """Test if a warning is raised when maximum iteration is reached."""
        p1 = Point(8.5067847, 47.4)
        # construct time that is far apart - exceeding the default max_iter
        t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
        t2 = pd.Timestamp("1981-01-01 00:00:00", tz="utc")

        list_dict = [
            {"user_id": 0, "started_at": t1, "finished_at": t2, "geom": p1},
        ]
        sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
        sp.index.name = "id"

        with pytest.warns(UserWarning):
            ti.analysis.tracking_quality._split_overlaps(sp, granularity="day")
