import os

import pytest
import pandas as pd
import trackintel as ti


@pytest.fixture
def testdata_stps_tpls_geolife_long():
    """Generate stps and tpls sequences of the original pfs for subsequent testing."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

    tpls["type"] = "tripleg"
    stps["type"] = "staypoint"
    stps_tpls = stps.append(tpls, ignore_index=True).sort_values(by="started_at")
    return stps_tpls


@pytest.fixture
def testdata_all_geolife_long():
    """Generate stps, tpls and trips of the original pfs for subsequent testing."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
    stps, tpls, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)

    return stps, tpls, trips


class TestTemporal_tracking_quality:
    """Tests for the temporal_tracking_quality() function."""

    def test_tracking_quality_all(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated total tracking quality is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long

        # calculate tracking quality for a sample user
        user_0 = stps_tpls.loc[stps_tpls["user_id"] == 0]
        extent = (user_0["finished_at"].max() - user_0["started_at"].min()).total_seconds()
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="all")

        assert quality_manual == quality.loc[quality["user_id"] == 0, "quality"].values[0]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_day(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated tracking quality per day is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long
        splitted_records = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

        # get the day relative to the start day
        start_date = splitted_records["started_at"].min().date()
        splitted_records["day"] = splitted_records["started_at"].apply(lambda x: (x.date() - start_date).days)
        # calculate tracking quality of the first day for the first user
        user_0 = splitted_records.loc[(splitted_records["user_id"] == 0) & (splitted_records["day"] == 0)]
        extent = 60 * 60 * 24
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent
        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="day")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["day"] == 0), "quality"].values[0]
        assert (quality["quality"] < 1).all()

    def test_tracking_quality_week(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated tracking quality per week is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

        # get the day relative to the start day
        start_date = splitted["started_at"].min().date()
        splitted["week"] = splitted["started_at"].apply(lambda x: (x.date() - start_date).days // 7)

        # calculate tracking quality of the first week for the first user
        user_0 = splitted.loc[splitted["user_id"] == 0]
        extent = 60 * 60 * 24 * 7
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="week")

        assert quality_manual == quality.loc[(quality["user_id"] == 0), "quality"].values[0]
        assert (quality["quality"] < 1).all()

    def test_tracking_quality_weekday(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated tracking quality per weekday is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

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
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="weekday")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["weekday"] == 3), "quality"].values[0]
        assert (quality["quality"] < 1).all()

    def test_tracking_quality_hour(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated tracking quality per hour is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="hour")

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
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="hour")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["hour"] == 2), "quality"].values[0]
        assert (quality["quality"] <= 1).all()

    def test_tracking_quality_error(self, testdata_stps_tpls_geolife_long):
        """Test if the an error is raised when passing unknown 'granularity' to temporal_tracking_quality()."""
        stps_tpls = testdata_stps_tpls_geolife_long

        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity=12345)
        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="random")

    def test_tracking_quality_wrong_datamodel(self, testdata_stps_tpls_geolife_long):
        """Test if the a keyerror is raised when passing incorrect datamodels."""
        # read positionfixes and feed to temporal_tracking_quality()
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        with pytest.raises(KeyError):
            ti.analysis.tracking_quality.temporal_tracking_quality(pfs)

        # generate locations and feed to temporal_tracking_quality()
        stps_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, tz="utc", index_col="id")
        _, locs = stps.as_staypoints.generate_locations(
            method="dbscan", epsilon=10, num_samples=0, distance_metric="haversine", agg_level="dataset"
        )
        with pytest.raises(KeyError):
            ti.analysis.tracking_quality.temporal_tracking_quality(locs)

    def test_tracking_quality_user_error(self, testdata_stps_tpls_geolife_long):
        """Test if the an error is raised when passing unknown 'granularity' to _get_tracking_quality_user()."""
        stps_tpls = testdata_stps_tpls_geolife_long
        user_0 = stps_tpls.loc[stps_tpls["user_id"] == 0]

        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality._get_tracking_quality_user(user_0, granularity=12345)
        with pytest.raises(AttributeError):
            ti.analysis.tracking_quality._get_tracking_quality_user(user_0, granularity="random")

    def test_staypoints_accessors(self, testdata_all_geolife_long):
        """Test tracking_quality calculation from staypoints accessor."""
        stps, _, _ = testdata_all_geolife_long

        # for staypoints
        stps_quality_accessor = stps.as_staypoints.temporal_tracking_quality()
        stps_quality_method = ti.analysis.tracking_quality.temporal_tracking_quality(stps)
        pd.testing.assert_frame_equal(stps_quality_accessor, stps_quality_method)

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


class TestSplit_overlaps:
    """Tests for the _split_overlaps() function."""

    def test_split_overlaps_days(self, testdata_stps_tpls_geolife_long):
        """Test if _split_overlaps() function can split records that span several days."""
        stps_tpls = testdata_stps_tpls_geolife_long

        # some of the records span several day
        multi_day_records = stps_tpls["finished_at"].dt.day - stps_tpls["started_at"].dt.day
        assert (multi_day_records > 0).any()

        # split the records according to day
        stps_tpls.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

        # no record spans several days after the split
        multi_day_records = (splitted["finished_at"] - pd.to_timedelta("1s")).dt.day - splitted["started_at"].dt.day
        assert (multi_day_records == 0).all()

    def test_split_overlaps_hours(self, testdata_stps_tpls_geolife_long):
        """Test if _split_overlaps() function can split records that span several hours."""
        stps_tpls = testdata_stps_tpls_geolife_long

        # some of the records span several hours
        hour_diff = stps_tpls["finished_at"].dt.hour - stps_tpls["started_at"].dt.hour
        assert (hour_diff > 0).any()

        # split the records according to hour
        stps_tpls.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="hour")

        # no record spans several hours after the split
        hour_diff = (splitted["finished_at"] - pd.to_timedelta("1s")).dt.hour - splitted["started_at"].dt.hour
        assert (hour_diff == 0).all()

    def test_split_overlaps_hours_case2(self, testdata_stps_tpls_geolife_long):
        """Test if _split_overlaps() function can split record that have the same hour but different days."""
        stps_tpls = testdata_stps_tpls_geolife_long

        # get the first two records
        head2 = stps_tpls.head(2).copy()
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

    def test_split_overlaps_duration(self, testdata_stps_tpls_geolife_long):
        """Test if the column 'duration' gets updated after using the _split_overlaps() function."""
        stps_tpls = testdata_stps_tpls_geolife_long
        # initiate the duration column
        stps_tpls["duration"] = stps_tpls["finished_at"] - stps_tpls["started_at"]
        stps_tpls.reset_index(inplace=True)

        # split the records according to day
        splitted_day = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

        # split the records according to hour
        splitted_hour = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="hour")

        # test "duration" is recalculated after the split
        assert splitted_day["duration"].sum() == stps_tpls["duration"].sum()
        assert splitted_hour["duration"].sum() == stps_tpls["duration"].sum()
