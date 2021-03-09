
import os
import pytest

import trackintel as ti


@pytest.fixture
def testdata_stps_tpls_geolife_long():
    """Generate stps and tpls sequences of the original pfs for subsequent testing."""
    pfs = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5 * 60)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

    tpls["type"] = "tripleg"
    stps["type"] = "staypoint"
    stps_tpls = stps.append(tpls, ignore_index=True).sort_values(by="started_at")
    return stps_tpls


class TestTemporal_tracking_quality:
    """Tests for the temporal_tracking_quality() function and its subfunctions."""

    def test_temporal_all(self, testdata_stps_tpls_geolife_long):
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

    def test_temporal_day(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated tracking quality per day is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long
        splitted_records = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

        # get the day relative to the start day
        start_date = splitted_records["started_at"].min().date()
        splitted_records["day"] = splitted_records["started_at"].apply(lambda x: (x.date() - start_date).days)
        # calculate tracking quality of the first day for the first user
        user_0 = splitted_records.loc[(splitted_records["user_id"] == 0) & (splitted_records["day"] == 0)]
        extent = 60 * 60 * 24 - 1
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent
        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="day")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["day"] == 0), "quality"].values[0]

    def test_temporal_hour(self, testdata_stps_tpls_geolife_long):
        """Test if the calculated tracking quality per hour is correct."""
        stps_tpls = testdata_stps_tpls_geolife_long

        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="hour")

        # get the day relative to the start day
        start_date = splitted["started_at"].min().date()
        splitted["day"] = splitted["started_at"].apply(lambda x: (x.date() - start_date).days)
        # get the hour of the record
        splitted["hour"] = splitted["started_at"].dt.hour

        # calculate tracking quality of a hour for the first user
        user_0 = splitted.loc[(splitted["user_id"] == 0) & (splitted["hour"] == 2)]
        extent = (60 * 60 - 1) * len(user_0["day"].unique())
        tracked = (user_0["finished_at"] - user_0["started_at"]).dt.total_seconds().sum()
        quality_manual = tracked / extent

        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="hour")

        assert quality_manual == quality.loc[(quality["user_id"] == 0) & (quality["hour"] == 2), "quality"].values[0]

    def test_temporal_split_overlaps_days(self, testdata_stps_tpls_geolife_long):
        """Test if _split_overlaps() function can split records that span several days."""
        stps_tpls = testdata_stps_tpls_geolife_long

        # some of the records spans several day
        multi_day_records = stps_tpls["finished_at"].dt.day - stps_tpls["started_at"].dt.day
        assert (multi_day_records > 0).any()

        # split the records according to day
        stps_tpls.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")

        # no record spans several day after the split
        multi_day_records = splitted["finished_at"].dt.day - splitted["started_at"].dt.day
        assert (multi_day_records == 0).all()

    def test_temporal_split_overlaps_hours(self, testdata_stps_tpls_geolife_long):
        """Test if _split_overlaps() function can split records that span several hours."""
        stps_tpls = testdata_stps_tpls_geolife_long

        # some of the records spans several hours
        multi_hour_records = stps_tpls["finished_at"].dt.hour - stps_tpls["started_at"].dt.hour
        assert (multi_hour_records > 0).any()

        # split the records according to hour
        stps_tpls.reset_index(inplace=True)
        splitted = ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="hour")

        # no record spans several hour after the split
        multi_hour_records = splitted["finished_at"].dt.hour - splitted["started_at"].dt.hour
        assert (multi_hour_records == 0).all()