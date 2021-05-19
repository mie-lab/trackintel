import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

import trackintel as ti


class TestSmoothen_triplegs:
    def test_smoothen_triplegs(self):
        tpls_file = os.path.join("tests", "data", "triplegs_with_too_many_points_test.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col=None)
        tpls_smoothed = ti.preprocessing.triplegs.smoothen_triplegs(tpls, tolerance=0.0001)
        line1 = tpls.iloc[0].geom
        line1_smoothed = tpls_smoothed.iloc[0].geom
        line2 = tpls.iloc[1].geom
        line2_smoothed = tpls_smoothed.iloc[1].geom

        assert line1.length == line1_smoothed.length
        assert line2.length == line2_smoothed.length
        assert len(line1.coords) == 10
        assert len(line2.coords) == 7
        assert len(line1_smoothed.coords) == 4
        assert len(line2_smoothed.coords) == 3


class TestGenerate_trips:
    """Tests for generate_trips() method."""

    def test_duplicate_columns(self):
        """Test if running the function twice, the generated column does not yield exception in join statement"""
        # load pregenerated trips
        trips_loaded = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")

        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps_run_1, tpls_run_1, _ = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)
        stps_run_2, tpls_run_2, _ = ti.preprocessing.triplegs.generate_trips(stps_run_1, tpls_run_1, gap_threshold=15)

        assert set(tpls_run_1.columns) == set(tpls_run_2.columns)
        assert set(stps_run_1.columns) == set(stps_run_2.columns)

    def test_generate_trips(self):
        """Test if we can generate the example trips based on example data."""
        # load pregenerated trips
        trips_loaded = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")

        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)
        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)

    def test_generate_trips_missing_link(self):
        """Test nan is assigned for missing link between stps and trips, and tpls and trips."""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, _ = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)
        assert pd.isna(stps["trip_id"]).any()
        assert pd.isna(stps["prev_trip_id"]).any()
        assert pd.isna(stps["next_trip_id"]).any()

    def test_generate_trips_dtype_consistent(self):
        """Test the dtypes for the generated columns."""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)

        assert stps["user_id"].dtype == trips["user_id"].dtype
        assert trips.index.dtype == "int64"

        assert stps["trip_id"].dtype == "Int64"
        assert stps["prev_trip_id"].dtype == "Int64"
        assert stps["next_trip_id"].dtype == "Int64"
        assert tpls["trip_id"].dtype == "Int64"

    def test_generate_trips_index_start(self):
        """Test the generated index start from 0 for different methods."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        _, _, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)

        assert (trips.index == np.arange(len(trips))).any()

    def test_generate_trips_gap_detection(self):
        """
        Test different gap cases:
        - activity - tripleg - activity [gap] activity - tripleg - activity
        - activity - tripleg -  [gap]  - tripleg - activity
        - activity - tripleg -  [gap]  activity - tripleg - activity
        - activity - tripleg -  [gap]  activity - tripleg - activity
        - activity - tripleg - activity [gap] - tripleg - tripleg - tripleg - activity
        - tripleg - [gap] - tripleg - tripleg - [gap] - tripleg
        Returns
        -------

        """
        gap_threshold = 15

        # load data and add dummy geometry
        stps_in = pd.read_csv(
            os.path.join("tests", "data", "trips", "staypoints_gaps.csv"),
            sep=";",
            index_col="id",
            parse_dates=[0, 1],
            infer_datetime_format=True,
            dayfirst=True,
        )
        stps_in["geom"] = Point(1, 1)
        stps_in = gpd.GeoDataFrame(stps_in, geometry="geom")
        stps_in = ti.io.read_staypoints_gpd(stps_in, tz="utc")

        assert stps_in.as_staypoints

        tpls_in = pd.read_csv(
            os.path.join("tests", "data", "trips", "triplegs_gaps.csv"),
            sep=";",
            index_col="id",
            parse_dates=[0, 1],
            infer_datetime_format=True,
            dayfirst=True,
        )
        tpls_in["geom"] = LineString([[1, 1], [2, 2]])
        tpls_in = gpd.GeoDataFrame(tpls_in, geometry="geom")
        tpls_in = ti.io.read_triplegs_gpd(tpls_in, tz="utc")

        assert tpls_in.as_triplegs

        # load ground truth data
        trips_loaded = pd.read_csv(os.path.join("tests", "data", "trips", "trips_gaps.csv"), index_col="id")
        trips_loaded["started_at"] = pd.to_datetime(trips_loaded["started_at"], utc=True)
        trips_loaded["finished_at"] = pd.to_datetime(trips_loaded["finished_at"], utc=True)

        stps_tpls_loaded = pd.read_csv(os.path.join("tests", "data", "trips", "stps_tpls_gaps.csv"), index_col="id")
        stps_tpls_loaded["started_at"] = pd.to_datetime(stps_tpls_loaded["started_at"], utc=True)
        stps_tpls_loaded["started_at_next"] = pd.to_datetime(stps_tpls_loaded["started_at_next"], utc=True)
        stps_tpls_loaded["finished_at"] = pd.to_datetime(stps_tpls_loaded["finished_at"], utc=True)

        # generate trips and a joint staypoint/triplegs dataframe
        stps_proc, tpls_proc, trips = ti.preprocessing.triplegs.generate_trips(stps_in, tpls_in, gap_threshold=15)
        stps_tpls = _create_debug_stps_tpls_data(stps_proc, tpls_proc, gap_threshold=gap_threshold)

        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        pd.testing.assert_frame_equal(stps_tpls_loaded, stps_tpls, check_dtype=False)

    def test_generate_trips_id_management(self):
        """Test if we can generate the example trips based on example data."""
        stps_tpls_loaded = pd.read_csv(os.path.join("tests", "data", "geolife_long", "stps_tpls.csv"), index_col="id")
        stps_tpls_loaded["started_at"] = pd.to_datetime(stps_tpls_loaded["started_at"])
        stps_tpls_loaded["started_at_next"] = pd.to_datetime(stps_tpls_loaded["started_at_next"])
        stps_tpls_loaded["finished_at"] = pd.to_datetime(stps_tpls_loaded["finished_at"])

        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        gap_threshold = 15
        stps, tpls, _ = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=gap_threshold)
        stps_tpls = _create_debug_stps_tpls_data(stps, tpls, gap_threshold=gap_threshold)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        pd.testing.assert_frame_equal(stps_tpls_loaded, stps_tpls, check_dtype=False)


def _create_debug_stps_tpls_data(stps, tpls, gap_threshold):
    """Preprocess stps and tpls for "test_generate_trips_*."""
    # create table with relevant information from triplegs and staypoints.
    tpls["type"] = "tripleg"
    stps["type"] = "staypoint"
    stps_tpls = stps[
        ["started_at", "finished_at", "user_id", "type", "activity", "trip_id", "prev_trip_id", "next_trip_id"]
    ].append(tpls[["started_at", "finished_at", "user_id", "type", "trip_id"]])

    # transform nan to bool
    stps_tpls["activity"] = stps_tpls["activity"] == True
    stps_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    stps_tpls["started_at_next"] = stps_tpls["started_at"].shift(-1)
    stps_tpls["activity_next"] = stps_tpls["activity"].shift(-1)

    stps_tpls["gap"] = (stps_tpls["started_at_next"] - stps_tpls["finished_at"]).dt.seconds / 60 > gap_threshold

    return stps_tpls
