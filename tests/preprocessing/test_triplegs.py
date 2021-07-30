import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from shapely import geometry
from shapely.geometry import LineString, Point

import trackintel as ti
from trackintel.preprocessing.triplegs import generate_trips


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
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps_run_1, tpls_run_1, _ = generate_trips(stps, tpls, gap_threshold=15)
        with pytest.warns(UserWarning):
            stps_run_2, tpls_run_2, _ = generate_trips(stps_run_1, tpls_run_1, gap_threshold=15)

        assert set(tpls_run_1.columns) == set(tpls_run_2.columns)
        assert set(stps_run_1.columns) == set(stps_run_2.columns)

    def test_generate_trips(self):
        """Test if we can generate the example trips based on example data."""
        # load pregenerated trips
        trips_loaded = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")

        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
        )
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, trips = generate_trips(stps, tpls, gap_threshold=15)
        trips = trips[
            ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id", "geom"]
        ]
        # test if generated trips are equal
        assert_geodataframe_equal(trips_loaded, trips)

    def test_trip_wo_geom(self):
        """Test if the add_geometry parameter shows correct behavior"""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
        )
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips dataframe with geometry
        _, _, trips = generate_trips(stps, tpls, gap_threshold=15)
        trips = pd.DataFrame(trips.drop(["geom"], axis=1))

        # generate trips without geometry
        _, _, trips_wo_geom = generate_trips(stps, tpls, gap_threshold=15, add_geometry=False)

        # test if generated trips are equal
        assert_frame_equal(trips_wo_geom, trips)

    def test_trip_coordinates(self):
        """Test if coordinates of start and destination are correct"""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
        )
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, trips = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)

        # Check start and destination points of all rows
        for i, row in trips.iterrows():
            start_point_trips = row["geom"][0]  # get origin Point in generated trips
            if not pd.isna(row["origin_staypoint_id"]):
                # compare to the Point in the staypoints
                correct_start_point = stps.loc[row["origin_staypoint_id"], "geom"]
            else:
                # check if it is the first point of the tripleg
                # get all triplegs on this trip
                tpls_on_trip = tpls[tpls["trip_id"] == row.name]
                # correct point is the first point on the tripleg
                correct_start_point, _ = tpls_on_trip.iloc[0]["geom"].boundary

            assert correct_start_point == start_point_trips

            dest_point_trips = row["geom"][1]  # get destination Point in generated trips
            if not pd.isna(row["destination_staypoint_id"]):
                correct_dest_point = stps.loc[row["destination_staypoint_id"], "geom"]
                # compare to the Point in the staypoints
            else:
                # check if it is the last point of the tripleg
                # get all triplegs on this trip
                tpls_on_trip = tpls[tpls["trip_id"] == row.name]
                # correct point is the first point on the tripleg
                _, correct_dest_point = tpls_on_trip.iloc[-1]["geom"].boundary

            assert correct_dest_point == dest_point_trips

    def test_accessor(self):
        """Test if the accessor leads to the same results as the explicit function."""
        # load pregenerated trips
        trips_loaded = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")

        # prepare data
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips using the explicit function import
        stps_expl, tpls_expl, trips_expl = ti.preprocessing.triplegs.generate_trips(stps, tpls, gap_threshold=15)

        # generate trips using the accessor
        stps_acc, tpls_acc, trips_acc = tpls.as_triplegs.generate_trips(stps, gap_threshold=15)

        # test if generated trips are equal
        assert_geodataframe_equal(trips_expl, trips_acc)
        assert_geodataframe_equal(stps_expl, stps_acc)
        assert_geodataframe_equal(tpls_expl, tpls_acc)

    def test_accessor_arguments(self):
        """Test if the accessor is robust to different ways to receive arguments"""
        # load pregenerated trips
        trips_loaded = ti.read_trips_csv(os.path.join("tests", "data", "geolife_long", "trips.csv"), index_col="id")

        # prepare data
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, spts = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        spts = spts.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(spts)

        # accessor with only arguments (not allowed)
        with pytest.raises(AssertionError):
            _, _, _ = tpls.as_triplegs.generate_trips(spts, 15)

        # accessor with only keywords
        spts_1, tpls_1, trips_1 = tpls.as_triplegs.generate_trips(spts=spts, gap_threshold=15)

        # accessor with mixed arguments/keywords
        spts_2, tpls_2, trips_2 = tpls.as_triplegs.generate_trips(spts, gap_threshold=15)

        # test if generated trips are equal (1,2)
        assert_geodataframe_equal(spts_1, spts_2)
        assert_geodataframe_equal(tpls_1, tpls_2)
        assert_geodataframe_equal(trips_1, trips_2)

    def test_generate_trips_missing_link(self):
        """Test nan is assigned for missing link between stps and trips, and tpls and trips."""
        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, _ = generate_trips(stps, tpls, gap_threshold=15)
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
        stps, tpls, trips = generate_trips(stps, tpls, gap_threshold=15)

        assert stps["user_id"].dtype == trips["user_id"].dtype
        assert trips.index.dtype == "int64"

        assert stps["trip_id"].dtype == "Int64"
        assert stps["prev_trip_id"].dtype == "Int64"
        assert stps["next_trip_id"].dtype == "Int64"
        assert tpls["trip_id"].dtype == "Int64"

    def test_compare_to_old_trip_function(self):
        """Test if we can generate the example trips based on example data."""
        # load pregenerated trips

        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        stps, tpls, trips = generate_trips(stps, tpls, gap_threshold=15)
        stps_, tpls_, trips_ = _generate_trips_old(stps, tpls, gap_threshold=15)
        trips.drop(columns=["geom"], inplace=True)

        # test if generated trips are equal
        # ignore column order and index dtype
        assert_frame_equal(trips, trips_, check_like=True, check_index_type=False)
        assert_frame_equal(stps, stps_, check_like=True, check_index_type=False)
        assert_frame_equal(tpls, tpls_, check_like=True, check_index_type=False)

    def test_generate_trips_index_start(self):
        """Test the generated index start from 0 for different methods."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        _, _, trips = generate_trips(stps, tpls, gap_threshold=15)

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

        # load ground truth data
        trips_loaded = ti.read_trips_csv(
            os.path.join("tests", "data", "trips", "trips_gaps.csv"), index_col="id", tz="utc"
        )

        stps_tpls_loaded = pd.read_csv(os.path.join("tests", "data", "trips", "stps_tpls_gaps.csv"), index_col="id")
        stps_tpls_loaded["started_at"] = pd.to_datetime(stps_tpls_loaded["started_at"], utc=True)
        stps_tpls_loaded["started_at_next"] = pd.to_datetime(stps_tpls_loaded["started_at_next"], utc=True)
        stps_tpls_loaded["finished_at"] = pd.to_datetime(stps_tpls_loaded["finished_at"], utc=True)

        # generate trips and a joint staypoint/triplegs dataframe
        stps_proc, tpls_proc, trips = generate_trips(stps_in, tpls_in, gap_threshold=gap_threshold, add_geometry=False)
        stps_tpls = _create_debug_stps_tpls_data(stps_proc, tpls_proc, gap_threshold=gap_threshold)

        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        assert_frame_equal(stps_tpls_loaded, stps_tpls, check_dtype=False)

    def test_generate_trips_id_management(self):
        """Test if we can generate the example trips based on example data."""
        stps_tpls_loaded = pd.read_csv(os.path.join("tests", "data", "geolife_long", "stps_tpls.csv"), index_col="id")
        stps_tpls_loaded["started_at"] = pd.to_datetime(stps_tpls_loaded["started_at"])
        stps_tpls_loaded["started_at_next"] = pd.to_datetime(stps_tpls_loaded["started_at_next"])
        stps_tpls_loaded["finished_at"] = pd.to_datetime(stps_tpls_loaded["finished_at"])

        # create trips from geolife (based on positionfixes)
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
        )
        stps = stps.as_staypoints.create_activity_flag(time_threshold=15)
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        # generate trips and a joint staypoint/triplegs dataframe
        gap_threshold = 15
        stps, tpls, _ = generate_trips(stps, tpls, gap_threshold=gap_threshold)
        stps_tpls = _create_debug_stps_tpls_data(stps, tpls, gap_threshold=gap_threshold)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        assert_frame_equal(stps_tpls_loaded, stps_tpls, check_dtype=False)

    def test_only_staypoints_in_trip(self):
        """Test that trips with only staypoints (non-activities) are deleted."""
        start = pd.Timestamp("2021-07-11 8:00:00")
        h = pd.to_timedelta("1h")
        spts_tpls = [
            {"activity": True, "type": "staypoint"},
            {"activity": False, "type": "staypoint"},
            {"activity": True, "type": "staypoint"},
            {"activity": False, "type": "tripleg"},
            {"activity": False, "type": "staypoint"},
            {"activity": True, "type": "staypoint"},
        ]
        for n, d in enumerate(spts_tpls):
            d["user_id"] = 0
            d["started_at"] = start + n * h
            d["finished_at"] = d["started_at"] + h
        spts_tpls = pd.DataFrame(spts_tpls)
        spts = spts_tpls[spts_tpls["type"] == "staypoint"]
        tpls = spts_tpls[spts_tpls["type"] == "tripleg"]
        spts_, tpls_, trips = generate_trips(spts, tpls, add_geometry=False)
        trip_id_truth = pd.Series([None, None, None, 0, None], dtype="Int64")
        trip_id_truth.index = spts_.index  # don't check index
        assert_series_equal(spts_["trip_id"], trip_id_truth, check_names=False)
        assert (tpls_["trip_id"] == 0).all()
        assert len(trips) == 1


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


def _generate_trips_old(stps_input, tpls_input, gap_threshold=15, print_progress=False):
    """Generate trips based on staypoints and triplegs.

    Parameters
    ----------
    stps_input : GeoDataFrame (as trackintel staypoints)
        The staypoints have to follow the standard definition for staypoints DataFrames.

    tpls_input : GeoDataFrame (as trackintel triplegs)
        The triplegs have to follow the standard definition for triplegs DataFrames.

    gap_threshold : float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than
        `gap_threshold` minutes, then a new trip begins after the gap.

    Returns
    -------
    staypoints: GeoDataFrame (as trackintel staypoints)
        the original staypoints with new columns ``[`trip_id`, `prev_trip_id`, `next_trip_id`]``.

    triplegs: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`trip_id`]``.

    trips: GeoDataFrame (as trackintel trips)
        The generated trips.

    Notes
    -----
    Trips are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities.
    The function returns altered versions of the input staypoints and triplegs. Staypoints receive the fields
    [`trip_id` `prev_trip_id` and `next_trip_id`], triplegs receive the field [`trip_id`].
    The following assumptions are implemented

        - All movement before the first and after the last activity is omitted
        - If we do not record a person for more than `gap_threshold` minutes,
          we assume that the person performed an activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination
        - There are no trips without a (recored) tripleg

    Examples
    --------
    >>> staypoints, triplegs, trips = generate_trips(staypoints, triplegs)
    """
    assert "activity" in stps_input.columns, "staypoints need the column 'activities' to be able to generate trips"

    # we copy the input because we need to add a temporary column
    tpls = tpls_input.copy()
    stps = stps_input.copy()

    # if the triplegs already have a column "trip_id", we drop it
    if "trip_id" in tpls:
        tpls.drop(columns="trip_id", inplace=True)

    # if the staypoints already have any of the columns "trip_id", "prev_trip_id", "next_trip_id", we drop them
    for col in ["trip_id", "prev_trip_id", "next_trip_id"]:
        if col in stps:
            stps.drop(columns=col, inplace=True)

    tpls["type"] = "tripleg"
    stps["type"] = "staypoint"

    # create table with relevant information from triplegs and staypoints.
    stps_tpls = pd.concat(
        [
            stps[["started_at", "finished_at", "user_id", "type", "activity"]],
            tpls[["started_at", "finished_at", "user_id", "type"]],
        ]
    )

    # create ID field from index
    stps_tpls["id"] = stps_tpls.index

    # transform nan to bool
    stps_tpls["activity"] = stps_tpls["activity"] == True

    stps_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    stps_tpls["started_at_next"] = stps_tpls["started_at"].shift(-1)
    stps_tpls["activity_next"] = stps_tpls["activity"].shift(-1)

    if print_progress:
        tqdm.pandas(desc="User trip generation")
        trips = (
            stps_tpls.groupby(["user_id"], group_keys=False, as_index=False)
            .progress_apply(_generate_trips_user, gap_threshold=gap_threshold)
            .reset_index(drop=True)
        )
    else:
        trips = (
            stps_tpls.groupby(["user_id"], group_keys=False, as_index=False)
            .apply(_generate_trips_user, gap_threshold=gap_threshold)
            .reset_index(drop=True)
        )

    # index management
    trips["id"] = np.arange(len(trips))
    trips.set_index("id", inplace=True)

    # assign trip_id to tpls
    trip2tpl_map = trips[["tpls"]].to_dict()["tpls"]
    ls = []
    for key, values in trip2tpl_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[tpls.index.name, "trip_id"]).set_index(tpls.index.name)
    tpls = tpls.join(temp, how="left")

    # assign trip_id to stps, for non-activity stps
    trip2spt_map = trips[["stps"]].to_dict()["stps"]
    ls = []
    for key, values in trip2spt_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[stps.index.name, "trip_id"]).set_index(stps.index.name)
    stps = stps.join(temp, how="left")

    # assign prev_trip_id to stps
    temp = trips[["destination_staypoint_id"]].copy()
    temp.rename(columns={"destination_staypoint_id": stps.index.name}, inplace=True)
    temp.index.name = "prev_trip_id"
    temp = temp.reset_index().set_index(stps.index.name)
    stps = stps.join(temp, how="left")

    # assign next_trip_id to stps
    temp = trips[["origin_staypoint_id"]].copy()
    temp.rename(columns={"origin_staypoint_id": stps.index.name}, inplace=True)
    temp.index.name = "next_trip_id"
    temp = temp.reset_index().set_index(stps.index.name)
    stps = stps.join(temp, how="left")

    # final cleaning
    tpls.drop(columns=["type"], inplace=True)
    stps.drop(columns=["type"], inplace=True)
    trips.drop(columns=["tpls", "stps"], inplace=True)

    ## dtype consistency
    # trips id (generated by this function) should be int64
    trips.index = trips.index.astype("int64")
    # trip id of stps and tpls can only be in Int64 (missing values)
    stps["trip_id"] = stps["trip_id"].astype("Int64")
    stps["prev_trip_id"] = stps["prev_trip_id"].astype("Int64")
    stps["next_trip_id"] = stps["next_trip_id"].astype("Int64")
    tpls["trip_id"] = tpls["trip_id"].astype("Int64")

    # user_id of trips should be the same as tpls
    trips["user_id"] = trips["user_id"].astype(tpls["user_id"].dtype)

    return stps, tpls, trips


def _generate_trips_user(df, gap_threshold):
    # function called after groupby: should only contain records of one user
    user_id = df["user_id"].unique()
    assert len(user_id) == 1
    user_id = user_id[0]

    unknown_activity = {"user_id": user_id, "activity": True, "id": np.nan}
    origin_activity = unknown_activity
    temp_trip_stack = []
    in_trip = False
    trip_ls = []

    for _, row in df.iterrows():

        # check if we can start a new trip
        # (we make sure that we start the trip with the most recent activity)
        if in_trip is False:
            # If there are several activities in a row, we skip until the last one
            if row["activity"] and row["activity_next"]:
                continue

            # if this is the last activity before the trip starts, reset the origin
            elif row["activity"]:
                origin_activity = row
                in_trip = True
                continue

            # if for non-activities we simply start the trip
            else:
                in_trip = True

        if in_trip is True:
            # during trip generation/recording

            # check if trip ends regularly
            is_gap = row["started_at_next"] - row["finished_at"] > datetime.timedelta(minutes=gap_threshold)
            if row["activity"] is True:

                # if there are no triplegs in the trip, set the current activity as origin and start over
                if not _check_trip_stack_has_tripleg(temp_trip_stack):
                    origin_activity = row
                    temp_trip_stack = list()
                    in_trip = True

                else:
                    # record trip
                    destination_activity = row
                    trip_ls.append(_create_trip_from_stack(temp_trip_stack, origin_activity, destination_activity))

                    # set values for next trip
                    if is_gap:
                        # if there is a gap after this trip the origin of the next trip is unknown
                        origin_activity = unknown_activity
                        destination_activity = None
                        temp_trip_stack = list()
                        in_trip = False

                    else:
                        # if there is no gap after this trip the origin of the next trip is the destination of the
                        # current trip
                        origin_activity = destination_activity
                        destination_activity = None
                        temp_trip_stack = list()
                        in_trip = False

            # check if gap during the trip
            elif is_gap:
                # in case of a gap, the destination of the current trip and the origin of the next trip
                # are unknown.

                # add current item to trip
                temp_trip_stack.append(row)

                # if the trip has no recored triplegs, we do not generate the current trip.
                if not _check_trip_stack_has_tripleg(temp_trip_stack):
                    origin_activity = unknown_activity
                    in_trip = True
                    temp_trip_stack = list()

                else:
                    # add tripleg to trip, generate trip, start new trip with unknown origin
                    destination_activity = unknown_activity

                    trip_ls.append(_create_trip_from_stack(temp_trip_stack, origin_activity, destination_activity))
                    origin_activity = unknown_activity
                    destination_activity = None
                    temp_trip_stack = list()
                    in_trip = True

            else:
                temp_trip_stack.append(row)

    # if user ends generate last trip with unknown destination
    if (len(temp_trip_stack) > 0) and (_check_trip_stack_has_tripleg(temp_trip_stack)):
        destination_activity = unknown_activity
        trip_ls.append(
            _create_trip_from_stack(
                temp_trip_stack,
                origin_activity,
                destination_activity,
            )
        )

    # print(trip_ls)
    trips = pd.DataFrame(trip_ls)
    return trips


def _check_trip_stack_has_tripleg(temp_trip_stack):
    """
    Check if a trip has at least 1 tripleg.

    Parameters
    ----------
    temp_trip_stack : list
        list of dictionary like elements (either pandas series or python dictionary).
        Contains all elements that will be aggregated into a trip

    Returns
    -------
    has_tripleg: Bool
    """
    has_tripleg = False
    for row in temp_trip_stack:
        if row["type"] == "tripleg":
            has_tripleg = True
            break

    return has_tripleg


def _create_trip_from_stack(temp_trip_stack, origin_activity, destination_activity):
    """
    Aggregate information of trip elements in a structured dictionary.

    Parameters
    ----------
    temp_trip_stack : list
        list of dictionary like elements (either pandas series or python dictionary).
        Contains all elements that will be aggregated into a trip

    origin_activity : dictionary like
        Either dictionary or pandas series

    destination_activity : dictionary like
        Either dictionary or pandas series

    Returns
    -------
    trip_dict_entry: dictionary

    """
    # this function return and empty dict if no tripleg is in the stack
    first_trip_element = temp_trip_stack[0]
    last_trip_element = temp_trip_stack[-1]

    # all data has to be from the same user
    assert origin_activity["user_id"] == last_trip_element["user_id"]

    # double check if trip requirements are fulfilled
    assert origin_activity["activity"] == True
    assert destination_activity["activity"] == True
    assert first_trip_element["activity"] == False

    trip_dict_entry = {
        "user_id": origin_activity["user_id"],
        "started_at": first_trip_element["started_at"],
        "finished_at": last_trip_element["finished_at"],
        "origin_staypoint_id": origin_activity["id"],
        "destination_staypoint_id": destination_activity["id"],
        "tpls": [tripleg["id"] for tripleg in temp_trip_stack if tripleg["type"] == "tripleg"],
        "stps": [tripleg["id"] for tripleg in temp_trip_stack if tripleg["type"] == "staypoint"],
    }

    return trip_dict_entry
