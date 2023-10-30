import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal

from shapely.geometry import LineString, Point
from tqdm import tqdm

import trackintel as ti
from trackintel.preprocessing.triplegs import generate_trips


@pytest.fixture
def example_triplegs():
    """Generate input data for trip generation from geolife positionfixes"""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
    )
    sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    return sp, tpls


@pytest.fixture
def example_triplegs_higher_gap_threshold():
    """Generate input data for trip generation, but with a higher gap threshold in stp generation"""
    # create trips from geolife (based on positionfixes) - with gap_threshold 1e6
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))

    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        method="sliding", dist_threshold=25, time_threshold=5, gap_threshold=1e6
    )
    sp = sp.as_staypoints.create_activity_flag(time_threshold=15)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    return sp, tpls


class TestGenerate_trips:
    """Tests for generate_trips() method."""

    def test_duplicate_columns(self, example_triplegs):
        """Test if running the function twice, the generated column does not yield exception in join statement"""
        sp, tpls = example_triplegs

        # generate trips and a joint staypoint/triplegs dataframe
        sp_run_1, tpls_run_1, _ = generate_trips(sp, tpls, gap_threshold=15)
        with pytest.warns(UserWarning):
            sp_run_2, tpls_run_2, _ = generate_trips(sp_run_1, tpls_run_1, gap_threshold=15)

        assert set(tpls_run_1.columns) == set(tpls_run_2.columns)
        assert set(sp_run_1.columns) == set(sp_run_2.columns)

    def test_generate_trips(self, example_triplegs_higher_gap_threshold):
        """Test if we can generate the example trips based on example data."""
        # load pregenerated trips
        path = os.path.join("tests", "data", "geolife_long", "trips.csv")
        trips_loaded = ti.read_trips_csv(path, index_col="id", geom_col="geom", crs="EPSG:4326")

        # create trips from geolife (based on positionfixes) - with gap_threshold 1e6
        sp, tpls = example_triplegs_higher_gap_threshold

        # generate trips and a joint staypoint/triplegs dataframe
        sp, tpls, trips = generate_trips(sp, tpls, gap_threshold=15)
        trips = trips[
            ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id", "geom"]
        ]
        # test if generated trips are equal
        assert_geodataframe_equal(trips_loaded, trips, check_less_precise=True)

    def test_trip_wo_geom(self, example_triplegs_higher_gap_threshold):
        """Test if the add_geometry parameter shows correct behavior"""
        sp, tpls = example_triplegs_higher_gap_threshold

        # generate trips dataframe with geometry
        _, _, trips = generate_trips(sp, tpls, gap_threshold=15)
        trips = pd.DataFrame(trips.drop(["geom"], axis=1))

        # generate trips without geometry
        _, _, trips_wo_geom = generate_trips(sp, tpls, gap_threshold=15, add_geometry=False)

        # test if generated trips are equal
        assert_frame_equal(trips_wo_geom, trips)

    def test_trip_coordinates(self, example_triplegs_higher_gap_threshold):
        """Test if coordinates of start and destination are correct"""
        sp, tpls = example_triplegs_higher_gap_threshold

        # generate trips and a joint staypoint/triplegs dataframe
        sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

        # Check start and destination points of all rows
        for i, row in trips.iterrows():
            start_point_trips = row["geom"].geoms[0]  # get origin Point in generated trips
            if not pd.isna(row["origin_staypoint_id"]):
                # compare to the Point in the staypoints
                correct_start_point = sp.loc[row["origin_staypoint_id"], "geom"]
            else:
                # check if it is the first point of the tripleg
                # get all triplegs on this trip
                tpls_on_trip = tpls[tpls["trip_id"] == row.name]
                # correct point is the first point on the tripleg
                correct_start_point, _ = tpls_on_trip.iloc[0]["geom"].boundary.geoms

            assert correct_start_point == start_point_trips

            dest_point_trips = row["geom"].geoms[1]  # get destination Point in generated trips
            if not pd.isna(row["destination_staypoint_id"]):
                correct_dest_point = sp.loc[row["destination_staypoint_id"], "geom"]
                # compare to the Point in the staypoints
            else:
                # check if it is the last point of the tripleg
                # get all triplegs on this trip
                tpls_on_trip = tpls[tpls["trip_id"] == row.name]
                # correct point is the first point on the tripleg
                _, correct_dest_point = tpls_on_trip.iloc[-1]["geom"].boundary.geoms

            assert correct_dest_point == dest_point_trips

    def test_accessor_triplegs(self, example_triplegs):
        """Test if the accessor leads to the same results as the explicit function."""
        sp, tpls = example_triplegs

        # generate trips using the explicit function import
        sp_expl, tpls_expl, trips_expl = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

        # generate trips using the accessor
        sp_acc, tpls_acc, trips_acc = tpls.as_triplegs.generate_trips(sp, gap_threshold=15)

        # test if generated trips are equal
        assert_geodataframe_equal(trips_expl, trips_acc)
        assert_geodataframe_equal(sp_expl, sp_acc)
        assert_geodataframe_equal(tpls_acc, tpls_expl)

    def test_accessor_staypoints(self, example_triplegs):
        """Test if the accessor leads to the same results as the explicit function."""
        sp, tpls = example_triplegs

        # generate trips using the explicit function import
        sp_expl, tpls_expl, trips_expl = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

        # generate trips using the accessor
        sp_acc, tpls_acc, trips_acc = sp.as_staypoints.generate_trips(tpls, gap_threshold=15)

        # test if generated trips are equal
        assert_geodataframe_equal(trips_expl, trips_acc)
        assert_geodataframe_equal(sp_expl, sp_acc)
        assert_geodataframe_equal(tpls_acc, tpls_expl)

    def test_generate_trips_missing_link(self, example_triplegs):
        """Test nan is assigned for missing link between sp and trips, and tpls and trips."""
        sp, tpls = example_triplegs

        # generate trips and a joint staypoint/triplegs dataframe
        sp, tpls, _ = generate_trips(sp, tpls, gap_threshold=15)
        assert pd.isna(sp["trip_id"]).any()
        assert pd.isna(sp["prev_trip_id"]).any()
        assert pd.isna(sp["next_trip_id"]).any()

    def test_generate_trips_dtype_consistent(self, example_triplegs):
        """Test the dtypes for the generated columns."""
        sp, tpls = example_triplegs

        # generate trips and a joint staypoint/triplegs dataframe
        sp, tpls, trips = generate_trips(sp, tpls, gap_threshold=15)

        assert sp["user_id"].dtype == trips["user_id"].dtype
        assert trips.index.dtype == "int64"

        assert sp["trip_id"].dtype == "Int64"
        assert sp["prev_trip_id"].dtype == "Int64"
        assert sp["next_trip_id"].dtype == "Int64"
        assert tpls["trip_id"].dtype == "Int64"

    def test_compare_to_old_trip_function(self, example_triplegs):
        """Test if we can generate the example trips based on example data."""
        sp, tpls = example_triplegs

        # generate trips and a joint staypoint/triplegs dataframe
        sp, tpls, trips = generate_trips(sp, tpls, gap_threshold=15)
        sp_, tpls_, trips_ = _generate_trips_old(sp, tpls, gap_threshold=15)
        trips.drop(columns=["geom"], inplace=True)

        # test if generated trips are equal
        # ignore column order and index dtype
        assert_frame_equal(trips, trips_, check_like=True, check_index_type=False)
        assert_frame_equal(sp, sp_, check_like=True, check_index_type=False)
        assert_frame_equal(tpls, tpls_, check_like=True, check_index_type=False)

    def test_generate_trips_index_start(self, example_triplegs):
        """Test the generated index start from 0 for different methods."""
        sp, tpls = example_triplegs

        # generate trips and a joint staypoint/triplegs dataframe
        _, _, trips = generate_trips(sp, tpls, gap_threshold=15)

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
        sp_in = pd.read_csv(
            os.path.join("tests", "data", "trips", "staypoints_gaps.csv"),
            sep=";",
            index_col="id",
            parse_dates=[0, 1],
            dayfirst=True,
        )
        sp_in["geom"] = Point(1, 1)
        sp_in = gpd.GeoDataFrame(sp_in, geometry="geom")
        sp_in = ti.io.read_staypoints_gpd(sp_in, tz="utc")

        tpls_in = pd.read_csv(
            os.path.join("tests", "data", "trips", "triplegs_gaps.csv"),
            sep=";",
            index_col="id",
            parse_dates=[0, 1],
            dayfirst=True,
        )
        tpls_in["geom"] = LineString([[1, 1], [2, 2]])
        tpls_in = gpd.GeoDataFrame(tpls_in, geometry="geom")
        tpls_in = ti.io.read_triplegs_gpd(tpls_in, tz="utc")

        # load ground truth data
        trips_loaded = ti.read_trips_csv(
            os.path.join("tests", "data", "trips", "trips_gaps.csv"), index_col="id", tz="utc"
        )

        sp_tpls_loaded = pd.read_csv(os.path.join("tests", "data", "trips", "sp_tpls_gaps.csv"), index_col="id")
        sp_tpls_loaded["started_at"] = pd.to_datetime(sp_tpls_loaded["started_at"], utc=True)
        sp_tpls_loaded["started_at_next"] = pd.to_datetime(sp_tpls_loaded["started_at_next"], utc=True)
        sp_tpls_loaded["finished_at"] = pd.to_datetime(sp_tpls_loaded["finished_at"], utc=True)

        # generate trips and a joint staypoint/triplegs dataframe
        sp_proc, tpls_proc, trips = generate_trips(sp_in, tpls_in, gap_threshold=gap_threshold, add_geometry=False)
        sp_tpls = _create_debug_sp_tpls_data(sp_proc, tpls_proc, gap_threshold=gap_threshold)

        # test if generated trips are equal
        pd.testing.assert_frame_equal(trips_loaded, trips)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        assert_frame_equal(sp_tpls_loaded, sp_tpls, check_dtype=False)

    def test_generate_trips_id_management(self, example_triplegs_higher_gap_threshold):
        """Test if we can generate the example trips based on example data."""
        sp_tpls_loaded = pd.read_csv(os.path.join("tests", "data", "geolife_long", "sp_tpls.csv"), index_col="id")
        sp_tpls_loaded["started_at"] = pd.to_datetime(sp_tpls_loaded["started_at"])
        sp_tpls_loaded["started_at_next"] = pd.to_datetime(sp_tpls_loaded["started_at_next"])
        sp_tpls_loaded["finished_at"] = pd.to_datetime(sp_tpls_loaded["finished_at"])

        sp, tpls = example_triplegs_higher_gap_threshold

        # generate trips and a joint staypoint/triplegs dataframe
        gap_threshold = 15
        sp, tpls, _ = generate_trips(sp, tpls, gap_threshold=gap_threshold)
        sp_tpls = _create_debug_sp_tpls_data(sp, tpls, gap_threshold=gap_threshold)

        # test if generated staypoints/triplegs are equal (especially important for trip ids)
        assert_frame_equal(sp_tpls_loaded, sp_tpls, check_dtype=False)

    def test_only_staypoints_in_trip(self):
        """Test that trips with only staypoints (non-activities) are deleted."""
        start = pd.Timestamp("2021-07-11 8:00:00", tz="utc")
        h = pd.to_timedelta("1h")
        sp_tpls = [
            {"is_activity": True, "type": "staypoint"},
            {"is_activity": False, "type": "staypoint"},
            {"is_activity": True, "type": "staypoint"},
            {"is_activity": False, "type": "tripleg"},
            {"is_activity": False, "type": "staypoint"},
            {"is_activity": True, "type": "staypoint"},
        ]
        for n, d in enumerate(sp_tpls):
            d["user_id"] = 0
            d["started_at"] = start + n * h
            d["finished_at"] = d["started_at"] + h
        sp_tpls = pd.DataFrame(sp_tpls)
        sp = sp_tpls[sp_tpls["type"] == "staypoint"].copy()
        tpls = sp_tpls[sp_tpls["type"] == "tripleg"].copy()
        sp["geom"] = Point(0, 0)
        tpls["geom"] = LineString([[1, 1], [2, 2]])
        sp = gpd.GeoDataFrame(sp, geometry="geom")
        tpls = gpd.GeoDataFrame(tpls, geometry="geom")
        sp_, tpls_, trips = generate_trips(sp, tpls, add_geometry=False)
        trip_id_truth = pd.Series([None, None, None, 0, None], dtype="Int64")
        trip_id_truth.index = sp_.index  # don't check index
        assert_series_equal(sp_["trip_id"], trip_id_truth, check_names=False)
        assert (tpls_["trip_id"] == 0).all()
        assert len(trips) == 1

    def test_sp_tpls_index(self):
        """Test if staypoint and tripleg index are identical before and after generating trips."""
        start = pd.Timestamp("2021-07-11 8:00:00", tz="utc")
        h = pd.to_timedelta("1h")
        sp_tpls = [
            {"is_activity": True, "type": "staypoint"},
            {"is_activity": False, "type": "tripleg"},
            {"is_activity": False, "type": "staypoint"},
            {"is_activity": False, "type": "tripleg"},
            {"is_activity": True, "type": "staypoint"},
        ]
        for n, d in enumerate(sp_tpls):
            d["user_id"] = 0
            d["started_at"] = start + n * h
            d["finished_at"] = d["started_at"] + h

        sp_tpls = pd.DataFrame(sp_tpls)
        sp = sp_tpls[sp_tpls["type"] == "staypoint"].copy()
        tpls = sp_tpls[sp_tpls["type"] != "staypoint"].copy()
        sp["geom"] = Point(0, 0)
        tpls["geom"] = LineString([(0, 0), (1, 1)])
        sp = gpd.GeoDataFrame(sp, geometry="geom")
        tpls = gpd.GeoDataFrame(tpls, geometry="geom")
        tpls.index.name = "something_long_and_obscure"
        sp.index.name = "even_obscurer"
        sp_, tpls_, _ = generate_trips(sp, tpls, add_geometry=False)
        assert_index_equal(tpls.index, tpls_.index)
        assert_index_equal(sp.index, sp_.index)

    def test_loop_linestring_case(self, example_triplegs):
        """Test corner case where a tripleg starts and ends at the same point"""
        # input data: preprocessed sp and tpls
        sp, tpls = example_triplegs

        # add a tripleg with same start as end, by modifying the first tripleg
        tpls.loc[0, "geom"] = LineString([(0, 0), (1, 1), (0, 0)])

        # generate trips and a joint staypoint/triplegs dataframe
        sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15)

        # test if start of first trip is (0,0)
        assert trips.loc[0, "geom"].geoms[0] == Point(0, 0)

    def test_keeping_all_columns_sp_tpls(self, example_triplegs):
        """Test if function does not drop columns in staypoints or triplegs."""
        sp_pre, tpls_pre = example_triplegs
        # these are the columns that are currently used in the code
        cols_sp = ["type", "sp_tpls_id", "temp_trip_id", "sp", "tpls", "prev_trip_id", "next_trip_id", "trip_id"]
        cols_tpls = cols_sp + ["is_activity"]
        sp_pre[cols_sp] = 11.06
        tpls_pre[cols_tpls] = 12.06
        sp_pre.as_staypoints
        tpls_pre.as_triplegs

        # catch warnings for overwriting columns
        with pytest.warns(UserWarning):
            sp_post, tpls_post, _ = generate_trips(sp_pre, tpls_pre, gap_threshold=15)

        assert_index_equal(sp_pre.columns, sp_pre.columns.intersection(sp_post.columns))
        assert_index_equal(tpls_pre.columns, tpls_pre.columns.intersection(tpls_post.columns))

    def test_missing_is_activity_column(self, example_triplegs):
        """Tests is AttributeError is raised on missing "is_activity" column of staypoints."""
        sp, tpls = example_triplegs
        sp.drop(columns="is_activity", inplace=True)
        error_msg = "staypoints need the column 'is_activity' to be able to generate trips"
        with pytest.raises(AttributeError, match=error_msg):
            generate_trips(sp, tpls)

    def test_crs(self, example_triplegs):
        """Test that the resulting GeoDataFrame has the correct crs or a warning or error is thrown if not set"""
        sp, tpls = example_triplegs
        # Case 1: sp crs None --> throw warning and set to tpls crs
        sp.crs = None
        with pytest.warns(UserWarning):
            _, _, trips = generate_trips(sp, tpls)
            assert trips.crs == tpls.crs
        # Case 2: Both crs None --> warn and set to None
        tpls.crs = None
        with pytest.warns(UserWarning):
            _, _, trips = generate_trips(sp, tpls)
            assert trips.crs is None
        # Case 3: tpls crs is None --> throw warning and set to sp crs
        sp.crs = "EPSG:4326"
        with pytest.warns(UserWarning):
            _, _, trips = generate_trips(sp, tpls)
            assert trips.crs == "EPSG:4326"
        # Case 4: Both crs set and correspond
        tpls.crs = "EPSG:2056"
        sp.crs = "EPSG:2056"
        _, _, trips = generate_trips(sp, tpls)
        assert trips.crs == "EPSG:2056"
        # Case 5: Both crs set but differ --> throw error
        sp.crs = "EPSG:4326"
        error_msg = "CRS of staypoints and triplegs differ. Geometry cannot be joined safely."
        with pytest.raises(AssertionError, match=error_msg):
            generate_trips(sp, tpls)

    def test_trips_type(self, example_triplegs):
        """Test if trips are really Trips"""
        sp, tpls = example_triplegs
        _, _, trips = generate_trips(sp, tpls)
        assert isinstance(trips, ti.TripsGeoDataFrame)


def _create_debug_sp_tpls_data(sp, tpls, gap_threshold):
    """Preprocess sp and tpls for "test_generate_trips_*."""
    # create table with relevant information from triplegs and staypoints.
    tpls["type"] = "tripleg"
    sp["type"] = "staypoint"
    cols_sp = ["started_at", "finished_at", "user_id", "type", "is_activity", "trip_id", "prev_trip_id", "next_trip_id"]
    cols_tpls = ["started_at", "finished_at", "user_id", "type", "trip_id"]
    sp_tpls = pd.concat((sp[cols_sp], tpls[cols_tpls]))

    # transform nan to bool
    sp_tpls["is_activity"] = sp_tpls["is_activity"].__eq__(True)
    sp_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    sp_tpls["started_at_next"] = sp_tpls["started_at"].shift(-1)
    sp_tpls["activity_next"] = sp_tpls["is_activity"].shift(-1)

    sp_tpls["gap"] = (sp_tpls["started_at_next"] - sp_tpls["finished_at"]).dt.seconds / 60 > gap_threshold

    return sp_tpls


def _generate_trips_old(sp_input, tpls_input, gap_threshold=15, print_progress=False):
    """Generate trips based on staypoints and triplegs.

    Parameters
    ----------
    sp_input : GeoDataFrame (as trackintel staypoints)
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
    assert "is_activity" in sp_input.columns, "staypoints need the column 'is_activity' to be able to generate trips"

    # we copy the input because we need to add a temporary column
    tpls = tpls_input.copy()
    sp = sp_input.copy()

    # if the triplegs already have a column "trip_id", we drop it
    if "trip_id" in tpls:
        tpls.drop(columns="trip_id", inplace=True)

    # if the staypoints already have any of the columns "trip_id", "prev_trip_id", "next_trip_id", we drop them
    for col in ["trip_id", "prev_trip_id", "next_trip_id"]:
        if col in sp:
            sp.drop(columns=col, inplace=True)

    tpls["type"] = "tripleg"
    sp["type"] = "staypoint"

    # create table with relevant information from triplegs and staypoints.
    sp_tpls = pd.concat(
        [
            sp[["started_at", "finished_at", "user_id", "type", "is_activity"]],
            tpls[["started_at", "finished_at", "user_id", "type"]],
        ]
    )

    # create ID field from index
    sp_tpls["id"] = sp_tpls.index

    # transform nan to bool
    sp_tpls["is_activity"] = sp_tpls["is_activity"].__eq__(True)

    sp_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    sp_tpls["started_at_next"] = sp_tpls["started_at"].shift(-1)
    sp_tpls["is_activity_next"] = sp_tpls["is_activity"].shift(-1)

    if print_progress:
        tqdm.pandas(desc="User trip generation")
        trips = (
            sp_tpls.groupby(["user_id"], group_keys=False, as_index=False)
            .progress_apply(_generate_trips_user, gap_threshold=gap_threshold)
            .reset_index(drop=True)
        )
    else:
        trips = (
            sp_tpls.groupby(["user_id"], group_keys=False, as_index=False)
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

    # assign trip_id to sp, for non-activity sp
    trip2spt_map = trips[["sp"]].to_dict()["sp"]
    ls = []
    for key, values in trip2spt_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[sp.index.name, "trip_id"]).set_index(sp.index.name)
    sp = sp.join(temp, how="left")

    # assign prev_trip_id to sp
    temp = trips[["destination_staypoint_id"]].copy()
    temp.rename(columns={"destination_staypoint_id": sp.index.name}, inplace=True)
    temp.index.name = "prev_trip_id"
    temp = temp.reset_index().set_index(sp.index.name)
    sp = sp.join(temp, how="left")

    # assign next_trip_id to sp
    temp = trips[["origin_staypoint_id"]].copy()
    temp.rename(columns={"origin_staypoint_id": sp.index.name}, inplace=True)
    temp.index.name = "next_trip_id"
    temp = temp.reset_index().set_index(sp.index.name)
    sp = sp.join(temp, how="left")

    # final cleaning
    tpls.drop(columns=["type"], inplace=True)
    sp.drop(columns=["type"], inplace=True)
    trips.drop(columns=["tpls", "sp"], inplace=True)

    ## dtype consistency
    # trips id (generated by this function) should be int64
    trips.index = trips.index.astype("int64")
    # trip id of sp and tpls can only be in Int64 (missing values)
    sp["trip_id"] = sp["trip_id"].astype("Int64")
    sp["prev_trip_id"] = sp["prev_trip_id"].astype("Int64")
    sp["next_trip_id"] = sp["next_trip_id"].astype("Int64")
    tpls["trip_id"] = tpls["trip_id"].astype("Int64")

    # user_id of trips should be the same as tpls
    trips["user_id"] = trips["user_id"].astype(tpls["user_id"].dtype)

    return sp, tpls, trips


def _generate_trips_user(df, gap_threshold):
    # function called after groupby: should only contain records of one user
    user_id = df["user_id"].unique()
    assert len(user_id) == 1
    user_id = user_id[0]

    unknown_activity = {"user_id": user_id, "is_activity": True, "id": np.nan}
    origin_activity = unknown_activity
    temp_trip_stack = []
    in_trip = False
    trip_ls = []

    for _, row in df.iterrows():
        # check if we can start a new trip
        # (we make sure that we start the trip with the most recent activity)
        if in_trip is False:
            # If there are several activities in a row, we skip until the last one
            if row["is_activity"] and row["activity_next"]:
                continue

            # if this is the last activity before the trip starts, reset the origin
            elif row["is_activity"]:
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
            if row["is_activity"] is True:
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

    origin_purpose : dictionary like
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
    assert origin_activity["is_activity"] is True
    assert destination_activity["is_activity"] is True
    assert first_trip_element["is_activity"] is False

    trip_dict_entry = {
        "user_id": origin_activity["user_id"],
        "started_at": first_trip_element["started_at"],
        "finished_at": last_trip_element["finished_at"],
        "origin_staypoint_id": origin_activity["id"],
        "destination_staypoint_id": destination_activity["id"],
        "tpls": [tripleg["id"] for tripleg in temp_trip_stack if tripleg["type"] == "tripleg"],
        "sp": [tripleg["id"] for tripleg in temp_trip_stack if tripleg["type"] == "staypoint"],
    }

    return trip_dict_entry
