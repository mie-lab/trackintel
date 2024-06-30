import datetime
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas import Timestamp
from shapely.geometry import Point

import trackintel as ti


@pytest.fixture
def geolife_pfs_sp_long():
    """Read geolife_long and generate sp."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    return pfs, sp


@pytest.fixture
def example_positionfixes():
    """Positionfixes for tests."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")

    list_dict = [
        {"user_id": 0, "tracked_at": t1, "geometry": p1},
        {"user_id": 0, "tracked_at": t2, "geometry": p2},
        {"user_id": 1, "tracked_at": t3, "geometry": p3},
    ]
    pfs = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    pfs.index.name = "id"

    return ti.Positionfixes(pfs)


@pytest.fixture
def example_positionfixes_isolated():
    """
    Positionfixes with isolated positionfixes.

    User1 has the same geometry but different timestamps.
    User2 has different geometry and timestamps.
    User3 has only one isolated positionfix.
    """
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-02 01:01:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 01:02:00", tz="utc")
    t4 = pd.Timestamp("1971-01-03 03:00:00", tz="utc")

    list_dict = [
        {"user_id": 0, "tracked_at": t1, "geometry": p1, "staypoint_id": 0},
        {"user_id": 0, "tracked_at": t2, "geometry": p2, "staypoint_id": pd.NA},
        {"user_id": 0, "tracked_at": t3, "geometry": p2, "staypoint_id": pd.NA},
        {"user_id": 0, "tracked_at": t4, "geometry": p3, "staypoint_id": 1},
        {"user_id": 1, "tracked_at": t1, "geometry": p1, "staypoint_id": 2},
        {"user_id": 1, "tracked_at": t2, "geometry": p2, "staypoint_id": pd.NA},
        {"user_id": 1, "tracked_at": t3, "geometry": p3, "staypoint_id": pd.NA},
        {"user_id": 1, "tracked_at": t4, "geometry": p3, "staypoint_id": 3},
        {"user_id": 2, "tracked_at": t1, "geometry": p1, "staypoint_id": 4},
        {"user_id": 2, "tracked_at": t2, "geometry": p2, "staypoint_id": pd.NA},
        {"user_id": 2, "tracked_at": t4, "geometry": p3, "staypoint_id": 5},
    ]
    pfs = ti.Positionfixes(data=list_dict, geometry="geometry", crs="EPSG:4326")
    pfs["staypoint_id"] = pfs["staypoint_id"].astype("Int64")
    pfs.index.name = "id"

    return pfs


class TestGenerate_staypoints:
    """Tests for generate_staypoints() method."""

    def test_empty_generation(self, example_positionfixes):
        """The function should run without error if the generation result is empty (no staypoint could be generated)."""
        # the pfs would not generate staypoints with the default parameters
        pfs = example_positionfixes

        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, sp = pfs.generate_staypoints()
        assert len(sp) == 0

    def test_parallel_computing(self):
        """The result obtained with parallel computing should be identical."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        # without parallel computing code
        pfs_ori, sp_ori = pfs.generate_staypoints(n_jobs=1)
        # using two cores
        pfs_para, sp_para = pfs.generate_staypoints(n_jobs=2)

        # the result of parallel computing should be identical
        assert_geodataframe_equal(pfs_ori, pfs_para)
        assert_geodataframe_equal(sp_ori, sp_para)

    def test_duplicate_pfs_warning(self, example_positionfixes):
        """Calling generate_staypoints with duplicate positionfixes should raise a warning."""
        duplicated_location = example_positionfixes.copy()
        duplicated_location.loc[0, "geometry"] = duplicated_location.loc[1, "geometry"]

        duplicated_time = example_positionfixes.copy()
        duplicated_time.loc[0, "tracked_at"] = duplicated_time.loc[1, "tracked_at"]

        # 0 and 1 pfs are now identical
        duplicated_location_and_time = example_positionfixes.copy()
        duplicated_location_and_time.iloc[0] = duplicated_location_and_time.iloc[1]

        warn_duplicates = "duplicates were dropped from your positionfixes."
        # generates warning for empty generation but not for duplicate pfs
        with pytest.warns(UserWarning) as record:
            example_positionfixes.generate_staypoints()
            duplicated_location.generate_staypoints()
            duplicated_time.generate_staypoints()
            assert not any(True for x in record if warn_duplicates in str(x.message))

        # capture all warnings in record -> we will maybe loose some warnings
        with pytest.warns(UserWarning) as record:
            duplicated_location_and_time.generate_staypoints()
            assert any(True for x in record if warn_duplicates in str(x.message))

    def test_duplicate_pfs_filtering(self, example_positionfixes):
        """Test that duplicate positionfixes are filtered in generate_staypoints."""
        pfs = example_positionfixes

        # set 0 and 1 pfs identical
        pfs.loc[0, "geometry"] = pfs.loc[1, "geometry"]
        pfs.loc[0, "tracked_at"] = pfs.loc[1, "tracked_at"]

        with pytest.warns(UserWarning):
            pfs_out, _ = pfs.generate_staypoints()

        # drop staypoint_id column of pfs_out and ensure that the second duplicate (id=1) is filtered
        assert_geodataframe_equal(pfs_out.drop(columns="staypoint_id"), pfs.iloc[[0, 2]])

    def test_duplicate_columns(self):
        """Test if running the function twice, the generated column does not yield exception in join statement"""

        # we run generate_staypoints twice in order to check that the extra column(tripleg_id) does
        # not cause any problems in the second run
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs_run_1, _ = pfs.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        pfs_run_2, _ = pfs_run_1.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        assert set(pfs_run_1.columns) == set(pfs_run_2.columns)

    def test_sliding_min(self):
        """Test if using small thresholds, stp extraction yields each pfs."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs, sp = pfs.generate_staypoints(method="sliding", dist_threshold=0, time_threshold=0, include_last=True)
        assert len(sp) == len(pfs)

    def test_sliding_max(self):
        """Test if using large thresholds, stp extraction yield no pfs."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        warn_string = "No staypoints can be generated, returning empty sp."
        max_time = pd.Timedelta.max.total_seconds() // 60
        with pytest.warns(UserWarning, match=warn_string):
            _, sp = pfs.generate_staypoints(method="sliding", dist_threshold=sys.maxsize, time_threshold=max_time)
        assert len(sp) == 0

    def test_missing_link(self):
        """Test nan is assigned for missing link between pfs and sp."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        warn_string = "No staypoints can be generated, returning empty sp."
        max_time = pd.Timedelta.max.total_seconds() // 60
        with pytest.warns(UserWarning, match=warn_string):
            pfs, _ = pfs.generate_staypoints(method="sliding", dist_threshold=sys.maxsize, time_threshold=max_time)

        assert pd.isna(pfs["staypoint_id"]).all()

    def test_dtype_consistent(self, geolife_pfs_sp_long):
        """Test the dtypes for the generated columns."""
        pfs, sp = geolife_pfs_sp_long

        assert pfs["user_id"].dtype == sp["user_id"].dtype
        assert pfs["staypoint_id"].dtype == "Int64"
        assert sp.index.dtype == "int64"

    def test_index_start(self, geolife_pfs_sp_long):
        """Test the generated index start from 0 for different methods."""
        _, sp = geolife_pfs_sp_long

        assert (sp.index == np.arange(len(sp))).all()

    def test_include_last(self):
        """Test if the include_last arguement will include the last pfs as stp."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))

        pfs_wo, sp_wo = pfs.generate_staypoints()
        pfs_include, sp_include = pfs.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        # sp_wo does not include the last staypoint
        assert len(sp_wo) == len(sp_include) - 1
        # the last pfs of pfs_include has stp connection
        assert not pfs_include.tail(1)["staypoint_id"].isna().all()
        assert pfs_wo.tail(1)["staypoint_id"].isna().all()

    def test_print_progress(self):
        """Test if the result from print progress agrees with the original."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs_ori, sp_ori = pfs.generate_staypoints(method="sliding", dist_threshold=100, time_threshold=5)
        pfs_print, sp_print = pfs.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5, print_progress=True
        )
        assert_geodataframe_equal(pfs_ori, pfs_print)
        assert_geodataframe_equal(sp_ori, sp_print)

    def test_temporal(self):
        """Test if the sp generation result follows predefined time_threshold and gap_threshold."""
        pfs_input, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))

        # the duration should be not longer than time_threshold
        time_threshold_ls = [3, 5, 10]

        # the missing time should not exceed gap_threshold
        gap_threshold_ls = [5, 10, 15, 20]
        for time_threshold in time_threshold_ls:
            for gap_threshold in gap_threshold_ls:
                pfs, sp = pfs_input.generate_staypoints(
                    method="sliding", dist_threshold=50, time_threshold=time_threshold, gap_threshold=gap_threshold
                )

                # all durations should be longer than the time_threshold
                duration_sp_min = (sp["finished_at"] - sp["started_at"]).dt.total_seconds() / 60
                assert (duration_sp_min > time_threshold).all()

                # all last pfs should be shorter than the gap_threshold
                # get the difference between pfs tracking time, and assign back to the previous pfs
                pfs["diff"] = ((pfs["tracked_at"] - pfs["tracked_at"].shift(1)).dt.total_seconds() / 60).shift(-1)
                # get the last pf of sp and check the gap size
                pfs.dropna(subset=["staypoint_id"], inplace=True)
                pfs.drop_duplicates(subset=["staypoint_id"], keep="last", inplace=True)
                assert (pfs["diff"] < gap_threshold).all()

    def test_gap_threshold(self):
        """Test for gap_threshold for consecutive pfs."""
        # two pfs far apart in time that could potentially form a sp spatially.
        pfs_dict = [
            {
                "latitude": 39.976,
                "longitude": 116.330683333333,
                "tracked_at": Timestamp("2007-08-09 01:35:46+0000", tz="UTC"),
                "user_id": 55,
            },
            {
                "latitude": 39.975818526,
                "longitude": 116.331600228,
                "tracked_at": Timestamp("2011-08-03 09:12:52+0000", tz="UTC"),
                "user_id": 55,
            },
        ]
        df = pd.DataFrame(pfs_dict)
        pfs = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="epsg:4326")
        pfs = ti.Positionfixes(pfs)

        # using the default gap_threshold will generate no sp
        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            _, sp = pfs.generate_staypoints(include_last=True)
            assert len(sp) == 0

        # using large gap_threshold one sp will be generated
        _, sp = pfs.generate_staypoints(gap_threshold=1e8, include_last=True)
        assert len(sp) == 1

    def test_str_userid(self, example_positionfixes):
        """Staypoint generation should also run without error if the user_id is a string"""
        # the pfs would not generate staypoints with the default parameters
        pfs = example_positionfixes
        pfs["user_id"] = pfs["user_id"].astype(str)
        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            pfs.generate_staypoints()

    def test_sp_type(self):
        """Test if sp are really Staypoints"""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        _, sp = pfs.generate_staypoints()
        assert isinstance(sp, ti.Staypoints)


class TestGenerate_staypoints_sliding_user:
    """Test for _generate_staypoints_sliding_user."""

    def test_unknown_distance_metric(self, example_positionfixes):
        """Test if the distance metric is unknown, an ValueError will be raised."""
        with pytest.raises(ValueError):
            example_positionfixes.generate_staypoints(
                method="sliding", dist_threshold=100, time_threshold=5, distance_metric="unknown"
            )


class TestCreate_new_staypoints:
    """Test __create_new_staypoints."""

    def test_planar_crs(self, geolife_pfs_sp_long):
        """Test if planar crs are handled as well"""
        pfs, _ = geolife_pfs_sp_long
        _, sp_wgs84 = pfs.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        pfs = pfs.set_crs(2056, allow_override=True)
        _, sp_lv95 = pfs.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        sp_lv95.set_crs(4326, allow_override=True, inplace=True)
        # planar and non-planar differ only if we experience a wrap in coords like [+180, -180]
        assert_geodataframe_equal(sp_wgs84, sp_lv95, check_less_precise=True)


class TestGenerate_triplegs_between_staypoints:
    """Tests for generate_triplegs() with 'between_staypoints' method."""

    def test_empty_generation(self, example_positionfixes_isolated):
        """The function should run without error if the generation result is empty (no tripleg could be generated)."""
        pfs = example_positionfixes_isolated

        # select subset of pfs such that no triplegs can be generated
        pfs = pfs.loc[pfs["user_id"] == 0]
        pfs = pfs.iloc[2:]

        warn_string = "No triplegs can be generated, returning empty tpls."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, tpls = pfs.generate_triplegs()
        assert len(tpls) == 0

    def test_noncontinuous_unordered_index(self, example_positionfixes_isolated):
        """The unordered and noncontinuous index of pfs shall not affect the generate_triplegs() result."""
        pfs = example_positionfixes_isolated

        # regenerate noncontinuous index
        pfs.index = range(0, pfs.shape[0] * 2, 2)
        pfs.index.name = "id"
        # shuffle index
        pfs = pfs.sample(frac=1)

        # a warning shall raise due to the duplicated positionfixes
        warn_string = "The positionfixes with ids .* lead to invalid tripleg geometries."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, tpls = pfs.generate_triplegs()

        # the index shall be reordered
        assert np.all(pfs.index == range(0, pfs.shape[0] * 2, 2))

        # only user 1 has generated a tripleg.
        # user 0 and 2 tripleg has invalid geometry (two identical points and only one point respectively)
        assert (tpls["user_id"].unique() == [1]) and (len(tpls) == 1)

    def test_invalid_isolates(self, example_positionfixes_isolated):
        """Triplegs generated from isolated duplicates are dropped."""
        pfs = example_positionfixes_isolated

        # a warning shall raise due to the duplicated positionfixes
        warn_string = "The positionfixes with ids .* lead to invalid tripleg geometries."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, tpls = pfs.generate_triplegs()

        # tripleg of user 0 is dropped. Tripleg of user 1 is available.
        assert len(tpls) == 1
        assert tpls.user_id.iloc[0] == 1

    def test_duplicate_columns(self, geolife_pfs_sp_long):
        """Test if running the function twice, the generated column does not yield exception in join statement."""

        # we run generate_triplegs twice in order to check that the extra column (tripleg_id) does
        # not cause any problems in the second run
        pfs, sp = geolife_pfs_sp_long

        pfs_run_1, _ = pfs.generate_triplegs(sp, method="between_staypoints")
        pfs_run_2, _ = pfs_run_1.generate_triplegs(sp, method="between_staypoints")
        assert set(pfs_run_1.columns) == set(pfs_run_2.columns)

    def test_user_without_sp(self, geolife_pfs_sp_long):
        """Check if it is safe to have users that have pfs but no sp."""
        pfs, sp = geolife_pfs_sp_long
        # manually change the first pfs' user_id, which has no stp correspondence
        pfs.loc[0, "user_id"] = 5000

        ## test for case 1
        _, tpls_1 = pfs.generate_triplegs(sp, method="between_staypoints")
        # result should be the same ommiting the first row
        _, tpls_2 = pfs.iloc[1:].generate_triplegs(sp, method="between_staypoints")
        assert_geodataframe_equal(tpls_1, tpls_2)

        ## test for case 2
        pfs.drop(columns="staypoint_id", inplace=True)

        warn_string = "Providing positionfixes without*"
        with pytest.warns(DeprecationWarning, match=warn_string):
            # manually change the first pfs' user_id, which has no stp correspondence
            _, tpls_1 = pfs.generate_triplegs(sp, method="between_staypoints")
            # result should be the same ommiting the first row
            _, tpls_2 = pfs.iloc[1:].generate_triplegs(sp, method="between_staypoints")

        assert_geodataframe_equal(tpls_1, tpls_2)

    def test_pfs_without_sp(self, geolife_pfs_sp_long):
        """Delete pfs that belong to staypoints and see if they are detected."""
        pfs, sp = geolife_pfs_sp_long

        _, tpls_case1 = pfs.generate_triplegs(sp, method="between_staypoints")
        # only keep pfs where staypoint id is nan
        pfs_no_sp = pfs[pd.isna(pfs["staypoint_id"])].drop(columns="staypoint_id")
        warn_string = "Providing positionfixes without*"
        with pytest.warns(DeprecationWarning, match=warn_string):
            _, tpls_case2 = pfs_no_sp.generate_triplegs(sp, method="between_staypoints")

        assert_geodataframe_equal(tpls_case1, tpls_case2)

    def test_stability(self, geolife_pfs_sp_long):
        """Checks if the results are same for different cases in tripleg_generation method."""
        pfs, sp = geolife_pfs_sp_long
        # case 1
        pfs_case1, tpls_case1 = pfs.generate_triplegs(sp, method="between_staypoints")
        # case 1 without sp
        pfs_case1_wo, tpls_case1_wo = pfs.generate_triplegs(method="between_staypoints")

        # case 2
        pfs = pfs.drop(columns="staypoint_id")
        warn_string = "Providing positionfixes without*"
        with pytest.warns(DeprecationWarning, match=warn_string):
            pfs_case2, tpls_case2 = pfs.generate_triplegs(sp, method="between_staypoints")

        assert_geodataframe_equal(pfs_case1.drop(columns="staypoint_id", axis=1), pfs_case2)
        assert_geodataframe_equal(pfs_case1, pfs_case1_wo)
        assert_geodataframe_equal(tpls_case1, tpls_case2)
        assert_geodataframe_equal(tpls_case1, tpls_case1_wo)

        with pytest.raises(TypeError, match="staypoints input must be provided for pfs without staypoint_id column"):
            pfs.generate_triplegs(staypoints=None, method="between_staypoints")

    def test_random_order(self, geolife_pfs_sp_long):
        """Checks if same tpls will be generated after random shuffling pfs."""
        pfs, sp = geolife_pfs_sp_long
        # ensure proper order of pfs
        pfs.sort_values(by=["user_id", "tracked_at"], inplace=True)

        # original order
        pfs_ori, tpls_ori = pfs.generate_triplegs(sp)

        # resample/shuffle pfs
        pfs_shuffle = pfs.sample(frac=1, random_state=0)
        pfs_shuffle, tpls_shuffle = pfs_shuffle.generate_triplegs(sp)

        # order should be the same -> pfs.sort_values within function
        # generated tpls index should be the same
        assert_geodataframe_equal(pfs_ori, pfs_shuffle)
        assert_geodataframe_equal(tpls_ori, tpls_shuffle)

    def test_pfs_index(self, geolife_pfs_sp_long):
        """Checks if same tpls will be generated after changing pfs index."""
        pfs, sp = geolife_pfs_sp_long

        # original index
        pfs_ori, tpls_ori = pfs.generate_triplegs(sp)

        # create discontinues index
        pfs.index = np.arange(len(pfs)) * 2
        pfs_index, tpls_index = pfs.generate_triplegs(sp)

        # generated tpls index should be the same
        assert_geodataframe_equal(pfs_ori.reset_index(drop=True), pfs_index.reset_index(drop=True))
        assert_geodataframe_equal(tpls_ori, tpls_index)

    def test_dtype_consistent(self, geolife_pfs_sp_long):
        """Test the dtypes for the generated columns."""
        pfs, sp = geolife_pfs_sp_long
        pfs, tpls = pfs.generate_triplegs(sp)
        assert pfs["user_id"].dtype == tpls["user_id"].dtype
        assert pfs["tripleg_id"].dtype == "Int64"
        assert tpls.index.dtype == "int64"

    def test_missing_link(self, geolife_pfs_sp_long):
        """Test nan is assigned for missing link between pfs and tpls."""
        pfs, sp = geolife_pfs_sp_long

        pfs, _ = pfs.generate_triplegs(sp, method="between_staypoints")

        assert pd.isna(pfs["tripleg_id"]).any()

    def test_index_start(self, geolife_pfs_sp_long):
        """Test the generated index start from 0 for different cases."""
        pfs, sp = geolife_pfs_sp_long

        _, tpls_case1 = pfs.generate_triplegs(sp)
        warn_string = "Providing positionfixes without*"
        with pytest.warns(DeprecationWarning, match=warn_string):
            _, tpls_case2 = pfs.drop("staypoint_id", axis=1).generate_triplegs(sp)

        assert (tpls_case1.index == np.arange(len(tpls_case1))).any()
        assert (tpls_case2.index == np.arange(len(tpls_case2))).any()

    def test_invalid_inputs(self, geolife_pfs_sp_long):
        """Test if ValueError will be raised after invalid method input."""
        pfs, sp = geolife_pfs_sp_long

        with pytest.raises(ValueError, match="Method unknown"):
            pfs.generate_triplegs(sp, method="random")
        with pytest.raises(ValueError, match="Method unknown"):
            pfs.generate_triplegs(sp, method=12345)

    def test_temporal(self, geolife_pfs_sp_long):
        """Test if the tpls generation result follows predefined gap_threshold."""
        pfs_input, sp = geolife_pfs_sp_long

        gap_threshold_ls = [0.1, 0.2, 1, 2]
        for gap_threshold in gap_threshold_ls:
            pfs, _ = pfs_input.generate_triplegs(sp, gap_threshold=gap_threshold)

            # conti_tpl checks whether the next pfs is in the same tpl
            pfs["conti_tpl"] = (pfs["tripleg_id"] - pfs["tripleg_id"].shift(1)).shift(-1)
            # get the time difference of pfs, and assign to the previous pfs
            pfs["diff"] = ((pfs["tracked_at"] - pfs["tracked_at"].shift(1)).dt.total_seconds() / 60).shift(-1)
            # we only take tpls that are splitted in the middle (tpl - tpl) and the second user
            pfs = pfs.loc[(pfs["conti_tpl"] == 1) & (pfs["user_id"] == 1)]
            # check if the cuts are appropriate
            assert (pfs["diff"] > gap_threshold).all()

    def test_sp_tpls_overlap(self, geolife_pfs_sp_long):
        """Tpls and sp should not overlap when generated using the default extract triplegs method."""
        pfs, sp = geolife_pfs_sp_long
        pfs, tpls = pfs.generate_triplegs(sp)

        sp = sp[["started_at", "finished_at", "user_id"]]
        tpls = tpls[["started_at", "finished_at", "user_id"]]
        sp_tpls = pd.concat((sp, tpls))
        sp_tpls.sort_values(by=["user_id", "started_at"], inplace=True)

        for user_id_this in sp["user_id"].unique():
            sp_tpls_this = sp_tpls[sp_tpls["user_id"] == user_id_this]
            diff = sp_tpls_this["started_at"] - sp_tpls_this["finished_at"].shift(1)
            # transform to numpy array and drop first values (always nan due to shift operation)
            diff = diff.values[1:]

            # all values have to greater or equal to zero. Otherwise there is an overlap
            assert all(diff >= np.timedelta64(datetime.timedelta()))

    def test_str_userid(self, example_positionfixes_isolated):
        """Tripleg generation should also work if the user IDs are strings."""
        pfs = example_positionfixes_isolated
        # remove isolated - not needed for this test
        pfs = pfs[~pfs.index.isin([1, 2])].copy()
        # set user ID to string
        pfs["user_id"] = pfs["user_id"].astype(str) + "not_numerical_interpretable_str"
        pfs, _ = pfs.generate_triplegs()

    def test_tpls_type(self, example_positionfixes_isolated):
        """Test that Tripleg generation returns correct type"""
        # remove isolated - not needed for this test
        pfs = example_positionfixes_isolated
        pfs = pfs[~pfs.index.isin([1, 2])].copy()

        _, tpls = pfs.generate_triplegs()
        assert isinstance(tpls, ti.Triplegs)


class TestGenerate_triplegs_overlap_staypoints:
    """Tests for generate_triplegs() with 'overlap_staypoints' method."""

    def test_sp_tpls_overlap(self, geolife_pfs_sp_long):
        """Triplegs should overlap staypoint with more than one positionfix."""
        pfs, sp = geolife_pfs_sp_long
        pfs, tpls = pfs.generate_triplegs(sp, method="overlap_staypoints")

        tpl_0 = tpls[tpls["user_id"] == 0].loc[2]
        sp = sp[sp["user_id"] == 0].loc[0]
        tpl_1 = tpls[tpls["user_id"] == 0].loc[3]

        assert tpl_0["finished_at"] == sp["started_at"]
        assert sp["finished_at"] == tpl_1["started_at"]

        # last point of tpl should correspond to next sp
        assert Point(tpl_0["geom"].coords[-1]) == sp["geom"]
        # first point of next tpl should correspond to previous sp
        assert sp["geom"] == Point(tpl_1["geom"].coords[0])

    def test_sp_one_pfs_tpls_overlap(self, geolife_pfs_sp_long):
        """Triplegs should overlap staypoint in time with only one positionfix, but only the first tripleg should
        spatially overlap the staypoint."""
        pfs, sp = geolife_pfs_sp_long
        pfs, tpls = pfs.generate_triplegs(sp, method="overlap_staypoints")

        tpl_0 = tpls[tpls["user_id"] == 1].loc[13]
        sp = sp[sp["user_id"] == 1].loc[6]
        tpl_1 = tpls[tpls["user_id"] == 1].loc[14]

        assert tpl_0["finished_at"] == sp["started_at"]
        assert sp["finished_at"] == tpl_1["started_at"]

        assert Point(tpl_0["geom"].coords[-1]) == sp["geom"]
        # no spatial overlap with second tripleg
        assert sp["geom"] != Point(tpl_1["geom"].coords[0])

    def test_str_userid(self, geolife_pfs_sp_long):
        """Tripleg generation should also work if the user IDs are strings."""
        pfs, sp = geolife_pfs_sp_long

        # set user ID to string
        pfs["user_id"] = pfs["user_id"].astype(str) + "not_numerical_interpretable_str"
        pfs, _ = pfs.generate_triplegs(staypoints=sp, method="overlap_staypoints")

    def test_staypoint_inputs(self, geolife_pfs_sp_long):
        """Test if TypeError will be raised when no staypoint is provided."""
        pfs, _ = geolife_pfs_sp_long

        with pytest.raises(TypeError, match="staypoints input must be provided for overlap_staypoints"):
            pfs.generate_triplegs(staypoints=None, method="overlap_staypoints")

    def test_pfs_format(self, geolife_pfs_sp_long):
        """Test if TypeError will be raised when pfs with no staypoint_id column is provided."""
        pfs, sp = geolife_pfs_sp_long

        with pytest.raises(TypeError, match="positionfixes must contain a staypoint_id column for overlap_staypoints"):
            pfs.drop(columns="staypoint_id").generate_triplegs(staypoints=sp, method="overlap_staypoints")
