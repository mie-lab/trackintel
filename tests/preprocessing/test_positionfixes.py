import datetime
import os
import sys
from pandas import Timestamp

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point

import trackintel as ti


@pytest.fixture
def geolife_pfs_sp_long():
    """Read geolife_long and generate sp."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
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

    # assert validity of positionfixes.
    pfs.as_positionfixes
    return pfs


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
    pfs = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    pfs.index.name = "id"

    # assert validity of positionfixes.
    pfs.as_positionfixes
    return pfs


class TestGenerate_staypoints:
    """Tests for generate_staypoints() method."""

    def test_empty_generation(self, example_positionfixes):
        """The function should run without error if the generation result is empty (no staypoint could be generated)."""
        # the pfs would not generate staypoints with the default parameters
        pfs = example_positionfixes

        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, sp = pfs.as_positionfixes.generate_staypoints()
        assert len(sp) == 0

    def test_parallel_computing(self):
        """The result obtained with parallel computing should be identical."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
        # without parallel computing code
        pfs_ori, sp_ori = pfs.as_positionfixes.generate_staypoints(n_jobs=1)
        # using two cores
        pfs_para, sp_para = pfs.as_positionfixes.generate_staypoints(n_jobs=2)

        # the result of parallel computing should be identical
        assert_geodataframe_equal(pfs_ori, pfs_para)
        assert_geodataframe_equal(sp_ori, sp_para)

    def test_duplicate_pfs_warning(self, example_positionfixes):
        """Calling generate_staypoints with duplicate positionfixes should raise a warning."""
        pfs_duplicate_loc = example_positionfixes.copy()
        pfs_duplicate_loc.loc[0, "geometry"] = pfs_duplicate_loc.loc[1, "geometry"]

        pfs_duplicate_t = example_positionfixes.copy()
        pfs_duplicate_t.loc[0, "tracked_at"] = pfs_duplicate_t.loc[1, "tracked_at"]

        # 0 and 1 pfs are now identical
        pfs_duplicate_all = pfs_duplicate_loc.copy()
        pfs_duplicate_all.loc[0, "tracked_at"] = pfs_duplicate_all.loc[1, "tracked_at"]

        warn_string = ".* duplicates were dropped from your positionfixes."
        with pytest.warns(None) as record:
            example_positionfixes.as_positionfixes.generate_staypoints()
            pfs_duplicate_loc.as_positionfixes.generate_staypoints()
            pfs_duplicate_t.as_positionfixes.generate_staypoints()

            # assert that no warning of the defined type is raised
            assert len([x for x in record if warn_string in str(x.message)]) == 0

        with pytest.warns(UserWarning, match=warn_string):
            pfs_duplicate_all.as_positionfixes.generate_staypoints()

    def test_duplicate_pfs_filtering(self, example_positionfixes):
        """Test that duplicate positionfixes are filtered in generate_staypoints."""
        pfs = example_positionfixes

        # set 0 and 1 pfs identical
        pfs.loc[0, "geometry"] = pfs.loc[1, "geometry"]
        pfs.loc[0, "tracked_at"] = pfs.loc[1, "tracked_at"]

        with pytest.warns(UserWarning):
            pfs_out, _ = pfs.as_positionfixes.generate_staypoints()

        # drop staypoint_id column of pfs_out and ensure that the second duplicate (id=1) is filtered
        assert_geodataframe_equal(pfs_out.drop(columns="staypoint_id"), pfs.iloc[[0, 2]])

    def test_duplicate_columns(self):
        """Test if running the function twice, the generated column does not yield exception in join statement"""

        # we run generate_staypoints twice in order to check that the extra column(tripleg_id) does
        # not cause any problems in the second run
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs_run_1, _ = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        pfs_run_2, _ = pfs_run_1.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        assert set(pfs_run_1.columns) == set(pfs_run_2.columns)

    def test_sliding_min(self):
        """Test if using small thresholds, stp extraction yields each pfs."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs, sp = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=0, time_threshold=0, include_last=True
        )
        assert len(sp) == len(pfs)

    def test_sliding_max(self):
        """Test if using large thresholds, stp extraction yield no pfs."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            _, sp = pfs.as_positionfixes.generate_staypoints(
                method="sliding", dist_threshold=sys.maxsize, time_threshold=sys.maxsize
            )
        assert len(sp) == 0

    def test_missing_link(self):
        """Test nan is assigned for missing link between pfs and sp."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, _ = pfs.as_positionfixes.generate_staypoints(
                method="sliding", dist_threshold=sys.maxsize, time_threshold=sys.maxsize
            )

        assert pd.isna(pfs["staypoint_id"]).any()

    def test_dtype_consistent(self, geolife_pfs_sp_long):
        """Test the dtypes for the generated columns."""
        pfs, sp = geolife_pfs_sp_long

        assert pfs["user_id"].dtype == sp["user_id"].dtype
        assert pfs["staypoint_id"].dtype == "Int64"
        assert sp.index.dtype == "int64"

    def test_index_start(self, geolife_pfs_sp_long):
        """Test the generated index start from 0 for different methods."""
        _, sp = geolife_pfs_sp_long

        assert (sp.index == np.arange(len(sp))).any()

    def test_include_last(self):
        """Test if the include_last arguement will include the last pfs as stp."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))

        pfs_wo, sp_wo = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=False
        )
        pfs_include, sp_include = pfs.as_positionfixes.generate_staypoints(
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
        pfs_ori, sp_ori = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5
        )
        pfs_print, sp_print = pfs.as_positionfixes.generate_staypoints(
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
                pfs, sp = pfs_input.as_positionfixes.generate_staypoints(
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
        pfs = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
        pfs.as_positionfixes

        # using the default gap_threshold will generate no sp
        warn_string = "No staypoints can be generated, returning empty sp."
        with pytest.warns(UserWarning, match=warn_string):
            _, sp = pfs.as_positionfixes.generate_staypoints(include_last=True)
            assert len(sp) == 0

        # using large gap_threshold one sp will be generated
        _, sp = pfs.as_positionfixes.generate_staypoints(gap_threshold=1e8, include_last=True)
        assert len(sp) == 1


class TestGenerate_triplegs:
    """Tests for generate_triplegs() method."""

    def test_empty_generation(self, example_positionfixes_isolated):
        """The function should run without error if the generation result is empty (no tripleg could be generated)."""
        pfs = example_positionfixes_isolated

        # select subset of pfs such that no triplegs can be generated
        pfs = pfs.loc[pfs["user_id"] == 0]
        pfs = pfs.iloc[2:]

        warn_string = "No triplegs can be generated, returning empty tpls."
        with pytest.warns(UserWarning, match=warn_string):
            pfs, tpls = pfs.as_positionfixes.generate_triplegs()
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
            pfs, tpls = pfs.as_positionfixes.generate_triplegs()

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
            pfs, tpls = pfs.as_positionfixes.generate_triplegs()

        # tripleg of user 0 is dropped. Tripleg of user 1 is available.
        assert len(tpls) == 1
        assert tpls.user_id.iloc[0] == 1

    def test_duplicate_columns(self, geolife_pfs_sp_long):
        """Test if running the function twice, the generated column does not yield exception in join statement."""

        # we run generate_triplegs twice in order to check that the extra column (tripleg_id) does
        # not cause any problems in the second run
        pfs, sp = geolife_pfs_sp_long

        pfs_run_1, _ = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        pfs_run_2, _ = pfs_run_1.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        assert set(pfs_run_1.columns) == set(pfs_run_2.columns)

    def test_user_without_sp(self, geolife_pfs_sp_long):
        """Check if it is safe to have users that have pfs but no sp."""
        pfs, sp = geolife_pfs_sp_long
        # manually change the first pfs' user_id, which has no stp correspondence
        pfs.loc[0, "user_id"] = 5000

        ## test for case 1
        _, tpls_1 = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        # result should be the same ommiting the first row
        _, tpls_2 = pfs.iloc[1:].as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        assert_geodataframe_equal(tpls_1, tpls_2)

        ## test for case 2
        pfs.drop(columns="staypoint_id", inplace=True)
        # manually change the first pfs' user_id, which has no stp correspondence
        _, tpls_1 = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        # result should be the same ommiting the first row
        _, tpls_2 = pfs.iloc[1:].as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        assert_geodataframe_equal(tpls_1, tpls_2)

    def test_pfs_without_sp(self, geolife_pfs_sp_long):
        """Delete pfs that belong to staypoints and see if they are detected."""
        pfs, sp = geolife_pfs_sp_long

        _, tpls_case1 = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        # only keep pfs where staypoint id is nan
        pfs_no_sp = pfs[pd.isna(pfs["staypoint_id"])].drop(columns="staypoint_id")
        _, tpls_case2 = pfs_no_sp.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

        assert_geodataframe_equal(tpls_case1, tpls_case2)

    def test_stability(self, geolife_pfs_sp_long):
        """Checks if the results are same for different cases in tripleg_generation method."""
        pfs, sp = geolife_pfs_sp_long
        # case 1
        pfs_case1, tpls_case1 = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")
        # case 1 without sp
        pfs_case1_wo, tpls_case1_wo = pfs.as_positionfixes.generate_triplegs(method="between_staypoints")

        # case 2
        pfs = pfs.drop(columns="staypoint_id")
        pfs_case2, tpls_case2 = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

        assert_geodataframe_equal(pfs_case1.drop(columns="staypoint_id", axis=1), pfs_case2)
        assert_geodataframe_equal(pfs_case1, pfs_case1_wo)
        assert_geodataframe_equal(tpls_case1, tpls_case2)
        assert_geodataframe_equal(tpls_case1, tpls_case1_wo)

    def test_random_order(self, geolife_pfs_sp_long):
        """Checks if same tpls will be generated after random shuffling pfs."""
        pfs, sp = geolife_pfs_sp_long
        # ensure proper order of pfs
        pfs.sort_values(by=["user_id", "tracked_at"], inplace=True)

        # original order
        pfs_ori, tpls_ori = pfs.as_positionfixes.generate_triplegs(sp)

        # resample/shuffle pfs
        pfs_shuffle = pfs.sample(frac=1, random_state=0)
        pfs_shuffle, tpls_shuffle = pfs_shuffle.as_positionfixes.generate_triplegs(sp)

        # order should be the same -> pfs.sort_values within function
        # generated tpls index should be the same
        assert_geodataframe_equal(pfs_ori, pfs_shuffle)
        assert_geodataframe_equal(tpls_ori, tpls_shuffle)

    def test_pfs_index(self, geolife_pfs_sp_long):
        """Checks if same tpls will be generated after changing pfs index."""
        pfs, sp = geolife_pfs_sp_long

        # original index
        pfs_ori, tpls_ori = pfs.as_positionfixes.generate_triplegs(sp)

        # create discontinues index
        pfs.index = np.arange(len(pfs)) * 2
        pfs_index, tpls_index = pfs.as_positionfixes.generate_triplegs(sp)

        # generated tpls index should be the same
        assert_geodataframe_equal(pfs_ori.reset_index(drop=True), pfs_index.reset_index(drop=True))
        assert_geodataframe_equal(tpls_ori, tpls_index)

    def test_dtype_consistent(self, geolife_pfs_sp_long):
        """Test the dtypes for the generated columns."""
        pfs, sp = geolife_pfs_sp_long
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        assert pfs["user_id"].dtype == tpls["user_id"].dtype
        assert pfs["tripleg_id"].dtype == "Int64"
        assert tpls.index.dtype == "int64"

    def test_missing_link(self, geolife_pfs_sp_long):
        """Test nan is assigned for missing link between pfs and tpls."""
        pfs, sp = geolife_pfs_sp_long

        pfs, _ = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

        assert pd.isna(pfs["tripleg_id"]).any()

    def test_index_start(self, geolife_pfs_sp_long):
        """Test the generated index start from 0 for different cases."""
        pfs, sp = geolife_pfs_sp_long

        _, tpls_case1 = pfs.as_positionfixes.generate_triplegs(sp)
        _, tpls_case2 = pfs.drop("staypoint_id", axis=1).as_positionfixes.generate_triplegs(sp)

        assert (tpls_case1.index == np.arange(len(tpls_case1))).any()
        assert (tpls_case2.index == np.arange(len(tpls_case2))).any()

    def test_invalid_inputs(self, geolife_pfs_sp_long):
        """Test if AttributeError will be raised after invalid method input."""
        pfs, sp = geolife_pfs_sp_long

        with pytest.raises(AttributeError, match="Method unknown"):
            pfs.as_positionfixes.generate_triplegs(sp, method="random")
        with pytest.raises(AttributeError, match="Method unknown"):
            pfs.as_positionfixes.generate_triplegs(sp, method=12345)

    def test_temporal(self, geolife_pfs_sp_long):
        """Test if the tpls generation result follows predefined gap_threshold."""
        pfs_input, sp = geolife_pfs_sp_long

        gap_threshold_ls = [0.1, 0.2, 1, 2]
        for gap_threshold in gap_threshold_ls:
            pfs, _ = pfs_input.as_positionfixes.generate_triplegs(sp, gap_threshold=gap_threshold)

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
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)

        sp = sp[["started_at", "finished_at", "user_id"]]
        tpls = tpls[["started_at", "finished_at", "user_id"]]
        sp_tpls = sp.append(tpls)
        sp_tpls.sort_values(by=["user_id", "started_at"], inplace=True)

        for user_id_this in sp["user_id"].unique():
            sp_tpls_this = sp_tpls[sp_tpls["user_id"] == user_id_this]
            diff = sp_tpls_this["started_at"] - sp_tpls_this["finished_at"].shift(1)
            # transform to numpy array and drop first values (always nan due to shift operation)
            diff = diff.values[1:]

            # all values have to greater or equal to zero. Otherwise there is an overlap
            assert all(diff >= np.timedelta64(datetime.timedelta()))
