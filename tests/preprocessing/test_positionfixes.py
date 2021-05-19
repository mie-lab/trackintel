import datetime
import os
import sys

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import numpy as np
import pandas as pd
import pytest

import trackintel as ti


@pytest.fixture
def geolife_pfs_stps_long():
    """Read geolife_long and generate stps."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    return pfs, stps


class TestGenerate_staypoints:
    """Tests for generate_staypoints() method."""

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
        pfs, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=0, time_threshold=0, include_last=True
        )
        assert len(stps) == len(pfs)

    def test_sliding_max(self):
        """Test if using large thresholds, stp extraction yield no pfs."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        _, stps = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=sys.maxsize, time_threshold=sys.maxsize
        )
        assert len(stps) == 0

    def test_missing_link(self):
        """Test nan is assigned for missing link between pfs and stps."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs, _ = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=sys.maxsize, time_threshold=sys.maxsize
        )

        assert pd.isna(pfs["staypoint_id"]).any()

    def test_dtype_consistent(self, geolife_pfs_stps_long):
        """Test the dtypes for the generated columns."""
        pfs, stps = geolife_pfs_stps_long

        assert pfs["user_id"].dtype == stps["user_id"].dtype
        assert pfs["staypoint_id"].dtype == "Int64"
        assert stps.index.dtype == "int64"

    def test_index_start(self, geolife_pfs_stps_long):
        """Test the generated index start from 0 for different methods."""
        _, stps = geolife_pfs_stps_long

        assert (stps.index == np.arange(len(stps))).any()

    def test_include_last(self):
        """Test if the include_last arguement will include the last pfs as stp."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))

        pfs_wo, stps_wo = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=False
        )
        pfs_include, stps_include = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5.0, include_last=True
        )
        # stps_wo does not include the last staypoint
        assert len(stps_wo) == len(stps_include) - 1
        # the last pfs of pfs_include has stp connection
        assert not pfs_include.tail(1)["staypoint_id"].isna().all()
        assert pfs_wo.tail(1)["staypoint_id"].isna().all()

    def test_print_progress(self):
        """Test if the result from print progress agrees with the original."""
        pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
        pfs_ori, stps_ori = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5
        )
        pfs_print, stps_print = pfs.as_positionfixes.generate_staypoints(
            method="sliding", dist_threshold=100, time_threshold=5, print_progress=True
        )
        assert_geodataframe_equal(pfs_ori, pfs_print)
        assert_geodataframe_equal(stps_ori, stps_print)

    def test_temporal(self):
        """Test if the stps generation result follows predefined time_threshold and gap_threshold."""
        pfs_input, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife_long"))

        # the duration should be not longer than time_threshold
        time_threshold_ls = [3, 5, 10]
        for time_threshold in time_threshold_ls:
            _, stps = pfs_input.as_positionfixes.generate_staypoints(
                method="sliding", dist_threshold=50, time_threshold=time_threshold
            )

            duration_stps_min = (stps["finished_at"] - stps["started_at"]).dt.total_seconds() / 60
            # all durations should be longer than the time_threshold
            assert (duration_stps_min > time_threshold).all()

        # the missing time should not exceed gap_threshold
        gap_threshold_ls = [10, 15, 20]
        for gap_threshold in gap_threshold_ls:
            pfs, _ = pfs_input.as_positionfixes.generate_staypoints(
                method="sliding", dist_threshold=50, time_threshold=time_threshold, gap_threshold=gap_threshold
            )
            # get the difference between pfs tracking time, and assign back to the previous pfs
            pfs["diff"] = ((pfs["tracked_at"] - pfs["tracked_at"].shift(1)).dt.total_seconds() / 60).shift(-1)
            # get the last pf of stps and check the gap size
            pfs.dropna(subset=["staypoint_id"], inplace=True)
            pfs.drop_duplicates(subset=["staypoint_id"], keep="last", inplace=True)
            # all last pfs should be shorter than the gap_threshold
            assert (pfs["diff"] < gap_threshold).all()


class TestGenerate_triplegs:
    """Tests for generate_triplegs() method."""

    def test_duplicate_columns(self, geolife_pfs_stps_long):
        """Test if running the function twice, the generated column does not yield exception in join statement"""

        # we run generate_triplegs twice in order to check that the extra column (tripleg_id) does
        # not cause any problems in the second run
        pfs, stps = geolife_pfs_stps_long

        pfs_run_1, _ = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        pfs_run_2, _ = pfs_run_1.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        assert set(pfs_run_1.columns) == set(pfs_run_2.columns)

    def test_user_without_stps(self, geolife_pfs_stps_long):
        """Check if it is safe to have users that have pfs but no stps."""
        pfs, stps = geolife_pfs_stps_long
        # test for case 1
        # manually change the first pfs' user_id, which has no stp correspondence
        pfs.loc[0, "user_id"] = 5000
        _, tpls_1 = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        # result should be the same ommiting the first row
        _, tpls_2 = pfs.iloc[1:].as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        assert_geodataframe_equal(tpls_1, tpls_2)

        # test for case 2
        pfs.drop(columns="staypoint_id", inplace=True)
        # manually change the first pfs' user_id, which has no stp correspondence
        _, tpls_1 = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        # result should be the same ommiting the first row
        _, tpls_2 = pfs.iloc[1:].as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        assert_geodataframe_equal(tpls_1, tpls_2)

    def test_pfs_without_stps(self, geolife_pfs_stps_long):
        """Delete pfs that belong to staypoints and see if they are detected."""
        pfs, stps = geolife_pfs_stps_long

        _, tpls_case1 = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        # only keep pfs where staypoint id is nan
        pfs_nostps = pfs[pd.isna(pfs["staypoint_id"])].drop(columns="staypoint_id")
        print(pfs_nostps)
        _, tpls_case2 = pfs_nostps.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

        assert_geodataframe_equal(tpls_case1, tpls_case2)

    def test_stability(self, geolife_pfs_stps_long):
        """Checks if the results are same for different cases in tripleg_generation method."""
        pfs, stps = geolife_pfs_stps_long
        # case 1
        pfs_case1, tpls_case1 = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")
        # case 1 without stps
        pfs_case1_wo, tpls_case1_wo = pfs.as_positionfixes.generate_triplegs(method="between_staypoints")

        # case 2
        pfs = pfs.drop(columns="staypoint_id")
        pfs_case2, tpls_case2 = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

        assert_geodataframe_equal(pfs_case1.drop(columns="staypoint_id", axis=1), pfs_case2)
        assert_geodataframe_equal(pfs_case1, pfs_case1_wo)
        assert_geodataframe_equal(tpls_case1, tpls_case2)
        assert_geodataframe_equal(tpls_case1, tpls_case1_wo)

    def test_random_order(self, geolife_pfs_stps_long):
        """Checks if same tpls will be generated after random shuffling pfs."""
        pfs, stps = geolife_pfs_stps_long
        # ensure proper order of pfs
        pfs.sort_values(by=["user_id", "tracked_at"], inplace=True)

        # original order
        pfs_ori, tpls_ori = pfs.as_positionfixes.generate_triplegs(stps)

        # resample/shuffle pfs
        pfs_shuffle = pfs.sample(frac=1, random_state=0)
        pfs_shuffle, tpls_shuffle = pfs_shuffle.as_positionfixes.generate_triplegs(stps)

        # order should be the same -> pfs.sort_values within function
        # generated tpls index should be the same
        assert_geodataframe_equal(pfs_ori, pfs_shuffle)
        assert_geodataframe_equal(tpls_ori, tpls_shuffle)

    def test_pfs_index(self, geolife_pfs_stps_long):
        """Checks if same tpls will be generated after changing pfs index."""
        pfs, stps = geolife_pfs_stps_long

        # original index
        pfs_ori, tpls_ori = pfs.as_positionfixes.generate_triplegs(stps)

        # create discontinues index
        pfs.index = np.arange(len(pfs)) * 2
        pfs_index, tpls_index = pfs.as_positionfixes.generate_triplegs(stps)

        # generated tpls index should be the same
        assert_geodataframe_equal(pfs_ori.reset_index(drop=True), pfs_index.reset_index(drop=True))
        assert_geodataframe_equal(tpls_ori, tpls_index)

    def test_dtype_consistent(self, geolife_pfs_stps_long):
        """Test the dtypes for the generated columns."""
        pfs, stps = geolife_pfs_stps_long
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)
        assert pfs["user_id"].dtype == tpls["user_id"].dtype
        assert pfs["tripleg_id"].dtype == "Int64"
        assert tpls.index.dtype == "int64"

    def test_missing_link(self, geolife_pfs_stps_long):
        """Test nan is assigned for missing link between pfs and tpls."""
        pfs, stps = geolife_pfs_stps_long

        pfs, _ = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

        assert pd.isna(pfs["tripleg_id"]).any()

    def test_index_start(self, geolife_pfs_stps_long):
        """Test the generated index start from 0 for different cases."""
        pfs, stps = geolife_pfs_stps_long

        _, tpls_case1 = pfs.as_positionfixes.generate_triplegs(stps)
        _, tpls_case2 = pfs.drop("staypoint_id", axis=1).as_positionfixes.generate_triplegs(stps)

        assert (tpls_case1.index == np.arange(len(tpls_case1))).any()
        assert (tpls_case2.index == np.arange(len(tpls_case2))).any()

    def test_invalid_inputs(self, geolife_pfs_stps_long):
        """Test if AttributeError will be raised after invalid method input."""
        pfs, stps = geolife_pfs_stps_long

        with pytest.raises(AttributeError, match="Method unknown"):
            pfs.as_positionfixes.generate_triplegs(stps, method="random")
        with pytest.raises(AttributeError, match="Method unknown"):
            pfs.as_positionfixes.generate_triplegs(stps, method=12345)

    def test_temporal(self, geolife_pfs_stps_long):
        """Test if the tpls generation result follows predefined gap_threshold."""
        pfs_input, stps = geolife_pfs_stps_long

        gap_threshold_ls = [0.1, 0.2, 1, 2]
        for gap_threshold in gap_threshold_ls:
            pfs, _ = pfs_input.as_positionfixes.generate_triplegs(stps, gap_threshold=gap_threshold)

            # conti_tpl checks whether the next pfs is in the same tpl
            pfs["conti_tpl"] = (pfs["tripleg_id"] - pfs["tripleg_id"].shift(1)).shift(-1)
            # get the time difference of pfs, and assign to the previous pfs
            pfs["diff"] = ((pfs["tracked_at"] - pfs["tracked_at"].shift(1)).dt.total_seconds() / 60).shift(-1)
            # we only take tpls that are splitted in the middle (tpl - tpl) and the second user
            pfs = pfs.loc[(pfs["conti_tpl"] == 1) & (pfs["user_id"] == 1)]
            # check if the cuts are appropriate
            assert (pfs["diff"] > gap_threshold).all()

    def test_stps_tpls_overlap(self, geolife_pfs_stps_long):
        """Tpls and stps should not overlap when generated using the default extract triplegs method."""
        pfs, stps = geolife_pfs_stps_long
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps)

        stps = stps[["started_at", "finished_at", "user_id"]]
        tpls = tpls[["started_at", "finished_at", "user_id"]]
        stps_tpls = stps.append(tpls)
        stps_tpls.sort_values(by=["user_id", "started_at"], inplace=True)

        for user_id_this in stps["user_id"].unique():
            stps_tpls_this = stps_tpls[stps_tpls["user_id"] == user_id_this]
            diff = stps_tpls_this["started_at"] - stps_tpls_this["finished_at"].shift(1)
            # transform to numpy array and drop first values (always nan due to shift operation)
            diff = diff.values[1:]

            # all values have to greater or equal to zero. Otherwise there is an overlap
            assert all(diff >= np.timedelta64(datetime.timedelta()))
