import datetime
import os

import numpy as np
import pandas as pd
import pytest

from geopandas.testing import assert_geoseries_equal

import trackintel as ti
from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs


@pytest.fixture
def read_geolife_modes():
    return read_geolife(os.path.join("tests", "data", "geolife_modes"))


@pytest.fixture
def read_geolife_triplegs_with_modes(read_geolife_modes):
    pfs, labels = read_geolife_modes
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

    return tpls, labels


@pytest.fixture
def matching_data():
    """generate test data for tripleg mode matching

    There are two labels given:
        Tripleg_0 overlaps from the left and is almost fully included in label_0
        Tripleg_1 is fully included in label_0
        Tripleg_2 overlaps and extents to the right but is almost not covered by label_0
        Tripleg_3 overlaps label_1 to the right and the left but is almost fully covered by it.
    """
    one_hour = datetime.timedelta(hours=1)
    one_min = datetime.timedelta(minutes=1)
    time_1 = pd.Timestamp("1970-01-01", tz="utc")

    triplegs = [
        {"id": 0, "started_at": time_1, "finished_at": time_1 + one_hour},
        {"id": 1, "started_at": time_1 + 2 * one_hour, "finished_at": time_1 + 3 * one_hour},
        {"id": 2, "started_at": time_1 + 4 * one_hour, "finished_at": time_1 + 5 * one_hour},
        {"id": 3, "started_at": time_1 + 6 * one_hour - one_min, "finished_at": time_1 + 7 * one_hour + one_min},
    ]

    labels_raw = [
        {"id": 0, "started_at": time_1 + one_min, "finished_at": time_1 + 4 * one_hour + one_min, "mode": "walk"},
        {"id": 1, "started_at": time_1 + 6 * one_hour, "finished_at": time_1 + 7 * one_hour, "mode": "bike"},
    ]

    triplegs = pd.DataFrame(triplegs).set_index("id")
    labels_raw = pd.DataFrame(labels_raw).set_index("id")

    return triplegs, labels_raw


@pytest.fixture()
def impossible_matching_data():
    """
    generate test data for tripleg mode matching where the labels and the tracking data are really far apart

    """

    one_hour = datetime.timedelta(hours=1)
    one_min = datetime.timedelta(minutes=1)
    time_1 = pd.Timestamp("1970-01-01", tz="utc")
    time_2 = pd.Timestamp("1980-01-01", tz="utc")

    triplegs = [{"id": 0, "started_at": time_1, "finished_at": time_1 + one_hour}]
    labels_raw = [
        {"id": 0, "started_at": time_2 + one_min, "finished_at": time_2 + 4 * one_hour + one_min, "mode": "walk"}
    ]

    triplegs = pd.DataFrame(triplegs).set_index("id")
    labels_raw = pd.DataFrame(labels_raw).set_index("id")

    return triplegs, labels_raw


class TestReadGeolife:
    def test_loop_read(self):
        """Use read_geolife reader, store posfix as .csv, load them again."""
        pfs, _ = read_geolife(os.path.join("tests", "data", "geolife"), print_progress=True)

        saved_file = os.path.join("tests", "data", "positionfixes_test.csv")
        pfs.as_positionfixes.to_csv(saved_file)
        pfs_reRead = ti.read_positionfixes_csv(saved_file, index_col="id", crs="epsg:4326")
        os.remove(saved_file)

        assert_geoseries_equal(pfs.geometry, pfs_reRead.geometry)

    def test_print_progress_flag(self, capsys):
        """Test if the print_progress bar controls the printing behavior."""
        g_path = os.path.join("tests", "data", "geolife")
        read_geolife(g_path, print_progress=True)
        captured_print = capsys.readouterr()
        assert captured_print.err != ""

        read_geolife(g_path, print_progress=False)
        captured_noprint = capsys.readouterr()
        assert captured_noprint.err == ""

        assert True

    def test_label_reading(self):
        """Test data types of the labels returned by read_geolife."""
        _, labels = read_geolife(os.path.join("tests", "data", "geolife_modes"))
        # the output is a dictionary
        assert isinstance(labels, dict)

        # it has the keys of the users 10 and 20, the values are pandas dataframes
        for key, value in labels.items():
            assert key in [10, 20, 178]
            assert isinstance(value, pd.DataFrame)

    def test_unavailable_label_reading(self):
        """Test data types of the labels returned by read_geolife from a dictionary without label files."""
        _, labels = read_geolife(os.path.join("tests", "data", "geolife_long"))

        # the output is a dictionary
        assert isinstance(labels, dict)

        # the values are pandas dataframes
        for key, value in labels.items():
            assert isinstance(value, pd.DataFrame)

    def test_wrong_folder_name(self):
        """Check if invalid folder names raise an exception."""
        geolife_path = os.path.join("tests", "data", "geolife")
        temp_dir = os.path.join(geolife_path, "123 - invalid folder ()%")
        os.mkdir(temp_dir)

        try:
            with pytest.raises(ValueError):
                _, _ = read_geolife(geolife_path)
        finally:
            os.rmdir(temp_dir)


class TestGeolife_add_modes_to_triplegs:
    def test_geolife_mode_matching(self, read_geolife_triplegs_with_modes):
        """Test that the matching runs with geolife.
        We only check that there are nan's and non nan's in the results."""

        tpls, labels = read_geolife_triplegs_with_modes
        tpls = geolife_add_modes_to_triplegs(tpls, labels)

        assert pd.isna(tpls["mode"]).any()
        assert (~pd.isna(tpls["mode"])).any()
        assert pd.isna(tpls["label_id"]).any()
        assert (~pd.isna(tpls["label_id"])).any()

        assert "started_at_s" not in tpls.columns

    def test_mode_matching(self, matching_data):
        # bring label data into right format. All labels belong to the same user
        tpls, labels_raw = matching_data
        tpls["user_id"] = 0
        labels = {0: labels_raw}

        tpls = geolife_add_modes_to_triplegs(tpls, labels)

        assert tpls.loc[0, "mode"] == "walk" and tpls.loc[0, "label_id"] == 0
        assert tpls.loc[1, "mode"] == "walk" and tpls.loc[1, "label_id"] == 0
        assert pd.isna(tpls.loc[2, "mode"]) and pd.isna(tpls.loc[2, "label_id"])
        assert tpls.loc[3, "mode"] == "bike" and tpls.loc[3, "label_id"] == 1

    def test_mode_matching_multi_user(self, matching_data):
        # bring label data into right format. All labels belong to the same user but we add an empty DataFrame with
        # labels in the end

        tpls, labels_raw = matching_data
        tpls["user_id"] = 0
        labels = {0: labels_raw, 1: pd.DataFrame(columns=labels_raw.columns)}

        tpls.loc[1, "user_id"] = 1

        tpls = geolife_add_modes_to_triplegs(tpls, labels)

        assert tpls.loc[0, "mode"] == "walk" and tpls.loc[0, "label_id"] == 0
        assert pd.isna(tpls.loc[1, "mode"]) and pd.isna(tpls.loc[1, "label_id"])
        assert pd.isna(tpls.loc[2, "mode"]) and pd.isna(tpls.loc[2, "label_id"])
        assert tpls.loc[3, "mode"] == "bike" and tpls.loc[3, "label_id"] == 1

    def test_impossible_matching(self, impossible_matching_data):
        # bring label data into right format. All labels belong to the same user
        tpls, labels_raw = impossible_matching_data
        tpls["user_id"] = 0
        labels = {0: labels_raw}

        tpls = geolife_add_modes_to_triplegs(tpls, labels)
        assert pd.isna(tpls.iloc[0]["mode"])
