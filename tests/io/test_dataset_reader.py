import datetime
import os

import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import Point

import trackintel as ti
from trackintel.io.dataset_reader import _get_df, _get_labels, geolife_add_modes_to_triplegs, read_geolife


@pytest.fixture
def read_geolife_modes():
    return read_geolife(os.path.join("tests", "data", "geolife_modes"))


@pytest.fixture
def read_geolife_triplegs_with_modes(read_geolife_modes):
    pfs, labels = read_geolife_modes
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

    return tpls, labels


@pytest.fixture
def matching_data():
    """Test data for tripleg mode matching.

    There are two labels given:
        Tripleg_0 overlaps from the left and is almost fully included in label_0
        Tripleg_1 is fully included in label_0
        Tripleg_2 overlaps and extents to the right but is almost not covered by label_0
        Tripleg_3 overlaps label_1 to the right and the left but is almost fully covered by it.
    """
    one_hour = pd.Timedelta("1h")
    one_min = pd.Timedelta("1min")
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
    triplegs["user_id"] = 0
    labels_raw = pd.DataFrame(labels_raw).set_index("id")

    return triplegs, labels_raw


@pytest.fixture()
def impossible_matching_data():
    """Test data for tripleg mode matching where the labels and the tracking data are really far apart."""
    one_hour = datetime.timedelta(hours=1)
    one_min = datetime.timedelta(minutes=1)
    time_1 = pd.Timestamp("1970-01-01", tz="utc")
    time_2 = pd.Timestamp("1980-01-01", tz="utc")

    triplegs = [{"id": 0, "started_at": time_1, "finished_at": time_1 + one_hour}]
    labels_raw = [
        {"id": 0, "started_at": time_2 + one_min, "finished_at": time_2 + 4 * one_hour + one_min, "mode": "walk"}
    ]

    triplegs = pd.DataFrame(triplegs).set_index("id")
    triplegs["user_id"] = 0
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

        assert_geodataframe_equal(pfs, pfs_reRead, check_like=True)

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

    def test_wrong_folder_name(self):
        """Check if invalid folder names raise an exception."""
        geolife_path = os.path.join("tests", "data", "geolife")
        temp_dir = os.path.join(geolife_path, "123 - invalid folder ()%")
        os.mkdir(temp_dir)

        try:
            with pytest.raises(ValueError):
                read_geolife(geolife_path)
        finally:
            os.rmdir(temp_dir)

    def test_no_user_folders(self):
        """Check if no user folders raise an exception."""
        geolife_path = os.path.join("tests", "data", "geolife", "000", "Trajectory")
        with pytest.raises(FileNotFoundError):
            read_geolife(geolife_path)


class Test_GetLabels:
    def test_example_data(self):
        """Read example data and test if it is valid."""
        geolife_path = os.path.join("tests", "data", "geolife_modes")
        uids = ["010", "020", "178"]
        labels = _get_labels(geolife_path, uids)

        assert len(labels) == 2
        assert all(key in [10, 20] for key in labels.keys())
        assert len(labels[10]) == 434
        assert len(labels[20]) == 223
        assert all(df.columns.tolist() == ["started_at", "finished_at", "mode"] for df in labels.values())


class Test_GetDf:
    def test_example_data(self):
        """Read example data and test if it is valid."""
        geolife_path = os.path.join("tests", "data", "geolife_modes")
        uids = ["010", "020", "178"]
        df_gen = _get_df(geolife_path, uids, False)

        s = 0
        df_lengths = [681, 818, 915, 1004, 66, 327, 256, 66, 84]
        columns = ["elevation", "tracked_at", "geom", "user_id"]
        for df in df_gen:
            assert len(df) in df_lengths
            assert df.columns.tolist() == columns
            assert isinstance(df["geom"][0], Point)
            s += 1
        assert s == len(df_lengths)


class TestGeolife_add_modes_to_triplegs:
    def test_duplicate_matching(self, matching_data):
        """Check each tripleg will only receive one largest overlapping ration mode label."""
        triplegs, labels_raw = matching_data

        # add one record to the labels_raw: mode bus which are 1 min shorter than mode bike
        labels_raw = (
            labels_raw.reset_index()
            .append(
                [
                    {
                        "id": 2,
                        # this record started 1 minute later than id=1 record - the final match ratio will be lower
                        "started_at": labels_raw.iloc[-1]["started_at"] + datetime.timedelta(minutes=1),
                        "finished_at": labels_raw.iloc[-1]["finished_at"],
                        "mode": "bus",
                    },
                ]
            )
            .set_index("id")
        )
        labels = {0: labels_raw}

        # the correct behaviour is to only choose one mode per tripleg id based on overlapping ratio
        # in this case choose mode bike
        tpls = geolife_add_modes_to_triplegs(triplegs, labels)

        # only one mode per tripleg should be assigned
        assert len(triplegs) == len(tpls)
        assert tpls.iloc[-1]["mode"] == "bike"

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

    def test_no_overlap(self, read_geolife_triplegs_with_modes):
        """Test that overlapping labels are not causing duplicate ids.
        user 10 was modified to have several overlapping labels"""

        tpls, labels = read_geolife_triplegs_with_modes
        tpls = geolife_add_modes_to_triplegs(tpls, labels)

        assert tpls.index.is_unique
        assert not tpls.duplicated(subset=["started_at", "finished_at"]).any()

    def test_mode_matching(self, matching_data):
        # bring label data into right format.
        tpls, labels_raw = matching_data
        labels = {0: labels_raw}

        tpls = geolife_add_modes_to_triplegs(tpls, labels)

        assert tpls.loc[0, "mode"] == "walk" and tpls.loc[0, "label_id"] == 0
        assert tpls.loc[1, "mode"] == "walk" and tpls.loc[1, "label_id"] == 0
        assert pd.isna(tpls.loc[2, "mode"]) and pd.isna(tpls.loc[2, "label_id"])
        assert tpls.loc[3, "mode"] == "bike" and tpls.loc[3, "label_id"] == 1

    def test_mode_matching_multi_user(self, matching_data):
        tpls, labels_raw = matching_data
        labels = {0: labels_raw}
        # explicitly change the user_id of the second record
        tpls.loc[1, "user_id"] = 1

        tpls = geolife_add_modes_to_triplegs(tpls, labels)

        assert tpls.loc[0, "mode"] == "walk" and tpls.loc[0, "label_id"] == 0
        assert pd.isna(tpls.loc[1, "mode"]) and pd.isna(tpls.loc[1, "label_id"])
        assert pd.isna(tpls.loc[2, "mode"]) and pd.isna(tpls.loc[2, "label_id"])
        assert tpls.loc[3, "mode"] == "bike" and tpls.loc[3, "label_id"] == 1

    def test_impossible_matching(self, impossible_matching_data):
        # bring label data into right format.
        tpls, labels_raw = impossible_matching_data
        labels = {0: labels_raw}

        tpls = geolife_add_modes_to_triplegs(tpls, labels)
        assert pd.isna(tpls.iloc[0]["mode"])
