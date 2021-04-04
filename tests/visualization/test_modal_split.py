import datetime
import os

import numpy as np
import pandas as pd
import pytest

from trackintel.analysis.modal_split import calculate_modal_split
from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs
from trackintel.visualization.modal_split import plot_modal_split


@pytest.fixture
def get_geolife_triplegs_with_modes():
    """Get modal split for a small part of the geolife dataset."""
    pfs, labels = read_geolife(os.path.join("tests", "data", "geolife_modes"))
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

    tpls_with_modes = geolife_add_modes_to_triplegs(tpls, labels)
    return tpls_with_modes


@pytest.fixture
def get_test_triplegs_with_modes():
    """Get modal split for randomly generated data."""
    n = 200
    day_1_h1 = pd.Timestamp("1970-01-01 00:00:00", tz="utc")
    one_day = datetime.timedelta(days=1)
    mode_list = ["car", "walk", "train", "bus", "bike", "walk", "bike"]
    df = pd.DataFrame(index=np.arange(n))
    df["mode"] = np.random.choice(mode_list, n)
    df["user_id"] = np.random.randint(1, 5, size=n)
    df["started_at"] = np.random.randint(1, 30, size=n) * one_day
    df["started_at"] = df["started_at"] + day_1_h1

    return df


class TestPlot_modal_split:
    def test_create_plot_geolife(self, get_geolife_triplegs_with_modes):
        """Check if we can run the plot function with geolife data without error"""
        modal_split = calculate_modal_split(get_geolife_triplegs_with_modes, freq="d", per_user=False)
        plot_modal_split(modal_split)
        assert True

    def test_check_dtype_error(self, get_geolife_triplegs_with_modes):
        """Check if error is thrown correctly when index is not datetime

        freq=None calculates the modal split over the whole period
        """
        modal_split = calculate_modal_split(get_geolife_triplegs_with_modes, freq=None, per_user=False)
        with pytest.raises(ValueError):
            plot_modal_split(modal_split)
        assert True

    def test_multi_user_error(self, get_test_triplegs_with_modes):
        """Create a modal split plot based on randomly generated test data"""
        modal_split = calculate_modal_split(get_test_triplegs_with_modes, freq="d", per_user=True, norm=True)
        with pytest.raises(ValueError):
            plot_modal_split(modal_split)

        # make sure that there is no error if the data was correctly created
        modal_split = calculate_modal_split(get_test_triplegs_with_modes, freq="d", per_user=False, norm=True)
        plot_modal_split(modal_split)

        assert True

    def test_create_plot_testdata(self, get_test_triplegs_with_modes):
        """Create a modal split plot based on randomly generated test data"""
        tmp_file = os.path.join("tests", "data", "modal_split_plot.png")

        modal_split = calculate_modal_split(get_test_triplegs_with_modes, freq="d", per_user=False, norm=True)

        modal_split = modal_split[["walk", "bike", "train", "car", "bus"]]  # change order for the looks of the plot
        plot_modal_split(
            modal_split, out_path=tmp_file, date_fmt_x_axis="%d", y_label="Percentage of daily count", x_label="days"
        )

        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
        os.remove(tmp_file.replace("png", "pdf"))
