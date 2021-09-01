import os

import numpy as np
import pandas as pd
import pytest

import trackintel as ti
from trackintel.analysis.labelling import _check_categories


class TestCreate_activity_flag:
    """Tests for create_activity_flag() method."""

    def test_create_activity_flag(self):
        """Test if 'activity' = True is assigned to staypoints."""
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp_test = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")

        activity_true = sp_test["activity"].copy()
        sp_test["activity"] = False

        sp_test = sp_test.as_staypoints.create_activity_flag(method="time_threshold", time_threshold=5.0)

        pd.testing.assert_series_equal(sp_test["activity"], activity_true)

    def test_wrong_input_parameter(self):
        """Test if an error will be raised when input wrong method."""
        sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        sp_test = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id")

        method = 12345
        with pytest.raises(AttributeError, match=f"Method {method} not known for creating activity flag."):
            sp_test.as_staypoints.create_activity_flag(method=method)

        method = "random"
        with pytest.raises(AttributeError, match=f"Method {method} not known for creating activity flag."):
            sp_test.as_staypoints.create_activity_flag(method=method)


class TestPredict_transport_mode:
    """Tests for predict_transport_mode() method."""

    def test_wrong_input_parameter(self):
        """Test if an error will be raised when input wrong method."""
        tpls_file = os.path.join("tests", "data", "triplegs_transport_mode_identification.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id")

        method = 12345
        with pytest.raises(AttributeError, match=f"Method {method} not known for predicting tripleg transport modes."):
            tpls.as_triplegs.predict_transport_mode(method=method)

        method = "random"
        with pytest.raises(AttributeError, match=f"Method {method} not known for predicting tripleg transport modes."):
            tpls.as_triplegs.predict_transport_mode(method=method)

    def test_check_empty_dataframe(self):
        """Assert that the method does not work for empty DataFrames."""
        tpls_file = os.path.join("tests", "data", "triplegs_transport_mode_identification.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id")
        empty_frame = tpls[0:0]
        with pytest.raises(AssertionError):
            empty_frame.as_triplegs.predict_transport_mode(method="simple-coarse")

    def test_simple_coarse_identification_no_crs(self):
        """
        Assert that the simple-coarse transport mode identification throws the correct
        warning and and yields the correct results for WGS84.
        """
        tpls_file = os.path.join("tests", "data", "triplegs_transport_mode_identification.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id")

        with pytest.warns(
            UserWarning,
            match="The CRS of your data is not defined.",
        ):
            tpls_transport_mode = tpls.as_triplegs.predict_transport_mode(method="simple-coarse")

        assert tpls_transport_mode.iloc[0]["mode"] == "slow_mobility"
        assert tpls_transport_mode.iloc[1]["mode"] == "motorized_mobility"
        assert tpls_transport_mode.iloc[2]["mode"] == "fast_mobility"

    def test_simple_coarse_identification_wgs_84(self):
        """Asserts the correct behaviour with data in wgs84."""
        tpls_file = os.path.join("tests", "data", "triplegs_transport_mode_identification.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id")
        tpls_2 = tpls.set_crs(epsg=4326)
        tpls_transport_mode_2 = tpls_2.as_triplegs.predict_transport_mode(method="simple-coarse")

        assert tpls_transport_mode_2.iloc[0]["mode"] == "slow_mobility"
        assert tpls_transport_mode_2.iloc[1]["mode"] == "motorized_mobility"
        assert tpls_transport_mode_2.iloc[2]["mode"] == "fast_mobility"

    def test_simple_coarse_identification_projected(self):
        """Asserts the correct behaviour with data in projected coordinate systems."""
        tpls_file = os.path.join("tests", "data", "triplegs_transport_mode_identification.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id")
        tpls_2 = tpls.set_crs(epsg=4326)
        tpls_3 = tpls_2.to_crs(epsg=2056)
        tpls_transport_mode_3 = tpls_3.as_triplegs.predict_transport_mode(method="simple-coarse")
        assert tpls_transport_mode_3.iloc[0]["mode"] == "slow_mobility"
        assert tpls_transport_mode_3.iloc[1]["mode"] == "motorized_mobility"
        assert tpls_transport_mode_3.iloc[2]["mode"] == "fast_mobility"

    def test_check_categories(self):
        """Asserts the correct identification of valid category dictionaries."""
        tpls_file = os.path.join("tests", "data", "triplegs_transport_mode_identification.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id")
        correct_dict = {2: "cat1", 7: "cat2", np.inf: "cat3"}

        assert _check_categories(correct_dict)
        with pytest.raises(ValueError):
            incorrect_dict = {10: "cat1", 5: "cat2", np.inf: "cat3"}
            tpls.as_triplegs.predict_transport_mode(method="simple-coarse", categories=incorrect_dict)
