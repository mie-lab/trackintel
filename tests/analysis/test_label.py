import os
import pandas as pd

import trackintel as ti


class TestCreate_activity_flag:
    """Tests for create_activity_flag() method."""

    def test_create_activity_flag(self):
        """Test if 'activity' = True is assigned to staypoints."""
        stps_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
        stps_test = ti.read_staypoints_csv(stps_file, tz="utc", index_col="id")

        activity_true = stps_test["activity"].copy()
        stps_test["activity"] = False

        stps_test = stps_test.as_staypoints.create_activity_flag()

        pd.testing.assert_series_equal(stps_test["activity"], activity_true)
