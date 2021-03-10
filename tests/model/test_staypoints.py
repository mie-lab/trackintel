import os
import pytest

import trackintel as ti


class TestStaypoints:
    def test_as_staypoints_accessor(self):
        stps_file = os.path.join("tests", "data", "staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id")
        assert stps.as_staypoints

        stps = stps.drop(["geom"], axis=1)
        with pytest.raises(AttributeError):
            stps.as_staypoints

    def test_staypoints_center(self):
        stps_file = os.path.join("tests", "data", "staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id")
        # check if stps has methods from gpd and contains (lat, lon) pairs as geometry
        assert len(stps.as_staypoints.center) == 2
