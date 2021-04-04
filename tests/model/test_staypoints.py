import os
import pytest
import numpy as np

from shapely.geometry import LineString

import trackintel as ti


@pytest.fixture
def testdata_stps():
    """Read stps test data from files."""
    stps_file = os.path.join("tests", "data", "staypoints.csv")
    stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id")
    return stps


class TestStaypoints:
    """Tests for the StaypointsAccessor."""

    def test_accessor_columns(self, testdata_stps):
        """Test if the as_staypoints accessor checks the required column for staypoints."""
        stps = testdata_stps.copy()
        assert stps.as_staypoints

        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of staypoints"):
            stps.drop(["user_id"], axis=1).as_staypoints

    def test_accessor_geometry(self, testdata_stps):
        """Test if the as_staypoints accessor requires geometry column."""
        stps = testdata_stps.copy()

        # geometery
        with pytest.raises(AttributeError, match="No geometry data set yet"):
            stps.drop(["geom"], axis=1).as_staypoints

    def test_accessor_geometry_type(self, testdata_stps):
        """Test if the as_staypoints accessor requires Point geometry."""
        stps = testdata_stps.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a Point"):
            stps["geom"] = LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])
            stps.as_staypoints

    def test_staypoints_center(self, testdata_stps):
        """Check if stps has center method and returns (lat, lon) pairs as geometry."""
        stps = testdata_stps
        assert len(stps.as_staypoints.center) == 2
