import os
import pytest
import numpy as np

from shapely.geometry import LineString

import trackintel as ti


@pytest.fixture
def testdata_sp():
    """Read sp test data from files."""
    sp_file = os.path.join("tests", "data", "staypoints.csv")
    sp = ti.read_staypoints_csv(sp_file, sep=";", index_col="id")
    return sp


class TestStaypoints:
    """Tests for the StaypointsAccessor."""

    def test_accessor_columns(self, testdata_sp):
        """Test if the as_staypoints accessor checks the required column for staypoints."""
        sp = testdata_sp.copy()
        assert sp.as_staypoints

        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of staypoints"):
            sp.drop(["user_id"], axis=1).as_staypoints

    def test_accessor_geometry(self, testdata_sp):
        """Test if the as_staypoints accessor requires geometry column."""
        sp = testdata_sp.copy()

        # geometery
        with pytest.raises(AttributeError, match="No geometry data set yet"):
            sp.drop(["geom"], axis=1).as_staypoints

    def test_accessor_geometry_type(self, testdata_sp):
        """Test if the as_staypoints accessor requires Point geometry."""
        sp = testdata_sp.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a Point"):
            sp["geom"] = LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])
            sp.as_staypoints

    def test_staypoints_center(self, testdata_sp):
        """Check if sp has center method and returns (lat, lon) pairs as geometry."""
        sp = testdata_sp.copy()
        assert len(sp.as_staypoints.center) == 2
