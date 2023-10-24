import os

import pytest
import geopandas as gpd
from shapely.geometry import LineString

import trackintel as ti
from trackintel import Locations


@pytest.fixture
def testdata_locs():
    """Read location test data from files."""
    sp_file = os.path.join("tests", "data", "geolife", "geolife_staypoints.csv")
    sp = ti.read_staypoints_csv(sp_file, tz="utc", index_col="id", crs="epsg:4326")
    sp, locs = sp.as_staypoints.generate_locations(
        method="dbscan", epsilon=10, num_samples=1, distance_metric="haversine", agg_level="dataset"
    )
    locs.as_locations
    # we want test the accessor of GeoDataFrames and not Locations
    return gpd.GeoDataFrame(locs)


class TestLocations:
    """Tests for the Locations class."""

    def test_accessor_column(self, testdata_locs):
        """Test if the as_locations accessor checks the required column for locations."""
        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of locations"):
            testdata_locs.drop(["user_id"], axis=1).as_locations

    def test_accessor_geometry_type(self, testdata_locs):
        """Test if the as_locations accessor requires Point geometry."""
        testdata_locs["center"] = LineString(
            [(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)]
        )
        with pytest.raises(ValueError, match="The center geometry must be a Point"):
            testdata_locs.as_locations

    def test_accessor_empty(self, testdata_locs):
        """Test if as_locations accessor raises error if data is empty."""
        with pytest.raises(ValueError, match="GeoDataFrame is empty with shape:"):
            testdata_locs.drop(testdata_locs.index).as_locations

    def test_accessor_recursive(self, testdata_locs):
        """Test if as_locations works recursivly"""
        locs = testdata_locs.as_locations
        assert type(locs) is Locations
        assert id(locs) == id(locs.as_locations)
